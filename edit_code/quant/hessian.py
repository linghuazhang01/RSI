"""Calibration pass on the CURRENT model (transformers): per-layer input Hessians + activation stats.

For each decoder layer and each distinct module input (attn_in, o_in, gdn_in, gdn_out_in, mlp_in,
down_in; see spec.INPUT_KEY) we accumulate H = sum_t x_t x_t^T (fp32) and per-channel max |x|,
over all calibration tokens. Hessians are written to <out>/L<layer>.<input_key>.safetensors
(fp32 `H`, `absmax`, scalar `n`) so GPTQ / AWQ variants can be (re)computed lazily later.
Memory: Qwen3.8-27B has ~1.6 GB of fp32 Hessians per layer (down_proj input is 17408-wide), so the
pass is windowed (`layers_per_pass`): each pass hooks a window of layers and stops the forward
after the window's last layer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load, save_file

from .spec import ModelSpec


class _StopForward(Exception):
    pass


def load_hf_model(model_dir: Path, device: str = "cpu", dtype=torch.bfloat16):
    """Load the checkpoint with the class named in config.json (Qwen3_5ForConditionalGeneration for
    Qwen3.8-27B). Only local files; no remote code."""
    import transformers
    cfg = json.load(open(Path(model_dir) / "config.json"))
    arch = (cfg.get("architectures") or ["AutoModelForCausalLM"])[0]
    cls = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
    model = cls.from_pretrained(str(model_dir), dtype=dtype, device_map=device if device != "cpu" else None,
                                local_files_only=True, trust_remote_code=False)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model


def tokenize_texts(model_dir: Path, texts: list[str], max_len: int) -> list[torch.Tensor]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        if len(ids) >= 2:
            out.append(torch.tensor([ids], dtype=torch.long))
    return out


def hessian_path(out_dir: Path, layer: int, input_key: str) -> Path:
    return Path(out_dir) / f"L{layer}.{input_key}.safetensors"


def load_hessian(out_dir: Path, layer: int, input_key: str, device="cpu") -> tuple[torch.Tensor, int, torch.Tensor]:
    # Read sequentially before strided access: remote mmap page faults are costly on JuiceFS.
    d = load(hessian_path(out_dir, layer, input_key).read_bytes())
    if str(device) != "cpu":
        d = {key: value.to(device) for key, value in d.items()}
    return d["H"], int(d["n"].item()), d["absmax"]


@torch.no_grad()
def collect_stats(model, spec: ModelSpec, batches: list[torch.Tensor], out_dir: Path, *,
                  layers_per_pass: int = 16, device: str | None = None, log=print) -> dict:
    """Run calibration batches through `model`, write Hessians, return activation statistics
    {"L.input_key": {...}}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or str(next(model.parameters()).device)
    reps: dict[int, dict[str, str]] = {}   # layer -> input_key -> representative module path
    for m in spec.modules:
        reps.setdefault(m.layer, {}).setdefault(m.input_key, m.module_path)
    stats = {}
    n_layers = spec.num_layers
    for start in range(0, n_layers, layers_per_pass):
        window = list(range(start, min(start + layers_per_pass, n_layers)))
        todo = [(l, k) for l in window for k in reps.get(l, {})
                if not hessian_path(out_dir, l, k).exists()]
        if not todo:
            continue
        acc = {}
        handles = []

        def make_hook(key):
            def hook(mod, args):
                x = args[0].detach()
                x = x.reshape(-1, x.shape[-1]).float()
                a = acc.get(key)
                if a is None:
                    a = acc[key] = {"H": torch.zeros(x.shape[1], x.shape[1], device=x.device),
                                    "absmax": torch.zeros(x.shape[1], device=x.device), "n": 0}
                a["H"].addmm_(x.t(), x)
                a["absmax"] = torch.maximum(a["absmax"], x.abs().amax(dim=0))
                a["n"] += x.shape[0]
            return hook

        for (l, k) in todo:
            mod = model.get_submodule(reps[l][k])
            handles.append(mod.register_forward_pre_hook(make_hook((l, k))))
        last = model.get_submodule(f"{spec.layers_prefix}.{window[-1]}")

        def stop(mod, args, out):
            raise _StopForward()
        handles.append(last.register_forward_hook(stop))
        t0 = time.time()
        try:
            for ids in batches:
                try:
                    model(input_ids=ids.to(device), use_cache=False)
                except _StopForward:
                    pass
        finally:
            for h in handles:
                h.remove()
        for (l, k), a in acc.items():
            save_file({"H": a["H"].cpu().contiguous(), "absmax": a["absmax"].cpu().contiguous(),
                       "n": torch.tensor([a["n"]], dtype=torch.int64)}, str(hessian_path(out_dir, l, k)))
        log(f"[hessian] layers {window[0]}-{window[-1]}: {len(acc)} inputs, {time.time() - t0:.1f}s")
        del acc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for l, ks in reps.items():
        for k in ks:
            H, n, absmax = load_hessian(out_dir, l, k)
            stats[f"{l}.{k}"] = activation_summary(H, n, absmax)
    return stats


def activation_summary(H: torch.Tensor, n: int, absmax: torch.Tensor, outlier_factor: float = 6.0) -> dict:
    rms = torch.sqrt(torch.clamp(torch.diag(H.float()) / max(n, 1), min=0))
    med = float(rms.median()) or 1e-12
    amax_med = float(absmax.float().median()) or 1e-12
    return {"tokens": int(n), "dim": int(rms.numel()), "rms_mean": float(rms.mean()), "rms_max": float(rms.max()),
            "rms_max_over_median": float(rms.max()) / med,
            "absmax": float(absmax.max()), "absmax_over_median": float(absmax.max()) / amax_med,
            "outlier_frac": float((rms > outlier_factor * med).float().mean())}
