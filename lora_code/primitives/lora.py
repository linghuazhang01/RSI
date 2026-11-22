"""LoRA representation, sandbox checks, PEFT save / load (vLLM-loadable) and merge into weights.

A LoRA is {short module name: (A [r, in], B [out, r])} with ΔW = B @ A (the PEFT scaling alpha/r is
folded into B; adapters are saved with lora_alpha == r so the scaling is exactly 1).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .naming import VLLM_LORA_LEAVES, parse_peft_key, parse_short, peft_key

VLLM_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


class LoRA:
    def __init__(self, factors: dict | None = None, meta: dict | None = None):
        self.factors = dict(factors or {})
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ basics
    def modules(self) -> list[str]:
        return sorted(self.factors, key=lambda s: (parse_short(s)[0], s))

    def rank(self, short: str) -> int:
        return int(self.factors[short][0].shape[0])

    def max_rank(self) -> int:
        return max((self.rank(m) for m in self.factors), default=0)

    def delta(self, short: str):
        A, B = self.factors[short]
        return B @ A

    def fro_norm(self, short: str) -> float:
        import torch
        A, B = self.factors[short]
        # ||BA||_F^2 = tr((B^T B)(A A^T))
        v = torch.trace((B.T @ B) @ (A @ A.T))
        return float(v.clamp_min(0).sqrt())

    def copy(self) -> "LoRA":
        return LoRA({k: (a.clone(), b.clone()) for k, (a, b) in self.factors.items()}, dict(self.meta))

    def add(self, short: str, A, B) -> None:
        """Append a low-rank term (concatenate ranks)."""
        import torch
        A, B = A.float().cpu(), B.float().cpu()
        if short in self.factors:
            A0, B0 = self.factors[short]
            self.factors[short] = (torch.cat([A0, A], 0), torch.cat([B0, B], 1))
        else:
            self.factors[short] = (A, B)

    def compact(self, short: str):
        """Exact thin SVD of ΔW = B A via two QR factorizations: returns U [out,r], S [r], V [in,r]."""
        import torch
        A, B = self.factors[short]
        Qb, Rb = torch.linalg.qr(B)
        Qa, Ra = torch.linalg.qr(A.T)
        u, s, vh = torch.linalg.svd(Rb @ Ra.T)
        return Qb @ u, s, Qa @ vh.T

    def set_from_svd(self, short: str, U, S, V, k: int | None = None, tol: float = 0.0) -> None:
        import torch
        if k is not None:
            U, S, V = U[:, :k], S[:k], V[:, :k]
        if tol > 0 and S.numel():
            keep = S > tol * S[0]
            U, S, V = U[:, keep], S[keep], V[:, keep]
        if S.numel() == 0:
            self.factors.pop(short, None)
            return
        r = torch.sqrt(S)
        self.factors[short] = ((V * r).T.contiguous(), (U * r).contiguous())

    def recompress(self, cap: int | None = None, tol: float = 1e-6) -> "LoRA":
        """Canonical form: each module re-factored by its thin SVD, ranks ≤ cap, near-zero directions dropped."""
        for m in list(self.factors):
            U, S, V = self.compact(m)
            self.set_from_svd(m, U, S, V, k=cap, tol=tol)
        return self

    def digest(self) -> str:
        h = hashlib.sha256()
        for m in self.modules():
            A, B = self.factors[m]
            h.update(m.encode())
            h.update(A.contiguous().float().numpy().tobytes())
            h.update(B.contiguous().float().numpy().tobytes())
        return h.hexdigest()[:16]

    def to_tensors(self) -> dict:
        out = {}
        for m, (A, B) in self.factors.items():
            out[f"{m}::A"] = A.contiguous()
            out[f"{m}::B"] = B.contiguous()
        return out

    @classmethod
    def from_tensors(cls, t: dict, meta=None) -> "LoRA":
        f = {}
        for k, v in t.items():
            m, which = k.split("::")
            a, b = f.get(m, (None, None))
            f[m] = (v.float(), b) if which == "A" else (a, v.float())
        return cls(f, meta)

    def save_raw(self, path: Path) -> None:
        from safetensors.torch import save_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.to_tensors(), str(path), metadata={"meta": json.dumps(self.meta, default=str)})

    @classmethod
    def load_raw(cls, path: Path) -> "LoRA":
        from safetensors import safe_open
        with safe_open(str(path), "pt") as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
            meta = json.loads((f.metadata() or {}).get("meta", "{}"))
        return cls.from_tensors(t, meta)


# ============================================================================ checks
def check_lora(lora: LoRA, store, *, rank_cap: int, tau: float, allowed_leaves=None) -> tuple[bool, dict]:
    """Sandbox checks: known modules, shapes, finite values, rank ≤ cap, ||ΔW||_F / ||W0||_F ≤ τ.
    Returns (ok, report) with per-module numbers (report['modules'][short] = {rank, rel_norm, norm})."""
    import torch
    allowed = set(allowed_leaves or VLLM_LORA_LEAVES)
    errors, mods = [], {}
    if not lora.factors:
        return True, {"errors": [], "modules": {}, "empty": True}
    for m, (A, B) in lora.factors.items():
        try:
            _, _, leaf = parse_short(m)
        except ValueError as e:
            errors.append(str(e))
            continue
        if leaf not in allowed:
            errors.append(f"{m}: module type {leaf} not allowed")
            continue
        if not store.has(m):
            errors.append(f"{m}: not in model")
            continue
        out_f, in_f = store.shape(m)
        if A.dim() != 2 or B.dim() != 2 or A.shape[1] != in_f or B.shape[0] != out_f or A.shape[0] != B.shape[1]:
            errors.append(f"{m}: bad shapes A{tuple(A.shape)} B{tuple(B.shape)} for W[{out_f},{in_f}]")
            continue
        if not (torch.isfinite(A).all() and torch.isfinite(B).all()):
            errors.append(f"{m}: non-finite values")
            continue
        r = A.shape[0]
        if r > rank_cap:
            errors.append(f"{m}: rank {r} > cap {rank_cap}")
        n = lora.fro_norm(m)
        rel = n / max(store.norm(m), 1e-12)
        if rel > tau:
            errors.append(f"{m}: ||dW||/||W0|| = {rel:.4f} > tau {tau}")
        mods[m] = {"rank": int(r), "norm": n, "rel_norm": rel}
    return not errors, {"errors": errors[:50], "n_errors": len(errors), "modules": mods}


# ============================================================================ cumulative adapter (merge.mode adapter)
def compose_cumulative(base, delta, budget: int | None = None, tol: float = 1e-6):
    """merge.mode adapter (2026-11-22): the adapter that serves M_0 + C + ΔW, i.e. the rank-concatenation of the
    cumulative LoRA C (None / empty = M_0) and the editor's delta ΔW. Exact (B_C A_C + B_Δ A_Δ, nothing rounded
    into bf16 weights). When a module's concatenated rank exceeds `budget` it is re-factored by its exact thin SVD
    (directions below tol·σ_max dropped) and, if still above the budget, truncated to the top `budget` singular
    directions; meta["recompressed"] records the modules and the dropped relative energy. The result is the very
    object that is registered on vLLM, gated, and (on accept) kept as the next C: gated = served.
    Never mutates base or delta. MockLoRA inputs give a MockLoRA (mock.m_stack)."""
    if getattr(delta, "kind", "lora") == "weights":
        raise TypeError("a weight edit cannot be composed into an adapter")
    if not hasattr(delta, "add"):               # MockLoRA (no factor tensors)
        from ..mock import m_stack
        return m_stack(base, delta, budget)
    out = base.copy() if base is not None else LoRA()
    for m in delta.modules():
        A, B = delta.factors[m]
        out.add(m, A, B)
    rec = {}
    if budget:
        import torch
        for m in out.modules():
            if out.rank(m) <= int(budget):
                continue
            U, S, V = out.compact(m)
            tot = float((S ** 2).sum())
            keep = S > tol * S[0] if S.numel() else S.bool()
            U, S, V = U[:, keep], S[keep], V[:, keep]
            k = min(int(budget), int(S.numel()))
            dropped = float((S[k:] ** 2).sum()) / max(tot, 1e-30)
            r = torch.sqrt(S[:k])
            out.factors[m] = ((V[:, :k] * r).T.contiguous(), (U[:, :k] * r).contiguous())
            rec[m] = {"rank": k, "dropped_energy": dropped}
    out.meta = {"cumulative": True, "components": list((base.meta if base is not None else {}).get("components", []))
                + [delta.meta.get("name") or delta.digest()]}
    if rec:
        out.meta["recompressed"] = {"modules": len(rec), "max_dropped_energy": max(v["dropped_energy"] for v in rec.values())}
    return out


def padded_rank(r: int) -> int:
    for v in VLLM_RANKS:
        if v >= r:
            return v
    return r


# ============================================================================ PEFT I/O
def save_peft(lora: LoRA, out_dir: Path, store, *, base_model: str = "", dtype: str = "bfloat16",
              pad_to_vllm_rank: bool = True) -> Path:
    """Write adapter_config.json + adapter_model.safetensors.

    - keys: base_model.model.<prefix>layers.<i>.<block>.<leaf>.lora_{A,B}.weight with prefix detected from
      the checkpoint (model.language_model. for Qwen3_5ForConditionalGeneration), i.e. unpacked HF names
      that vLLM packs itself (qkv_proj, gate_up_proj, in_proj_qkvz, in_proj_ba).
    - one global rank r (vLLM reads a single `r`): modules with smaller rank are zero-padded; r is rounded
      up to a vLLM-supported value; lora_alpha = r so PEFT/vLLM scaling is 1 (scale already folded into B).
    """
    import torch
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = max(1, lora.max_rank())
    if pad_to_vllm_rank:
        r = padded_rank(r)
    tdt = getattr(torch, dtype)
    tensors, leaves = {}, set()
    for m in lora.modules():
        A, B = lora.factors[m]
        k = A.shape[0]
        if k < r:
            A = torch.cat([A, A.new_zeros(r - k, A.shape[1])], 0)
            B = torch.cat([B, B.new_zeros(B.shape[0], r - k)], 1)
        tensors[peft_key(store.prefix, m, "A")] = A.to(tdt).contiguous()
        tensors[peft_key(store.prefix, m, "B")] = B.to(tdt).contiguous()
        leaves.add(parse_short(m)[2])
    if not tensors:  # vLLM rejects empty adapters; write a zero rank-1 term on one module
        m = store.modules()[0]
        out_f, in_f = store.shape(m)
        tensors[peft_key(store.prefix, m, "A")] = torch.zeros(r, in_f, dtype=tdt)
        tensors[peft_key(store.prefix, m, "B")] = torch.zeros(out_f, r, dtype=tdt)
        leaves.add(parse_short(m)[2])
    save_file(tensors, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": str(base_model),
        "r": int(r), "lora_alpha": int(r), "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
        "use_rslora": False, "use_dora": False, "init_lora_weights": True, "inference_mode": True,
        # leaf names (not full paths): vLLM validates leaves against the model's supported LoRA modules;
        # with PEFT, leaf names only match language-model linears (vision uses qkv/proj/linear_fc*).
        "target_modules": sorted(leaves), "modules_to_save": None,
        "layers_to_transform": None, "rank_pattern": {}, "alpha_pattern": {},
    }
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
    with open(out_dir / "lora_meta.json", "w") as f:
        json.dump({"meta": lora.meta, "ranks": {m: lora.rank(m) for m in lora.modules()},
                   "padded_rank": r}, f, indent=1, sort_keys=True, default=str)
    return out_dir


def load_peft(path: Path) -> tuple[LoRA, str]:
    """Read a PEFT LoRA adapter (ours or a standard PEFT one). Returns (LoRA, checkpoint prefix);
    the scaling lora_alpha / r (or / sqrt(r) with rslora) is folded into B; zero-padding is dropped."""
    import math

    from safetensors import safe_open

    path = Path(path)
    cfg = json.load(open(path / "adapter_config.json"))
    r = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    scaling = alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)
    parts, prefix = {}, None
    with safe_open(str(path / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            p = parse_peft_key(k)
            if p is None:
                continue
            pre, short, which = p
            prefix = pre
            parts.setdefault(short, {})[which] = f.get_tensor(k).float()
    lo = LoRA()
    for m, d in parts.items():
        A, B = d["A"], d["B"] * scaling
        keep = (A.abs().sum(1) > 0) | (B.abs().sum(0) > 0)
        if keep.any():
            lo.factors[m] = (A[keep].contiguous(), B[:, keep].contiguous())
    return lo, prefix or ""


# ============================================================================ merge
NON_WEIGHT_SKIP = {".tmp"}


def merge_into_model(loras, src_dir: Path, out_dir: Path, *, store=None, device: str = "cpu",
                     link_unchanged: bool = True) -> dict:
    """Write a new checkpoint W' = W + Σ ΔW for the given LoRA(s) into out_dir.

    Shards that contain an edited tensor are rewritten (fp32 accumulation, cast back to the stored dtype);
    unchanged shards are hard-linked (fallback copy); all other files (config, tokenizer, chat template,
    preprocessor / generation configs, index) are copied. Returns a report with the realised relative
    error of each edited tensor after rounding to the stored dtype (bf16 rounding can swallow tiny ΔW)."""
    import torch

    from .weights import WeightStore

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    if isinstance(loras, LoRA):
        loras = [loras]
    store = store or WeightStore(src_dir)
    edits = {}
    for lo in loras:
        for m in lo.modules():
            edits.setdefault(store.key(m), []).append((lo, m))
    report = {"src": str(src_dir), "modules": {}}

    def transform(k, t):
        w = t.to(device=device, dtype=torch.float32)
        dw = torch.zeros_like(w)
        for lo, m in edits[k]:
            A, B = lo.factors[m]
            dw += B.to(device) @ A.to(device)
        new = (w + dw).to(t.dtype)
        realised = new.to(torch.float32) - w
        rel_err = float((realised - dw).norm() / max(float(dw.norm()), 1e-30))
        report["modules"][k] = {"dw_rel": float(dw.norm() / max(float(w.norm()), 1e-30)),
                                "realised_rel_err": rel_err}
        return new.cpu()

    def finalize(rep):
        errs = [v["realised_rel_err"] for v in rep["modules"].values()]
        rep["max_realised_rel_err"] = max(errs) if errs else 0.0
        rep["mean_realised_rel_err"] = sum(errs) / len(errs) if errs else 0.0

    return _rewrite_checkpoint(src_dir, out_dir, store, edits, transform, report,
                               report_name="merge_report.json", link_unchanged=link_unchanged, finalize=finalize)


def _rewrite_checkpoint(src_dir: Path, out_dir: Path, store, keys, transform, report: dict, *, report_name: str,
                        link_unchanged: bool = True, finalize=None) -> dict:
    """Shared checkpoint writer (merge_into_model, wedit.edit_into_model).

    keys: checkpoint keys to rewrite; transform(key, tensor) -> new CPU tensor (stored dtype) and may fill
    report. Shards holding one of the keys are rewritten with every other tensor unchanged; other shards are
    hard-linked (fallback copy); non-weight files are copied; report['shards_rewritten'] is set, finalize(report)
    is called, the report is written to <out>/<report_name>; everything goes to <out>.partial first, then is
    renamed over out_dir."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    by_shard = {}
    for key in keys:
        by_shard.setdefault(store.weight_map[key], []).append(key)
    report["shards_rewritten"] = sorted(by_shard)
    for shard in sorted(set(store.weight_map.values())):
        src = src_dir / shard
        dst = tmp / shard
        if shard not in by_shard:
            _link_or_copy(src, dst, link_unchanged)
            continue
        tensors = {}
        with safe_open(str(src), "pt", device="cpu") as f:
            meta = f.metadata() or {"format": "pt"}
            for k in f.keys():
                t = f.get_tensor(k)
                if k in keys:
                    t = transform(k, t)
                tensors[k] = t.contiguous()
        save_file(tensors, str(dst), metadata=meta)
    for p in src_dir.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and not any(p.name.endswith(s) for s in NON_WEIGHT_SKIP):
            shutil.copy2(p, tmp / p.name)
    if finalize is not None:
        finalize(report)
    with open(tmp / report_name, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return report


def _link_or_copy(src: Path, dst: Path, link: bool) -> None:
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)
