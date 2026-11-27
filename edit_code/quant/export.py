"""Export a candidate checkpoint in one of two formats (candidates.format, 2026-11-27):

dequant (default; "fake quant"): same architecture, config and tokenizer files (copied), same shard layout and tensor
    names; every quantized linear weight (and the embedding / LM head when not "bf16") is replaced by its dequantized
    value in the original dtype. No `quantization_config` is added, so vLLM loads it as a normal BF16 model: the
    numbers of the quantized model, BF16 speed and memory.

packed (real quantization): every quantized linear module is stored as GPTQ-format integer tensors
    <module>.qweight int32 [in*b/32, out]   codes, one LSB-first bitstream along the input dim per output column
    <module>.qzeros  int32 [in/g, out*b/32] the TRUE integer zero points (not zero-1), bitstream along the output dim
    <module>.scales  bf16  [in/g, out]
    and config.json gets a gptq quantization_config with per-module bits / group sizes in its "dynamic" field
    ("+:<regex>" = this vLLM module's setting, "-:<regex>" = keep BF16). Served with `vllm serve --quantization
    humming` (vLLM 0.24.0 HummingConfig + humming-kernels 0.1.6: INT2-8 with zero points on bf16 activations; it
    reads exactly these tensors, verified in the source 2026-11-27, no g_idx needed). Weight = bf16((code - zero) *
    bf16(scale)): the same numbers as the dequant export of the same cache entries when the scales are bf16 (the cache
    computes them so under action_space.packed). Only configs that schema.packed_problems accepts can be written (one
    setting per fused vLLM module, int32-aligned shapes, rtn / gptq, BF16 embedding and LM head). vLLM does not check
    that a quantized module's tensors were loaded, so validate_packed re-derives every module's setting from the
    written quantization_config (vLLM's resolution rules) and checks the tensors before the export is accepted.

Both write `fake_quant.json` (the name kept for both: older code and runs look for it) with the resolved config,
its analytic packed size, "format" and "vllm_args" (the extra `vllm serve` arguments the checkpoint needs:
server_args()).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import quantizers as Q
from .cache import QuantCache
from .schema import module_assignment, packed_problems
from .sizes import EMBED_GROUP, config_bytes, special_precisions
from .spec import ModelSpec, checkpoint_tensors, shard_files, vllm_module_prefix

MARKER = "fake_quant.json"
FORMATS = ("dequant", "packed")
PACKED_VLLM_ARGS = ["--quantization", "humming"]
HUMMING_PIN = "humming-kernels[cu13]==0.1.6"    # what vllm 0.24.0 requirements/cuda.txt pins

SKIP_COPY_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".tmp")


# safetensors header dtype -> torch dtype, for the weights an export replaces (other dtypes: read the tensor)
_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def copy_model_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in Path(src).iterdir():
        if p.is_file() and not p.name.endswith(SKIP_COPY_SUFFIXES) and p.name != MARKER:
            shutil.copy2(p, dst / p.name)


def export_fake_quant(src_dir: Path, out_dir: Path, norm: dict, spec: ModelSpec, cache: QuantCache, *,
                      meta: dict | None = None, log=print) -> dict:
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    t0 = time.time()
    copy_model_files(src_dir, tmp_dir)
    assign = module_assignment(norm, spec)
    special = special_precisions(spec, norm["embed"], norm["lm_head"])
    idx = src_dir / "model.safetensors.index.json"
    n_q = 0
    for fn in shard_files(src_dir):
        tensors = {}
        with safe_open(str(src_dir / fn), framework="pt") as f:
            smeta = f.metadata() or {}
            for k in f.keys():
                if k in assign and assign[k]["bits"] < 16:
                    # replaced: dtype from the shard header, the stored weight is not read (2026-11-25)
                    dtype = _ST_DTYPES.get(f.get_slice(k).get_dtype())
                    if dtype is None:
                        dtype = f.get_tensor(k).dtype
                    t = cache.dequant(k, assign[k], dtype=dtype)
                    n_q += 1
                else:
                    t = f.get_tensor(k)
                    if k in special:
                        q = Q.rtn(t.to(cache.device), special[k], EMBED_GROUP)
                        t = Q.dequant(q, dtype=t.dtype).cpu()
                        n_q += 1
                tensors[k] = t.contiguous()
        save_file(tensors, str(tmp_dir / fn), metadata=dict(smeta, format=smeta.get("format", "pt")))
        del tensors
    if idx.exists():
        shutil.copy2(idx, tmp_dir / idx.name)
    info = {"config": norm, "bytes": config_bytes(norm, spec), "quantized_tensors": n_q, "source": str(src_dir),
            "seconds": time.time() - t0, "format": "dequant", "vllm_args": [],
            "scale_dtype": str(cache.scale_dtype).replace("torch.", ""), **(meta or {})}
    with open(tmp_dir / MARKER, "w") as f:
        json.dump(info, f, indent=1)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)
    log(f"[export] {n_q} tensors fake-quantized -> {out_dir} ({info['seconds']:.1f}s, "
        f"{info['bytes']['gib']:.2f} GiB packed)")
    return info


# ------------------------------------------------------------------------------------------------ both formats
def export_checkpoint(fmt: str, src_dir: Path, out_dir: Path, norm: dict, spec: ModelSpec, cache: QuantCache, *,
                      meta: dict | None = None, log=print) -> dict:
    """Write the candidate checkpoint in format `fmt` ("dequant" | "packed")."""
    if fmt == "dequant":
        return export_fake_quant(src_dir, out_dir, norm, spec, cache, meta=meta, log=log)
    if fmt == "packed":
        return export_packed(src_dir, out_dir, norm, spec, cache, meta=meta, log=log)
    raise ValueError(f"unknown checkpoint format {fmt!r} (candidates.format: {' | '.join(FORMATS)})")


def read_marker(model_dir) -> dict | None:
    p = Path(model_dir) / MARKER
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def server_args(model_dir) -> list[str]:
    """Extra `vllm serve` arguments of an exported checkpoint (the packed format: --quantization humming); [] for a
    dequant export, M_0, a mock model or anything without a marker."""
    info = read_marker(model_dir) or {}
    return [str(a) for a in info.get("vllm_args") or []]


LAUNCHER_EXEC = "'''exec'"        # pip / uv console-script launcher line: '''exec' "<python>" "$0" "$@"


def vllm_python() -> str:
    """The Python interpreter of the vLLM that rsicore launches (RSI_VLLM_PYTHON overrides): the interpreter of a
    `python -m vllm...` command, else the one a `vllm` console script runs: its shebang, the pip / uv launcher form
    (#!/bin/sh plus an exec line, written for long or spaced paths) or #!/usr/bin/env python3; else this one."""
    import os
    import shlex
    import shutil as _sh
    if os.environ.get("RSI_VLLM_PYTHON"):
        return os.environ["RSI_VLLM_PYTHON"]
    try:
        from rsicore.servers import _vllm_cmd
        cmd = _vllm_cmd()
    except Exception:  # noqa: BLE001 - rsicore missing: the current interpreter
        return sys.executable
    if len(cmd) > 1:
        return cmd[0]
    try:
        head = Path(cmd[0]).read_bytes()[:2048].decode("utf-8", "replace").splitlines()
    except OSError:
        return sys.executable
    if not head or not head[0].startswith("#!"):
        return sys.executable
    words = shlex.split(head[0][2:].strip())
    if not words:
        return sys.executable
    name = Path(words[0]).name
    if name in ("sh", "bash", "dash", "zsh"):
        for line in head[1:4]:
            if line.startswith(LAUNCHER_EXEC):
                parts = shlex.split(line[len(LAUNCHER_EXEC):].strip())
                if parts and Path(parts[0]).exists():
                    return parts[0]
        return sys.executable
    if name == "env":
        prog = next((w for w in words[1:] if not w.startswith("-")), None)
        return (_sh.which(prog) if prog else None) or sys.executable
    return words[0] if Path(words[0]).exists() else sys.executable


def humming_check(python: str | None = None) -> None:
    """Raise unless the vLLM environment can import humming (the kernels of the packed format)."""
    py = python or vllm_python()
    r = subprocess.run([py, "-c", "import humming"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"candidates.format packed needs the humming kernels in the vLLM environment ({py}; "
                           f"override with RSI_VLLM_PYTHON): pip install \"{HUMMING_PIN}\"\n"
                           f"{(r.stderr or '').strip()[-400:]}")


# ------------------------------------------------------------------------------------------------ packed format
_DECODER_ALLOW = (r"(?:language_model\.)?model\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|"
                  r"linear_attn\.(?:in_proj_qkvz|in_proj_ba|out_proj)|mlp\.(?:gate_up_proj|down_proj))")
NOT_CONVERT = ["visual.", "mtp."]


def prefix_regex(prefix: str) -> str:
    """Anchored regex of one vLLM module prefix; the leading language_model. is optional (multimodal and text-only
    checkpoints name the decoder language_model.model.layers / model.layers)."""
    if prefix.startswith("language_model.model."):
        prefix = prefix[len("language_model."):]
    if prefix.startswith("model."):
        return r"^(?:language_model\.)?" + re.escape(prefix) + "$"
    return "^" + re.escape(prefix) + "$"


def fused_settings(norm: dict, spec: ModelSpec) -> dict[str, dict]:
    """vLLM module prefix -> the resolved setting of its modules (equal for every module of a fused module: checked)."""
    out: dict[str, dict] = {}
    assign = module_assignment(norm, spec)
    for m in spec.modules:
        s = assign[m.name]
        key = (s["bits"], s["group_size"] if s["bits"] < 16 else 0)
        p = m.vllm_prefix
        if p in out and (out[p]["bits"], out[p]["group_size"] if out[p]["bits"] < 16 else 0) != key:
            raise ValueError(f"{p}: its modules have different settings ({out[p]} vs {s}); see schema.packed_problems")
        out.setdefault(p, s)
    return out


def packed_quant_config(norm: dict, spec: ModelSpec) -> dict | None:
    """The quantization_config of the packed checkpoint (None: nothing quantized, a plain BF16 checkpoint)."""
    fused = fused_settings(norm, spec)
    qset = Counter((s["bits"], s["group_size"]) for s in fused.values() if s["bits"] < 16)
    if not qset:
        return None
    (b0, g0), _ = qset.most_common(1)[0]
    dynamic = {"-:^(?!" + _DECODER_ALLOW + "$)": {}}     # only the decoder's linear modules can be quantized
    for p, s in fused.items():
        if s["bits"] >= 16:
            dynamic["-:" + prefix_regex(p)] = {}
    for p, s in fused.items():
        if s["bits"] < 16:
            dynamic["+:" + prefix_regex(p)] = {"bits": int(s["bits"]), "group_size": int(s["group_size"])}
    return {"quant_method": "gptq", "bits": int(b0), "group_size": int(g0), "sym": False, "desc_act": False,
            "checkpoint_format": "gptq_v2",       # true zero points (humming reads them as they are; it ignores this key)
            "modules_to_not_convert": list(NOT_CONVERT), "dynamic": dynamic}


def resolve_humming(qcfg: dict | None, prefix: str) -> dict | None:
    """The weight setting vLLM 0.24.0 gives the module `prefix` under this quantization_config with --quantization
    humming (HummingConfig.is_layer_skipped / get_layer_weight_schema, humming.py:206-249): None = BF16 (skipped),
    else {"bits", "group_size", "sym", "desc_act"}. Ignore lists are substring tests, "lm_head" is always skipped,
    "-:" regexes (re.match) skip, the first matching "+:" regex is merged over the top-level config."""
    if not qcfg:
        return None
    ignored = []
    for k in ("ignored_layers", "ignore", "modules_to_not_convert"):
        if qcfg.get(k):
            ignored = qcfg[k]
            break
    if any(x in prefix for x in ignored) or "lm_head" in prefix:
        return None
    dyn = qcfg.get("dynamic") or {}
    for rx in dyn:
        if rx[:1] == "-" and re.match(rx[2:], prefix):
            return None
    layer = dict(qcfg)
    for rx, over in dyn.items():
        if rx[:1] == "+" and re.match(rx[2:], prefix):
            layer = dict(qcfg, **over)
            break
    return {"bits": int(layer["bits"]), "group_size": int(layer["group_size"]), "sym": bool(layer.get("sym", True)),
            "desc_act": bool(layer.get("desc_act", False))}


def pack_rows_int32(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """uint8 codes [R, C] (< 2^bits) -> int32 [R, C*bits/32]: each row one LSB-first bitstream (value j of a row at
    bit j*bits, crossing word boundaries for 3 bit), the layout humming reads (pack_weight.cuh)."""
    R, C = codes.shape
    if (C * bits) % 32:
        raise ValueError(f"{C} values of {bits} bits do not fill whole 32-bit words")
    by = Q.pack(codes.contiguous(), bits)[: R * C * bits // 8].contiguous()
    return by.view(torch.int32).reshape(R, C * bits // 32)


def packed_tensors(cache: QuantCache, module: str, setting: dict) -> dict[str, torch.Tensor]:
    """{"qweight", "qzeros", "scales"} of one module from its cache entry (the codes are reused as stored: the
    cache's packing is the same row-major LSB-first stream)."""
    b, gs, q = int(setting["bits"]), int(setting["group_size"]), setting["quantizer"]
    d = cache.raw(module, q, b, gs)
    if "col_scale" in d:
        raise ValueError(f"{module}: an AWQ entry (per-channel scales) has no packed form")
    if d["scale"].dtype != torch.bfloat16:
        raise ValueError(f"{module}: cache entry has {d['scale'].dtype} scales; the packed format needs bf16 scales "
                         "(open the cache with scale_dtype=torch.bfloat16, i.e. action_space.packed)")
    o, i, zo, zg = (int(x) for x in d["shape"].tolist())
    if i % gs or zg != i // gs or zo != o:
        raise ValueError(f"{module}: {o} x {i} with group size {gs} does not pack (in % group_size != 0)")
    qweight = d["codes"][: o * i * b // 8].contiguous().view(torch.int32).reshape(o, i * b // 32).t().contiguous()
    zero = Q.unpack(d["zero"], b, zo * zg).reshape(zo, zg)
    return {"qweight": qweight, "qzeros": pack_rows_int32(zero.t().contiguous(), b),
            "scales": d["scale"].t().contiguous()}


def unpack_rows_int32(words: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Reference reader of pack_rows_int32 (int32 [R, W] -> int64 values [R, n]), written from humming's
    extract_packed_value (kernel/process.cuh): value j = bits [j*b, j*b+b) of the row's little-endian stream."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    R = w.shape[0]
    out = torch.empty(R, n, dtype=torch.int64)
    mask = (1 << bits) - 1
    for j in range(n):
        start = j * bits
        k, o = divmod(start, 32)
        v = w[:, k] >> o
        if o + bits > 32:
            v = v | (w[:, k + 1] << (32 - o))
        out[:, j] = v & mask
    return out


def validate_packed(ckpt_dir: Path, norm: dict, spec: ModelSpec) -> dict:
    """Check a written packed checkpoint the way vLLM will load it: every decoder module's vLLM setting (resolved
    from config.json's quantization_config with vLLM's rules) equals the config's, a quantized module has exactly
    qweight / qzeros / scales of the right shapes and dtypes (no .weight), a BF16 module its .weight (no qweight),
    and every other 2-D weight of the checkpoint (vision tower, embedding, LM head) resolves to BF16.
    Raises ValueError listing every problem; returns {"quantized_modules", "bf16_modules"}."""
    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    qcfg = cfg.get("quantization_config")
    have = checkpoint_tensors(ckpt_dir)
    assign = module_assignment(norm, spec)
    problems, nq, nb = [], 0, 0
    known = set()
    for m in spec.modules:
        s, p, base = assign[m.name], m.vllm_prefix, m.module_path
        known.add(base)
        got = resolve_humming(qcfg, p)
        if s["bits"] >= 16:
            nb += 1
            if got is not None:
                problems.append(f"{p}: BF16 in the config but vLLM would quantize it ({got})")
            if base + ".weight" not in have or base + ".qweight" in have:
                problems.append(f"{base}: a BF16 module needs .weight and no .qweight")
            continue
        nq += 1
        b, gs = int(s["bits"]), int(s["group_size"])
        if got is None or (got["bits"], got["group_size"], got["sym"], got["desc_act"]) != (b, gs, False, False):
            problems.append(f"{p}: config {b} bit g{gs} asymmetric, vLLM would use {got}")
        out_f, in_f = int(m.shape[0]), int(m.shape[1])
        want = {"qweight": ("I32", [in_f * b // 32, out_f]), "qzeros": ("I32", [in_f // gs, out_f * b // 32]),
                "scales": ("BF16", [in_f // gs, out_f])}
        for suf, (dt, shp) in want.items():
            t = have.get(f"{base}.{suf}")
            if t is None or t["dtype"] != dt or list(t["shape"]) != shp:
                problems.append(f"{base}.{suf}: want {dt} {shp}, have {None if t is None else (t['dtype'], t['shape'])}")
        if base + ".weight" in have:
            problems.append(f"{base}: quantized module still has a .weight")
    for name, t in have.items():
        if not name.endswith(".weight") or len(t["shape"]) != 2:
            continue
        base = name[: -len(".weight")]
        if base in known or name.startswith("mtp."):
            continue
        got = resolve_humming(qcfg, vllm_module_prefix(base))
        if got is not None:
            problems.append(f"{base}: kept BF16 but vLLM would quantize {vllm_module_prefix(base)} ({got})")
    if problems:
        raise ValueError(f"packed checkpoint {ckpt_dir} would not load as intended ({len(problems)} problems):\n  - "
                         + "\n  - ".join(problems[:40]))
    return {"quantized_modules": nq, "bf16_modules": nb}


def export_packed(src_dir: Path, out_dir: Path, norm: dict, spec: ModelSpec, cache: QuantCache, *,
                  meta: dict | None = None, log=print) -> dict:
    """The packed (real quantization) checkpoint of a resolved config; see the module docstring."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    problems = packed_problems(norm, spec)
    if problems:
        raise ValueError("config cannot be written as a packed checkpoint:\n  - " + "\n  - ".join(problems))
    if cache.scale_dtype != torch.bfloat16:
        raise ValueError("the packed format needs the bf16-scale layer cache (action_space.packed)")
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    t0 = time.time()
    copy_model_files(src_dir, tmp_dir)
    assign = module_assignment(norm, spec)
    qcfg = packed_quant_config(norm, spec)
    weight_map, total, n_q = {}, 0, 0
    for fn in shard_files(src_dir):
        tensors = {}
        with safe_open(str(src_dir / fn), framework="pt") as f:
            smeta = f.metadata() or {}
            for k in f.keys():
                if k in assign and assign[k]["bits"] < 16:
                    base = k[: -len(".weight")]
                    for suf, t in packed_tensors(cache, k, assign[k]).items():
                        tensors[f"{base}.{suf}"] = t.contiguous()
                    n_q += 1
                else:
                    tensors[k] = f.get_tensor(k).contiguous()
        save_file(tensors, str(tmp_dir / fn), metadata=dict(smeta, format=smeta.get("format", "pt")))
        for k, t in tensors.items():
            weight_map[k] = fn
            total += t.numel() * t.element_size()
        del tensors
    if (src_dir / "model.safetensors.index.json").exists() or len(set(weight_map.values())) > 1:
        with open(tmp_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
    cfg = json.loads((tmp_dir / "config.json").read_text())
    cfg.pop("quantization_config", None)
    if qcfg is not None:
        cfg["quantization_config"] = qcfg
    with open(tmp_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    check = validate_packed(tmp_dir, norm, spec)
    info = {"config": norm, "bytes": config_bytes(norm, spec), "quantized_tensors": n_q, "source": str(src_dir),
            "seconds": time.time() - t0, "format": "packed", "vllm_args": list(PACKED_VLLM_ARGS) if qcfg else [],
            "scale_dtype": "bfloat16", "checkpoint_bytes": total, "validated": check, **(meta or {})}
    with open(tmp_dir / MARKER, "w") as f:
        json.dump(info, f, indent=1)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)
    log(f"[export] {n_q} modules packed (GPTQ format, humming) -> {out_dir} ({info['seconds']:.1f}s, "
        f"{total / 2 ** 30:.2f} GiB on disk, {info['bytes']['gib']:.2f} GiB analytic)")
    return info
