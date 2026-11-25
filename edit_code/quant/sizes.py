"""Analytic size of a quantization config in its real packed form.

Per quantized linear weight [out, in] with b bits and group size g (groups along the input dim):
    codes       out * in * b / 8                    bytes
    scales      out * ceil(in / g) * 2              (fp16, one per group)
    zero points out * ceil(in / g) * b / 8          (asymmetric, b-bit packed, GPTQ-style)
    awq only    in * 2                              (fp16 per-input-channel scale)
BF16 (bits 16) modules count 2 bytes per weight. Embedding / LM head: "bf16", or 8/4-bit RTN with
group size 128. Everything else in the checkpoint that vLLM loads (norms, conv1d, A_log, dt_bias,
the vision tower) is counted at its stored dtype; `mtp.*` tensors are skipped by vLLM and not
counted. Real deployment formats may add a few KB of metadata; this is not included.
"""
from __future__ import annotations

import math

from ..common import GIB
from .spec import ModelSpec

EMBED_GROUP = 128


def weight_bytes(out_f: int, in_f: int, bits: int, group_size: int, quantizer: str = "rtn",
                 dtype_bytes: int = 2) -> float:
    if bits >= 16:
        return float(out_f * in_f * dtype_bytes)
    ng = math.ceil(in_f / group_size)
    b = out_f * in_f * bits / 8 + out_f * ng * (2 + bits / 8)
    if quantizer == "awq":
        b += in_f * 2
    return float(b)


def precision_bytes(t: dict | None, prec) -> float:
    if not t:
        return 0.0
    out_f, in_f = int(t["shape"][0]), int(t["shape"][1])
    if prec == "bf16":
        return float(out_f * in_f * t.get("dtype_bytes", 2))
    return weight_bytes(out_f, in_f, int(prec), EMBED_GROUP, "rtn")


def special_precisions(spec: ModelSpec, embed, lm_head) -> dict[str, int]:
    """{tensor name: bits} of the embedding / LM head when not "bf16" (RTN at EMBED_GROUP in the export and the
    EvoPress worker)."""
    return {t["name"]: int(prec) for t, prec in ((spec.embed, embed), (spec.lm_head, lm_head))
            if t and prec != "bf16"}


def modules_bytes(mods, setting: dict, dtype_bytes: int = 2) -> float:
    """Packed bytes of these modules under one setting."""
    total = 0.0
    for m in mods:
        total += weight_bytes(m.shape[0], m.shape[1], setting["bits"], setting.get("group_size") or 128,
                              setting.get("quantizer", "rtn"), dtype_bytes)
    return total


def group_bytes(spec: ModelSpec, key: str, setting: dict) -> float:
    return modules_bytes(spec.groups()[key], setting, spec.weight_dtype_bytes)


def config_bytes(norm: dict, spec: ModelSpec) -> dict:
    """Size breakdown of a normalized config (see schema.normalize)."""
    groups = spec.groups()
    lin = 0.0
    by_group: dict[str, float] = {}
    for key, ms in groups.items():
        s = norm["groups"][key]
        b = modules_bytes(ms, s, spec.weight_dtype_bytes)
        lin += b
        g = key.split(".", 1)[1]
        by_group[g] = by_group.get(g, 0.0) + b
    emb = precision_bytes(spec.embed, norm["embed"])
    head = precision_bytes(spec.lm_head, norm["lm_head"]) if spec.lm_head is not None else 0.0
    total = lin + emb + head + spec.other_bytes
    lin_params = sum(m.numel for m in spec.modules)
    return {"total_bytes": int(total), "gib": total / GIB, "linear_bytes": int(lin), "embed_bytes": int(emb),
            "lm_head_bytes": int(head), "other_bytes": int(spec.other_bytes),
            "linear_avg_bits": 8 * lin / max(lin_params, 1),
            "by_group_gib": {g: v / GIB for g, v in sorted(by_group.items())}}


def bf16_bytes(spec: ModelSpec) -> float:
    lin = sum(m.numel for m in spec.modules) * spec.weight_dtype_bytes
    return lin + precision_bytes(spec.embed, "bf16") + precision_bytes(spec.lm_head, "bf16") + spec.other_bytes
