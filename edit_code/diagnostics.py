"""Fixed diagnostics given to the editor (identical program in every generation; METHOD_DESIGN §2.2).

profile_table     one row per (group key, bits): whole-model DeltaKL(M_0 || M_0 with ONLY that module group
                  RTN-quantized to `bits`), full-vocabulary token KL by teacher forcing on the first
                  diagnostics.profile_contexts generation-0 KL texts (prompt + M_0's answer, cut at
                  profile_max_len tokens); plus p99 token KL, top-1 agreement drop, params.
hessian_table     one row per (group key, quantizer, bits): Hessian-weighted output error tr(dW H dW^T) of
                  that group alone, from the cached quantized weights and the calibration Hessians (every
                  calibration token; no forward pass). Built by QuantProblem.hessian_table.
activation_table  one row per linear module: input RMS / max-over-median / outlier fraction / red flag
                  (from the calibration pass, quant/hessian.py).
bytes_table       one row per group key: params and packed MiB for each (bits, group size).
All three are cached per generation in <gen>/diagnostics/*.json. In mock mode they are synthetic
with the same schema (edit_code/mock.py). behavior_sensitivity (METHOD_DESIGN §2.2) is future work.
"""
from __future__ import annotations

import time
from pathlib import Path

from .common import read_json, write_json
from .quant import sizes
from .quant.spec import ModelSpec


# ----------------------------------------------------------------------------- real (torch)
def profile_table_real(model, spec: ModelSpec, batches, refs, *, bits_list=(2, 3, 4, 8), group_size: int = 128,
                       log=print) -> list[dict]:
    import torch
    from .quant import quantizers as Q
    from .quant.kl import kl_to_reference
    rows = []
    gp = spec.group_params()
    t0 = time.time()
    groups = spec.groups()
    for gi, (key, mods) in enumerate(groups.items()):
        params = {m.name: model.get_parameter(m.name) for m in mods}
        orig = {n: p.data.clone() for n, p in params.items()}
        try:
            for b in bits_list:
                with torch.no_grad():
                    for n, p in params.items():
                        p.data.copy_(Q.rtn(orig[n], b, group_size).dequant(p.dtype, device=p.device))
                r = kl_to_reference(model, batches, refs)
                layer, g = key.split(".", 1)
                rows.append({"key": key, "layer": int(layer), "group": g, "bits": b, "group_size": group_size,
                             "quantizer": "rtn", "dkl_mean": r["mean"], "dkl_p99": r["p99"],
                             "top1_drop": 1.0 - r["top1_agree"], "params": gp[key]})
        finally:
            with torch.no_grad():
                for n, p in params.items():
                    p.data.copy_(orig[n])
        if gi % 25 == 0:
            log(f"[profile] {gi + 1}/{len(groups)} groups, {time.time() - t0:.0f}s")
    return rows


def activation_table_real(spec: ModelSpec, stats: dict, *, red_absmax_ratio: float = 50.0,
                          red_outlier_frac: float = 0.01) -> list[dict]:
    rows = []
    for m in spec.modules:
        s = stats.get(f"{m.layer}.{m.input_key}")
        if not s:
            continue
        rows.append({"layer": m.layer, "group": m.group, "module": m.proj, "input": m.input_key,
                     "rms_mean": round(s["rms_mean"], 5), "rms_max_over_median": round(s["rms_max_over_median"], 3),
                     "absmax_over_median": round(s["absmax_over_median"], 3),
                     "outlier_frac": round(s["outlier_frac"], 6),
                     "red_flag": bool(s["absmax_over_median"] > red_absmax_ratio or s["outlier_frac"] > red_outlier_frac)})
    return rows


# ----------------------------------------------------------------------------- shared
def bytes_table(spec: ModelSpec, bits_list=(2, 3, 4, 8), group_sizes=(32, 64, 128)) -> list[dict]:
    rows = []
    for key, mods in spec.groups().items():
        layer, g = key.split(".", 1)
        mib = {}
        for b in bits_list:
            for gs in group_sizes:
                mib[f"b{b}_g{gs}"] = round(sum(sizes.weight_bytes(m.shape[0], m.shape[1], b, gs) for m in mods) / 2 ** 20, 3)
        mib["bf16"] = round(sum(m.numel for m in mods) * spec.weight_dtype_bytes / 2 ** 20, 3)
        rows.append({"key": key, "layer": int(layer), "group": g, "params": sum(m.numel for m in mods),
                     "modules": [m.proj for m in mods], "mib": mib})
    return rows


def summarize_profile(rows: list[dict], top: int = 12) -> dict:
    """Compact view for the prompt: most / least sensitive groups at 3 and 2 bits."""
    out = {}
    for b in sorted({r["bits"] for r in rows}):
        rs = sorted((r for r in rows if r["bits"] == b), key=lambda r: -r["dkl_mean"])
        if not rs:
            continue
        out[f"b{b}"] = {"total_dkl_if_all": round(sum(r["dkl_mean"] for r in rs), 5),
                        "most_sensitive": [(r["key"], round(r["dkl_mean"], 6)) for r in rs[:top]],
                        "least_sensitive": [(r["key"], round(r["dkl_mean"], 6)) for r in rs[-5:]]}
        by_group = {}
        for r in rs:
            by_group[r["group"]] = by_group.get(r["group"], 0.0) + r["dkl_mean"]
        out[f"b{b}"]["by_group_sum"] = {g: round(v, 5) for g, v in sorted(by_group.items(), key=lambda x: -x[1])}
    return out


def load_tables(diag_dir: Path) -> dict | None:
    d = Path(diag_dir)
    if not (d / "profile_table.json").exists():
        return None
    return {"profile": read_json(d / "profile_table.json"), "activation": read_json(d / "activation_table.json"),
            "bytes": read_json(d / "bytes_table.json"), "meta": read_json(d / "meta.json", {}),
            "hessian": read_json(d / "hessian_table.json")}   # None until QuantProblem.prepare builds it


def save_tables(diag_dir: Path, profile, activation, bytes_rows, meta: dict) -> dict:
    d = Path(diag_dir)
    write_json(d / "profile_table.json", profile)
    write_json(d / "activation_table.json", activation)
    write_json(d / "bytes_table.json", bytes_rows)
    write_json(d / "meta.json", meta)
    return {"profile": profile, "activation": activation, "bytes": bytes_rows, "meta": meta}
