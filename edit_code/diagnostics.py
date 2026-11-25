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
All four are computed once from M_0 and cached in prep/diagnostics/*.json (hessian_table after the
layer cache). In mock mode they are synthetic with the same schema (edit_code/mock.py). behavior_sensitivity (METHOD_DESIGN §2.2) is future work.
"""
from __future__ import annotations

import time
from pathlib import Path

from .common import exit_with_parent, read_json, write_json
from .quant import sizes
from .quant.spec import ModelSpec


# ----------------------------------------------------------------------------- real (torch)
def profile_table_real(model, spec: ModelSpec, batches, refs, *, bits_list=(2, 3, 4, 8), group_size: int = 128,
                       log=print, groups: list[str] | None = None, kl=None, rows_dir: Path | None = None,
                       meta: dict | None = None) -> list[dict]:
    """groups: only these group keys (one worker's share when the table is split over several GPUs).
    kl: kl_to_reference (default; batches = [1, T] contexts) or kl_to_reference_packed (batches = pack() output).
    rows_dir (2026-11-21): one <key>.json per finished group ({meta, rows}); groups already there with the same
    meta are read instead of recomputed, so an interrupted prep resumes where it stopped."""
    import torch
    from .quant import quantizers as Q
    from .quant.kl import kl_to_reference
    kl = kl or kl_to_reference
    rows = []
    gp = spec.group_params()
    t0 = time.time()
    keep = set(groups) if groups is not None else None
    groups = {k: v for k, v in spec.groups().items() if keep is None or k in keep}
    done = 0
    for gi, (key, mods) in enumerate(groups.items()):
        if rows_dir is not None:
            prev = saved_profile_rows(rows_dir, key, meta)
            if prev is not None:
                rows += prev
                continue
        params = {m.name: model.get_parameter(m.name) for m in mods}
        orig = {n: p.data.clone() for n, p in params.items()}
        mine = []
        try:
            for b in bits_list:
                with torch.no_grad():
                    for n, p in params.items():
                        p.data.copy_(Q.rtn(orig[n], b, group_size).dequant(p.dtype, device=p.device))
                r = kl(model, batches, refs)
                layer, g = key.split(".", 1)
                mine.append({"key": key, "layer": int(layer), "group": g, "bits": b, "group_size": group_size,
                             "quantizer": "rtn", "dkl_mean": r["mean"], "dkl_p99": r["p99"],
                             "top1_drop": 1.0 - r["top1_agree"], "params": gp[key]})
        finally:
            with torch.no_grad():
                for n, p in params.items():
                    p.data.copy_(orig[n])
        rows += mine
        if rows_dir is not None:
            write_json(Path(rows_dir) / f"{key}.json", {"meta": meta, "rows": mine})
        done += 1
        if done == 1 or done % 10 == 0 or gi == len(groups) - 1:
            el = time.time() - t0
            log(f"[profile] {gi + 1}/{len(groups)} groups ({done} computed here), {el:.0f}s, "
                f"{el / done:.1f}s/group")
    return rows


def profile_meta(texts: list[str], dcfg: dict) -> dict:
    """What a saved profile row depends on (contexts, their length cap, bits, group size)."""
    import hashlib
    return {"texts_sha1": hashlib.sha1("\x00".join(texts).encode()).hexdigest(), "n_texts": len(texts),
            "max_len": int(dcfg["profile_max_len"]), "bits": list(dcfg["profile_bits"]),
            "group_size": int(dcfg["profile_group_size"])}


def saved_profile_rows(rows_dir: Path, key: str, meta: dict) -> list | None:
    """The rows saved in rows_dir/<key>.json when they were computed with `meta` (profile_meta), else None."""
    rec = read_json(Path(rows_dir) / f"{key}.json")
    return rec["rows"] if rec and rec.get("meta") == meta else None


def warn_slow_kernels(log=print) -> None:
    """Without flash-linear-attention (and causal-conv1d) transformers runs the Gated-DeltaNet layers of Qwen3.5 /
    Qwen3.8 with a pure-torch fallback, several times slower."""
    import importlib.util
    miss = [m for m, pkg in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d"))
            if importlib.util.find_spec(m) is None]
    if miss:
        log(f"[profile] WARNING: {', '.join(miss)} not installed: transformers uses the slow torch fallback for the "
            f"linear-attention layers (pip install flash-linear-attention causal-conv1d into the GPU venv)")


def profile_rows(model, model_dir: Path, spec: ModelSpec, texts: list[str], dcfg: dict, *, rows_dir: Path,
                 groups: list[str] | None = None, log=print) -> list[dict]:
    """The profile table on `texts` with an already-loaded model: contexts packed into batches of
    diagnostics.profile_batch_tokens tokens (default 8192), one reference pass, then every group x bits."""
    from .quant.hessian import tokenize_texts
    from .quant.kl import kl_to_reference_packed, pack, reference_packed
    meta = profile_meta(texts, dcfg)
    pb = tokenize_texts(model_dir, texts, dcfg["profile_max_len"])
    packed = pack(pb, int(dcfg.get("profile_batch_tokens") or 8192))
    t0 = time.time()
    refs = reference_packed(model, packed)
    log(f"[profile] {len(pb)} contexts in {len(packed)} batches, reference pass {time.time() - t0:.1f}s")
    rows = profile_table_real(model, spec, packed, refs, bits_list=dcfg["profile_bits"],
                              group_size=dcfg["profile_group_size"], log=log, groups=groups,
                              kl=kl_to_reference_packed, rows_dir=rows_dir, meta=meta)
    del refs
    return rows


def profile_worker(model_dir: str, device: str, spec_path: str, texts: list[str], dcfg: dict, rows_dir: str,
                   groups: list[str], tag: str = "") -> None:  # pragma: no cover - GPU path (own process)
    """One prep worker of the profile table (2026-11-21): a separate PROCESS per GPU (the earlier threads shared one
    GIL), loads its own HF model and writes rows_dir/<key>.json for each of its groups."""
    import torch
    from .quant.hessian import load_hf_model

    def log(msg):
        print(f"[prep:{tag}:{device}] {msg}", flush=True)

    exit_with_parent(f"[prep:{tag}:{device}] parent process gone, exiting")   # the run was killed (kill <pid>)
    meta = profile_meta(texts, dcfg)
    todo = [k for k in groups if saved_profile_rows(rows_dir, k, meta) is None]
    if not todo:
        log(f"[profile] all {len(groups)} groups already done")
        return
    warn_slow_kernels(log)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    log(f"[profile] loading {model_dir}; {len(todo)}/{len(groups)} groups to compute")
    model = load_hf_model(Path(model_dir), device)
    profile_rows(model, Path(model_dir), ModelSpec.from_json(read_json(spec_path)), texts, dcfg,
                 rows_dir=Path(rows_dir), groups=todo, log=log)


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
    """Compact view for the prompt: most / least sensitive groups per profiled bit width."""
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
