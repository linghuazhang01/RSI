"""Fixed diagnostics table (METHOD_DESIGN §3.2, RSI_LOOP_DESIGN §3.2), identical program in every generation.

One row per (layer, module) of the CURRENT model:
  w_norm, w_sigma (top singular values of W), and for every adapter already part of the model (merged, or in the cumulative adapter)
  (the stack, one entry per earlier generation): norm, rel_norm (||ΔW||/||W0||, W0 = base θ0), sv (singular
  values of ΔW), eff_rank (exp of the entropy of σ²), intruder (mean max-alignment of ΔW's singular
  directions with θ0's top-k left singular subspace; low = intruder), relchange_q (quantiles of
  |ΔW|/|W0|), act_ratio_target / act_ratio_retention (mean ||ΔW h||/||W h|| on target `train` prompts vs
  retention `kl` prompts) and their ratio.
Cached per generation in gen_XX/diagnostics.json; render() gives the compact text shown to the editor.
"""
from __future__ import annotations

import math
from pathlib import Path

from .common import read_json, write_json


def eff_rank(sv) -> float:
    s2 = [x * x for x in sv if x > 0]
    t = sum(s2)
    if t <= 0:
        return 0.0
    p = [x / t for x in s2]
    return math.exp(-sum(x * math.log(x) for x in p if x > 0))


def compute(store, stack, *, base_store=None, hf=None, target_texts=(), retention_texts=(), svd_k: int = 8,
            intruder_k: int = 32, relchange: bool = True, device: str = "cpu", act_items: int = 32) -> dict:
    """Real diagnostics (torch). `stack` = [(gen, LoRA)], `base_store` = WeightStore of θ0 (defaults to store)."""
    import torch

    base_store = base_store or store
    rows = []
    ratios_t, ratios_r = {}, {}
    if hf is not None and stack:
        for g, lo in stack:
            if target_texts:
                ratios_t[g] = hf.act_ratio(list(target_texts)[:act_items], lo)
            if retention_texts:
                ratios_r[g] = hf.act_ratio(list(retention_texts)[:act_items], lo)
    for m in store.modules():
        inf = store.info(m)
        _, S, _ = store.svd(m, svd_k)
        row = dict(inf, w_norm=round(store.norm(m), 4), w_sigma=[round(float(x), 4) for x in S[:4]], stack=[])
        for g, lo in stack:
            if m not in lo.factors:
                continue
            U, s, _ = lo.compact(m)
            n = float(torch.sqrt((s ** 2).sum()))
            w0n = base_store.norm(m)
            U0, _, _ = base_store.svd(m, intruder_k)
            align = (U.T @ U0).abs().max(dim=1).values if s.numel() else torch.zeros(0)
            wts = (s / s.sum()) if s.numel() and float(s.sum()) > 0 else s
            ent = {"gen": g, "norm": round(n, 5), "rel_norm": round(n / max(w0n, 1e-12), 6),
                   "sv": [round(float(x), 5) for x in s[:4]], "eff_rank": round(eff_rank(s.tolist()), 3),
                   "intruder": round(float((align * wts).sum()), 4) if s.numel() else None}
            if relchange:
                dW = lo.delta(m).to(device)
                W0 = base_store.weight(m, device=device)
                r = (dW.abs() / (W0.abs() + 1e-8)).flatten()
                if r.numel() > 2_000_000:
                    idx = torch.randperm(r.numel(), generator=torch.Generator().manual_seed(0))[:2_000_000].to(r.device)
                    r = r[idx]
                q = torch.quantile(r.float(), torch.tensor([0.5, 0.9, 0.99], device=r.device))
                ent["relchange_q"] = {"q50": float(q[0]), "q90": float(q[1]), "q99": float(q[2])}
            if g in ratios_t:
                ent["act_ratio_target"] = round(ratios_t[g].get(m, 0.0), 6)
            if g in ratios_r:
                ent["act_ratio_retention"] = round(ratios_r[g].get(m, 0.0), 6)
            row["stack"].append(ent)
        rows.append(row)
    return {"model": str(getattr(store, "model_dir", "")), "n_modules": len(rows), "rows": rows,
            "stack_gens": [g for g, _ in stack]}


def finalize(diag: dict) -> dict:
    for r in diag["rows"]:
        for e in r.get("stack", []):
            t, q = e.get("act_ratio_target"), e.get("act_ratio_retention")
            if t is not None and q:
                e["act_ratio_ratio"] = round(t / q, 4)
    diag["summary"] = layer_summary(diag)
    return diag


def layer_summary(diag: dict) -> list[dict]:
    by = {}
    for r in diag["rows"]:
        d = by.setdefault(r["layer"], {"layer": r["layer"], "layer_type": r.get("layer_type"), "n": 0,
                                        "w_sigma1_attn": [], "w_sigma1_mlp": [], "stack_rel": 0.0,
                                        "intruder": [], "act_ratio_ratio": []})
        d["n"] += 1
        (d["w_sigma1_mlp"] if r["group"] == "mlp" else d["w_sigma1_attn"]).append(r["w_sigma"][0] if r["w_sigma"] else 0)
        for e in r.get("stack", []):
            d["stack_rel"] += e.get("rel_norm", 0) ** 2
            if e.get("intruder") is not None:
                d["intruder"].append(e["intruder"])
            if e.get("act_ratio_ratio") is not None:
                d["act_ratio_ratio"].append(e["act_ratio_ratio"])
    out = []
    for L in sorted(by):
        d = by[L]
        avg = (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)
        out.append({"layer": L, "layer_type": d["layer_type"], "modules": d["n"],
                    "w_sigma1_attn": avg(d["w_sigma1_attn"]), "w_sigma1_mlp": avg(d["w_sigma1_mlp"]),
                    "stack_rel_norm": round(math.sqrt(d["stack_rel"]), 5), "stack_intruder": avg(d["intruder"]),
                    "stack_act_ratio_ratio": avg(d["act_ratio_ratio"])})
    return out


def render(diag: dict, max_modules: int = 12, adapter: bool = False) -> str:
    """Compact text table for the editor prompt (layer summary + most-changed modules). adapter (merge.mode adapter,
    2026-11-22): the accepted adapters are in the cumulative adapter, not merged."""
    held = ("adapters accepted in generations {} (kept as one cumulative adapter, part of the current model)"
            if adapter else "adapters already merged from generations {}")
    lines = [f"Diagnostics: {diag['n_modules']} editable modules; " + held.format(diag.get('stack_gens') or 'none') + ".",
             "layer | type | sigma1(attn) | sigma1(mlp) | stack ||dW||/||W0|| | stack intruder | act ratio tgt/ret"]
    for s in diag.get("summary", []):
        lines.append(f"{s['layer']:>3} | {('full' if s['layer_type'] == 'full_attention' else 'gdn'):4} | "
                     f"{s['w_sigma1_attn']} | {s['w_sigma1_mlp']} | {s['stack_rel_norm']} | "
                     f"{s['stack_intruder']} | {s['stack_act_ratio_ratio']}")
    changed = [(r, r["stack"][-1]) for r in diag["rows"] if r.get("stack")]
    if changed:
        changed.sort(key=lambda x: -x[1].get("rel_norm", 0))
        lines.append("most-changed modules (latest adapter): module | rel_norm | eff_rank | intruder | relchange q99 | act ratio tgt/ret")
        for r, e in changed[:max_modules]:
            lines.append(f"{r['module']} | {e.get('rel_norm')} | {e.get('eff_rank')} | {e.get('intruder')} | "
                         f"{(e.get('relchange_q') or {}).get('q99')} | {e.get('act_ratio_ratio')}")
    lines.append("Full rows: api.diagnostics(module=..., layer=..., top=..., sort=...).")
    return "\n".join(lines)


def load_or_compute(path: Path, fn) -> dict:
    got = read_json(path)
    if got is not None:
        return got
    diag = finalize(fn())
    write_json(path, diag)
    return diag
