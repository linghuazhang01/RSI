"""Raw LoRA training (mode=edit, generation 0 only) and the non-LLM branch (RSI_LOOP_DESIGN §3.5).

2026-11-20: there is no SFT baseline and nothing else is ever trained. The only training of a run is the
raw LoRA of mode=edit: the fixed `raw` recipe (rank 16 by default, all attention / MLP projections except
in_proj_a/b), trained once in generation 0 on raw.n_train of that generation's train items; the editor edits
it in every generation.
Non-LLM search (the 4th branch), same evaluate budget as the editor, by method.mode: edit = random rule-based
edits of the raw LoRA (random_edit_search); scratch = random gradient-free constructions
(random_construct_search); weights = random structured edits of the dense weights (random_weight_search).
Best by screen, top-3 re-scored on the full signal.
All points report the target gain vs the current model.
"""
from __future__ import annotations

import math
import random

from . import mock as M

DEFAULT_RAW = {"rank": 16, "lr": 1e-4, "epochs": 1.0, "steps": None, "batch_size": 4,
               "leaves": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                          "gate_proj", "up_proj", "down_proj"]}
RAW_KEYS = ("rank", "lr", "epochs", "steps", "batch_size", "leaves")


def raw_recipe(cfg: dict) -> tuple[dict, int | None, float]:
    """mode=edit: (recipe, n_train, gpu_seconds) of the raw LoRA. Null raw.* values take DEFAULT_RAW;
    raw.n_train null = all of generation 0's train items; raw.gpu_seconds null = budget.gpu_seconds."""
    rc = cfg.get("raw", {}) or {}
    r = dict(DEFAULT_RAW)
    for k in RAW_KEYS:
        if rc.get(k) is not None:
            r[k] = rc[k]
    gb = rc.get("gpu_seconds")
    gb = float(gb) if gb is not None else float(cfg.get("budget", {}).get("gpu_seconds", 7200))
    n = int(rc["n_train"]) if rc.get("n_train") is not None else None
    return r, n, gb


def subset_ids(ids, n: int | None, seed: int) -> list:
    """Deterministic subset of n train ids (all of them when n is None or >= len)."""
    ids = sorted(ids)
    if n is None or n >= len(ids):
        return ids
    return sorted(random.Random(f"{seed}:raw").sample(ids, max(1, int(n))))


def modules_for(store, leaves=None, layers=None) -> list[str]:
    out = []
    for m in store.modules():
        inf = store.info(m)
        if leaves and inf["leaf"] not in leaves:
            continue
        if layers and not (layers[0] <= inf["layer"] < layers[1]):
            continue
        out.append(m)
    return out


def train_lora(ws, rec: dict, gpu_budget_s: float, *, seed: int = 42, modules=None, ids=None):
    """Train one LoRA with the recipe (the raw LoRA of mode=edit). Returns (LoRA, info). Uses ws.train /
    ws.hf (or the mock model). ids: train items to use (default: all of ws.train)."""
    mods = modules or modules_for(ws.store, rec["leaves"], rec.get("layers"))
    ids = ws.train.check(list(ids)) if ids is not None else ws.train.ids()
    bs = int(rec["batch_size"])
    n_ex = max(1, len([e for e in ws.train.get(ids) if e.get("target")]) if not ws.mock else len(ids))
    steps = int(rec["steps"] or max(1, math.ceil(n_ex * float(rec["epochs"]) / bs)))
    if ws.mock:
        spe = float(ws.cfg.get("mock", {}).get("sec_per_example", 0.6))
        steps = min(steps, max(1, int(gpu_budget_s / (bs * spe))))
        final = M.m_sgd(None, mods, ids, steps, float(rec["lr"]), int(rec["rank"]))
        return final, {"steps": steps, "modules": len(mods), "gpu_seconds": round(steps * bs * spe, 1),
                           "rank": int(rec["rank"]), "lr": float(rec["lr"])}
    ex = ws._examples(ids)
    t0 = ws.hf.gpu_seconds
    final, info = ws.hf.sgd(ex, None, modules=mods, rank=int(rec["rank"]), steps=steps, lr=float(rec["lr"]),
                            batch_size=bs, warmup=int(rec.get("warmup", 0)), time_budget_s=gpu_budget_s,
                            init_seed=seed)
    info.pop("checkpoints", None)
    info.update({"steps_planned": steps, "gpu_seconds": round(ws.hf.gpu_seconds - t0, 1), "lr": float(rec["lr"])})
    info["loss_first"], info["loss_last"] = (info["losses"][0], info["losses"][-1]) if info["losses"] else (None, None)
    info.pop("losses", None)
    return final, info


def point(ws, lo, label: str, level: str = "screen") -> dict:
    """Evaluate one LoRA (not charged to the editor budget) and summarise the trade-off."""
    sc, fb = ws.evaluate_lora(lo, level, charge=False)
    tgt = fb.get("target") or {}
    return {"label": label, "score": sc, "target_gain": (tgt.get("vs_parent") or {}).get("delta"),
            "check_error": fb.get("error"), "name": ws.lora_name(lo) if lo.modules() else None}


def _apply(ws, tool: str, lo, **kw):
    """Run a workspace operator directly (not through ws.api: baselines are not charged to the editor)."""
    h = ws._new("L", lo)
    out = getattr(ws, "t_" + tool)(lora=h, **kw)
    return ws._get(out["lora"])


def random_edit_search(ws, raw, *, evals: int, seed: int, top_full: int = 3,
                       scales=(0.5, 0.75, 1.0, 1.25)) -> tuple[object, dict]:
    """mode=edit non-LLM branch: random rule-based edits of the raw LoRA (the operators the editor has,
    without an LLM): truncate_rank -> relchange_prune -> intruder_scale -> orthogonalize (against the stack) ->
    global scale. Same evaluate budget as the editor. Returns (best LoRA, log)."""
    rng = random.Random(f"{seed}:nonllm-edit")
    tried, seen = [], set()
    for _ in range(evals * 4):
        if len(tried) >= evals:
            break
        spec = {"truncate": rng.choice([None, 4, 8, 16, 32]), "prune": rng.choice([None, 0.01, 0.03, 0.05]),
                "intruder": rng.choice([None, 0.3, 0.5, 0.7]),
                "orth": bool(ws.stack) and rng.random() < 0.5, "scale": rng.choice(list(scales))}
        key = tuple(sorted(spec.items()))
        if key in seen:
            continue
        seen.add(key)
        lo = raw
        try:
            if spec["truncate"]:
                lo = _apply(ws, "truncate_rank", lo, k=spec["truncate"])
            if spec["prune"]:
                lo = _apply(ws, "relchange_prune", lo, frac=spec["prune"])
            if spec["intruder"]:
                lo = _apply(ws, "intruder_scale", lo, lam=spec["intruder"])
            if spec["orth"]:
                lo = _apply(ws, "orthogonalize", lo, against="stack")
            if spec["scale"] != 1.0:
                lo = _apply(ws, "scale", lo, s=spec["scale"])
        except Exception as e:  # noqa: BLE001
            tried.append(({"label": "edit", **spec, "score": None, "error": str(e)[:200]}, None))
            continue
        p = point(ws, lo, "edit " + ",".join(f"{k}={v}" for k, v in spec.items() if v))
        p.update(spec)
        tried.append((p, lo))
    return _pick_best(ws, tried, top_full)


def random_construct_search(ws, *, evals: int, seed: int, top_full: int = 3) -> tuple[object, dict]:
    """mode=scratch non-LLM branch: random gradient-free constructions from zero (singular-direction scaling
    of the current weights on a random set of MLP / attention output projections). Returns (best LoRA, log)."""
    rng = random.Random(f"{seed}:nonllm-scratch")
    pool = [m for m in ws.store.modules() if ws.store.info(m)["leaf"] in ("down_proj", "up_proj", "o_proj", "out_proj")]
    svds, tried = {}, []
    for i in range(evals):
        n = rng.choice([4, 8, 16])
        mods = sorted(rng.sample(pool, min(n, len(pool))), key=pool.index)
        k = rng.choice([1, 2, 4])
        amp = rng.choice([0.02, 0.05, 0.1])
        sign = rng.choice([1, -1])
        lo = ws._empty()
        for m in mods:
            if m not in svds:
                svds[m] = ws.t_svd(module=m, k=4)["handle"]
            factors = [1 + sign * amp * rng.uniform(0.5, 1.0) for _ in range(k)]
            h = ws._new("L", lo)
            lo = ws._get(ws.t_scale_singular(lora=h, module=m, svd=svds[m], idx=list(range(k)), factors=factors)["lora"])
        p = point(ws, lo, f"construct n={n},k={k},amp={sign * amp}")
        p.update({"modules": len(mods), "k": k, "amp": sign * amp})
        tried.append((p, lo))
    return _pick_best(ws, tried, top_full)


def _wapply(ws, tool: str, ed, **kw):
    """Run a workspace weight-edit tool directly (not charged to the editor)."""
    h = ws._new("W", ed)
    out = getattr(ws, "t_" + tool)(edit=h, **kw)
    return ws._get(out["edit"], "edit")


def random_weight_search(ws, *, evals: int, seed: int, top_full: int = 3) -> tuple[object, dict]:
    """mode=weights non-LLM branch: random structured edits of the current weights on a random set of MLP /
    attention output projections: row (output-channel) / column (input-channel) scaling of random channels, or
    whole-module scaling, with factors near 1 (sized to stay under weights.norm_cap_tau), optionally plus a
    singular-direction low-rank addition (scale_singular -> w_add_lora). Returns (best edit, log)."""
    rng = random.Random(f"{seed}:nonllm-weights")
    pool = [m for m in ws.store.modules() if ws.store.info(m)["leaf"] in ("down_proj", "up_proj", "o_proj", "out_proj")]
    tau = float(getattr(ws, "tau_w", 0.05))
    svds, tried = {}, []
    for i in range(evals):
        n = rng.choice([2, 4, 8])
        mods = sorted(rng.sample(pool, min(n, len(pool))), key=pool.index)
        kind = rng.choice(["rows", "cols", "scale"])
        sign = rng.choice([1, -1])
        amp = rng.choice([0.25, 0.5, 0.8])          # target relative change per module, as a fraction of tau
        singular = rng.random() < 0.3
        spec = {"kind": kind, "modules": len(mods), "amp": round(sign * amp * tau, 5), "singular": singular}
        ed = ws._empty_edit()
        try:
            if kind == "scale":
                ed = _wapply(ws, "w_scale", ed, modules=mods, s=1 + sign * amp * tau)
            else:
                k = rng.choice([8, 32, 128])
                spec["channels"] = k
                for m in mods:
                    out_f, in_f = ws.store.shape(m)
                    dim = out_f if kind == "rows" else in_f
                    idx = sorted(rng.sample(range(dim), min(k, dim)))
                    c = amp * tau * math.sqrt(dim / len(idx))   # per-channel deviation for that relative change
                    factors = [round(1 + sign * c * rng.uniform(0.5, 1.0), 6) for _ in idx]
                    if kind == "rows":
                        ed = _wapply(ws, "w_scale_rows", ed, module=m, factors=factors, rows=idx)
                    else:
                        ed = _wapply(ws, "w_scale_cols", ed, module=m, factors=factors, cols=idx)
            if singular:
                lo = ws._empty()
                for m in mods[:2]:
                    if m not in svds:
                        svds[m] = ws.t_svd(module=m, k=4)["handle"]
                    f = [1 + sign * rng.choice([0.02, 0.05]) for _ in range(2)]
                    h = ws._new("L", lo)
                    lo = ws._get(ws.t_scale_singular(lora=h, module=m, svd=svds[m], idx=[0, 1], factors=f)["lora"])
                ed = _wapply(ws, "w_add_lora", ed, lora=ws._new("L", lo), alpha=1.0)
        except Exception as e:  # noqa: BLE001  (e.g. a channel scaling over the cap)
            tried.append(({"label": "weights", **spec, "score": None, "error": str(e)[:200]}, None))
            continue
        p = point(ws, ed, "weights " + ",".join(f"{k}={v}" for k, v in spec.items()))
        p.update(spec)
        tried.append((p, ed))
    return _pick_best(ws, tried, top_full)


def _pick_best(ws, tried, top_full: int) -> tuple[object, dict]:
    """Best of `tried` [(point, LoRA)] by screen score, top_full re-scored on the full signal draw."""
    ok = [(p, lo) for p, lo in tried if p.get("score") is not None and lo is not None]
    ok.sort(key=lambda x: -x[0]["score"])
    best, best_full = None, -1e9
    for p, lo in ok[:top_full]:
        sc, _ = ws.evaluate_lora(lo, "full", charge=False)
        p["full_score"] = sc
        if sc is not None and sc > best_full:
            best, best_full = (p, lo), sc
    if best is None and ok:
        best = ok[0]
    log = {"points": [p for p, _ in tried], "best": best[0] if best else None}
    return (best[1] if best else None), log
