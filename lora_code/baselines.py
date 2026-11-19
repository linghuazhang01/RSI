"""Non-LLM branches and strong baselines (RSI_LOOP_DESIGN §3.5, METHOD_DESIGN §3.6).

- SFT baseline: fixed recipe (rank 16 by default, all attention/MLP projections except in_proj_a/b), trained on
  this generation's `train` items with the SAME gradient budget (GPU-seconds) as the editor; checkpoints at
  25/50/75/100% are kept for the checkpoint frontier.
- Global-scale frontier: SFT LoRA × s for a grid of s.
- Checkpoint frontier: the SFT checkpoints.
- Raw LoRA (method.mode=edit): the `raw` recipe (null fields = SFT recipe; raw.n_train = how many train items),
  trained once per generation; the editor edits it. Identical to the SFT recipe -> the SFT LoRA is reused.
- Non-LLM search (the 4th branch), same evaluate budget as the editor, by method.mode:
  free: random search over SFT hyper-parameters (lr, rank, module set, steps fraction) × checkpoint × global
  scale (same total gradient budget split over the sampled configurations); edit: random rule-based edits of
  the raw LoRA (random_edit_search); scratch: random gradient-free constructions (random_construct_search).
  Best by screen, top-3 re-scored on the full signal.
All points report target gain, worst retention delta and token-KL vs the current model.
"""
from __future__ import annotations

import math
import random

from . import mock as M

DEFAULT_SFT = {"rank": 16, "lr": 1e-4, "epochs": 1.0, "steps": None, "batch_size": 4,
               "leaves": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                          "gate_proj", "up_proj", "down_proj"], "ckpt_fracs": [0.25, 0.5, 0.75, 1.0]}


def recipe(cfg: dict) -> dict:
    r = dict(DEFAULT_SFT)
    r.update(cfg.get("sft", {}) or {})
    return r


RAW_KEYS = ("rank", "lr", "epochs", "steps", "batch_size", "leaves")


def raw_recipe(cfg: dict) -> tuple[dict, int | None, float, bool]:
    """mode=edit: (recipe, n_train, gpu_seconds, same_as_sft) of the raw LoRA the editor starts from.
    Null raw.* values take the SFT recipe's; same_as_sft=True means the SFT LoRA can be reused as is."""
    sft = recipe(cfg)
    rc = cfg.get("raw", {}) or {}
    r = dict(sft)
    for k in RAW_KEYS:
        if rc.get(k) is not None:
            r[k] = rc[k]
    gb_default = float(cfg.get("budget", {}).get("gpu_seconds", 7200))
    gb = float(rc["gpu_seconds"]) if rc.get("gpu_seconds") is not None else gb_default
    n = int(rc["n_train"]) if rc.get("n_train") is not None else None
    same = r == sft and n is None and gb == gb_default
    return r, n, gb, same


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


def train_sft(ws, rec: dict, gpu_budget_s: float, *, seed: int = 42, modules=None, ids=None):
    """Returns (final LoRA, [(frac, LoRA)], info). Uses ws.train / ws.hf (or the mock model).
    ids: train items to use (default: all of ws.train)."""
    mods = modules or modules_for(ws.store, rec["leaves"], rec.get("layers"))
    ids = ws.train.check(list(ids)) if ids is not None else ws.train.ids()
    bs = int(rec["batch_size"])
    n_ex = max(1, len([e for e in ws.train.get(ids) if e.get("target")]) if not ws.mock else len(ids))
    steps = int(rec["steps"] or max(1, math.ceil(n_ex * float(rec["epochs"]) / bs)))
    fracs = list(rec.get("ckpt_fracs") or [1.0])
    if ws.mock:
        spe = float(ws.cfg.get("mock", {}).get("sec_per_example", 0.6))
        steps = min(steps, max(1, int(gpu_budget_s / (bs * spe))))
        final = M.m_sgd(None, mods, ids, steps, float(rec["lr"]), int(rec["rank"]))
        ck = [(f, M.m_sgd(None, mods, ids, max(1, int(steps * f)), float(rec["lr"]), int(rec["rank"]))) for f in fracs]
        return final, ck, {"steps": steps, "modules": len(mods), "gpu_seconds": round(steps * bs * spe, 1),
                           "rank": int(rec["rank"]), "lr": float(rec["lr"])}
    ex = ws._examples(ids)
    t0 = ws.hf.gpu_seconds
    final, info = ws.hf.sgd(ex, None, modules=mods, rank=int(rec["rank"]), steps=steps, lr=float(rec["lr"]),
                            batch_size=bs, warmup=int(rec.get("warmup", 0)), time_budget_s=gpu_budget_s,
                            checkpoints=fracs, init_seed=seed)
    ck = info.pop("checkpoints", [])
    info.update({"steps_planned": steps, "gpu_seconds": round(ws.hf.gpu_seconds - t0, 1), "lr": float(rec["lr"])})
    info["loss_first"], info["loss_last"] = (info["losses"][0], info["losses"][-1]) if info["losses"] else (None, None)
    info.pop("losses", None)
    return final, ck, info


def scaled(ws, lo, s: float):
    if ws.mock:
        return M.m_scale(lo, s)
    from .primitives.ops import scale
    return scale(lo, s)


def point(ws, lo, label: str, level: str = "screen") -> dict:
    """Evaluate one LoRA (not charged to the editor budget) and summarise the trade-off."""
    sc, fb = ws.evaluate_lora(lo, level, charge=False)
    tgt = fb.get("target") or {}
    ret = fb.get("retention") or {}
    ds = [(v.get("vs_parent") or {}).get("delta") for v in ret.values()]
    ds = [d for d in ds if d is not None]
    return {"label": label, "score": sc, "target_gain": (tgt.get("vs_parent") or {}).get("delta"),
            "worst_retention_delta": min(ds) if ds else None, "kl": fb.get("kl_vs_parent"),
            "check_error": fb.get("error"), "name": ws.lora_name(lo) if lo.modules() else None}


def frontiers(ws, sft_final, sft_ckpts, scales) -> dict:
    return {"global_scale": [point(ws, scaled(ws, sft_final, s), f"scale={s}") for s in scales],
            "checkpoint": [point(ws, lo, f"ckpt={f:.2f}") for f, lo in sft_ckpts]}


def random_search(ws, rec: dict, *, evals: int, gpu_budget_s: float, n_configs: int, scales, seed: int,
                  top_full: int = 3) -> tuple[object, dict]:
    """Same evaluate budget and total gradient budget as the editor. Returns (best LoRA, log)."""
    rng = random.Random(f"{seed}:nonllm")
    leaves_opts = {"mlp": ["gate_proj", "up_proj", "down_proj"],
                   "attn": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"],
                   "all": rec["leaves"]}
    configs, pool = [], []
    per = gpu_budget_s / max(n_configs, 1)
    for i in range(n_configs):
        c = dict(rec)
        c["lr"] = float(10 ** rng.uniform(math.log10(3e-5), math.log10(1e-3)))
        c["rank"] = rng.choice([8, 16, 32, 64])
        key = rng.choice(list(leaves_opts))
        c["leaves"] = leaves_opts[key]
        c["epochs"] = float(rec["epochs"]) * rng.choice([0.5, 1.0])
        final, ck, info = train_sft(ws, c, per, seed=seed + 1 + i)
        configs.append({"i": i, "lr": c["lr"], "rank": c["rank"], "modules": key, "epochs": c["epochs"], "train": info})
        for f, lo in ck:
            pool.append((i, f, lo))
    combos = [(i, f, s) for (i, f, _) in pool for s in scales]
    rng.shuffle(combos)
    lookup = {(i, f): lo for (i, f, lo) in pool}
    tried = []
    for i, f, s in combos[:evals]:
        lo = scaled(ws, lookup[(i, f)], s)
        p = point(ws, lo, f"cfg={i},ckpt={f:.2f},scale={s}")
        p.update({"cfg": i, "ckpt": f, "scale": s})
        tried.append((p, lo))
    best, log = _pick_best(ws, tried, top_full)
    log["configs"] = configs
    return best, log


def _apply(ws, tool: str, lo, **kw):
    """Run a workspace operator directly (not through ws.api: baselines are not charged to the editor)."""
    h = ws._new("L", lo)
    out = getattr(ws, "t_" + tool)(lora=h, **kw)
    return ws._get(out["lora"])


def random_edit_search(ws, raw, *, evals: int, seed: int, top_full: int = 3, scales=(0.5, 0.75, 1.0, 1.25),
                       use_regmean: bool = True) -> tuple[object, dict]:
    """mode=edit non-LLM branch: random rule-based edits of the raw LoRA (the operators the editor has,
    without an LLM): truncate_rank -> relchange_prune -> intruder_scale -> regmean_preserve -> orthogonalize
    (against the stack) -> global scale. Same evaluate budget as the editor. Returns (best LoRA, log)."""
    rng = random.Random(f"{seed}:nonllm-edit")
    tried, seen = [], set()
    for _ in range(evals * 4):
        if len(tried) >= evals:
            break
        spec = {"truncate": rng.choice([None, 4, 8, 16, 32]), "prune": rng.choice([None, 0.01, 0.03, 0.05]),
                "intruder": rng.choice([None, 0.3, 0.5, 0.7]),
                "regmean": rng.choice([None, 0.5, 1.0, 2.0]) if use_regmean else None,
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
            if spec["regmean"]:
                lo = _apply(ws, "regmean_preserve", lo, alpha=spec["regmean"])
            if spec["orth"]:
                lo = _apply(ws, "orthogonalize", lo, against="stack")
            if spec["scale"] != 1.0:
                lo = _apply(ws, "scale", lo, s=spec["scale"])
        except Exception as e:  # noqa: BLE001  (e.g. GPU budget for regmean exhausted)
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
