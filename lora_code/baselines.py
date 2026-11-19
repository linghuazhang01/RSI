"""Non-LLM branches and strong baselines (RSI_LOOP_DESIGN §3.5, METHOD_DESIGN §3.6).

- SFT baseline: fixed recipe (rank 16 by default, all attention/MLP projections except in_proj_a/b), trained on
  this generation's `train` items with the SAME gradient budget (GPU-seconds) as the editor; checkpoints at
  25/50/75/100% are kept for the checkpoint frontier.
- Global-scale frontier: SFT LoRA × s for a grid of s.
- Checkpoint frontier: the SFT checkpoints.
- Non-LLM search (the 4th branch): random search over SFT hyper-parameters (lr, rank, module set, steps
  fraction) × checkpoint × global scale, with the same evaluate budget as the editor and the same total
  gradient budget split over the sampled configurations; best by screen, top-3 re-scored on the full signal.
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


def train_sft(ws, rec: dict, gpu_budget_s: float, *, seed: int = 42, modules=None):
    """Returns (final LoRA, [(frac, LoRA)], info). Uses ws.train / ws.hf (or the mock model)."""
    mods = modules or modules_for(ws.store, rec["leaves"], rec.get("layers"))
    ids = ws.train.ids()
    bs = int(rec["batch_size"])
    n_ex = max(1, len([e for e in ws.train.examples.values() if e.get("target")]) if not ws.mock else len(ids))
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
    ok = [(p, lo) for p, lo in tried if p["score"] is not None]
    ok.sort(key=lambda x: -x[0]["score"])
    best, best_full = None, -1e9
    for p, lo in ok[:top_full]:
        sc, _ = ws.evaluate_lora(lo, "full", charge=False)
        p["full_score"] = sc
        if sc is not None and sc > best_full:
            best, best_full = (p, lo), sc
    if best is None and ok:
        best = ok[0]
    log = {"configs": configs, "points": [p for p, _ in tried], "best": best[0] if best else None}
    return (best[1] if best else None), log
