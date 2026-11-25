"""Raw LoRA training (mode=edit, generation 0 only).

2026-11-20: there is no SFT baseline and nothing else is ever trained. The only training of a run is the
raw LoRA of mode=edit: the fixed `raw` recipe (rank 16 by default, all attention / MLP projections except
in_proj_a/b), trained once in generation 0 on raw.n_train of that generation's train items; the editor edits
it in every generation.
A point reports the target gain vs the current model on the fixed training set.
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
    sc, fb = ws.evaluate_lora(lo, level)
    tgt = fb.get("target") or {}
    return {"label": label, "score": sc, "target_gain": (tgt.get("vs_parent") or {}).get("delta"),
            "check_error": fb.get("error"), "name": ws.lora_name(lo) if lo.modules() else None}
