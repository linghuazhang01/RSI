"""Non-LLM baselines on the same build / evaluate path and the same diagnostics (METHOD_DESIGN §2.7).

uniform     best uniform (bits, group size) GPTQ config that fits the budget (embed/head bf16 -> 8 -> 4)
kl_greedy   solve_allocation over the profile table with unit weights (KL knapsack)
evo         EvoPress-style (1+lambda) evolutionary search from kl_greedy: mutations move one group up a
            bit level and others down until the budget holds; fitness = screen score; uses
            `baselines.evo_evaluations` evaluations (separate from the editor's budget)
Results: <gen dir>/baselines/<method>.json + summary.json. Run from run.py after each generation's
search when `baselines.methods` is set (e.g. --set 'baselines.methods=[uniform,kl_greedy,evo]').
Baselines keep their own evaluation path (2026-11-23: search candidates are never screened): QuantProblem.evaluate at
the screen stage (and full with baselines.full), score = retention vs the generation's incumbent on the same items;
they are not gated and never become the incumbent.
"""
from __future__ import annotations

import random

from .allocation import solve_allocation
from .common import write_json
from .quant import schema, sizes


def uniform_config(problem) -> dict:
    sp = problem.space
    for prec in ("bf16", 8, 4):
        if prec not in sp["embed"]:
            continue
        for b in sorted([b for b in sp["bits"] if b < 16], reverse=True):
            for gs in sorted(sp["group_sizes"], reverse=True):
                qz = "gptq" if "gptq" in sp["quantizers"] else sp["quantizers"][0]
                cfg = {"default": {"bits": b, "group_size": gs, "quantizer": qz}, "embed": prec, "lm_head": prec}
                if problem.size(cfg)["gib"] <= problem.budget_gib:
                    # with the largest fitting width, prefer the smallest group size that still fits
                    for g2 in sorted(sp["group_sizes"]):
                        c2 = {"default": {"bits": b, "group_size": g2, "quantizer": qz}, "embed": prec, "lm_head": prec}
                        if problem.size(c2)["gib"] <= problem.budget_gib:
                            return c2
                    return cfg
    raise ValueError("no uniform config fits the budget")


def kl_greedy_config(problem, cfg: dict) -> dict:
    sp = problem.space
    qz = "gptq" if "gptq" in sp["quantizers"] else sp["quantizers"][0]
    best = None
    for prec in ("bf16", 8):
        c = solve_allocation(problem.spec, problem.tables["profile"], problem.budget_gib, quantizer=qz,
                             group_sizes=cfg.get("solver", {}).get("group_sizes", [128]), embed=prec, lm_head=8,
                             gs_factor=cfg.get("solver", {}).get("group_size_factor"), action_space=sp)
        if "INFEASIBLE" not in c.get("note", ""):
            return c
        best = c
    return best


def _mutate(problem, norm: dict, rng: random.Random, ladder: list[int]) -> dict:
    groups = {k: dict(v) for k, v in norm["groups"].items()}
    keys = list(groups)
    up = rng.choice(keys)
    b = groups[up]["bits"]
    higher = [x for x in ladder if x > b]
    if not higher:
        return norm
    groups[up] = dict(groups[up], bits=higher[0], group_size=groups[up]["group_size"] or 128,
                      quantizer=groups[up]["quantizer"] if groups[up]["quantizer"] != "none" else "gptq")
    if groups[up]["bits"] >= 16:
        groups[up] = {"bits": 16, "group_size": 0, "quantizer": "none"}
    new = {"groups": groups, "embed": norm["embed"], "lm_head": norm["lm_head"]}
    for _ in range(200):
        if sizes.config_bytes(new, problem.spec)["gib"] <= problem.budget_gib:
            return new
        down = rng.choice(keys)
        if down == up:
            continue
        b = groups[down]["bits"]
        lower = [x for x in ladder if x < b]
        if lower:
            q = groups[down]["quantizer"] if groups[down]["quantizer"] != "none" else "gptq"
            groups[down] = {"bits": lower[-1], "group_size": groups[down]["group_size"] or 128, "quantizer": q}
    return norm


def run_baselines(problem, cfg: dict, methods: list[str], out_dir, *, stage_full: bool = False, log=print) -> dict:
    out = {}
    for m in methods:
        try:
            if m == "uniform":
                c = uniform_config(problem)
                r = problem.evaluate(c, "screen", f"base_{m}", origin="baseline")
                res = {"config": c, "screen": r}
            elif m == "kl_greedy":
                c = kl_greedy_config(problem, cfg)
                r = problem.evaluate(c, "screen", f"base_{m}", origin="baseline")
                res = {"config": c, "screen": r}
            elif m == "evo":
                rng = random.Random(problem.rc.seed)
                ladder = sorted(problem.space["bits"])
                cur_cfg = kl_greedy_config(problem, cfg)
                cur = problem.normalize(cur_cfg)
                cur_r = problem.evaluate(cur_cfg, "screen", "base_evo_0", origin="baseline")
                trace = [{"i": 0, "score": cur_r["score"], "hash": cur_r["hash"]}]
                n = int(cfg["baselines"].get("evo_evaluations", 24))
                for i in range(1, n):
                    child = cur
                    for _ in range(rng.randint(1, 3)):
                        child = _mutate(problem, child, rng, ladder)
                    child_cfg = schema.compress(child, problem.spec)
                    r = problem.evaluate(child_cfg, "screen", f"base_evo_{i}", origin="baseline")
                    trace.append({"i": i, "score": r["score"], "hash": r["hash"]})
                    if r["score"] is not None and (cur_r["score"] is None or r["score"] >= cur_r["score"]):
                        cur, cur_r = child, r
                res = {"config": schema.compress(cur, problem.spec), "screen": cur_r, "trace": trace}
            else:
                raise ValueError(f"unknown baseline {m!r}")
            if stage_full:
                res["full"] = problem.evaluate(res["config"], "full", f"base_{m}", origin="baseline")
            res["bytes_gib"] = round(problem.size(res["config"])["gib"], 4)
            out[m] = {"score": res["screen"]["score"], "bytes_gib": res["bytes_gib"],
                      "full_score": (res.get("full") or {}).get("score")}
        except Exception as e:  # noqa: BLE001 - one failing baseline must not stop the run
            res = {"error": f"{type(e).__name__}: {e}"}
            out[m] = res
            log(f"[baselines] {m} failed: {e}")
        write_json(out_dir / f"{m}.json", res)
    return out
