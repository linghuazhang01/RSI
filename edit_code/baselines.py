"""Classic (non-LLM) baselines for Paper A (2026-11-23), scored exactly like the method: built at the same size
budget, gated ONCE on the same gate set (simple_<d>_eval, n 4, same seed) paired with Q_0's stored gate records, and
held-out evaluated on simple_<d>_test (run.Runner.run_baselines, after the generations; baselines.methods).

  uniform_rtn   budget.reference (uniform W4 g128, embed / lm_head 8) with RTN instead of GPTQ; uniform GPTQ is Q_0
  uniform_awq   the same with AWQ(-lite). Its per-input-channel fp16 scales make it a few MiB larger than the budget;
                it is evaluated anyway and the overshoot is reported (over_budget_mib)
  hawq          HAWQ-style mixed precision: Hessian-aware sensitivity (the hessian table: tr(dW H dW^T) per token
                when only that group is GPTQ-quantized) + a multiple-choice knapsack at the budget
  kl_greedy     sensitivity = the profile table's whole-model KL to M_0 when only that group is RTN-quantized, +
                the same knapsack (solve_allocation with unit weights)
  evopress      EvoPress-style evolutionary search (Sieberling et al.): from Q_0, level-switch mutations at the
                budget (GPTQ g128 levels = the action space's bits), fitness = KL to M_0 on the calibration contexts
                (edit_code.evopress: one HF model per GPU), two-stage selection; NOT selected on the gate set

The knapsack / search baselines keep embed and lm_head at the reference's precision (8 bit), like Q_0.
"""
from __future__ import annotations

import copy
import random

from .allocation import solve_allocation
from .quant import schema, sizes

METHODS = ("uniform_rtn", "uniform_awq", "hawq", "kl_greedy", "evopress")


def check_methods(methods) -> list[str]:
    out = [str(m) for m in (methods or [])]
    bad = [m for m in out if m not in METHODS]
    if bad:
        raise ValueError(f"baselines.methods: unknown {bad}; choose from {list(METHODS)} (2026-11-23: uniform / evo "
                         "were replaced by uniform_rtn / uniform_awq / evopress)")
    return out


def uniform_config(reference: dict, quantizer: str) -> dict:
    """budget.reference with every quantizer replaced (default, rules and groups)."""
    c = copy.deepcopy(reference)

    def fix(s):
        if isinstance(s, dict) and int(s.get("bits", 16)) < 16:
            s["quantizer"] = quantizer
    fix(c.get("default"))
    for r in c.get("rules") or []:
        fix(r)
    for s in (c.get("groups") or {}).values():
        fix(s)
    c["note"] = f"baseline uniform_{quantizer}: budget.reference with {quantizer}"
    return c


def _knapsack(problem, rows: list[dict], reference: dict, budget_gib: float, label: str) -> dict:
    sp = problem.space
    qz = "gptq" if "gptq" in sp["quantizers"] else sp["quantizers"][0]
    c = solve_allocation(problem.spec, rows, budget_gib, quantizer=qz, group_sizes=[128],
                         embed=reference.get("embed", "bf16"), lm_head=reference.get("lm_head", 8), action_space=sp)
    if "INFEASIBLE" in c.get("note", ""):
        raise ValueError(f"{label}: no allocation fits the budget ({c['note']})")
    c["note"] = f"baseline {label}: " + c.get("note", "")
    return c


def hawq_config(problem, reference: dict, budget_gib: float) -> dict:
    """HAWQ-style: damage(group, bits) = Hessian-weighted output error per token of GPTQ g128 at that width (the
    hessian table); 8 bit is extrapolated (/4 per bit) and 16 bit is 0, as in solve_allocation."""
    rows = [{"key": r["key"], "bits": int(r["bits"]), "group_size": int(r["group_size"]),
             "dkl_mean": float(r["herr_per_token"])}
            for r in problem.tables.get("hessian") or [] if r.get("quantizer") == "gptq"]
    if not rows:
        raise ValueError("hawq: the hessian table has no gptq rows")
    return _knapsack(problem, rows, reference, budget_gib, "hawq")


def kl_greedy_config(problem, reference: dict, budget_gib: float) -> dict:
    return _knapsack(problem, problem.tables["profile"], reference, budget_gib, "kl_greedy")


# ----------------------------------------------------------------------------- EvoPress-style search
def levels(problem) -> list[int]:
    return sorted({int(b) for b in problem.space["bits"]})


def mutate(spec, budget_gib: float, norm: dict, rng: random.Random, ladder: list[int]) -> dict:
    """Level switch: one group one level up, then random other groups one level down until the config fits the
    budget again (GPTQ g128 at every level below 16). Returns norm unchanged when it cannot."""
    groups = {k: dict(v) for k, v in norm["groups"].items()}
    keys = sorted(groups)
    up = rng.choice(keys)
    higher = [x for x in ladder if x > int(groups[up]["bits"])]
    if not higher:
        return norm
    groups[up] = _level(higher[0])
    new = {"groups": groups, "embed": norm["embed"], "lm_head": norm["lm_head"]}
    for _ in range(400):
        if sizes.config_bytes(new, spec)["gib"] <= budget_gib + 1e-9:
            return new
        down = rng.choice(keys)
        if down == up:
            continue
        lower = [x for x in ladder if x < int(groups[down]["bits"])]
        if lower:
            groups[down] = _level(lower[-1])
    return norm


def _level(bits: int) -> dict:
    return {"bits": 16, "group_size": 0, "quantizer": "none"} if bits >= 16 else \
        {"bits": int(bits), "group_size": 128, "quantizer": "gptq"}


def evopress_search(problem, reference: dict, budget_gib: float, fitness, *, generations: int, offspring: int,
                    survivors: int, max_mutations: int, seed: int, state: dict | None = None, checkpoint=None,
                    log=print) -> dict:
    """(1 + offspring) evolution with two-stage selection. fitness(norms, stage) -> [KL] with stage "small" (a
    quarter of the calibration contexts) or "full". Deterministic given the seed: results are collected per
    generation before selecting. state / checkpoint: resume (checkpoint(state) after every generation)."""
    ladder = levels(problem)
    if state:
        rng = random.Random()
        st = state["rng"]
        rng.setstate((st[0], tuple(st[1]), st[2]))
        parent, f_parent, trace, start = state["parent"], state["f_parent"], state["trace"], state["gen"]
    else:
        rng = random.Random(seed)
        parent = problem.normalize(reference)
        parent = {"groups": {k: _level(int(s["bits"])) for k, s in parent["groups"].items()},
                  "embed": parent["embed"], "lm_head": parent["lm_head"]}
        f_parent = fitness([parent], "full")[0]
        trace, start = [{"gen": 0, "fitness": f_parent, "evaluations": 1}], 0
    n_eval = trace[-1]["evaluations"]
    for g in range(start + 1, generations + 1):
        kids, seen = [], {schema_key(parent)}
        for _ in range(offspring * 4):
            if len(kids) >= offspring:
                break
            child = parent
            for _ in range(rng.randint(1, max_mutations)):
                child = mutate(problem.spec, budget_gib, child, rng, ladder)
            k = schema_key(child)
            if k not in seen:
                seen.add(k)
                kids.append(child)
        if not kids:
            break
        f_small = fitness(kids, "small")
        order = sorted(range(len(kids)), key=lambda i: (f_small[i], i))[:max(1, survivors)]
        f_full = fitness([kids[i] for i in order], "full")
        n_eval += len(kids) + len(order)
        best = min(range(len(order)), key=lambda j: (f_full[j], j))
        if f_full[best] < f_parent:
            parent, f_parent = kids[order[best]], f_full[best]
        trace.append({"gen": g, "fitness": f_parent, "best_child": f_full[best], "evaluations": n_eval})
        if g == 1 or g % 5 == 0 or g == generations:
            log(f"[baselines] evopress gen {g}/{generations}: KL {f_parent:.5f} ({n_eval} evaluations)")
        if checkpoint is not None:
            st = rng.getstate()
            checkpoint({"gen": g, "parent": parent, "f_parent": f_parent, "trace": trace,
                        "rng": [st[0], list(st[1]), st[2]]})
    cfg = schema.compress(parent, problem.spec)
    cfg["note"] = f"baseline evopress: KL {f_parent:.5f} after {trace[-1]['gen']} generations, {n_eval} evaluations"
    return {"config": cfg, "fitness": f_parent, "trace": trace, "evaluations": n_eval}


def schema_key(norm: dict) -> str:
    import json
    return json.dumps([sorted((k, int(s["bits"])) for k, s in norm["groups"].items()), norm["embed"],
                       norm["lm_head"]], sort_keys=True)


def mock_fitness(problem, norms: list[dict], stage: str) -> list[float]:
    """Mock runs: 1 - quality of the mock model the config gives (no GPU)."""
    from . import mock as M
    lam = problem.cfg.get("mock", {}).get("lam", 1.2)
    return [round(1.0 - M.apply_config(problem.spec, problem.source_state, n, lam)["quality"], 8) for n in norms]
