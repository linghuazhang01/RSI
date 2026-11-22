"""Mock mode for Paper A: no weights. A synthetic "true damage" model drives everything:

* each module group g has a sensitivity; quantizing it to (bits, group size, quantizer) adds damage
  dkl(g, setting) in nats (re-quantizing at >= the parent's bits adds only 10% "re-rounding" noise);
* a mock model state = accumulated damage per group + current bits; quality = base * exp(-lam * D);
* the profile table reports the incremental damage (+-10% deterministic noise) relative to the
  CURRENT model, with the same schema as the real table;
* candidate quality is registered in the mock registry so the rsicore mock server answers with
  that accuracy; KL feedback is computed from the same damage.

Also provides the mock editor policy: a program that reads model_info/profile_table and fits the
budget, varying its strategy across threads and steps.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .common import write_json, read_json
from .quant.spec import ModelSpec

GROUP_SENS = {"qk": 1.0, "v": 1.6, "o": 1.0, "gdn_qkv": 1.0, "gdn_z": 0.6, "gdn_ba": 3.0, "gdn_out": 1.0,
              "gate_up": 0.8, "down": 1.4}
BITS_DAMAGE = {16: 0.0, 8: 0.0005, 4: 0.01, 3: 0.04, 2: 0.25}
GS_FACTOR = {0: 1.0, 128: 1.0, 64: 0.8, 32: 0.65}
QZ_FACTOR = {"none": 0.0, "rtn": 1.0, "awq": 0.8, "gptq": 0.7}
EMBED_DAMAGE = {"bf16": 0.0, 8: 0.001, 4: 0.03}
DOMAIN_KL = {"math": 1.2, "code": 1.1, "knowledge": 0.8, "instruction_following": 0.9, "tool_calling": 1.1}
SCALE = 0.011
REQUANT = 0.1


def _h(*parts) -> float:
    """deterministic pseudo-random number in [0, 1)."""
    s = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(s[:12], 16) / float(16 ** 12)


def layer_factor(layer: int, n_layers: int) -> float:
    if layer < 2:
        return 2.5
    if layer >= n_layers - 4:
        return 2.0
    return 0.8 + 0.4 * _h("layer", layer)


def group_weight(spec: ModelSpec, key: str, _cache={}) -> float:
    ck = (id(spec), key)
    if ck not in _cache:
        gp = spec.group_params()
        mean = sum(gp.values()) / len(gp)
        layer, g = key.split(".", 1)
        _cache[ck] = GROUP_SENS[g] * layer_factor(int(layer), spec.num_layers) * math.sqrt(gp[key] / mean) * SCALE
    return _cache[ck]


def setting_damage(spec: ModelSpec, key: str, s: dict) -> float:
    b = int(s["bits"])
    if b >= 16:
        return 0.0
    return group_weight(spec, key) * BITS_DAMAGE[b] * GS_FACTOR[int(s.get("group_size") or 128)] * \
        QZ_FACTOR[s.get("quantizer", "rtn")]


def fresh_state(base_quality: float) -> dict:
    return {"quality": base_quality, "base_quality": base_quality, "damage": {}, "bits": {},
            "embed": "bf16", "lm_head": "bf16", "embed_damage": 0.0, "generation": 0}


def increment(spec: ModelSpec, parent: dict, key: str, s: dict) -> float:
    pb = int(parent["bits"].get(key, 16))
    d = setting_damage(spec, key, s)
    return d if int(s["bits"]) < pb else REQUANT * d


def apply_config(spec: ModelSpec, parent: dict, norm: dict, lam: float = 1.2) -> dict:
    """State of the model obtained by quantizing `parent` with the normalized config."""
    dmg, bits = dict(parent["damage"]), dict(parent["bits"])
    inc_total = 0.0
    for key, s in norm["groups"].items():
        inc = increment(spec, parent, key, s)
        dmg[key] = dmg.get(key, 0.0) + inc
        bits[key] = min(int(bits.get(key, 16)), int(s["bits"]))
        inc_total += inc
    emb = parent["embed_damage"]
    for part in ("embed", "lm_head"):
        p = norm[part]
        inc = EMBED_DAMAGE[p] * (1.0 if _prec_rank(p) < _prec_rank(parent[part]) else REQUANT)
        emb += inc
        inc_total += inc
    total = sum(dmg.values()) + emb
    q = parent["base_quality"] * math.exp(-lam * total)
    return {"quality": q, "base_quality": parent["base_quality"], "damage": dmg, "bits": bits,
            "embed": _min_prec(parent["embed"], norm["embed"]), "lm_head": _min_prec(parent["lm_head"], norm["lm_head"]),
            "embed_damage": emb, "generation": parent.get("generation", 0) + 1,
            "kl_vs_parent": inc_total, "kl_vs_original": total}


def _prec_rank(p) -> int:
    return 16 if p == "bf16" else int(p)


def _min_prec(a, b):
    return a if _prec_rank(a) <= _prec_rank(b) else b


def candidate_kl(spec: ModelSpec, parent: dict, norm: dict, domains: list[str]) -> dict:
    st = apply_config(spec, parent, norm)
    inc = st["kl_vs_parent"]
    out = {"overall": round(inc, 5), "vs_original": round(st["kl_vs_original"], 5)}
    for d in domains:
        out[d] = round(inc * DOMAIN_KL.get(d, 1.0), 5)
    return out


# ------------------------------------------------------------------------------ diagnostics
def profile_table(spec: ModelSpec, parent: dict, bits_list, group_size: int = 128) -> list[dict]:
    rows = []
    gp = spec.group_params()
    for key in spec.group_keys():
        layer, g = key.split(".", 1)
        for b in bits_list:
            s = {"bits": b, "group_size": group_size, "quantizer": "rtn"}
            d = increment(spec, parent, key, s) * (0.9 + 0.2 * _h("prof", key, b, parent.get("generation", 0)))
            rows.append({"key": key, "layer": int(layer), "group": g, "bits": b, "group_size": group_size,
                         "quantizer": "rtn", "dkl_mean": round(d, 7), "dkl_p99": round(d * 6, 7),
                         "top1_drop": round(min(1.0, d * 3), 6), "params": gp[key]})
    return rows


def activation_table(spec: ModelSpec, generation: int = 0) -> list[dict]:
    rows = []
    for m in spec.modules:
        r = _h("act", m.name)
        ratio = 3 + 40 * r ** 4 + (60 if (m.group == "down" and m.layer in (1, spec.num_layers - 2)) else 0)
        rows.append({"layer": m.layer, "group": m.group, "module": m.proj, "input": m.input_key,
                     "rms_mean": round(0.5 + r, 4), "rms_max_over_median": round(ratio / 2, 3),
                     "absmax_over_median": round(ratio, 3), "outlier_frac": round(0.02 * r ** 3, 5),
                     "red_flag": bool(ratio > 30)})
    return rows


# ------------------------------------------------------------------------------ registry
def register(registry: Path, name: str, quality: float) -> None:
    reg = read_json(registry, {}) or {}
    reg[name] = round(float(quality), 6)
    write_json(registry, reg)


# ------------------------------------------------------------------------------ editor policy
_MARK = re.compile(r"\[thread=(\d+) step=(\d+)(?: task=(\w+))?(?: gen=(\d+))?\]")

PROGRAM_TEMPLATE = '''
def main(api):
    """mock editor: strategy={strategy}, margin={margin}, protect_last={protect}"""
    info = api.model_info()
    budget = info["budget_gib"] * {margin}
    tools = info["tools"]
    rules = []
    if {protect} > 0:
        n = info["num_layers"]
        rules.append({{"layers": "%d-%d" % (n - {protect}, n - 1), "groups": ["down", "gate_up"], "bits": 8}})
    if "{strategy}" == "solver" and "solve_allocation" in tools:
        weights = {{"*.down": {down_w}, "*.v": 1.5, "*.gdn_ba": 3.0}}
        cfg = api.solve_allocation(budget_gib=budget, weights=weights, quantizer="{quantizer}",
                                   fixed_rules=rules)
        return cfg
    # hand-written greedy: start every group at the lowest bit width, upgrade the groups with the
    # largest profiled damage per byte while the budget allows
    prof = api.profile_table()
    dmg = {{}}
    for r in prof:
        dmg[(r["key"], r["bits"])] = r["dkl_mean"]
    keys = sorted({{r["key"] for r in prof}})
    ladder = [b for b in (2, 3, 4, 8) if b in info["allowed_bits"]]
    lo = ladder[0] if "{strategy}" != "uniform" else ladder[1]
    choice = {{k: lo for k in keys}}
    def cfg_of(ch):
        return {{"default": {{"bits": lo, "group_size": {gs}, "quantizer": "{quantizer}"}},
                "rules": rules, "groups": {{k: {{"bits": b}} for k, b in ch.items() if b != lo}},
                "embed": {embed}, "lm_head": 8}}
    if api.bytes(config=cfg_of(choice))["gib"] > budget:
        return cfg_of(choice)
    improved = True
    while improved:
        improved = False
        best, best_gain = None, 0.0
        for k in keys:
            b = choice[k]
            nxt = [x for x in ladder if x > b]
            if not nxt:
                continue
            gain = dmg.get((k, b), 0.0) - dmg.get((k, nxt[0]), 0.0)
            if gain > best_gain:
                best, best_gain = (k, nxt[0]), gain
        if best is not None:
            trial = dict(choice)
            trial[best[0]] = best[1]
            if api.bytes(config=cfg_of(trial))["gib"] <= budget:
                choice = trial
                improved = True
            else:
                dmg[(best[0], choice[best[0]])] = 0.0   # cannot afford: skip this group
                improved = True
    {maybe_eval}
    if "screen_results" in tools:           # the incumbent's screen run (objective simple, 2026-11-21)
        rows = api.screen_results(filter="wrong", limit=2) or api.screen_results(limit=1)
        print("screen_results", [(r["id"], r["scores"]) for r in rows])
        if rows:
            out = api.screen_output(id=rows[0]["id"], sample=0)
            print("screen_output", out["total_chars"], "thinking" if out["has_thinking"] else "no thinking")
    if "group_error" in tools:              # analysis tools (free): where the error concentrates
        ge = api.group_error(config=cfg_of(choice), top=3)
        print("group_error top", ge["top"])
    if "compare" in tools:                  # item flips vs the incumbent (ids only)
        cmp = api.compare(cid="incumbent")
        print("compare vs incumbent", {{k: v["net"] for k, v in cmp["datasets"].items()}})
    return cfg_of(choice)
'''


PROBE_EVAL = """try:
        if "evaluate" not in tools:         # program_evaluate_calls 0 (gate per candidate): no evaluate tool
            raise api.ToolError("no evaluate tool")
        fb = api.evaluate(config=cfg_of(choice))
        print("probe evaluate:", fb.get("score"))
        again = api.evaluate(config=cfg_of(choice))
        print("probe again cached:", again.get("cached"))
        other = dict(cfg_of(choice), lm_head=4)
        api.evaluate(config=other)
        print("probe second config evaluated")
    except api.BudgetExceeded as e:
        print("probe cap:", str(e)[:120])
    except api.ToolError as e:
        print("probe tool error:", str(e)[:120])"""


REFINE_TEMPLATE = """
def main(api):
    # mock editor: strategy=refine (bit exchange around the incumbent, profile damage per MiB), rounds=__ROUNDS__, up_max=__UPMAX__
    info = api.model_info()
    budget = info["budget_gib"]
    dmg = {}
    for r in api.profile_table():
        dmg[(r["key"], r["bits"])] = r["dkl_mean"]
    mib = {r["key"]: r["mib"] for r in api.bytes_table()}
    keys = sorted(mib)
    inc = info.get("incumbent_config") or {}
    base = int((inc.get("default") or {}).get("bits", 3))
    bits = {k: base for k in keys}
    for k, s in (inc.get("groups") or {}).items():
        if k in bits and "bits" in s:
            bits[k] = int(s["bits"])
    ladder = [b for b in (2, 3, 4, 8) if b in info["allowed_bits"]]

    def d(k, b):
        return dmg.get((k, b), 0.0)

    def m(k, b):
        return mib[k]["b%d_g128" % b] if b < 16 else mib[k]["bf16"]

    def cfg_of():
        return {"default": {"bits": base, "group_size": 128, "quantizer": "__QUANT__"},
                "groups": {k: {"bits": b} for k, b in bits.items() if b != base},
                "embed": inc.get("embed", 8), "lm_head": inc.get("lm_head", 8)}

    slack = (budget - api.bytes(config=cfg_of())["gib"]) * 1024 * 0.999
    def options():
        ups, downs = [], []
        for k in keys:
            nxt = [b for b in ladder if b > bits[k] and b <= __UPMAX__]
            if nxt:
                cost, gain = m(k, nxt[0]) - m(k, bits[k]), d(k, bits[k]) - d(k, nxt[0])
                if cost > 0 and gain > 0:
                    ups.append((gain / cost, k, nxt[0], cost, gain))
            low = [b for b in ladder if b < bits[k]]
            if low:
                save, loss = m(k, bits[k]) - m(k, low[-1]), d(k, low[-1]) - d(k, bits[k])
                if save > 0:
                    downs.append((loss / save, k, low[-1], save, loss))
        return sorted(ups, reverse=True), sorted(downs)

    for _ in range(__ROUNDS__):
        ups, downs = options()
        bought = [u for u in ups if u[3] <= slack][:1]
        if bought:                                  # free slack: take the best affordable upgrade
            _, k, b, cost, _ = bought[0]
            bits[k] = b
            slack -= cost
            continue
        moved = False
        for _, k_dn, b_dn, save, loss in downs[:5]:  # downgrade one group, spend the savings on upgrades
            room, gain, buy = slack + save, 0.0, []
            for _, k, b, cost, g in ups:
                if k != k_dn and cost <= room:
                    buy.append((k, b, cost))
                    room -= cost
                    gain += g
            if gain > loss:
                bits[k_dn] = b_dn
                for k, b, cost in buy:
                    bits[k] = b
                slack = room
                moved = True
                break
        if not moved:
            break
    cfg = cfg_of()
    if not api.bytes(config=cfg)["fits_budget"]:
        return inc
    if "group_error" in info["tools"]:      # analysis tools (free): where the error concentrates
        print("group_error top", api.group_error(config=cfg, top=3)["top"])
    if "compare" in info["tools"]:          # item flips vs the incumbent (ids only)
        cmp = api.compare(cid="incumbent")
        print("compare vs incumbent", {k: v["net"] for k, v in cmp["datasets"].items()})
    return cfg
"""


def mock_program(thread: int, step: int, gen: int = 0) -> str:
    """Varies with (thread, step, gen) so later rounds try configs earlier rounds did not."""
    strategies = ["refine", "solver", "refine", "greedy"]
    strategy = strategies[(thread + gen) % len(strategies)]
    if strategy == "refine":
        return (REFINE_TEMPLATE.replace("__ROUNDS__", str(8 * (1 + step + gen)))
                .replace("__UPMAX__", str([4, 8][(step + gen) % 2])).replace("__QUANT__", "gptq"))
    margin = round(0.995 - 0.01 * ((step + gen) % 3), 3)
    protect = [0, 4, 2, 0][(thread + step + gen) % 4]
    quantizer = ["gptq", "gptq", "awq", "rtn"][(thread + 2 * step + gen) % 4]
    gs = [128, 128, 64, 128][(step + thread + gen) % 4]
    maybe_eval = ""
    if strategy == "greedy" and step == 1:
        maybe_eval = ('if "evaluate" in tools:\n        fb = api.evaluate(config=cfg_of(choice))\n'
                      '        print("probe evaluate:", fb.get("score"))')
    elif strategy == "greedy" and step == 0 and thread >= 2:
        # 2026-11-21: the k x 1 layout with api.evaluate (never reached by the default mock): evaluate, the same
        # config again (free with free screens: cached) and a second config (over a per-program cap of 1)
        maybe_eval = PROBE_EVAL
    return PROGRAM_TEMPLATE.format(strategy=strategy, margin=margin, protect=protect, quantizer=quantizer,
                                   gs=gs, down_w=1.0 + 0.5 * step, embed=repr("bf16" if step % 2 == 0 else 8),
                                   maybe_eval=maybe_eval)


def mock_policy(history: list[dict]) -> str:
    """rsicore Editor mock_policy: history = chat messages; returns the program text."""
    last = ""
    for m in reversed(history or []):
        if m.get("role") == "user":
            last = m.get("content") or ""
            break
    mt = _MARK.search(last)
    thread, step = (int(mt.group(1)), int(mt.group(2))) if mt else (0, 0)
    gen = int(mt.group(4)) if mt and mt.group(4) else 0
    return mock_program(thread, step, gen)


def save_state(path: Path, state: dict) -> None:
    write_json(path, state)


def load_state(path: Path) -> dict:
    return json.load(open(path))
