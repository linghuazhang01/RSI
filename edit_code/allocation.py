"""solve_allocation: multiple-choice knapsack over the profile table (greedy on the lower convex hull
of each group's (bytes, damage) options; Lagrangian-greedy, optimal up to one item for convex hulls).

damage(key, bits, gs) = profile dkl_mean(key, bits)  [RTN, profile group size]
                        x gs_factor[gs] / gs_factor[profile gs]       (heuristic, configs/base.yaml)
                        x weight(key)                                  (editor-chosen multipliers)
bits 16 (keep BF16) has damage 0. Used by the editor tool `solve_allocation`, the KL-greedy baseline,
the HAWQ baseline (Hessian rows) and the mock editor.
Under action_space.packed (2026-11-27) the knapsack items are the fused vLLM modules: qk + v and gdn_qkv + gdn_z of a
layer get one setting (their damages add up), and settings the packed format cannot express are not offered.
"""
from __future__ import annotations

import fnmatch
import heapq

from .quant import schema, sizes
from .quant.spec import TIED_GROUPS, ModelSpec

DEFAULT_GS_FACTOR = {128: 1.0, 64: 0.8, 32: 0.65}


def match_keys(pattern: str, spec: ModelSpec) -> list[str]:
    """Group keys matching a selector: '17.down', 'down', '*.down', '56-63.down', '3.*', 'all', or a
    shell glob over keys."""
    keys = spec.group_keys()
    if pattern in ("all", "*"):
        return keys
    if pattern in keys:
        return [pattern]
    probs: list[str] = []
    if "." in pattern:
        lsel, g = pattern.rsplit(".", 1)
        layers = schema.parse_layers("all" if lsel == "*" else lsel, spec.num_layers, probs, pattern)
        groups = schema.parse_groups(g, probs, pattern)
        if not probs:
            want = {f"{i}.{gg}" for i in layers for gg in groups}
            return [k for k in keys if k in want]
    elif pattern in schema.GROUP_NAMES:
        return [k for k in keys if k.split(".", 1)[1] == pattern]
    return [k for k in keys if fnmatch.fnmatch(k, pattern)]


def damage_lookup(profile_rows: list[dict]) -> tuple[dict, int]:
    d, gs = {}, 128
    for r in profile_rows:
        d[(r["key"], int(r["bits"]))] = float(r["dkl_mean"])
        gs = int(r.get("group_size", gs))
    return d, gs


def _damage(dl: dict, key: str, bits: int, gs: int, prof_gs: int, gsf: dict) -> float:
    if bits >= 16:
        return 0.0
    if (key, bits) in dl:
        base = dl[(key, bits)]
    else:  # bits not profiled: geometric interpolation / extrapolation between profiled widths
        have = sorted(b for (k, b) in dl if k == key)
        if not have:
            return 0.0
        lo = max([b for b in have if b < bits], default=None)
        hi = min([b for b in have if b > bits], default=None)
        if lo is not None and hi is not None:
            t = (bits - lo) / (hi - lo)
            a, c = max(dl[(key, lo)], 1e-12), max(dl[(key, hi)], 1e-12)
            base = a ** (1 - t) * c ** t
        elif hi is not None:
            base = dl[(key, hi)] * 4 ** (hi - bits)
        else:
            base = dl[(key, lo)] / 4 ** (bits - lo)
    return base * gsf.get(gs, 1.0) / gsf.get(prof_gs, 1.0)


def _units(spec: ModelSpec, packed: bool) -> list[tuple[str, ...]]:
    """Knapsack items in group-key order: (key,) each, or (a, b) for the TIED_GROUPS of a layer when packed."""
    keys = spec.group_keys()
    if not packed:
        return [(k,) for k in keys]
    partner = {}
    for i in range(spec.num_layers):
        for a, b in TIED_GROUPS:
            ka, kb = f"{i}.{a}", f"{i}.{b}"
            if ka in keys and kb in keys:
                partner[ka], partner[kb] = (ka, kb), (ka, kb)
    out, seen = [], set()
    for k in keys:
        u = partner.get(k, (k,))
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def solve_allocation(spec: ModelSpec, profile_rows: list[dict], budget_gib: float, *, weights: dict | None = None,
                     bits: list[int] | None = None, group_sizes: list[int] | None = None, quantizer: str = "gptq",
                     fixed: dict | None = None, embed="bf16", lm_head=None, gs_factor: dict | None = None,
                     action_space: dict | None = None) -> dict:
    """Return an editor-style config (dict) that fits budget_gib (if feasible) and minimises the sum
    of weighted profiled damages. `fixed` = {group key: setting} kept as is. lm_head None = 8 when the action space
    allows it, else "bf16"."""
    space = dict(schema.DEFAULT_ACTION_SPACE, **(action_space or {}))
    packed = bool(space.get("packed"))
    if lm_head is None:
        lm_head = 8 if 8 in space["lm_head"] else "bf16"
    if quantizer not in space["quantizers"]:
        raise ValueError(f"solve_allocation: quantizer {quantizer!r} not in {space['quantizers']}")
    bits = [b for b in (bits or space["bits"]) if b in space["bits"]]
    group_sizes = [g for g in (group_sizes or [128]) if g in space["group_sizes"]]
    if not bits or not group_sizes:
        raise ValueError("solve_allocation: no allowed bits / group sizes")
    gsf = {int(k): float(v) for k, v in (gs_factor or DEFAULT_GS_FACTOR).items()}
    dl, prof_gs = damage_lookup(profile_rows)
    w = {k: 1.0 for k in spec.group_keys()}
    for pat, mult in (weights or {}).items():
        hit = match_keys(pat, spec)
        if not hit:
            raise ValueError(f"solve_allocation: weight pattern {pat!r} matches no group key")
        for k in hit:
            w[k] *= float(mult)
    fixed = dict(fixed or {})
    options = []
    for b in bits:
        if b >= 16:
            options.append((16, 0, "none"))
        else:
            for g in group_sizes:
                options.append((b, g, quantizer))
    budget = budget_gib * 2 ** 30
    base = sizes.precision_bytes(spec.embed, embed) + \
        (sizes.precision_bytes(spec.lm_head, lm_head) if spec.lm_head is not None else 0) + spec.other_bytes
    groups = spec.groups()
    units = _units(spec, packed)               # knapsack items: one group key, or a tied pair (packed)
    for unit in units:                         # a fixed key fixes its tied partner too (same setting)
        hit = [k for k in unit if k in fixed]
        for k in unit:
            if hit and k not in fixed:
                fixed[k] = dict(fixed[hit[0]])
    hulls, choice = {}, {}
    used = base
    for unit in units:
        if unit[0] in fixed:
            for k in unit:
                used += sizes.modules_bytes(groups[k], fixed[k], spec.weight_dtype_bytes)
            continue
        mods = [m for k in unit for m in groups[k]]
        pts = []
        for (b, g, q) in options:
            opt = {"bits": b, "group_size": g, "quantizer": q}
            if packed and schema.setting_packable(mods, opt):
                continue                       # not expressible in the packed format (e.g. 3-bit in_proj_b)
            pts.append((sizes.modules_bytes(mods, opt, spec.weight_dtype_bytes),
                        sum(w[k] * _damage(dl, k, b, g, prof_gs, gsf) for k in unit), opt))
        if not pts:
            raise ValueError(f"solve_allocation: no allowed setting for {', '.join(unit)}")
        key = unit
        pts.sort(key=lambda p: (p[0], p[1]))
        hull = []   # lower convex hull, decreasing damage with increasing bytes
        for p in pts:
            if hull and p[1] >= hull[-1][1]:
                continue
            while len(hull) >= 2:
                (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
                if (y2 - y1) * (p[0] - x1) >= (p[1] - y1) * (x2 - x1):  # hull[-1] above the chord
                    hull.pop()
                else:
                    break
            hull.append(p)
        hulls[key] = hull
        choice[key] = 0
        used += hull[0][0]
    feasible = used <= budget
    heap = []

    def push(key):
        i = choice[key]
        h = hulls[key]
        if i + 1 < len(h):
            db, dd = h[i + 1][0] - h[i][0], h[i][1] - h[i + 1][1]
            heapq.heappush(heap, (-(dd / max(db, 1e-9)), key, i))
    for key in hulls:
        push(key)
    while heap:
        _, key, i = heapq.heappop(heap)
        if choice[key] != i:
            continue
        h = hulls[key]
        db = h[i + 1][0] - h[i][0]
        if used + db <= budget:
            used += db
            choice[key] = i + 1
            push(key)
    groups_cfg = {k: dict(v) for k, v in fixed.items()}
    for unit, i in choice.items():
        for k in unit:
            groups_cfg[k] = dict(hulls[unit][i][2])
    for s in groups_cfg.values():
        if s.get("bits", 0) >= 16:
            s.clear()
            s["bits"] = 16
    from collections import Counter
    common = Counter(tuple(sorted(s.items())) for s in groups_cfg.values() if s.get("bits", 16) < 16)
    default = dict(common.most_common(1)[0][0]) if common else \
        {"bits": 16, "group_size": group_sizes[0], "quantizer": quantizer}
    cfg = {"default": default, "groups": {k: s for k, s in groups_cfg.items() if s != default},
           "embed": embed, "lm_head": lm_head}
    norm = schema.normalize(cfg, spec, action_space)
    # common.AUTO_NOTE parses this text (program_note strips it): keep the two in step
    cfg["note"] = (f"solve_allocation: {sizes.config_bytes(norm, spec)['gib']:.3f} GiB of {budget_gib:.3f}; "
                   f"predicted weighted damage {sum(hulls[k][i][1] for k, i in choice.items()):.5f}"
                   + ("" if feasible else " (INFEASIBLE: cheapest options exceed the budget)"))
    return cfg
