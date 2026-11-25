"""solve_allocation: multiple-choice knapsack over the profile table (greedy on the lower convex hull
of each group's (bytes, damage) options; Lagrangian-greedy, optimal up to one item for convex hulls).

damage(key, bits, gs) = profile dkl_mean(key, bits)  [RTN, profile group size]
                        x gs_factor[gs] / gs_factor[profile gs]       (heuristic, configs/base.yaml)
                        x weight(key)                                  (editor-chosen multipliers)
bits 16 (keep BF16) has damage 0. Used by the editor tool `solve_allocation`, the KL-greedy baseline,
the HAWQ baseline (Hessian rows) and the mock editor.
"""
from __future__ import annotations

import fnmatch
import heapq

from .quant import schema, sizes
from .quant.spec import ModelSpec

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


def solve_allocation(spec: ModelSpec, profile_rows: list[dict], budget_gib: float, *, weights: dict | None = None,
                     bits: list[int] | None = None, group_sizes: list[int] | None = None, quantizer: str = "gptq",
                     fixed: dict | None = None, embed="bf16", lm_head=8, gs_factor: dict | None = None,
                     action_space: dict | None = None) -> dict:
    """Return an editor-style config (dict) that fits budget_gib (if feasible) and minimises the sum
    of weighted profiled damages. `fixed` = {group key: setting} kept as is."""
    space = dict(schema.DEFAULT_ACTION_SPACE, **(action_space or {}))
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
    hulls, choice = {}, {}
    used = base
    for key in groups:
        if key in fixed:
            used += sizes.modules_bytes(groups[key], fixed[key], spec.weight_dtype_bytes)
            continue
        pts = []
        for (b, g, q) in options:
            opt = {"bits": b, "group_size": g, "quantizer": q}
            pts.append((sizes.modules_bytes(groups[key], opt, spec.weight_dtype_bytes),
                        w[key] * _damage(dl, key, b, g, prof_gs, gsf), opt))
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
    for key, i in choice.items():
        groups_cfg[key] = dict(hulls[key][i][2])
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
