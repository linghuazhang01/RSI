"""Quantization-config schema: what the editor's program returns, validation, resolution.

A config is a JSON object:

    {
      "default":  {"bits": 4, "group_size": 128, "quantizer": "gptq"},   # required
      "rules": [                                                          # optional, applied in order
        {"layers": "56-63", "groups": ["down", "gate_up"], "bits": 8},
        {"layers": [0, 1, -1], "groups": "all", "group_size": 64}
      ],
      "groups": {"17.down": {"bits": 3}, "o": {"quantizer": "rtn"}},      # optional, applied last
      "embed": "bf16",          # "bf16" | 8 | 4      (RTN, group size 128)
      "lm_head": 8              # "bf16" | 8 | 4      (ignored when tied to the embedding)
    }

* `bits` in {2, 3, 4, 8} or 16 (= keep BF16); `group_size` in {32, 64, 128}; `quantizer` in
  {"rtn", "gptq", "awq"} (subset allowed by the action space in configs/base.yaml).
* Layer selectors: an int (negative = from the end), "a-b" (inclusive range), "all", or a list of
  those. Group selectors: a group name, "all", or a list. `groups` keys: "<layer>.<group>",
  "<group>" (all layers), "<a-b>.<group>" or "*.<group>". Pattern keys are applied before exact keys.
* Missing fields inherit from the setting already in force for that group.

`normalize()` returns a fully resolved config ({"groups": {key: setting} for every group, ...}),
which is what is hashed, sized, cached and exported.
"""
from __future__ import annotations

from .spec import GROUP_NAMES, ModelSpec

DEFAULT_ACTION_SPACE = {"bits": [2, 3, 4, 8, 16], "group_sizes": [32, 64, 128],
                        "quantizers": ["rtn", "gptq", "awq"], "embed": ["bf16", 8, 4],
                        "lm_head": ["bf16", 8, 4]}
SETTING_KEYS = ("bits", "group_size", "quantizer")


class ConfigError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("invalid quantization config:\n  - " + "\n  - ".join(problems))


def parse_layers(sel, n_layers: int, problems: list[str], where: str) -> list[int]:
    if sel is None or sel == "all" or sel == "*":
        return list(range(n_layers))
    items = sel if isinstance(sel, list) else [sel]
    out = []
    for it in items:
        if isinstance(it, bool):
            problems.append(f"{where}: bad layer selector {it!r}")
            continue
        if isinstance(it, int):
            i = it + n_layers if it < 0 else it
            if not 0 <= i < n_layers:
                problems.append(f"{where}: layer {it} out of range 0..{n_layers - 1}")
                continue
            out.append(i)
        elif isinstance(it, str) and it.strip().lstrip("-").isdigit():
            out += parse_layers(int(it), n_layers, problems, where)
        elif isinstance(it, str) and "-" in it.strip()[1:]:
            a, b = it.strip().split("-", 1) if not it.startswith("-") else (None, None)
            try:
                a, b = int(a), int(b)
            except (TypeError, ValueError):
                problems.append(f"{where}: bad layer range {it!r} (use 'a-b')")
                continue
            if not (0 <= a <= b < n_layers):
                problems.append(f"{where}: layer range {it!r} outside 0..{n_layers - 1}")
                continue
            out += list(range(a, b + 1))
        elif it in ("all", "*"):
            out += list(range(n_layers))
        else:
            problems.append(f"{where}: bad layer selector {it!r}")
    return sorted(set(out))


def parse_groups(sel, problems: list[str], where: str) -> list[str]:
    if sel is None or sel == "all" or sel == "*":
        return list(GROUP_NAMES)
    items = sel if isinstance(sel, list) else [sel]
    out = []
    for g in items:
        if g in ("all", "*"):
            out += GROUP_NAMES
        elif g in GROUP_NAMES:
            out.append(g)
        else:
            problems.append(f"{where}: unknown group {g!r} (groups: {', '.join(GROUP_NAMES)})")
    return out


def _check_setting(s: dict, space: dict, problems: list[str], where: str, partial: bool) -> dict:
    if not isinstance(s, dict):
        problems.append(f"{where}: expected an object with bits/group_size/quantizer, got {type(s).__name__}")
        return {}
    out = {}
    for k, v in s.items():
        if k in ("layers", "groups", "note", "comment"):
            continue
        if k not in SETTING_KEYS:
            problems.append(f"{where}: unknown key {k!r} (allowed: {', '.join(SETTING_KEYS)})")
            continue
        if k == "bits":
            if isinstance(v, bool) or not isinstance(v, int) or v not in space["bits"]:
                problems.append(f"{where}: bits={v!r} not in {space['bits']}")
                continue
        if k == "group_size":
            if isinstance(v, bool) or not isinstance(v, int) or v not in space["group_sizes"]:
                problems.append(f"{where}: group_size={v!r} not in {space['group_sizes']}")
                continue
        if k == "quantizer":
            if v not in space["quantizers"]:
                problems.append(f"{where}: quantizer={v!r} not in {space['quantizers']}")
                continue
        out[k] = v
    if not partial:
        for k in SETTING_KEYS:
            if k not in out:
                problems.append(f"{where}: missing {k!r}")
    return out


def _check_precision(v, allowed, problems, where):
    if isinstance(v, str) and v.lower() in ("bf16", "16"):
        v = "bf16"
    elif isinstance(v, str) and v.strip().isdigit():
        v = int(v)
    if v == 16:
        v = "bf16"
    if v not in allowed:
        problems.append(f"{where}: {v!r} not in {allowed}")
        return "bf16"
    return v


def normalize(config, spec: ModelSpec, action_space: dict | None = None) -> dict:
    """Validate and fully resolve a config. Raises ConfigError listing every problem."""
    space = dict(DEFAULT_ACTION_SPACE, **(action_space or {}))
    problems: list[str] = []
    if not isinstance(config, dict):
        raise ConfigError([f"config must be a JSON object (dict), got {type(config).__name__}"])
    unknown = set(config) - {"default", "rules", "groups", "embed", "lm_head", "calibration", "note", "name"}
    if unknown:
        problems.append(f"unknown top-level keys {sorted(unknown)} (allowed: default, rules, groups, embed, lm_head)")
    default = _check_setting(config.get("default"), space, problems, "default", partial=False) \
        if "default" in config else (problems.append("missing 'default' setting") or {})
    keys = spec.group_keys()
    present = {}
    for k in keys:
        layer, g = k.split(".", 1)
        present.setdefault(g, set()).add(int(layer))
    resolved = {k: dict(default) for k in keys}
    n = spec.num_layers

    def apply(layers, groups, setting):
        hit = 0
        for g in groups:
            for i in layers:
                key = f"{i}.{g}"
                if key in resolved:
                    resolved[key].update(setting)
                    hit += 1
        return hit

    rules = config.get("rules") or []
    if not isinstance(rules, list):
        problems.append("'rules' must be a list")
        rules = []
    for j, r in enumerate(rules):
        where = f"rules[{j}]"
        if not isinstance(r, dict):
            problems.append(f"{where}: must be an object")
            continue
        s = _check_setting(r, space, problems, where, partial=True)
        layers = parse_layers(r.get("layers", "all"), n, problems, where)
        groups = parse_groups(r.get("groups", "all"), problems, where)
        if s and layers and groups and apply(layers, groups, s) == 0:
            problems.append(f"{where}: matches no module group (check layer types: groups qk/v/o exist only in "
                            f"full-attention layers, gdn_* only in linear-attention layers)")
    gmap = config.get("groups") or {}
    if not isinstance(gmap, dict):
        problems.append("'groups' must be an object {key: setting}")
        gmap = {}
    exact, patterns = [], []
    for key, s in gmap.items():
        (exact if key in resolved else patterns).append((key, s))
    for key, s in patterns + exact:
        where = f"groups[{key!r}]"
        st = _check_setting(s, space, problems, where, partial=True)
        if key in resolved:
            resolved[key].update(st)
            continue
        if "." in key:
            lsel, g = key.rsplit(".", 1)
            layers = parse_layers(lsel if lsel != "*" else "all", n, problems, where)
        else:
            g, layers = key, list(range(n))
        groups = parse_groups(g, problems, where)
        if st and groups and layers and apply(layers, groups, st) == 0:
            problems.append(f"{where}: matches no module group (existing keys look like '3.down', '3.gdn_qkv', '7.qk')")
    embed = _check_precision(config.get("embed", "bf16"), space["embed"], problems, "embed")
    lm_head = _check_precision(config.get("lm_head", "bf16"), space["lm_head"], problems, "lm_head")
    if spec.lm_head is None:
        lm_head = embed  # tied
    if problems:
        raise ConfigError(problems)
    for k, s in resolved.items():
        if s.get("bits") == 16:
            resolved[k] = {"bits": 16, "group_size": 0, "quantizer": "none"}
    return {"groups": resolved, "embed": embed, "lm_head": lm_head}


def compress(norm: dict, spec: ModelSpec) -> dict:
    """Shortest equivalent editor-style config of a resolved config (for prompts / logs): the most
    common setting becomes the default, the rest are per-group overrides."""
    from collections import Counter
    c = Counter(tuple(sorted(s.items())) for s in norm["groups"].values())
    dflt = dict(c.most_common(1)[0][0])
    if dflt.get("bits") == 16:
        dflt = {"bits": 16, "group_size": 128, "quantizer": "rtn"}
    groups = {}
    for k, s in norm["groups"].items():
        if s != dflt and not (s.get("bits") == 16 and dflt.get("bits") == 16):
            groups[k] = {"bits": 16} if s.get("bits") == 16 else dict(s)
    out = {"default": dflt, "groups": groups, "embed": norm["embed"], "lm_head": norm["lm_head"]}
    return out


def module_assignment(norm: dict, spec: ModelSpec) -> dict[str, dict]:
    """tensor name -> setting (for every quantizable linear module)."""
    out = {}
    for m in spec.modules:
        out[m.name] = norm["groups"][f"{m.layer}.{m.group}"]
    return out
