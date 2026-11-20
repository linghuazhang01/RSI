"""Mock mode for Paper B (no weights, no GPU): synthetic model geometry, LoRA specs with a `quality`
model, synthetic diagnostics with the real schema, and a mock editor policy that writes valid programs.

Quality model (deterministic): a mock LoRA carries `gain` (target improvement) and `damage` (retention
damage); every primitive / operator changes them the way the literature suggests (training the raw LoRA
adds gain with growing damage; truncation, intruder scaling, RegMean, pruning and orthogonalisation trade a little
gain for less damage; a global scale trades both). The mock server answers with probability `quality`
(registry: lora / served name -> quality, a float or a per-domain dict, see eval/rsicore/mock_quality.py).
Per-domain quality of a model with a LoRA for target domain T: q[T] = parent[T] + gain + noise,
q[d] = parent[d] − damage + noise for every other domain ("default" covers the domains not listed), so a
candidate can raise the target while lowering retention; merged successors inherit the dict.
"""
from __future__ import annotations

import json
import math
import random
import re

from .common import stable_hash
from .primitives.naming import LEAVES

# Qwen3.8-27B-like geometry (64 layers, 3 GDN : 1 full attention)
N_LAYERS = 64
HIDDEN, INTER = 5120, 17408
SHAPES = {
    "q_proj": (12288, HIDDEN), "k_proj": (1024, HIDDEN), "v_proj": (1024, HIDDEN), "o_proj": (HIDDEN, 6144),
    "in_proj_qkv": (10240, HIDDEN), "in_proj_z": (6144, HIDDEN), "in_proj_b": (48, HIDDEN),
    "in_proj_a": (48, HIDDEN), "out_proj": (HIDDEN, 6144),
    "gate_proj": (INTER, HIDDEN), "up_proj": (INTER, HIDDEN), "down_proj": (HIDDEN, INTER),
}
FULL = ["q_proj", "k_proj", "v_proj", "o_proj"]
LINEAR = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


def layer_type(i: int) -> str:
    return "full_attention" if i % 4 == 3 else "linear_attention"


class MockStore:
    """Duck-types the parts of WeightStore used by the workspace / checks."""
    prefix = "model.language_model."
    fingerprint = "mock"

    def __init__(self, n_layers: int = N_LAYERS):
        self.n_layers = n_layers
        self._mods = []
        for i in range(n_layers):
            for leaf in (FULL if layer_type(i) == "full_attention" else LINEAR) + MLP:
                self._mods.append(f"layers.{i}.{LEAVES[leaf][0]}.{leaf}")

    @property
    def num_layers(self):
        return self.n_layers

    def modules(self):
        return list(self._mods)

    def has(self, m):
        return m in set(self._mods)

    def shape(self, m):
        return SHAPES[m.rsplit(".", 1)[1]]

    def info(self, m):
        i = int(m.split(".")[1])
        leaf = m.rsplit(".", 1)[1]
        out_f, in_f = self.shape(m)
        return {"module": m, "layer": i, "block": LEAVES[leaf][0], "leaf": leaf, "group": LEAVES[leaf][1],
                "layer_type": layer_type(i), "out": out_f, "in": in_f}

    def norm(self, m):
        h = int(stable_hash("norm", m), 16) % 1000 / 1000
        return 40.0 + 30.0 * h


# ============================================================================ mock LoRA
class MockLoRA:
    """modules: {short: {"rank": r, "rel": ||ΔW||/||W0||}}; gain / damage drive the mock quality."""

    def __init__(self, modules=None, gain=0.0, damage=0.0, ops=None, meta=None):
        self.modules_ = dict(modules or {})
        self.gain = float(gain)
        self.damage = float(damage)
        self.ops = list(ops or [])
        self.meta = dict(meta or {})

    # LoRA-like interface
    @property
    def factors(self):
        return self.modules_

    def modules(self):
        return sorted(self.modules_, key=lambda s: (int(s.split(".")[1]), s))

    def rank(self, m):
        return int(self.modules_[m]["rank"])

    def max_rank(self):
        return max((self.rank(m) for m in self.modules_), default=0)

    def copy(self):
        return MockLoRA({k: dict(v) for k, v in self.modules_.items()}, self.gain, self.damage, list(self.ops),
                        dict(self.meta))

    def digest(self):
        return stable_hash(json.dumps(self.to_json(), sort_keys=True), n=16)

    def to_json(self):
        return {"modules": self.modules_, "gain": round(self.gain, 6), "damage": round(self.damage, 6),
                "ops": self.ops, "meta": self.meta}

    @classmethod
    def from_json(cls, d):
        return cls(d.get("modules"), d.get("gain", 0), d.get("damage", 0), d.get("ops"), d.get("meta"))

    def _noise(self, *salt) -> float:
        return (int(stable_hash("q", self.digest(), *salt), 16) % 2001 - 1000) / 1000 * 0.008

    def quality(self, parent_q, salt: str = "", target: str | None = None):
        """target=None: scalar parent + gain - damage + noise (old single-quality model).
        target=<domain>: per-domain dict {"default", <domains...>} from parent_q (float or dict): the target
        domain gains `gain`, every other key loses `damage`."""
        def clip(x):
            return round(max(0.02, min(0.98, x)), 6)
        if target is None:
            from rsicore.mock_quality import scalar
            return clip(scalar(parent_q) + self.gain - self.damage + self._noise(salt))
        from rsicore.mock_quality import as_map
        par = as_map(parent_q)
        par.setdefault(target, par["default"])
        out = {}
        for k, v in par.items():
            if k == target:
                out[k] = clip(v + self.gain + self._noise(salt, k))
            else:
                out[k] = clip(v - self.damage + (self._noise(salt) if k == "default" else self._noise(salt, k)))
        return out

    def touch(self, mods, rank, rel, mode="add"):
        for m in mods:
            cur = self.modules_.get(m, {"rank": 0, "rel": 0.0})
            r = max(cur["rank"], int(rank)) if mode == "max" else cur["rank"] + int(rank)
            self.modules_[m] = {"rank": min(512, r), "rel": float(math.sqrt(cur["rel"] ** 2 + rel ** 2))}


def _coverage(mods) -> float:
    """Fraction of the 'useful' capacity touched: MLP and attention projections across layers."""
    if not mods:
        return 0.0
    layers = {int(m.split(".")[1]) for m in mods}
    groups = {LEAVES[m.rsplit(".", 1)[1]][1] for m in mods}
    return min(1.0, len(layers) / 32) * (0.6 + 0.4 * (len(groups) == 2)) + 0.25 * ("mlp" in groups)


# ---------------------------------------------------------------- mock primitive semantics
def m_sgd(base, mods, ids, steps, lr, rank):
    lo = base.copy() if base is not None else MockLoRA()
    x = steps * lr / 0.008                      # 40 steps @ 2e-4 = 1.0
    data = min(1.0, math.sqrt(max(len(ids), 1) / 512))
    cov = min(1.0, _coverage(mods))
    lo.gain += 0.10 * (1 - math.exp(-x)) * (0.5 + 0.5 * data) * (0.4 + 0.6 * cov)
    lo.damage += 0.030 * x ** 1.3 * (rank / 16) ** 0.25 * (0.5 + 0.5 * cov)
    lo.touch(mods, rank, 0.02 * math.sqrt(x), mode="max")   # like peft: adapter rank = max(rank, warm-start ranks)
    lo.ops.append(["sgd", steps, lr, rank, len(mods)])
    return lo


def m_construct(base, mod, strength, kind):
    lo = base.copy()
    a = abs(float(strength))
    lo.gain += 0.006 * min(1.0, a)
    lo.damage += 0.008 * a ** 2
    lo.touch([mod], 1, 0.01 * a)
    lo.ops.append([kind, mod, strength])
    return lo


def m_scale(base, s):
    lo = base.copy()
    lo.gain *= s if s <= 1 else 1 + 0.25 * (s - 1)
    lo.damage *= s ** 2
    for v in lo.modules_.values():
        v["rel"] *= abs(s)
    lo.ops.append(["scale", s])
    return lo


def m_truncate(base, k):
    lo = base.copy()
    lo.gain *= 0.94 + 0.06 * min(1.0, k / 16)
    lo.damage *= 0.55 + 0.45 * min(1.0, k / 32)
    for v in lo.modules_.values():
        v["rank"] = min(v["rank"], int(k))
    lo.ops.append(["truncate_rank", k])
    return lo


def m_intruder(base, lam):
    lo = base.copy()
    lo.damage *= 0.4 + 0.6 * lam
    lo.gain *= 0.93 + 0.07 * lam
    lo.ops.append(["intruder_scale", lam])
    return lo


def m_regmean(base, alpha):
    lo = base.copy()
    f = alpha / (1 + alpha)
    lo.damage *= 1 - 0.5 * f
    lo.gain *= 1 - 0.06 * f
    lo.ops.append(["regmean_preserve", alpha])
    return lo


def m_prune(base, frac):
    lo = base.copy()
    lo.damage *= max(0.5, 1 - 4 * frac)
    lo.gain *= 1 - 0.5 * frac
    lo.ops.append(["relchange_prune", frac])
    return lo


def m_orth(base, n):
    lo = base.copy()
    if n:
        lo.damage *= 0.85
        lo.gain *= 0.97
    lo.ops.append(["orthogonalize", n])
    return lo


def m_combine(loras, weights):
    out = MockLoRA()
    for lo, w in zip(loras, weights):
        out.gain += w * lo.gain
        out.damage += abs(w) ** 1.5 * lo.damage
        for m, v in lo.modules_.items():
            out.touch([m], v["rank"], abs(w) * v["rel"])
    out.ops.append(["combine", list(weights)])
    return out


def m_ties(loras, weights, density):
    out = m_combine(loras, [w / max(len(loras), 1) for w in weights])
    out.gain *= 1.05
    out.damage *= 0.8
    out.ops.append(["ties", density])
    return out


# ============================================================================ diagnostics
def mock_diagnostics(store: MockStore, stack: list, seed: int = 42) -> dict:
    rows = []
    for m in store.modules():
        inf = store.info(m)
        h = int(stable_hash(seed, "diag", m), 16)
        r = random.Random(h)
        s1 = 8 + 6 * r.random() + 0.05 * inf["layer"]
        row = dict(inf, w_norm=round(store.norm(m), 3), w_sigma=[round(s1 * 0.8 ** i, 3) for i in range(4)],
                   stack=[])
        for g, lo in stack:
            if m not in lo.modules_:
                continue
            v = lo.modules_[m]
            rr = random.Random(h + 7 * g)
            sv = sorted((rr.random() * v["rel"] * 50 for _ in range(min(4, v["rank"]))), reverse=True)
            row["stack"].append({
                "gen": g, "norm": round(v["rel"] * row["w_norm"], 4), "rel_norm": round(v["rel"], 5),
                "sv": [round(x, 4) for x in sv], "eff_rank": round(1 + rr.random() * min(8, v["rank"]), 2),
                "intruder": round(0.3 + 0.5 * rr.random(), 3),
                "relchange_q": {"q50": round(v["rel"] * 0.5, 5), "q90": round(v["rel"] * 1.8, 5),
                                "q99": round(v["rel"] * 4.0, 5)},
                "act_ratio_target": round(v["rel"] * (1.2 + rr.random()), 5),
                "act_ratio_retention": round(v["rel"] * (0.6 + rr.random()), 5),
            })
        rows.append(row)
    return {"model": "mock", "n_modules": len(rows), "rows": rows, "stack_gens": [g for g, _ in stack]}


# ============================================================================ mock editor policy
def make_mock_policy(seed: int, skill: float = 0.7, tag: str = "main", mode: str = "scratch", retention: bool = True):
    """Return policy(history) -> program text. Choices vary with (thread, step) parsed from the last user
    message ("[thread=K step=S]"); a higher `skill` (the editor model's mock quality) picks better
    hyper-parameters and protective operators more often. mode (method.mode): edit = edits of the raw LoRA
    (EDIT_PROGRAM); scratch = gradient-free constructions from zero (SCRATCH_PROGRAM)."""

    def policy(history):
        last = ""
        for msg in reversed(history or []):
            if msg.get("role") == "user":
                last = msg.get("content", "")
                break
        m = re.search(r"\[thread=(\d+) step=(\d+)\]", last)
        th, st = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        rng = random.Random(f"{seed}:{tag}:{th}:{st}:{len(last) // 500}")
        good = rng.random() < skill
        if mode == "edit":
            recipe = EDIT_RECIPES[(th * 3 + st + rng.randrange(len(EDIT_RECIPES))) % len(EDIT_RECIPES)]
            if recipe == "regmean" and not retention:   # no regmean_preserve without retention texts
                recipe = "intruder"
            return EDIT_PROGRAM.format(recipe=recipe, th=th, st=st,
                                       k=rng.choice([8, 16] if good else [2, 4]),
                                       lam=rng.choice([0.3, 0.5] if good else [0.1, 0.9]),
                                       alpha=rng.choice([1.0, 2.0] if good else [0.2]),
                                       s=rng.choice([0.8, 0.9] if good else [0.5, 1.4]),
                                       boost=rng.choice([1.6, 2.0] if good else [1.05]),
                                       lo=rng.choice([0, 8, 16]), hi=rng.choice([48, 56, 64]))
        recipe = SCRATCH_RECIPES[(th + st + rng.randrange(len(SCRATCH_RECIPES))) % len(SCRATCH_RECIPES)]
        if recipe == "singular_regmean" and not retention:
            recipe = "singular_scale"
        return SCRATCH_PROGRAM.format(recipe=recipe, th=th, st=st, n=rng.choice([8, 16] if good else [2, 32]),
                                      f=rng.choice([1.04, 1.06] if good else [1.3, 0.7]),
                                      alpha=rng.choice([0.3, 0.5] if good else [2.0]),
                                      lo=rng.choice([8, 16]), hi=rng.choice([48, 56]))

    return policy


EDIT_RECIPES = ["trunc_scale", "intruder", "regmean", "prune_orth", "drop_attn", "layer_scale", "trunc_intruder",
                "boost", "boost"]

EDIT_PROGRAM = '''\
# mock editor program (mode=edit): thread {th} step {st}, recipe {recipe}
def main(api):
    recipe = "{recipe}"
    raw = api.raw_lora()
    lora = raw["lora"]
    if recipe in ("trunc_scale", "trunc_intruder"):
        lora = api.truncate_rank(lora=lora, k={k})["lora"]
    if recipe in ("intruder", "trunc_intruder"):
        lora = api.intruder_scale(lora=lora, lam={lam})["lora"]
    if recipe == "regmean":
        lora = api.regmean_preserve(lora=lora, alpha={alpha})["lora"]
    if recipe == "prune_orth":
        lora = api.relchange_prune(lora=lora, frac=0.03)["lora"]
        lora = api.orthogonalize(lora=lora)["lora"]
    if recipe == "drop_attn":
        mods = [m["module"] for m in api.modules() if m["group"] != "mlp" and not ({lo} <= m["layer"] < {hi})]
        if mods:
            lora = api.drop(lora=lora, modules=mods)["lora"]
    if recipe in ("trunc_scale", "layer_scale"):
        lora = api.scale(lora=lora, s={s})["lora"]
    if recipe == "boost":                       # strengthen the LoRA (raises the target, costs retention)
        lora = api.scale(lora=lora, s={boost})["lora"]
    info = api.inspect(lora=lora)
    print("modules", info["modules"], "max_rank", info["max_rank"])
    return {{"lora": lora}}
'''

SCRATCH_RECIPES = ["singular", "singular_regmean", "singular_scale"]

SCRATCH_PROGRAM = '''\
# mock editor program (mode=scratch): thread {th} step {st}, recipe {recipe}
def main(api):
    recipe = "{recipe}"
    mods = [m["module"] for m in api.modules() if m["leaf"] == "down_proj" and {lo} <= m["layer"] < {hi}]
    mods = mods[: {n}]
    lora = api.new_lora()["lora"]
    for m in mods:
        s = api.svd(module=m, k=4)
        lora = api.scale_singular(lora=lora, module=m, svd=s["handle"], idx=[0, 1], factors=[{f}, {f}])["lora"]
    if recipe == "singular_regmean":
        lora = api.regmean_preserve(lora=lora, alpha={alpha})["lora"]
    if recipe == "singular_scale":
        lora = api.scale(lora=lora, s=0.8)["lora"]
    try:                                    # cheap screening + item flips vs the current model (free tools)
        tl = api.train_loss(lora=lora, n=16)
        print("train_loss delta", tl["delta"])
        print("compare parent", api.compare(cid="parent")["datasets"].keys())
    except Exception as e:  # noqa: BLE001  (e.g. nothing evaluated yet, tool switched off)
        print("analysis tools:", type(e).__name__, str(e)[:120])
    return {{"lora": lora}}
'''
