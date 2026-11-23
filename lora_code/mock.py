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
mode=weights: MockWeightEdit (kind "weights", no rank) with the m_w_* primitives (small multiplicative tweaks near 1
give a little gain, large ones cost damage ~ change^2; an added LoRA brings its gain / damage scaled by alpha;
moving toward the original weights shrinks both) and the WEIGHTS_PROGRAM mock editor program.
tensor_compute (2026-11-21): MockStore also has the full tensor catalogue (norms, dt_bias / A_log, conv1d, embed /
final norm / lm_head) with 27B shapes; a mock tensor_compute really runs the code in the sandbox on seeded synthetic
tensors (SYNTH_SRC; needs torch), and its outputs become a MockLoRA (m_tensor_lora) or MockWeightEdit entries
(m_w_delta) whose rel comes from the actual factors / deltas.
"""
from __future__ import annotations

import json
import math
import random
import re

from .common import stable_hash
from .primitives.naming import LEAVES, is_edit_excluded, linear_short, tensor_layer

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


VOCAB = 248320
HEAD_DIM, LIN_HEADS, LIN_V_DIM, CONV_DIM, CONV_K = 256, 48, 128, 10240, 4
FULL_TENSORS = {"self_attn.q_norm.weight": (HEAD_DIM,), "self_attn.k_norm.weight": (HEAD_DIM,)}
LINEAR_TENSORS = {"linear_attn.conv1d.weight": (CONV_DIM, 1, CONV_K), "linear_attn.dt_bias": (LIN_HEADS,),
                  "linear_attn.A_log": (LIN_HEADS,), "linear_attn.norm.weight": (LIN_V_DIM,)}
LAYER_TENSORS = {"input_layernorm.weight": (HIDDEN,), "post_attention_layernorm.weight": (HIDDEN,)}
GLOBAL_TENSORS = {"embed_tokens.weight": (VOCAB, HIDDEN), "norm.weight": (HIDDEN,), "lm_head.weight": (VOCAB, HIDDEN)}
ZERO_CENTRED = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")


def layer_type(i: int) -> str:
    return "full_attention" if i % 4 == 3 else "linear_attention"


# Seeded synthetic tensors of the mock tensor_compute (2026-11-21): exec'd in the parent (to know the "current"
# tensor of a write) and embedded in the sandbox wrapper (the child cannot import lora_code), so both see the same
# values. spec = {"shape", "seed", "mean", "std"}.
SYNTH_SRC = """
def synth_tensor(torch, spec):
    g = torch.Generator().manual_seed(int(spec["seed"]))
    return torch.randn(*spec["shape"], generator=g) * float(spec["std"]) + float(spec["mean"])
"""


def synth_tensor(spec):
    import torch
    ns = {}
    exec(SYNTH_SRC, ns)
    return ns["synth_tensor"](torch, spec)


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

    # ---- tensor catalogue (2026-11-21), same interface as WeightStore
    def _catalogue(self) -> dict:
        if getattr(self, "_tensors", None) is None:
            t = {}
            t["embed_tokens.weight"] = GLOBAL_TENSORS["embed_tokens.weight"]
            for i in range(self.n_layers):
                extra = FULL_TENSORS if layer_type(i) == "full_attention" else LINEAR_TENSORS
                for suffix, shape in {**LAYER_TENSORS, **extra}.items():
                    t[f"layers.{i}.{suffix}"] = shape
            for m in self._mods:
                t[m + ".weight"] = self.shape(m)
            t["norm.weight"] = GLOBAL_TENSORS["norm.weight"]
            t["lm_head.weight"] = GLOBAL_TENSORS["lm_head.weight"]
            self._tensors = t
        return self._tensors

    def _canon(self, name):
        return name + ".weight" if self.has(name) else name

    def tensor_names(self):
        return sorted(self._catalogue(), key=lambda n: (-1 if n.startswith("embed") else
                                                        (10 ** 6 if tensor_layer(n) is None else tensor_layer(n)), n))

    def has_tensor(self, name):
        return self._canon(name) in self._catalogue()

    def tensor_key(self, name):
        n = self._canon(name)
        if n not in self._catalogue():
            raise KeyError(f"tensor {name!r} not in model (see api.tensors())")
        return n if n.startswith("lm_head.") else self.prefix + n

    def tensor_shape(self, name):
        self.tensor_key(name)
        return tuple(self._catalogue()[self._canon(name)])

    def tensor_dtype(self, name):
        return "float32" if self._canon(name).endswith("A_log") else "bfloat16"

    def editable(self, name):
        return self.has_tensor(name) and not is_edit_excluded(self._canon(name))

    def edit_name(self, name):
        n = self._canon(name)
        self.tensor_key(n)
        s = linear_short(n)
        return s if s is not None and self.has(s) else n

    def tensor_info(self, name):
        n = self._canon(name)
        return {"name": n, "shape": list(self.tensor_shape(n)), "dtype": self.tensor_dtype(n),
                "editable": not is_edit_excluded(n), "layer": tensor_layer(n)}

    def synth_spec(self, name) -> dict:
        """Seeded synthetic values of a tensor: linear weights with ||W||_F = norm(module); zero-centred RMSNorm
        weights near 0; linear_attn.norm near 1; dt_bias / A_log positive; conv1d / embeddings small."""
        n = self._canon(name)
        shape = list(self.tensor_shape(n))
        numel = 1
        for x in shape:
            numel *= x
        short = linear_short(n)
        if short is not None and self.has(short):
            mean, std = 0.0, self.norm(short) / math.sqrt(numel)
        elif n.endswith(ZERO_CENTRED) or n == "norm.weight":
            mean, std = 0.0, 0.05
        elif n.endswith("linear_attn.norm.weight"):
            mean, std = 1.0, 0.05
        elif n.endswith("dt_bias"):
            mean, std = 1.0, 0.5
        elif n.endswith("A_log"):
            mean, std = 1.5, 0.5
        elif n.endswith("conv1d.weight"):
            mean, std = 0.0, 0.1
        else:
            mean, std = 0.0, 0.02
        return {"shape": shape, "seed": int(stable_hash("synth", n), 16) % (2 ** 31), "mean": mean, "std": std}


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
        return type(self)({k: dict(v) for k, v in self.modules_.items()}, self.gain, self.damage, list(self.ops),
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


def m_stack(base, delta, budget=None):
    """merge.mode adapter: the cumulative mock adapter C (+) dW (ranks add per module, capped at `budget`). Mock
    quality is never computed from it: a candidate's quality is its delta's, relative to the current model."""
    out = base.copy() if base is not None else MockLoRA()
    out.gain += delta.gain
    out.damage += delta.damage
    for m, v in delta.modules_.items():
        out.touch([m], v["rank"], v["rel"])
    if budget:
        for v in out.modules_.values():
            v["rank"] = min(int(budget), v["rank"])
    out.ops.append(["stack", delta.digest()])
    out.meta = {"cumulative": True,
                "components": list((base.meta if base is not None else {}).get("components", [])) + [delta.digest()]}
    return out


def m_ties(loras, weights, density):
    out = m_combine(loras, [w / max(len(loras), 1) for w in weights])
    out.gain *= 1.05
    out.damage *= 0.8
    out.ops.append(["ties", density])
    return out


# ============================================================================ mock weight edit (mode=weights)
class MockWeightEdit(MockLoRA):
    """mode=weights: a structured edit of the dense weights (no rank). modules_: {short: {"rank": 0, "rel":
    ||W'-W||_F/||W||_F}}; gain / damage drive the mock quality exactly like MockLoRA."""
    kind = "weights"

    def max_rank(self):
        return 0

    def to_json(self):
        return dict(super().to_json(), kind=self.kind)


def _wrel(cur: float, d: float) -> float:
    return float(math.sqrt(cur ** 2 + d ** 2))


def _w_touch(lo, mods, d: float):
    for m in mods:
        cur = lo.modules_.get(m, {"rank": 0, "rel": 0.0})
        lo.modules_[m] = {"rank": 0, "rel": _wrel(cur["rel"], d)}


def _w_effect(lo, n_mods: int, d: float, dmax: float, coef: float):
    """Small relative changes (d ~ 1-2% per module) give a little gain; large ones cost damage ~ d^2."""
    lo.gain += coef * min(1.0, d / 0.02) * min(1.0, math.sqrt(max(n_mods, 1) / 8))
    lo.damage += 4.0 * d ** 2 * math.sqrt(max(n_mods, 1)) + 0.002 * dmax ** 2


def m_w_scale(edit, mods, s):
    """W <- s W on every module of mods: relative change |s-1| per module."""
    lo = edit.copy()
    d = abs(float(s) - 1)
    _w_effect(lo, len(mods), d, d, 0.006)
    _w_touch(lo, mods, d)
    lo.ops.append(["w_scale", list(mods), float(s)])
    return lo


def _channel_change(factors, idx, dim):
    """(relative Frobenius change, max |f-1|) of scaling channels idx (None = all) by factors (list or float),
    assuming channels of equal norm."""
    if isinstance(factors, (int, float)):
        n = dim if idx is None else len(idx)
        dev = [abs(float(factors) - 1)] * max(n, 1)
    else:
        dev = [abs(float(f) - 1) for f in factors]
    dim = max(int(dim or len(dev)), 1)
    return math.sqrt(sum(x * x for x in dev) / dim), max(dev, default=0.0)


def _m_w_channels(edit, mod, factors, idx, dim, kind):
    lo = edit.copy()
    if dim is None:
        dim = len(factors) if isinstance(factors, list) else 1
    d, dmax = _channel_change(factors, idx, dim)
    _w_effect(lo, 1, d, dmax, 0.008)
    _w_touch(lo, [mod], d)
    n = dim if idx is None else len(idx)
    lo.ops.append([kind, [mod], n, round(d, 6)])
    return lo


def m_w_rows(edit, mod, factors, idx=None, dim=None):
    """Scale output channels (rows of W) idx (None = all) of one module by factors."""
    return _m_w_channels(edit, mod, factors, idx, dim, "w_rows")


def m_w_cols(edit, mod, factors, idx=None, dim=None):
    """Scale input channels (columns of W) idx (None = all) of one module by factors."""
    return _m_w_channels(edit, mod, factors, idx, dim, "w_cols")


def m_w_add_lora(edit, lora, alpha=1.0):
    """W <- W + alpha * (B A) of a (mock) LoRA: its gain / damage scaled by alpha (damage by alpha^2)."""
    lo = edit.copy()
    a = float(alpha)
    lo.gain += a * lora.gain
    lo.damage += a * a * lora.damage
    for m, v in lora.modules_.items():
        cur = lo.modules_.get(m, {"rank": 0, "rel": 0.0})
        lo.modules_[m] = {"rank": 0, "rel": _wrel(cur["rel"], abs(a) * v["rel"])}
    lo.ops.append(["w_add_lora", sorted(lora.modules_), a, lora.digest()])
    return lo


def m_w_toward(edit, mods, alpha, base_rel=None, undo_gain=0.0):
    """W <- W + alpha (W_0 - W) on mods: shrinks the edit there by (1-alpha) (damage by (1-alpha)^2 on that share)
    and, where the current weights differ from M_0 (base_rel[m] = ||W-W_0||/||W||, earlier merges), undoes alpha of
    that difference (losing `undo_gain` of the merged gain)."""
    lo = edit.copy()
    a = min(max(float(alpha), 0.0), 1.0)
    base_rel = base_rel or {}
    tot = sum(v["rel"] ** 2 for v in lo.modules_.values())
    share = (sum(lo.modules_[m]["rel"] ** 2 for m in mods if m in lo.modules_) / tot) if tot > 0 else 0.0
    f = a * share
    lo.gain *= 1 - f
    lo.damage *= (1 - f) ** 2
    lo.gain -= float(undo_gain)
    for m in mods:
        cur = lo.modules_.get(m, {"rank": 0, "rel": 0.0})["rel"]
        rel = math.sqrt(((1 - a) * cur) ** 2 + (a * float(base_rel.get(m, 0.0))) ** 2)
        if rel > 0:
            lo.modules_[m] = {"rank": 0, "rel": rel}
        else:                                   # fully back to M_0 there: nothing left to edit
            lo.modules_.pop(m, None)
    lo.ops.append(["w_toward_original", list(mods), a])
    return lo


def m_tensor_lora(info: dict) -> "MockLoRA":
    """A LoRA built by tensor_compute from explicit factors (2026-11-21): info = {module: (rank, rel)} with rel =
    ||B A||_F / ||W||_F of the actual factors; gain / damage like the constructors (m_construct, strength rel/0.01)."""
    lo = MockLoRA()
    for m, (rank, rel) in sorted(info.items()):
        a = float(rel) / 0.01
        lo.gain += 0.006 * min(1.0, a)
        lo.damage += 0.008 * a ** 2
        lo.touch([m], int(rank), float(rel))
    lo.ops.append(["tensor_compute", sorted(info)])
    return lo


def m_w_delta(edit, name: str, rel: float, ndim: int, tau_ratio: float = 0.1):
    """W <- W + D on one tensor (a tensor_compute write, 2026-11-21). rel = the capped measure: ||D||_F/||W||_F for
    >= 2-D tensors, rms(D) / max(rms(W), 1) for 1-D ones (scaled by tau_ratio = norm_cap_tau / vector_cap_tau for the
    gain / damage model). modules_[name] keeps ndim so the mock check applies the right cap."""
    lo = edit.copy()
    d = float(rel) if ndim >= 2 else float(rel) * tau_ratio
    _w_effect(lo, 1, d, d, 0.006)
    cur = lo.modules_.get(name, {"rank": 0, "rel": 0.0})
    lo.modules_[name] = {"rank": 0, "rel": _wrel(cur["rel"], float(rel)), "ndim": int(ndim)}
    lo.ops.append(["w_delta", [name], round(float(rel), 6)])
    return lo


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
            })
        rows.append(row)
    return {"model": "mock", "n_modules": len(rows), "rows": rows, "stack_gens": [g for g, _ in stack]}


# ============================================================================ mock editor policy
def make_mock_policy(seed: int, skill: float = 0.7, tag: str = "main", mode: str = "scratch",
                     tensor_compute: bool = False, paper: bool = False):
    """Return policy(history) -> program text. Choices vary with (thread, step) parsed from the last user
    message ("[thread=K step=S]"); a higher `skill` (the editor model's mock quality) picks better
    hyper-parameters and protective operators more often. mode (method.mode): edit = edits of the raw LoRA
    (EDIT_PROGRAM); scratch = gradient-free constructions from zero (SCRATCH_PROGRAM); weights = structured edits
    of the dense weights returned as {"edit": handle} (WEIGHTS_PROGRAM). tensor_compute (2026-11-21, scratch, the
    tool is on): thread 1 writes TENSOR_PROGRAM instead (api.tensor_compute vectors / LoRA factors); every other
    choice is unchanged. paper (2026-11-21, paper and screen tools on; scratch / weights): thread 1 writes PAPER_PROGRAM instead (screen_results / screen_output, acts on
    its own screen answers, direction / unit / tensor_compute(vectors=...), project_out, save_edit / load_edit).
    There is no regmean_preserve tool (2026-11-22): the recipes "regmean" / "singular_regmean" are written as
    "intruder" / "singular_scale" (the recipe lists keep their length, so the other choices are unchanged; the
    templates have no branch for them, 2026-11-23)."""

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
            if recipe == "regmean":                     # no regmean_preserve tool
                recipe = "intruder"
            return EDIT_PROGRAM.format(recipe=recipe, th=th, st=st,
                                       k=rng.choice([8, 16] if good else [2, 4]),
                                       lam=rng.choice([0.3, 0.5] if good else [0.1, 0.9]),
                                       alpha=rng.choice([1.0, 2.0] if good else [0.2]),
                                       s=rng.choice([0.8, 0.9] if good else [0.5, 1.4]),
                                       boost=rng.choice([1.6, 2.0] if good else [1.05]),
                                       lo=rng.choice([0, 8, 16]), hi=rng.choice([48, 56, 64]))
        if paper and th == 1 and mode in ("scratch", "weights"):
            key = "edit" if mode == "weights" else "lora"
            return PAPER_PROGRAM.format(th=th, st=st, mode=mode, key=key, layer=rng.choice([40, 44, 48]),
                                        new="api.new_edit()['edit']" if mode == "weights" else "api.new_lora()['lora']",
                                        alpha=rng.choice([0.05, 0.1] if good else [0.5]))
        if mode == "weights":
            recipe = WEIGHTS_RECIPES[(th + 2 * st + rng.randrange(len(WEIGHTS_RECIPES))) % len(WEIGHTS_RECIPES)]
            return WEIGHTS_PROGRAM.format(recipe=recipe, th=th, st=st, n=rng.choice([4, 8] if good else [2, 16]),
                                          f=rng.choice([1.04, 1.06] if good else [1.3, 0.7]),
                                          alpha=rng.choice([0.5, 1.0] if good else [1.0, 1.5]),
                                          rf=rng.choice([1.2, 1.3, 0.8] if good else [1.6, 0.4]),
                                          s=rng.choice([0.99, 1.01] if good else [1.03, 0.96]),
                                          lo=rng.choice([8, 16]), hi=rng.choice([48, 56]))
        recipe = SCRATCH_RECIPES[(th + st + rng.randrange(len(SCRATCH_RECIPES))) % len(SCRATCH_RECIPES)]
        if recipe == "singular_regmean":
            recipe = "singular_scale"
        if tensor_compute and th == 1:
            return TENSOR_PROGRAM.format(th=th, st=st, n=rng.choice([8, 16] if good else [2, 32]),
                                         f=rng.choice([1.04, 1.06] if good else [1.3, 0.7]),
                                         lo=rng.choice([8, 16]), hi=rng.choice([48, 56]))
        return SCRATCH_PROGRAM.format(recipe=recipe, th=th, st=st, n=rng.choice([8, 16] if good else [2, 32]),
                                      f=rng.choice([1.04, 1.06] if good else [1.3, 0.7]),
                                      alpha=rng.choice([0.3, 0.5] if good else [2.0]),
                                      lo=rng.choice([8, 16]), hi=rng.choice([48, 56]))

    return policy


# "regmean" / "singular_regmean" are never written (make_mock_policy rewrites those recipes); they stay in the lists
# so the other recipe choices are unchanged (2026-11-22).
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


# (2026-11-21) scratch with the tensor tool: one tensor_compute call whose vectors feed rank1 and whose LoRA factors
# are combined in
TENSOR_PROGRAM = '''\
# mock editor program (mode=scratch): thread {th} step {st}, recipe tensor_compute
def main(api):
    mods = [m["module"] for m in api.modules() if m["leaf"] == "down_proj" and {lo} <= m["layer"] < {hi}][: {n}]
    lora = api.new_lora()["lora"]
    for m in mods:
        s = api.svd(module=m, k=4)
        lora = api.scale_singular(lora=lora, module=m, svd=s["handle"], idx=[0, 1], factors=[{f}, {f}])["lora"]
    small = [m["module"] for m in api.modules() if m["leaf"] == "in_proj_b"][0]
    code = ("U, S, V = torch.svd_lowrank(T[%r], q=4)\\n"
            "vectors = {{'u': U[:, 0], 'v': V[:, 0]}}\\n"
            "lora = {{%r: (V[:, :1].T.contiguous(), (U[:, :1] * S[:1] * 0.002).contiguous())}}\\n"
            "result = [float(x) for x in S[:2]]") % (small, small)
    tc = api.tensor_compute(code=code, tensors=[small])
    print("top singular values", tc.get("json"), "sizes", tc.get("vector_sizes"), "lora", tc.get("lora_check"))
    lora = api.rank1(lora=lora, module=small, u=tc["vectors"]["u"], v=tc["vectors"]["v"], s=0.5)["lora"]
    lora = api.combine(loras=[lora, tc["lora"]], weights=[1.0, 1.0])["lora"]
    return {{"lora": lora}}
'''


# (2026-11-21) the simple flow: the paper-style recipe on the current model's own screen answers
PAPER_PROGRAM = '''\
# mock editor program (mode={mode}): thread {th} step {st}, recipe paper
def main(api):
    rows = api.screen_results(filter="all", limit=200)
    wrong = [r["id"] for r in rows if 2 * sum(1 for s in r["scores"] if s is not None and s > 0.5) <= len(r["scores"])][:8]
    right = [r["id"] for r in rows if r["scores"] and all(s is not None and s > 0.5 for s in r["scores"])][:8]
    if rows:
        out = api.screen_output(id=rows[0]["id"], sample=0)
        print("screen output", out["total_chars"], "chars, thinking", out["has_thinking"])
    mats = api.output_matrices(layer={layer})
    mod = [m["module"] for m in mats if m["kind"] == "down_proj"][0]
    target = {new}
    if wrong and right:
        a = api.acts(module=mod, ids=wrong, source="own")
        b = api.acts(module=mod, ids=right, source="own", sample="all")
        d = api.direction(a=a["handle"] + ".out_mean", b=b["handle"] + ".out_mean")
        u = api.unit(ref=d["vector"])
        tc = api.tensor_compute(code="result = float(V[%r] @ V[%r])" % (d["vector"], u["vector"]),
                                vectors=[d["vector"], u["vector"]])
        print("cos(direction, unit)", tc.get("json"))
        target = api.project_out(target=target, module=mod, u=u["vector"], alpha={alpha})["{key}"]
    api.save_edit(handle=target, name="best")
    api.save_edit(handle=target, name="scratch_copy")
    api.drop_edit(name="scratch_copy")
    print("saved", [e["name"] for e in api.list_edits()])
    return {{"{key}": api.load_edit(name="best")["{key}"]}}
'''


WEIGHTS_RECIPES = ["rows", "cols", "rows_scale", "scale", "toward"]

WEIGHTS_PROGRAM = '''\
# mock editor program (mode=weights): thread {th} step {st}, recipe {recipe}
def main(api):
    recipe = "{recipe}"
    down = [m["module"] for m in api.modules() if m["leaf"] == "down_proj"]
    mods = [m for m in down if {lo} <= int(m.split(".")[1]) < {hi}][: {n}]
    lora = api.new_lora()["lora"]                # singular-direction edits, added below as a dense dW
    for m in mods:
        s = api.svd(module=m, k=4)
        lora = api.scale_singular(lora=lora, module=m, svd=s["handle"], idx=[0, 1], factors=[{f}, {f}])["lora"]
    e = api.new_edit()["edit"]
    e = api.w_add_lora(edit=e, lora=lora, alpha={alpha})["edit"]
    one = mods[0] if mods else down[0]
    if recipe in ("rows", "rows_scale"):         # scale a few output channels of one module
        e = api.w_scale_rows(edit=e, module=one, factors=[{rf}] * 16, rows=list(range(16)))["edit"]
    if recipe == "cols":                         # scale a few input channels of one module
        e = api.w_scale_cols(edit=e, module=one, factors=[{rf}] * 16, cols=list(range(16)))["edit"]
    if recipe in ("rows_scale", "scale"):        # whole-module scaling near 1
        e = api.w_scale(edit=e, modules=mods[:2] or [one], s={s})["edit"]
    if recipe == "toward":                       # move one module halfway back to the original weights
        e = api.w_toward_original(edit=e, modules=[one], alpha=0.5)["edit"]
    info = api.inspect_edit(edit=e)
    print("modules", info["modules"], "check_ok", info["check_ok"], "total_rel_norm", info["total_rel_norm"])
    try:                                         # cheap screening (forward passes only)
        print("train_loss delta", api.train_loss(edit=e, n=16)["delta"])
    except Exception as ex:  # noqa: BLE001  (e.g. tool switched off, budget exhausted)
        print("train_loss:", type(ex).__name__, str(ex)[:120])
    return {{"edit": e}}
'''
