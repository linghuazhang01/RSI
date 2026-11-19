"""The editor's tool surface for one generation (or one E_t session): a Workspace holding tensors behind
string handles, registered on an rsicore.sandbox.ToolAPI so programs call `api.<tool>(**kwargs)`.

Handles: L* = LoRA, S* = SVD of a weight, A* = activation statistics, G* = low-rank gradient,
H<g> = adapter accepted in generation g (the stack), C:<cid> = an earlier candidate of this generation.
Vector references (for rank1 / from_direction): "S3.u0", "S3.v1", "A2.in_mean", "A2.out_mean",
"A2.pc0", "G4.<module>.u0" / ".v0".

Budgets (RSI_LOOP_DESIGN §3.3): `evaluate` calls (an identical LoRA is evaluated once and only charged
once) and `gpu_seconds` for grad / sgd / acts / regmean_preserve (measured on gpu1; estimated in mock).
All accounting is done by the workspace's rsicore.sandbox.ToolAPI (`ws.api`, created with the workspace):
evaluate = cost_fn (1 per non-empty LoRA) + dedupe_key_fn (LoRA name = digest; identical LoRAs are free,
also between program calls and the search's screen evaluation, which goes through the same tool);
GPU tools = cost {"gpu_seconds": 0} (refused once exhausted) + cost_fn (measured seconds, capped at what
was left before the call so a successful call keeps its result; the uncapped total is ws.gpu_measured).
Data access: acts / grad / sgd / train_example only accept this generation's `train` ids (TrainData).
Tool set by method.mode (tools_for_mode): free = all; edit = no sgd / grad / from_grad, plus raw_lora (the
raw LoRA under handle "RAW"); scratch = no sgd / grad / from_grad and no raw LoRA. Unregistered tools are
refused by the ToolAPI.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

from . import mock as M
from .common import stable_hash, write_json

TOOL_DOCS = {
    "modules": "modules(layer=None, leaf=None, group=None) -> list of {module, layer, layer_type, leaf, group, out, in}",
    "diagnostics": "diagnostics(module=None, layer=None, top=20, sort='w_norm') -> rows of the fixed diagnostics table",
    "train_ids": "train_ids(n=None) -> ids of this generation's train items (the only items gradient/activation tools accept)",
    "train_example": "train_example(id) -> {prompt, target} of one train item (truncated)",
    "svd": "svd(module, k=8) -> {handle, S, shape}: top-k singular triplets of the CURRENT weight W",
    "acts": "acts(module, ids=None, k=4) -> {handle, in_norm, out_norm, pc_var}: activation stats on train items",
    "grad": "grad(modules, ids=None, k=4, lora=None) -> {handle, modules:{m:{fro,S}}}: top-k SVD of dLoss/dW (costs gpu_seconds)",
    "new_lora": "new_lora() -> {lora}: an empty LoRA",
    "rank1": "rank1(lora, module, u, v, s) -> {lora}: add s * u v^T (u,v vector refs, normalised)",
    "from_direction": "from_direction(lora, module, direction, key, alpha) -> {lora}: dW = alpha d k^T/|k|^2 so dW k = alpha d",
    "scale_singular": "scale_singular(lora, module, svd, idx, factors) -> {lora}: dW = sum_i (f_i-1) s_i u_i v_i^T",
    "from_grad": "from_grad(lora, grad, lr, k=None, modules=None) -> {lora}: dW = -lr * (low-rank gradient)",
    "sgd": "sgd(lora=None, ids=None, steps=50, lr=1e-4, rank=16, modules=None, batch_size=4) -> {lora, loss_first, loss_last, steps_done, gpu_seconds}",
    "scale": "scale(lora, s, modules=None) -> {lora}",
    "interpolate": "interpolate(a, b, lam) -> {lora}: (1-lam) a + lam b",
    "combine": "combine(loras, weights=None) -> {lora}: weighted sum",
    "drop": "drop(lora, modules) -> {lora}",
    "truncate_rank": "truncate_rank(lora, k, modules=None) -> {lora}",
    "intruder_scale": "intruder_scale(lora, lam=0.5, eps=0.3, k0=32, modules=None) -> {lora}",
    "relchange_prune": "relchange_prune(lora, frac=0.03, modules=None) -> {lora}",
    "regmean_preserve": "regmean_preserve(lora, alpha=1.0, modules=None) -> {lora} (costs gpu_seconds)",
    "orthogonalize": "orthogonalize(lora, against='stack') -> {lora}",
    "ties": "ties(loras, weights=None, density=0.2) -> {lora}",
    "inspect": "inspect(lora) -> {modules, max_rank, check, top, intruder_mean}",
    "stack": "stack() -> list of {handle, gen, modules, max_rank} (adapters merged in earlier generations)",
    "raw_lora": "raw_lora() -> {lora: 'RAW', modules, max_rank, train}: the LoRA trained for you this generation "
                "with the fixed recipe (the starting point you edit)",
    "candidate": "candidate(cid) -> {lora} of an earlier candidate of this generation",
    "evaluate": "evaluate(lora) -> structured screen feedback (costs 1 evaluate; identical LoRAs are free)",
    "budget": "budget() -> {spent, left}",
}


def tools_for_mode(mode: str) -> dict:
    """TOOL_DOCS available to the editor in method.mode: free = everything except raw_lora; edit = no
    training / gradient tools, plus raw_lora; scratch = no training / gradient tools, no raw_lora."""
    from .common import GRADIENT_TOOLS
    out = dict(TOOL_DOCS)
    if mode in ("edit", "scratch"):
        for t in GRADIENT_TOOLS:
            out.pop(t, None)
    if mode != "edit":
        out.pop("raw_lora", None)
    return out


class BudgetError(Exception):
    pass


def _budget_exc():
    try:
        from rsicore.sandbox import BudgetExceeded
        return BudgetExceeded
    except Exception:  # noqa: BLE001
        return BudgetError


class Workspace:
    def __init__(self, *, mock: bool, store, train, cfg: dict, out_dir: Path, evaluator=None, hf=None,
                 stack=None, retention_texts=None, budget: dict | None = None, name_prefix: str = "c",
                 diagnostics: dict | None = None, seed: int = 42, mode: str = "free", raw=None,
                 raw_info: dict | None = None):
        self.mock = mock
        self.mode = mode                        # method.mode: which tools are registered (tools_for_mode)
        self.raw = raw                          # mode=edit: the raw LoRA (handle "RAW")
        self.raw_info = dict(raw_info or {})
        self.store = store
        self.train = train
        self.cfg = cfg
        self.lcfg = cfg.get("lora", {})
        self.out_dir = Path(out_dir)
        self.evaluator = evaluator
        self.hf = hf
        self.stack = list(stack or [])          # [(gen, LoRA|MockLoRA)]
        self.retention_texts = list(retention_texts or [])
        from rsicore.sandbox import ToolAPI
        self.api = ToolAPI(dict(budget or {"evaluate": 32, "gpu_seconds": 7200}))   # the budget accountant
        self.budget = self.api.budget
        self.gpu_measured = 0.0
        self.name_prefix = name_prefix
        self.diag = diagnostics
        self.seed = seed
        self._h: dict[str, object] = {}
        self._owner: dict[str, str] = {}
        self._n = 0
        self._lock = threading.RLock()
        self._local = threading.local()
        self._evaluated: dict[str, tuple] = {}
        self._gram_t, self._gram_r = {}, {}
        self.tool_log: dict[str, list] = {}
        self._registered = False
        self.rank_cap = int(self.lcfg.get("rank_cap", 64))
        self.tau = float(self.lcfg.get("norm_cap_tau", 0.1))
        self.allowed = self.lcfg.get("allowed_leaves")
        for g, lo in self.stack:
            self._h[f"H{g}"] = lo
        if raw is not None:
            self._h["RAW"] = raw

    # ------------------------------------------------------------------ bookkeeping
    @property
    def spent(self) -> dict:
        return self.api.spent

    def left(self, kind: str) -> float:
        return self.budget.get(kind, math.inf) - self.api.spent.get(kind, 0)

    def begin_program(self, cid: str) -> None:
        self._local.cid = cid
        self.tool_log[cid] = []

    def end_program(self, cid: str, keep: tuple = ()) -> None:
        with self._lock:
            for h in [h for h, o in self._owner.items() if o == cid and h not in keep]:
                self._h.pop(h, None)
                self._owner.pop(h, None)
        self._local.cid = None

    def _cid(self):
        return getattr(self._local, "cid", None) or "_"

    def _new(self, prefix: str, obj) -> str:
        with self._lock:
            self._n += 1
            h = f"{prefix}{self._n}"
            self._h[h] = obj
            self._owner[h] = self._cid()
            return h

    def _get(self, h, kind: str | None = None):
        if h is None:
            return None
        if isinstance(h, dict) and "lora" in h:
            h = h["lora"]
        if isinstance(h, str) and h.startswith("C:"):
            return self.load_candidate(h[2:])
        if h not in self._h:
            raise KeyError(f"unknown handle {h!r}")
        obj = self._h[h]
        if kind == "lora" and not self._is_lora(obj):
            raise TypeError(f"handle {h!r} is not a LoRA")
        return obj

    def _is_lora(self, obj):
        from_mock = isinstance(obj, M.MockLoRA)
        if self.mock:
            return from_mock
        from .primitives.lora import LoRA
        return isinstance(obj, LoRA)

    def _empty(self):
        if self.mock:
            return M.MockLoRA()
        from .primitives.lora import LoRA
        return LoRA()

    def _mods(self, modules) -> list[str]:
        if modules is None:
            return []
        if isinstance(modules, str):
            modules = [modules]
        allm = self.store.modules()
        out = []
        for pat in modules:
            if pat in ("all", "*"):
                out += allm
            elif self.store.has(pat):
                out.append(pat)
            else:
                hits = [m for m in allm if m.endswith("." + pat) or pat in m]
                if not hits:
                    raise KeyError(f"no module matches {pat!r}")
                out += hits
        return sorted(set(out), key=allm.index)

    def _ids(self, ids, default_n=None):
        if ids is None:
            ids = self.train.ids()
            if default_n:
                ids = ids[:default_n]
        return self.train.check(ids)

    def _log(self, name, kwargs):
        lst = self.tool_log.setdefault(self._cid(), [])
        lst.append(name)

    def _gpu(self, fn, est_seconds: float):
        """Run fn and record its gpu seconds (mock: the estimate) for the tool's cost_fn."""
        if self.left("gpu_seconds") <= 0:
            raise _budget_exc()("gpu_seconds budget exhausted")
        if self.mock:
            self._local.gpu = getattr(self._local, "gpu", 0.0) + est_seconds
            return fn()
        t0 = self.hf.gpu_seconds
        try:
            return fn()
        finally:
            self._local.gpu = getattr(self._local, "gpu", 0.0) + (self.hf.gpu_seconds - t0)

    def _gpu_cost(self, kwargs, result) -> dict:
        """cost_fn of the GPU tools: seconds measured by _gpu during this call, capped at the remaining budget."""
        used = float(getattr(self._local, "gpu", 0.0))
        self._local.gpu = 0.0
        self.gpu_measured += used
        return {"gpu_seconds": min(used, max(self.left("gpu_seconds"), 0.0))}

    def _eval_key(self, kwargs):
        """dedupe_key_fn of evaluate: the LoRA's served name (digest), or "__keep__" for an empty LoRA."""
        lo = self._get(kwargs.get("lora"), "lora")
        return self.lora_name(lo) if lo.modules() else "__keep__"

    def _eval_cost(self, kwargs, result) -> dict:
        """cost_fn of evaluate: 1 per non-empty LoRA (evaluating "keep" is free)."""
        return {} if self._eval_key(kwargs) == "__keep__" else {"evaluate": 1}

    # ------------------------------------------------------------------ registration
    GPU_TOOLS = ("acts", "grad", "sgd", "regmean_preserve")

    def build_api(self):
        if self._registered:
            return self.api
        docs = tools_for_mode(self.mode)
        if self.mode == "edit" and self.raw is None:
            docs.pop("raw_lora", None)
        for name, doc in docs.items():
            fn = self._wrap(name, getattr(self, "t_" + name))
            if name == "evaluate":
                self.api.register(name, fn, cost={"evaluate": 0}, cost_fn=self._eval_cost,
                                  dedupe_key_fn=self._eval_key, doc=doc)
            elif name in self.GPU_TOOLS:
                self.api.register(name, fn, cost={"gpu_seconds": 0}, cost_fn=self._gpu_cost, doc=doc)
            else:
                self.api.register(name, fn, doc=doc)
        self._registered = True
        return self.api

    def _wrap(self, name, fn):
        def call(**kwargs):
            with self._lock:
                self._log(name, kwargs)
                return fn(**kwargs)
        call.__name__ = name
        return call

    # ------------------------------------------------------------------ info tools
    def t_modules(self, layer=None, leaf=None, group=None):
        out = []
        for m in self.store.modules():
            inf = self.store.info(m)
            if layer is not None and inf["layer"] != int(layer):
                continue
            if leaf is not None and inf["leaf"] != leaf:
                continue
            if group is not None and inf["group"] != group:
                continue
            out.append(inf)
        return out

    def t_diagnostics(self, module=None, layer=None, top=20, sort="w_norm"):
        rows = (self.diag or {}).get("rows", [])
        if module is not None:
            rows = [r for r in rows if r["module"] == module]
        if layer is not None:
            rows = [r for r in rows if r["layer"] == int(layer)]
        if sort:
            def key(r):
                v = r.get(sort)
                if v is None and r.get("stack"):
                    v = r["stack"][-1].get(sort)
                return -(v if isinstance(v, (int, float)) else 0)
            rows = sorted(rows, key=key)
        return rows[: int(top)] if top else rows

    def t_train_ids(self, n=None):
        ids = self.train.ids()
        return ids[: int(n)] if n else ids

    def t_train_example(self, id):  # noqa: A002
        ex = self.train.get([id])[0]
        msgs = ex.get("messages") or []
        return {"prompt": "\n".join(m.get("content", "") for m in msgs)[:2000],
                "target": (ex.get("target") or "")[:2000], "source": ex.get("source")}

    def t_budget(self):
        return {"spent": dict(self.api.spent), "left": self.api.remaining()}

    def t_stack(self):
        return [{"handle": f"H{g}", "gen": g, "modules": len(lo.modules()), "max_rank": lo.max_rank()}
                for g, lo in self.stack]

    def t_raw_lora(self):
        if self.raw is None:
            raise KeyError("no raw LoRA in this session")
        return {"lora": "RAW", "modules": len(self.raw.modules()), "max_rank": self.raw.max_rank(),
                "train": self.raw_info}

    def t_candidate(self, cid):
        lo = self.load_candidate(cid)
        if lo is None:
            raise KeyError(f"candidate {cid!r} has no saved LoRA")
        return {"lora": self._new("L", lo)}

    # ------------------------------------------------------------------ weight-space tools
    def t_svd(self, module, k=8):
        k = int(min(max(1, k), 64))
        if self.mock:
            s1 = 10 + (int(stable_hash(module), 16) % 100) / 20
            S = [round(s1 * 0.85 ** i, 4) for i in range(k)]
            return {"handle": self._new("S", {"module": module, "S": S}), "S": S, "shape": list(self.store.shape(module))}
        U, S, V = self.store.svd(module, k)
        return {"handle": self._new("S", {"module": module, "U": U, "S": S, "V": V}), "S": [float(x) for x in S],
                "shape": list(self.store.shape(module))}

    def t_acts(self, module, ids=None, k=4):
        ids = self._ids(ids, default_n=64)
        if not self.store.has(module):
            raise KeyError(f"unknown module {module!r}")
        if self.mock:
            def run():
                return {"module": module, "in_norm": 12.0, "out_norm": 3.0, "pc_var": [4.0 / (i + 1) for i in range(k)]}
            st = self._gpu(run, est_seconds=2 + 0.05 * len(ids))
            return {"handle": self._new("A", st), "in_norm": st["in_norm"], "out_norm": st["out_norm"],
                    "pc_var": st["pc_var"], "tokens": 200 * len(ids)}
        texts = self.train.texts(ids)
        st = self._gpu(lambda: self.hf.acts(texts, [module], k=int(k))[module], est_seconds=0)
        st["module"] = module
        return {"handle": self._new("A", st), "in_norm": st["in_norm"], "out_norm": st["out_norm"],
                "pc_var": st["pc_var"], "tokens": st["tokens"]}

    def t_grad(self, modules, ids=None, k=4, lora=None):
        ids = self._ids(ids, default_n=64)
        mods = self._mods(modules)
        if not mods:
            raise ValueError("grad needs modules")
        lo = self._get(lora, "lora") if lora else None
        k = int(min(max(1, k), 32))
        if self.mock:
            def run():
                return {m: {"S": [1.0 / (i + 1) for i in range(k)], "fro": 1.5} for m in mods}
            g = self._gpu(run, est_seconds=5 + 0.2 * len(ids))
        else:
            ex = self._examples(ids)
            g = self._gpu(lambda: self.hf.grad(ex, mods, lora=lo, k=k), est_seconds=0)
        h = self._new("G", g)
        return {"handle": h, "modules": {m: {"fro": float(v["fro"]), "S": [float(x) for x in v["S"]][:k]}
                                         for m, v in g.items()}}

    def _examples(self, ids):
        ex = [e for e in self.train.get(ids) if e.get("target")]
        if not ex:
            raise ValueError("none of these train items has a training target")
        return ex

    def _vec(self, ref: str, dim: int):
        """Resolve a vector reference to a tensor of size dim (mock: None after validation)."""
        if not isinstance(ref, str) or "." not in ref:
            if isinstance(ref, list) and len(ref) == dim:
                if self.mock:
                    return None
                import torch
                return torch.tensor(ref, dtype=torch.float32)
            raise ValueError(f"bad vector reference {ref!r}")
        h, rest = ref.split(".", 1)
        obj = self._get(h)
        if self.mock:
            return None
        if h.startswith("S"):
            side, i = rest[0], int(rest[1:])
            v = obj["U"][:, i] if side == "u" else obj["V"][:, i]
        elif h.startswith("A"):
            if rest in ("in_mean", "out_mean"):
                v = obj[rest]
            elif rest.startswith("pc"):
                v = obj["pcs"][int(rest[2:])]
            else:
                raise ValueError(f"bad activation field {rest!r}")
        elif h.startswith("G"):
            mod, tail = rest.rsplit(".", 1)
            side, i = tail[0], int(tail[1:])
            v = obj[mod]["U"][:, i] if side == "u" else obj[mod]["V"][:, i]
        else:
            raise ValueError(f"handle {h!r} has no vectors")
        if v.numel() != dim:
            raise ValueError(f"vector {ref!r} has size {v.numel()}, need {dim}")
        return v

    # ------------------------------------------------------------------ constructors
    def t_new_lora(self):
        return {"lora": self._new("L", self._empty())}

    def t_rank1(self, lora, module, u, v, s):
        base = self._get(lora, "lora")
        out_f, in_f = self.store.shape(module)
        uu, vv = self._vec(u, out_f), self._vec(v, in_f)
        if self.mock:
            return {"lora": self._new("L", M.m_construct(base, module, s, "rank1"))}
        from .primitives.construct import rank1
        lo = base.copy()
        rank1(lo, module, uu, vv, float(s), shape=(out_f, in_f))
        return {"lora": self._new("L", lo)}

    def t_from_direction(self, lora, module, direction, key, alpha):
        base = self._get(lora, "lora")
        out_f, in_f = self.store.shape(module)
        d, kk = self._vec(direction, out_f), self._vec(key, in_f)
        if self.mock:
            return {"lora": self._new("L", M.m_construct(base, module, alpha, "from_direction"))}
        from .primitives.construct import from_direction
        lo = base.copy()
        from_direction(lo, module, d, kk, float(alpha), shape=(out_f, in_f))
        return {"lora": self._new("L", lo)}

    def t_scale_singular(self, lora, module, svd, idx, factors):
        base = self._get(lora, "lora")
        s = self._get(svd)
        if s["module"] != module:
            raise ValueError(f"svd handle is for {s['module']}, not {module}")
        if self.mock:
            f = factors if isinstance(factors, list) else [factors] * len(idx)
            strength = sum(abs(float(x) - 1) for x in f) * 5
            return {"lora": self._new("L", M.m_construct(base, module, strength, "scale_singular"))}
        from .primitives.construct import scale_singular
        lo = base.copy()
        scale_singular(lo, module, s["U"], s["S"], s["V"], idx, factors)
        return {"lora": self._new("L", lo)}

    def t_from_grad(self, lora, grad, lr, k=None, modules=None):
        base = self._get(lora, "lora")
        g = self._get(grad)
        mods = self._mods(modules) if modules else list(g)
        if self.mock:
            return {"lora": self._new("L", M.m_from_grad(base, mods, float(lr), int(k or 4)))}
        from .primitives.construct import from_lowrank_grad
        lo = base.copy()
        for m in mods:
            from_lowrank_grad(lo, m, g[m]["U"], g[m]["S"], g[m]["V"], float(lr), k=k)
        return {"lora": self._new("L", lo)}

    def t_sgd(self, lora=None, ids=None, steps=50, lr=1e-4, rank=16, modules=None, batch_size=4):
        ids = self._ids(ids)
        base = self._get(lora, "lora") if lora else None
        if modules:
            mods = self._mods(modules)
        elif base is not None and base.modules():
            mods = list(base.modules())
        else:  # default: the SFT recipe's module set
            from .baselines import modules_for, recipe
            mods = modules_for(self.store, recipe(self.cfg)["leaves"])
        steps, rank, batch_size = int(steps), int(min(rank, self.rank_cap)), int(max(1, min(batch_size, 16)))
        lr = float(lr)
        if steps <= 0 or not (0 < lr < 1):
            raise ValueError("need steps > 0 and 0 < lr < 1")
        if self.mock:
            est = steps * batch_size * float(self.cfg.get("mock", {}).get("sec_per_example", 0.6))
            if est > self.left("gpu_seconds"):
                frac = max(self.left("gpu_seconds"), 0) / est
                steps = max(1, int(steps * frac))
                est = self.left("gpu_seconds")
            lo = self._gpu(lambda: M.m_sgd(base, mods, ids, steps, lr, rank), est_seconds=est)
            return {"lora": self._new("L", lo), "loss_first": 1.2, "loss_last": round(1.2 * math.exp(-steps * lr * 40), 4),
                    "steps_done": steps, "gpu_seconds": round(est, 1)}
        ex = self._examples(ids)
        t0 = self.hf.gpu_seconds
        lo, info = self._gpu(lambda: self.hf.sgd(ex, base, modules=mods, rank=rank, steps=steps, lr=lr,
                                                  batch_size=batch_size, time_budget_s=self.left("gpu_seconds")),
                             est_seconds=0)
        return {"lora": self._new("L", lo), "loss_first": info["losses"][0] if info["losses"] else None,
                "loss_last": info["losses"][-1] if info["losses"] else None, "steps_done": info["steps_done"],
                "gpu_seconds": round(self.hf.gpu_seconds - t0, 1)}

    # ------------------------------------------------------------------ operators
    def _op(self, mock_fn, real_fn, *los):
        if self.mock:
            return {"lora": self._new("L", mock_fn(*los))}
        out = real_fn(*los)
        if out.max_rank() > self.rank_cap:
            out.recompress(self.rank_cap)
        return {"lora": self._new("L", out)}

    def t_scale(self, lora, s, modules=None):
        from .primitives import ops
        lo = self._get(lora, "lora")
        return self._op(lambda a: M.m_scale(a, float(s)), lambda a: ops.scale(a, float(s), modules), lo)

    def t_interpolate(self, a, b, lam):
        from .primitives import ops
        A, B = self._get(a, "lora"), self._get(b, "lora")
        return self._op(lambda x, y: M.m_combine([x, y], [1 - float(lam), float(lam)]),
                        lambda x, y: ops.interpolate(x, y, float(lam), cap=self.rank_cap), A, B)

    def t_combine(self, loras, weights=None):
        from .primitives import ops
        los = [self._get(h, "lora") for h in loras]
        w = [float(x) for x in (weights or [1.0] * len(los))]
        if self.mock:
            return {"lora": self._new("L", M.m_combine(los, w))}
        return {"lora": self._new("L", ops.combine(los, w, cap=self.rank_cap))}

    def t_drop(self, lora, modules):
        from .primitives import ops
        lo = self._get(lora, "lora")
        if self.mock:
            out = lo.copy()
            for m in self._mods(modules):
                out.modules_.pop(m, None)
            out.ops.append(["drop", modules])
            return {"lora": self._new("L", out)}
        return {"lora": self._new("L", ops.drop(lo, modules))}

    def t_truncate_rank(self, lora, k, modules=None):
        from .primitives import ops
        lo = self._get(lora, "lora")
        return self._op(lambda a: M.m_truncate(a, int(k)), lambda a: ops.truncate_rank(a, int(k), modules), lo)

    def t_intruder_scale(self, lora, lam=0.5, eps=0.3, k0=32, modules=None):
        from .primitives import ops
        lo = self._get(lora, "lora")
        return self._op(lambda a: M.m_intruder(a, float(lam)),
                        lambda a: ops.intruder_scale(a, self.store, float(lam), float(eps), int(k0), modules), lo)

    def t_relchange_prune(self, lora, frac=0.03, modules=None):
        from .primitives import ops
        lo = self._get(lora, "lora")
        frac = float(min(max(frac, 0.0), 0.5))
        return self._op(lambda a: M.m_prune(a, frac),
                        lambda a: ops.relchange_prune(a, self.store, frac, modules,
                                                      device=getattr(self.hf, "device", "cpu") if self.hf else "cpu"), lo)

    def t_regmean_preserve(self, lora, alpha=1.0, modules=None):
        from .primitives import ops
        lo = self._get(lora, "lora")
        if self.mock:
            self._gpu(lambda: None, est_seconds=20)
            return {"lora": self._new("L", M.m_regmean(lo, float(alpha)))}
        mods = [m for m in (self._mods(modules) if modules else lo.modules()) if m in lo.factors]
        need = [m for m in mods if m not in self._gram_t]
        if need:
            n = int(self.cfg.get("tools", {}).get("gram_items", 64))
            tt = self.train.texts(self.train.ids()[:n])
            rt = self.retention_texts[:n]
            if not rt:
                raise ValueError("no retention contexts available for regmean_preserve")
            self._gram_t.update(self._gpu(lambda: self.hf.grams(tt, need), est_seconds=0))
            self._gram_r.update(self._gpu(lambda: self.hf.grams(rt, need), est_seconds=0))
        out = ops.regmean_preserve(lo, self._gram_t, self._gram_r, float(alpha), modules=mods)
        return {"lora": self._new("L", out)}

    def t_orthogonalize(self, lora, against="stack"):
        from .primitives import ops
        lo = self._get(lora, "lora")
        if against == "stack":
            others = [x for _, x in self.stack]
        else:
            others = [self._get(h, "lora") for h in (against if isinstance(against, list) else [against])]
        if self.mock:
            return {"lora": self._new("L", M.m_orth(lo, len(others)))}
        return {"lora": self._new("L", ops.orthogonalize(lo, others))}

    def t_ties(self, loras, weights=None, density=0.2):
        from .primitives import ops
        los = [self._get(h, "lora") for h in loras]
        w = [float(x) for x in (weights or [1.0] * len(los))]
        if self.mock:
            return {"lora": self._new("L", M.m_ties(los, w, float(density)))}
        return {"lora": self._new("L", ops.ties(los, w, float(density), cap=self.rank_cap,
                                                device=getattr(self.hf, "device", "cpu") if self.hf else "cpu"))}

    # ------------------------------------------------------------------ inspection / checks
    def check(self, lo) -> tuple[bool, dict]:
        if self.mock:
            errs, mods = [], {}
            for m, v in lo.modules_.items():
                if not self.store.has(m):
                    errs.append(f"{m}: not in model")
                if v["rank"] > self.rank_cap:
                    errs.append(f"{m}: rank {v['rank']} > cap {self.rank_cap}")
                if v["rel"] > self.tau:
                    errs.append(f"{m}: ||dW||/||W0|| = {v['rel']:.4f} > tau {self.tau}")
                mods[m] = {"rank": v["rank"], "rel_norm": v["rel"]}
            return not errs, {"errors": errs[:50], "n_errors": len(errs), "modules": mods}
        from .primitives.lora import check_lora
        return check_lora(lo, self.store, rank_cap=self.rank_cap, tau=self.tau, allowed_leaves=self.allowed)

    def report(self, lo, max_intruder_modules: int = 16) -> dict:
        ok, rep = self.check(lo)
        mods = rep.get("modules", {})
        top = sorted(mods.items(), key=lambda kv: -kv[1].get("rel_norm", 0))[:8]
        out = {"modules": len(mods), "max_rank": lo.max_rank(), "check_ok": ok, "check_errors": rep.get("errors", [])[:5],
               "top_rel_norm": [{"module": m, **{k: round(v, 5) if isinstance(v, float) else v for k, v in d.items()}}
                                for m, d in top],
               "total_rel_norm": round(math.sqrt(sum(d.get("rel_norm", 0) ** 2 for d in mods.values())), 5)}
        if self.mock:
            out["intruder_mean"] = round(0.35 + 0.1 * (lo.damage > lo.gain), 3)
            out["ops"] = lo.ops[-6:]
        elif mods:
            from .primitives.ops import intruder_scores
            al = []
            for m, _ in top[:max_intruder_modules]:
                U0, _, _ = self.store.svd(m, 32)
                _, a = intruder_scores(lo, m, U0)
                al += a
            out["intruder_mean"] = round(sum(al) / len(al), 4) if al else None
            if self.stack:
                out["stack_overlap"] = _overlap(lo, [x for _, x in self.stack])
        return out

    def t_inspect(self, lora):
        lo = self._get(lora, "lora")
        return self.report(lo)

    # ------------------------------------------------------------------ evaluation
    def lora_name(self, lo) -> str:
        return f"{self.name_prefix}-{lo.digest()}"

    def evaluate_lora(self, lo, level: str = "screen", charge: bool = True) -> tuple[float | None, dict]:
        """Register + score `lo` (empty LoRA = keep the model unchanged). Identical LoRAs are cached.
        charge=True at level 'screen' goes through the `evaluate` tool of ws.api (budget + dedupe there)."""
        if charge and level == "screen":
            self.build_api()
            h = self._new("L", lo)
            try:
                fb = self.api.call("evaluate", {"lora": h})
            finally:
                with self._lock:
                    self._h.pop(h, None)
                    self._owner.pop(h, None)
            return fb.get("score"), fb
        empty = not lo.modules()
        name = None if empty else self.lora_name(lo)
        key = (name, level)
        if key in self._evaluated:
            return self._evaluated[key]
        rep = self.report(lo) if not empty else {"modules": 0, "keep": True}
        if not empty and not rep["check_ok"]:
            fb = {"error": "LoRA failed the sandbox checks", "lora_stats": rep}
            self._evaluated[key] = (None, fb)
            return None, fb
        if name is not None and not self.evaluator.known(name):
            self.evaluator.register(lo, name)
        pending = getattr(self._local, "eval_pending", 0)   # the evaluate tool charges after this returns
        cost = {"evaluate_left": self.left("evaluate") - pending, "gpu_seconds_left": round(self.left("gpu_seconds"), 1),
                "gpu_seconds_spent": round(self.spent.get("gpu_seconds", 0), 1)}
        score, fb = self.evaluator.feedback(name, level, lora_report=rep, cost=cost)
        self._evaluated[key] = (score, fb)
        return score, fb

    def t_evaluate(self, lora):
        lo = self._get(lora, "lora")
        self._local.eval_pending = 1 if lo.modules() else 0
        try:
            _, fb = self.evaluate_lora(lo, "screen", charge=False)   # ws.api charges (cost_fn) and dedupes
        finally:
            self._local.eval_pending = 0
        return fb

    def screen_candidate(self, cid: str) -> tuple[float | None, dict]:
        """screen_fn helper: charged screen evaluation of a saved candidate; budget exhausted -> (None, error)."""
        try:
            return self.evaluate_lora(self.load_candidate(cid), "screen", charge=True)
        except _budget_exc() as e:
            return None, {"error": f"not evaluated: {e}"}

    # ------------------------------------------------------------------ candidates
    def cand_path(self, cid: str) -> Path:
        return self.out_dir / "loras" / (f"{cid}.json" if self.mock else f"{cid}.safetensors")

    def save_candidate(self, cid: str, lo) -> Path:
        p = self.cand_path(cid)
        if self.mock:
            write_json(p, lo.to_json())
        else:
            lo.save_raw(p)
        return p

    def load_candidate(self, cid: str):
        p = self.cand_path(cid)
        if not p.exists():
            return None
        if self.mock:
            return M.MockLoRA.from_json(json.load(open(p)))
        from .primitives.lora import LoRA
        return LoRA.load_raw(p)

    def resolve_result(self, result):
        """main(api) return value -> LoRA object (empty = keep), or raise ValueError."""
        if result is None:
            raise ValueError("main(api) returned None; return {'lora': handle} (or {'lora': None} to keep the model)")
        h = result.get("lora", "__missing__") if isinstance(result, dict) else result
        if h == "__missing__":
            raise ValueError("main(api) must return {'lora': <handle>}")
        if h is None:
            return self._empty()
        return self._get(h, "lora")


def _overlap(lo, others) -> float | None:
    """Mean normalised principal-angle overlap (cos² averaged) between ΔW column spaces and the stack."""
    import torch
    vals = []
    for m, (A, B) in lo.factors.items():
        for o in others:
            if m not in o.factors:
                continue
            q1, _ = torch.linalg.qr(B)
            q2, _ = torch.linalg.qr(o.factors[m][1])
            s = torch.linalg.svdvals(q1.T @ q2)
            vals.append(float((s ** 2).mean()))
    return round(sum(vals) / len(vals), 4) if vals else None


def program_build(ws: Workspace, program: str, cid: str, *, timeout_s: int, mem_gb: int, workdir: Path):
    """build_fn for rsicore.search.inner_search: run the program in the sandbox, resolve and check the LoRA,
    save it. Returns (artifact | None, error | None)."""
    from rsicore.sandbox import run_program

    if not program or "def main" not in program:
        return None, "no program (the reply must contain a ```python block defining main(api))"
    ws.begin_program(cid)
    t0 = time.time()
    try:
        res = run_program(program, ws.api, timeout_s=timeout_s, mem_gb=mem_gb, workdir=Path(workdir))
        if not res.ok:
            return None, f"program failed: {res.error}\n{(res.stderr or '')[-1500:]}"
        try:
            lo = ws.resolve_result(res.result)
        except (KeyError, TypeError, ValueError) as e:
            return None, f"bad return value: {e}"
        ok, rep = ws.check(lo)
        if not ok:
            return None, "sandbox checks failed: " + "; ".join(rep["errors"][:8])
        path = ws.save_candidate(cid, lo)
        tools = ws.tool_log.get(cid, [])
        art = {"cid": cid, "path": str(path), "keep": not lo.modules(), "modules": len(lo.modules()),
               "max_rank": lo.max_rank(), "digest": lo.digest(), "tools_used": _count(tools),
               "strategy": classify(tools), "stdout": (res.stdout or "")[-1000:], "seconds": round(time.time() - t0, 1)}
        return art, None
    finally:
        ws.end_program(cid)


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def classify(tools) -> str:
    """Primitive-usage category for behaviour analysis (RSI_LOOP_DESIGN §3.6)."""
    s = set(tools)
    construct = s & {"rank1", "from_direction", "scale_singular", "from_grad"}
    train = "sgd" in s
    stack_ops = s & {"interpolate", "combine", "truncate_rank", "intruder_scale", "relchange_prune",
                     "regmean_preserve", "orthogonalize", "ties", "scale", "drop"}
    if "raw_lora" in s:   # mode=edit: what was done to the raw LoRA
        if construct and stack_ops:
            return "raw+construct+ops"
        if construct:
            return "raw+construct"
        return "raw+ops" if stack_ops else "raw_as_is"
    if not (construct or train or stack_ops):
        return "keep"
    if train and not construct and not stack_ops:
        return "pure_sft"
    if construct and not train:
        return "pure_construct" if not stack_ops else "construct+ops"
    if construct and train:
        return "construct+sft"
    return "sft+ops" if train else "ops_only"
