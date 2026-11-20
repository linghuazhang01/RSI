"""Dense weight edits for Paper B's "weights" mode (any rank, structured operations).

A WeightEdit is {short module name: [op, ...]}; the ops of a module are applied in order to its fp32 weight W
[out, in]:

  {"op": "scale", "s": s}                        W <- s * W
  {"op": "rows", "idx": [i, ...] | None, "f": f}  W[idx, :] *= f   (output channels; idx None = all rows;
                                                  f a scalar or one factor per index; repeated indices compound)
  {"op": "cols", "idx": [j, ...] | None, "f": f}  W[:, idx] *= f   (input channels, same rules)
  {"op": "lowrank", "A": [r, in], "B": [out, r]}  W <- W + B @ A   (fp32 CPU tensors)
  {"op": "toward_base", "alpha": a}               W <- W + a * (W_base - W)   (W_base = original model M_0)

check_wedit is the sandbox check (modules exist, indices / lengths / shapes valid, result finite,
||W' - W||_F / ||W||_F <= tau); edit_into_model writes the edited checkpoint the same way merge_into_model
writes a merged LoRA (shared shard writer in lora.py). torch / safetensors are imported lazily.
"""
from __future__ import annotations

import hashlib
import json
import operator
from pathlib import Path

from .naming import SHORT_RE

OPS = ("scale", "rows", "cols", "lowrank", "toward_base")
_TENSOR_FIELDS = ("A", "B")


def _sort_key(short: str):
    m = SHORT_RE.match(short)
    return (int(m.group(1)) if m else 10 ** 9, short)


def _is_scalar(f) -> bool:
    if isinstance(f, (int, float)):
        return True
    return hasattr(f, "dim") and f.dim() == 0


def _normalize(op: dict) -> dict:
    """Canonical op dict (floats / int lists / fp32 contiguous CPU tensors, extra keys dropped).
    Structural problems (unknown op, missing field, wrong type) raise ValueError; value problems
    (index range, factor count, factor shapes) are left to check_wedit / apply."""
    if not isinstance(op, dict) or op.get("op") not in OPS:
        raise ValueError(f"bad op {op!r}; expected a dict with 'op' in {OPS}")
    kind = op["op"]
    try:
        if kind == "scale":
            return {"op": kind, "s": float(op["s"])}
        if kind == "toward_base":
            return {"op": kind, "alpha": float(op["alpha"])}
        if kind in ("rows", "cols"):
            idx = op.get("idx")
            if idx is not None:
                idx = [operator.index(i) for i in idx]
            f = op["f"]
            f = float(f) if _is_scalar(f) else [float(x) for x in f]
            return {"op": kind, "idx": idx, "f": f}
        A, B = op["A"], op["B"]
        if not (hasattr(A, "dim") and hasattr(B, "dim")):
            raise TypeError("A and B must be tensors")
        return {"op": kind, "A": A.detach().float().cpu().contiguous(), "B": B.detach().float().cpu().contiguous()}
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"bad {kind} op: {e!r}") from None


def _plain(op: dict) -> dict:
    """Op without tensors (JSON-safe)."""
    if op["op"] == "lowrank":
        return {"op": "lowrank", "rank": int(op["A"].shape[0])}
    return {k: (list(v) if isinstance(v, list) else v) for k, v in op.items()}


class WeightEdit:
    kind = "weights"

    def __init__(self, ops: dict | None = None, meta: dict | None = None):
        self.ops: dict[str, list[dict]] = {}
        self.meta = dict(meta or {})
        for short, lst in (ops or {}).items():
            for op in lst:
                self.add(short, op)

    # ------------------------------------------------------------------ basics
    def modules(self) -> list[str]:
        return sorted(self.ops, key=_sort_key)

    def max_rank(self) -> int:
        return 0

    def add(self, short: str, op: dict) -> None:
        self.ops.setdefault(str(short), []).append(_normalize(op))

    def copy(self) -> "WeightEdit":
        new = WeightEdit(meta=dict(self.meta))
        for m, lst in self.ops.items():
            new.ops[m] = [{k: (v.clone() if hasattr(v, "clone") else (list(v) if isinstance(v, list) else v))
                           for k, v in op.items()} for op in lst]
        return new

    def needs_base(self, short: str | None = None) -> bool:
        mods = [short] if short is not None else list(self.ops)
        return any(op["op"] == "toward_base" for m in mods for op in self.ops.get(m, []))

    def digest(self) -> str:
        h = hashlib.sha256()
        for m in self.modules():
            h.update(m.encode())
            for i, op in enumerate(self.ops[m]):
                plain = {k: v for k, v in op.items() if k not in _TENSOR_FIELDS}
                h.update(f"|{i}|".encode())
                h.update(json.dumps(plain, sort_keys=True).encode())
                for k in _TENSOR_FIELDS:
                    if k in op:
                        t = op[k]
                        h.update(f"{k}{tuple(t.shape)}".encode())
                        h.update(t.contiguous().float().numpy().tobytes())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------ math
    def apply(self, short: str, W, W_base=None):
        """New fp32 tensor (on W's device) with the module's ops applied in order; W is not modified."""
        import torch
        out = W.to(torch.float32).clone()
        for op in self.ops.get(short, []):
            out = _apply_op(op, out, W_base)
        return out

    def summary(self) -> dict:
        return {m: [_plain(op) for op in self.ops[m]] for m in self.modules()}

    # ------------------------------------------------------------------ I/O
    def save_raw(self, path) -> None:
        from safetensors.torch import save_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors, spec = {}, {}
        for m in self.modules():
            spec[m] = []
            for i, op in enumerate(self.ops[m]):
                spec[m].append({k: v for k, v in op.items() if k not in _TENSOR_FIELDS})
                for k in _TENSOR_FIELDS:
                    if k in op:
                        tensors[f"{m}|{i}|{k}"] = op[k].contiguous()
        save_file(tensors, str(path), metadata={"kind": self.kind, "ops": json.dumps(spec),
                                                "meta": json.dumps(self.meta, default=str)})

    @classmethod
    def load_raw(cls, path) -> "WeightEdit":
        from safetensors import safe_open
        with safe_open(str(path), "pt") as f:
            md = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        spec = json.loads(md.get("ops", "{}"))
        edit = cls(meta=json.loads(md.get("meta", "{}")))
        for m, lst in spec.items():
            for i, op in enumerate(lst):
                op = dict(op)
                for k in _TENSOR_FIELDS:
                    if f"{m}|{i}|{k}" in tensors:
                        op[k] = tensors[f"{m}|{i}|{k}"]
                edit.add(m, op)
        return edit


# ============================================================================ op math / validation
def _factor_vector(op: dict, n: int, device):
    """Per-channel multiplier g [n] for a rows / cols op (repeated indices compound)."""
    import torch
    idx, f = op["idx"], op["f"]
    if idx is None:
        if isinstance(f, list):
            if len(f) != n:
                raise ValueError(f"{op['op']}: {len(f)} factors for {n} channels (idx=None)")
            return torch.tensor(f, dtype=torch.float32, device=device)
        return torch.full((n,), float(f), dtype=torch.float32, device=device)
    if isinstance(f, list) and len(f) != len(idx):
        raise ValueError(f"{op['op']}: {len(f)} factors for {len(idx)} indices")
    bad = [i for i in idx if not 0 <= i < n]
    if bad:
        raise ValueError(f"{op['op']}: indices out of range [0, {n}): {bad[:5]}")
    g = [1.0] * n
    vals = f if isinstance(f, list) else [float(f)] * len(idx)
    for i, v in zip(idx, vals):
        g[i] *= v
    return torch.tensor(g, dtype=torch.float32, device=device)


def _apply_op(op: dict, W, W_base):
    import torch
    kind = op["op"]
    if kind == "scale":
        return W.mul_(op["s"])
    if kind == "rows":
        return W.mul_(_factor_vector(op, W.shape[0], W.device)[:, None])
    if kind == "cols":
        return W.mul_(_factor_vector(op, W.shape[1], W.device)[None, :])
    if kind == "lowrank":
        A, B = op["A"], op["B"]
        if A.dim() != 2 or B.dim() != 2 or A.shape[0] != B.shape[1] or B.shape[0] != W.shape[0] \
                or A.shape[1] != W.shape[1]:
            raise ValueError(f"lowrank: bad shapes A{tuple(A.shape)} B{tuple(B.shape)} for W{tuple(W.shape)}")
        return W.add_(B.to(W.device, torch.float32) @ A.to(W.device, torch.float32))
    if kind == "toward_base":
        if W_base is None:
            raise ValueError("toward_base needs the base weight W_base (original model M_0)")
        Wb = W_base.to(W.device, torch.float32)
        if tuple(Wb.shape) != tuple(W.shape):
            raise ValueError(f"toward_base: base shape {tuple(Wb.shape)} != {tuple(W.shape)}")
        return W.add_(Wb - W, alpha=op["alpha"])
    raise ValueError(f"unknown op {kind!r}")


def _op_errors(short: str, i: int, op: dict, shape: tuple[int, int]) -> list[str]:
    """Static checks of one op against the weight shape (no tensor math beyond factor shapes)."""
    out_f, in_f = shape
    kind, where = op["op"], f"{short} op {i} ({op['op']})"
    errs = []
    if kind in ("rows", "cols"):
        n = out_f if kind == "rows" else in_f
        idx, f = op["idx"], op["f"]
        if idx is None:
            if isinstance(f, list) and len(f) != n:
                errs.append(f"{where}: {len(f)} factors for {n} channels (idx=None)")
        else:
            bad = [j for j in idx if not 0 <= j < n]
            if bad:
                errs.append(f"{where}: indices out of range [0, {n}): {bad[:5]}")
            if isinstance(f, list) and len(f) != len(idx):
                errs.append(f"{where}: {len(f)} factors for {len(idx)} indices")
    elif kind == "lowrank":
        A, B = op["A"], op["B"]
        if A.dim() != 2 or B.dim() != 2 or A.shape[0] != B.shape[1] or B.shape[0] != out_f or A.shape[1] != in_f:
            errs.append(f"{where}: bad shapes A{tuple(A.shape)} B{tuple(B.shape)} for W[{out_f},{in_f}]")
    return errs


# ============================================================================ conversions / checks
def lora_to_edit(lora) -> WeightEdit:
    """One lowrank op per module of a LoRA (ΔW = B @ A)."""
    edit = WeightEdit(meta=dict(getattr(lora, "meta", {}) or {}))
    for m in lora.modules():
        A, B = lora.factors[m]
        edit.add(m, {"op": "lowrank", "A": A, "B": B})
    return edit


def check_wedit(edit: WeightEdit, store, base_store=None, *, tau: float, max_modules: int | None = None,
                device="cpu") -> tuple[bool, dict]:
    """Sandbox checks: modules exist, indices / factor counts / low-rank shapes valid, toward_base has a
    base model, applied W' finite, ||W' - W||_F / ||W||_F <= tau, at most max_modules modules.
    Returns (ok, report) with report['modules'][short] = {rel_norm, ops}."""
    import torch
    errors, mods = [], {}
    modules = edit.modules()
    if not modules:
        return True, {"errors": [], "n_errors": 0, "modules": {}, "empty": True}
    if max_modules is not None and len(modules) > max_modules:
        errors.append(f"{len(modules)} modules edited > max_modules {max_modules}")
    for m in modules:
        ops = edit.ops[m]
        if not store.has(m):
            errors.append(f"{m}: not in model")
            continue
        shape = tuple(store.shape(m))
        errs = [e for i, op in enumerate(ops) for e in _op_errors(m, i, op, shape)]
        W_base = None
        if edit.needs_base(m):
            if base_store is None:
                errs.append(f"{m}: toward_base needs the base model (no base_store given)")
            elif not base_store.has(m) or tuple(base_store.shape(m)) != shape:
                errs.append(f"{m}: not in the base model or shape differs")
        if errs:
            errors.extend(errs)
            continue
        W = store.weight(m, device=device)
        if edit.needs_base(m):
            W_base = base_store.weight(m, device=device)
        try:
            new = edit.apply(m, W, W_base)
        except (ValueError, RuntimeError) as e:
            errors.append(f"{m}: {e}")
            continue
        if not bool(torch.isfinite(new).all()):
            errors.append(f"{m}: non-finite values after the edit")
            continue
        rel = float((new - W).norm()) / max(float(W.norm()), 1e-12)
        if rel > tau:
            errors.append(f"{m}: ||W'-W||/||W|| = {rel:.4f} > tau {tau}")
        mods[m] = {"rel_norm": rel, "ops": len(ops)}
        del new
    return not errors, {"errors": errors[:50], "n_errors": len(errors), "modules": mods}


# ============================================================================ write checkpoint
def edit_into_model(edit: WeightEdit, src_dir, out_dir, *, store=None, base_store=None, device="cpu",
                    link_unchanged: bool = True) -> dict:
    """Write a new checkpoint with W' = edit.apply(W) for every edited tensor into out_dir.

    Same layout rules as merge_into_model: shards holding an edited tensor are rewritten (fp32 compute,
    cast back to the stored dtype), other shards hard-linked (fallback copy), non-weight files copied,
    written to <out>.partial then renamed. report['modules'][key] = {short, rel_change (realised after
    the cast: ||W'_stored - W|| / ||W||), intended_rel_change (fp32, before the cast)}; also written to
    out_dir/edit_report.json."""
    import torch

    from .lora import _rewrite_checkpoint
    from .weights import WeightStore

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    store = store or WeightStore(src_dir)
    if edit.needs_base() and base_store is None:
        raise ValueError("edit has toward_base ops but no base_store was given")
    shorts = {store.key(m): m for m in edit.modules()}
    report = {"src": str(src_dir), "modules": {}}

    def transform(k, t):
        m = shorts[k]
        w = t.to(device=device, dtype=torch.float32)
        wb = base_store.weight(m, device="cpu").to(device) if edit.needs_base(m) else None
        new32 = edit.apply(m, w, wb)
        new = new32.to(t.dtype)
        wn = max(float(w.norm()), 1e-30)
        report["modules"][k] = {"short": m, "rel_change": float((new.to(torch.float32) - w).norm()) / wn,
                                "intended_rel_change": float((new32 - w).norm()) / wn}
        return new.cpu()

    def finalize(rep):
        rep["edit_digest"] = edit.digest()
        rep["n_ops"] = sum(len(v) for v in edit.ops.values())

    return _rewrite_checkpoint(src_dir, out_dir, store, shorts, transform, report, report_name="edit_report.json",
                               link_unchanged=link_unchanged, finalize=finalize)
