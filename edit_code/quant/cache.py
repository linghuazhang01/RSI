"""Run-level layer quantization cache, built once from M_0 (run_dir/prep/cache).

Every (module, quantizer, bits, group size) of the original model M_0 is quantized at most once and
stored on disk as packed integer codes + fp16 scales + packed zero points (+ AWQ channel scales):
    <root>/<module tensor name>/<quantizer>_b<bits>_g<gs>.safetensors
A candidate config is then assembled by dequantizing cached entries (seconds per layer), instead of
running GPTQ for the whole model per candidate (EvoPress-style). Missing entries are computed
lazily (RTN needs only the weight; GPTQ / AWQ need the Hessians from quant/hessian.py) and then
persisted. `warm()` precomputes a configured set of variants.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import quantizers as Q
from .hessian import hessian_path, load_hessian
from .spec import ModelSpec, checkpoint_tensors


class WeightReader:
    """Lazy tensor access to a safetensors checkpoint (keeps file handles open)."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.index = checkpoint_tensors(self.model_dir)
        self._handles = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> torch.Tensor:
        fn = self.index[name]["file"]
        with self._lock:
            h = self._handles.get(fn)
            if h is None:
                h = self._handles[fn] = safe_open(str(self.model_dir / fn), framework="pt")
            return h.get_tensor(name)


class QuantCache:
    # (2026-11-23) several caches (one per candidate slot GPU) may export at the same time: a missing entry is computed
    # once, by whichever cache takes its lock first; the others wait and load the stored entry. Keyed by path.
    _entry_locks: dict = {}
    _entry_locks_guard = threading.Lock()

    @classmethod
    def _entry_lock(cls, path: Path) -> threading.Lock:
        with cls._entry_locks_guard:
            lk = cls._entry_locks.get(str(path))
            if lk is None:
                lk = cls._entry_locks[str(path)] = threading.Lock()
            return lk

    def __init__(self, root: Path, spec: ModelSpec, model_dir: Path, *, hessian_dir: Path | None = None,
                 device: str = "cpu", damp: float = 0.01, lazy: bool = True, log=print):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.reader = WeightReader(model_dir)
        self.hessian_dir = Path(hessian_dir) if hessian_dir else None
        self.device = device
        self.damp = damp
        self.lazy = lazy
        self.log = log
        self.stats = {"hits": 0, "computed": 0, "seconds": 0.0}

    # ------------------------------------------------------------------------------ paths
    def path(self, module: str, quantizer: str, bits: int, gs: int) -> Path:
        return self.root / module / f"{quantizer}_b{bits}_g{gs}.safetensors"

    def has(self, module: str, quantizer: str, bits: int, gs: int) -> bool:
        return self.path(module, quantizer, bits, gs).exists()

    def can_compute(self, module: str, quantizer: str) -> bool:
        if quantizer == "rtn":
            return True
        m = self.spec.module(module)
        return self.hessian_dir is not None and hessian_path(self.hessian_dir, m.layer, m.input_key).exists()

    # ------------------------------------------------------------------------------ io
    def _save(self, path: Path, q: Q.QResult):
        path.parent.mkdir(parents=True, exist_ok=True)
        t = {"codes": Q.pack(q.codes.cpu(), q.bits), "scale": q.scale.cpu().contiguous(),
             "zero": Q.pack(q.zero.cpu(), q.bits),
             "shape": torch.tensor(list(q.codes.shape) + list(q.zero.shape), dtype=torch.int64)}
        if q.col_scale is not None:
            t["col_scale"] = q.col_scale.cpu().contiguous()
        tmp = path.with_suffix(f".tmp{os.getpid()}_{threading.get_ident()}")   # unique per thread too
        save_file(t, str(tmp), metadata={"bits": str(q.bits), "group_size": str(q.group_size)})
        os.replace(tmp, path)

    def _load(self, path: Path) -> Q.QResult:
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
            d = {k: f.get_tensor(k) for k in f.keys()}
        bits, gs = int(meta["bits"]), int(meta["group_size"])
        o, i, zo, zg = d["shape"].tolist()
        return Q.QResult(codes=Q.unpack(d["codes"], bits, o * i).reshape(o, i), scale=d["scale"],
                         zero=Q.unpack(d["zero"], bits, zo * zg).reshape(zo, zg), bits=bits, group_size=gs,
                         col_scale=d.get("col_scale"))

    # ------------------------------------------------------------------------------ main api
    def weight(self, module: str) -> torch.Tensor:
        return self.reader.get(module)

    def compute(self, module: str, quantizer: str, bits: int, gs: int) -> Q.QResult:
        W = self.reader.get(module).to(self.device)
        H, n = None, 0
        if quantizer != "rtn":
            if not self.can_compute(module, quantizer):
                raise RuntimeError(f"no Hessian for {module}: cannot compute {quantizer}")
            m = self.spec.module(module)
            H, n, _ = load_hessian(self.hessian_dir, m.layer, m.input_key, device=self.device)
        return Q.quantize(W, quantizer, bits, gs, H=H, n_tokens=n, damp=self.damp)

    def get(self, module: str, quantizer: str, bits: int, gs: int) -> Q.QResult:
        p = self.path(module, quantizer, bits, gs)
        if p.exists():
            self.stats["hits"] += 1
            return self._load(p)
        if not self.lazy:
            raise KeyError(f"cache miss {module} {quantizer} b{bits} g{gs} (lazy computation disabled)")
        with self._entry_lock(p):
            if p.exists():                 # computed by another cache (another slot) while this one waited
                self.stats["hits"] += 1
                return self._load(p)
            t0 = time.time()
            q = self.compute(module, quantizer, bits, gs)
            self._save(p, q)
            self.stats["computed"] += 1
            self.stats["seconds"] += time.time() - t0
            return q

    def dequant(self, module: str, setting: dict, dtype=torch.bfloat16) -> torch.Tensor:
        """Fake-quant weight for one module under a resolved setting (bits 16 = original)."""
        if setting["bits"] >= 16:
            return self.reader.get(module).to(dtype)
        q = self.get(module, setting["quantizer"], setting["bits"], setting["group_size"])
        return Q.dequant(q, dtype=dtype, device=self.device).cpu()

    def warm(self, variants: list[dict], *, groups: list[str] | None = None, modules: list[str] | None = None) -> dict:
        """Precompute variants [{"quantizer", "bits", "group_size"}] for every module (or those of
        the listed group names, or only the listed module names: one worker's share)."""
        t0 = time.time()
        n = 0
        keep = set(modules) if modules is not None else None
        for m in self.spec.modules:
            if groups and m.group not in groups:
                continue
            if keep is not None and m.name not in keep:
                continue
            for v in variants:
                if v["bits"] >= 16:
                    continue
                if not self.has(m.name, v["quantizer"], v["bits"], v["group_size"]):
                    if v["quantizer"] != "rtn" and not self.can_compute(m.name, v["quantizer"]):
                        continue
                    self.get(m.name, v["quantizer"], v["bits"], v["group_size"])
                    n += 1
        self.log(f"[cache] warmed {n} entries in {time.time() - t0:.1f}s ({self.root})")
        return {"computed": n, "seconds": time.time() - t0}

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*.safetensors"))
