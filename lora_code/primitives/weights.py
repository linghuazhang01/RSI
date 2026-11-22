"""Lazy, read-only access to a model's safetensors shards (no full model load).

WeightStore(model_dir) lists the editable linear modules of the language model (see naming.py), reads
single weight matrices on demand (fp32, on any device) and computes / caches top-k SVDs.
Tensor catalogue (2026-11-21, tensor_compute): tensor_names() lists EVERY language-model tensor by its checkpoint key
minus the prefix (norms, linear_attn.dt_bias / A_log / conv1d, the linear weights as '<module>.weight', embed_tokens,
norm, lm_head); modules() stays linear-only for the LoRA tools. Weight-edit names (edit_name): a linear weight is
addressed by its module short name, any other tensor by its tensor name.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path

from .naming import LEAVES, is_edit_excluded, linear_short, parse_key, parse_short, parse_tensor_key, tensor_layer


class WeightStore:
    def __init__(self, model_dir, *, device: str = "cpu", cache_dir=None, max_cached: int = 8):
        import torch  # noqa: F401  (fail early with a clear error when torch is missing)

        self.model_dir = Path(model_dir)
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._lru: OrderedDict = OrderedDict()
        self._lru_lock = threading.Lock()     # tensor_compute reads outside the workspace lock (2026-11-21)
        self._max_cached = max_cached
        idx = self.model_dir / "model.safetensors.index.json"
        if idx.exists():
            self.weight_map = json.load(open(idx))["weight_map"]
        else:
            single = self.model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no safetensors checkpoint in {self.model_dir}")
            from safetensors import safe_open
            with safe_open(str(single), "pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        cfg = json.load(open(self.model_dir / "config.json"))
        self.config = cfg
        self.text_config = cfg.get("text_config", cfg)
        self.layer_types = self.text_config.get("layer_types")
        self.prefix = None
        self._key = {}
        for key in self.weight_map:
            p = parse_key(key)
            if p is None:
                continue
            pre, short = p
            if self.prefix is None:
                self.prefix = pre
            if pre != self.prefix:
                continue
            self._key[short] = key
        if not self._key:
            raise ValueError(f"no editable language-model linear weights found in {self.model_dir}")
        self._tensors = {}                      # tensor name -> checkpoint key (every language-model tensor)
        for key in self.weight_map:
            p = parse_tensor_key(key, self.prefix)
            if p is not None:
                self._tensors[p[1]] = key
        self._shapes = {}
        self._tmeta = None                      # tensor name -> (shape, dtype), read lazily from the shard headers
        self.fingerprint = self._fingerprint()

    # ------------------------------------------------------------------ listing
    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        for f in sorted(set(self.weight_map.values())):
            st = (self.model_dir / f).stat()
            h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
        return h.hexdigest()[:16]

    def modules(self) -> list[str]:
        return sorted(self._key, key=lambda s: (parse_short(s)[0], s))

    def has(self, short: str) -> bool:
        return short in self._key

    def key(self, short: str) -> str:
        if short not in self._key:
            raise KeyError(f"module {short!r} not in model (example: {self.modules()[0]!r})")
        return self._key[short]

    def full_name(self, short: str) -> str:
        """nn.Module path inside the HF model (checkpoint key without '.weight')."""
        return self.key(short)[: -len(".weight")]

    def shape(self, short: str) -> tuple[int, int]:
        if short not in self._shapes:
            from safetensors import safe_open
            key = self.key(short)
            with safe_open(str(self.model_dir / self.weight_map[key]), "pt") as f:
                self._shapes[short] = tuple(f.get_slice(key).get_shape())
        return self._shapes[short]

    def info(self, short: str) -> dict:
        layer, block, leaf = parse_short(short)
        lt = self.layer_types[layer] if self.layer_types and layer < len(self.layer_types) else None
        out_f, in_f = self.shape(short)
        return {"module": short, "layer": layer, "block": block, "leaf": leaf, "group": LEAVES[leaf][1],
                "layer_type": lt, "out": out_f, "in": in_f}

    @property
    def num_layers(self) -> int:
        return 1 + max(parse_short(s)[0] for s in self._key)

    # ------------------------------------------------------------------ tensor catalogue (2026-11-21)
    def tensor_names(self) -> list[str]:
        return sorted(self._tensors, key=_tensor_sort)

    def has_tensor(self, name: str) -> bool:
        return self._canon(name) in self._tensors

    def _canon(self, name: str) -> str:
        """A linear module short name stands for its '.weight' tensor."""
        return name + ".weight" if name in self._key else name

    def tensor_key(self, name: str) -> str:
        n = self._canon(name)
        if n not in self._tensors:
            raise KeyError(f"tensor {name!r} not in model (see api.tensors(); example: {self.tensor_names()[0]!r})")
        return self._tensors[n]

    def edit_name(self, name: str) -> str:
        """Name of a tensor inside a WeightEdit: the module short name for a linear weight, else the tensor name."""
        n = self._canon(name)
        self.tensor_key(n)
        return linear_short(n) if linear_short(n) in self._key else n

    def editable(self, name: str) -> bool:
        return self.has_tensor(name) and not is_edit_excluded(self._canon(name))

    def _meta(self) -> dict:
        if self._tmeta is None:
            from safetensors import safe_open
            meta, by_shard = {}, {}
            for n, k in self._tensors.items():
                by_shard.setdefault(self.weight_map[k], []).append((n, k))
            for shard, items in by_shard.items():
                with safe_open(str(self.model_dir / shard), "pt") as f:
                    for n, k in items:
                        sl = f.get_slice(k)
                        meta[n] = (tuple(sl.get_shape()), _DTYPES.get(sl.get_dtype(), str(sl.get_dtype()).lower()))
            self._tmeta = meta
        return self._tmeta

    def tensor_shape(self, name: str) -> tuple:
        self.tensor_key(name)
        return self._meta()[self._canon(name)][0]

    def tensor_dtype(self, name: str) -> str:
        self.tensor_key(name)
        return self._meta()[self._canon(name)][1]

    def tensor_info(self, name: str) -> dict:
        n = self._canon(name)
        self.tensor_key(n)
        shape, dtype = self._meta()[n]
        return {"name": n, "shape": list(shape), "dtype": dtype, "editable": not is_edit_excluded(n),
                "layer": tensor_layer(n)}

    def tensor_source(self, name: str) -> tuple[str, str]:
        """(resolved shard path, checkpoint key) for a reader outside this process (the tensor_compute sandbox;
        merged models hard-link their shards, so the real path is what the sandbox must be allowed to read)."""
        key = self.tensor_key(name)
        return str((self.model_dir / self.weight_map[key]).resolve()), key

    def tensor(self, name: str, *, device="cpu", dtype=None):
        """fp32 copy of any language-model tensor (a linear weight goes through weight(), with its cache)."""
        import torch
        from safetensors import safe_open
        n = self._canon(name)
        short = linear_short(n)
        if short in self._key:
            return self.weight(short, device=device, dtype=dtype).clone()
        key = self.tensor_key(n)
        with safe_open(str(self.model_dir / self.weight_map[key]), "pt", device="cpu") as f:
            t = f.get_tensor(key)
        return t.to(device=device, dtype=dtype or torch.float32)

    # ------------------------------------------------------------------ tensors
    def weight(self, short: str, *, device=None, dtype=None):
        import torch
        from safetensors import safe_open

        dtype = dtype or torch.float32
        device = device or self.device
        ck = (short, str(device), str(dtype))
        with self._lru_lock:
            w = self._lru.get(ck)
            if w is not None:
                self._lru.move_to_end(ck)
                return w
        key = self.key(short)
        with safe_open(str(self.model_dir / self.weight_map[key]), "pt", device="cpu") as f:
            w = f.get_tensor(key)
        w = w.to(device=device, dtype=dtype)
        with self._lru_lock:
            self._lru[ck] = w
            self._lru.move_to_end(ck)
            while len(self._lru) > self._max_cached:
                self._lru.popitem(last=False)
        return w

    def norm(self, short: str) -> float:
        return float(self.weight(short).norm())

    def svd(self, short: str, k: int = 16):
        """Top-k singular triplets of W (U [out,k], S [k], V [in,k]); exact for small matrices,
        randomized (torch.svd_lowrank, 6 power iterations) otherwise; cached on disk per model."""
        cache = None if self.cache_dir is None else self.cache_dir / f"svd_{self.fingerprint}" / f"{short}.k{k}.pt"
        if cache is not None and cache.exists():
            return _top_svd(None, k, cache)
        return _top_svd(self.weight(short), k, cache)


class OverlayStore:
    """merge.mode adapter (2026-11-22): the CURRENT weights W_g = W_0 + B_C A_C of a model served as M_0 + the
    cumulative adapter C, computed in fp32 on read (nothing is written or rounded to bf16). Duck-types WeightStore:
    weight / tensor / norm / svd of a module C touches see W_0 + ΔW_C; every other tensor and all metadata
    (model_dir, prefix, shapes, tensor_source) come from the base store. tensor_source of an overlaid tensor still
    points at W_0 on disk: readers outside the process (tensor_compute) must take is_overlaid() tensors from tensor().
    SVDs of overlaid modules are cached per module factor digest (cache_dir/svd_<base fp>_<digest>/)."""

    def __init__(self, base: "WeightStore", lora):
        self.base = base
        self.lora = lora
        self._norms = {}                         # W_g is not cached (the base store caches W_0); its norms are
        self._mdig = {}
        self.fingerprint = f"{base.fingerprint}+{lora.digest()}"

    def __getattr__(self, name):                 # metadata and untouched tensors: the base store
        if name in ("base", "lora", "__setstate__", "__getstate__"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def is_overlaid(self, name: str) -> bool:
        short = linear_short(self.base._canon(name))
        return short in self.lora.factors

    def _digest(self, m: str) -> str:
        if m not in self._mdig:
            A, B = self.lora.factors[m]
            h = hashlib.sha256()
            h.update(A.contiguous().float().cpu().numpy().tobytes())
            h.update(B.contiguous().float().cpu().numpy().tobytes())
            self._mdig[m] = h.hexdigest()[:16]
        return self._mdig[m]

    def weight(self, short: str, *, device=None, dtype=None):
        import torch
        if short not in self.lora.factors:
            return self.base.weight(short, device=device, dtype=dtype)
        device = device or self.base.device
        w0 = self.base.weight(short, device=device, dtype=torch.float32)
        A, B = self.lora.factors[short]
        return (w0 + B.to(device=device, dtype=torch.float32) @ A.to(device=device, dtype=torch.float32)).to(
            dtype or torch.float32)

    def tensor(self, name: str, *, device="cpu", dtype=None):
        short = linear_short(self.base._canon(name))
        if short in self.lora.factors:
            return self.weight(short, device=device, dtype=dtype).clone()
        return self.base.tensor(name, device=device, dtype=dtype)

    def norm(self, short: str) -> float:
        if short not in self.lora.factors:
            return self.base.norm(short)
        if short not in self._norms:             # check_lora reads it for every module of every candidate
            self._norms[short] = float(self.weight(short).norm())
        return self._norms[short]

    def svd(self, short: str, k: int = 16):
        if short not in self.lora.factors:
            return self.base.svd(short, k)
        return _top_svd(self.weight(short), k, None if self.base.cache_dir is None else
                        self.base.cache_dir / f"svd_{self.base.fingerprint}_{self._digest(short)}" / f"{short}.k{k}.pt")


def _top_svd(w, k: int, cache):
    """Top-k singular triplets (U [out,k], S [k], V [in,k]) of w, cached at `cache` (a .pt path) when given."""
    import torch
    if cache is not None and cache.exists():
        d = torch.load(cache, map_location="cpu", weights_only=True)
        return d["U"], d["S"], d["V"]
    m = min(w.shape)
    if m <= 2048 or k >= m // 4:
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        U, S, V = U[:, :k], S[:k], Vh[:k].T
    else:
        torch.manual_seed(0)
        U, S, V = torch.svd_lowrank(w, q=min(m, k + 16), niter=6)
        U, S, V = U[:, :k], S[:k], V[:, :k]
    U, S, V = U.float().cpu().contiguous(), S.float().cpu().contiguous(), V.float().cpu().contiguous()
    if cache is not None:
        # several tool devices share this cache (2026-11-22): write a private file, then rename it into place
        import os
        import uuid
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f".{cache.name}.{uuid.uuid4().hex}")
        torch.save({"U": U, "S": S, "V": V}, tmp)
        os.replace(tmp, cache)
    return U, S, V


_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64", "I32": "int32",
           "U8": "uint8", "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}


def _tensor_sort(name: str):
    layer = tensor_layer(name)
    return (-1 if name.startswith("embed_tokens") else (10 ** 6 if layer is None else layer), name)
