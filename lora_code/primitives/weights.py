"""Lazy, read-only access to a model's safetensors shards (no full model load).

WeightStore(model_dir) lists the editable linear modules of the language model (see naming.py), reads
single weight matrices on demand (fp32, on any device) and computes / caches top-k SVDs.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

from .naming import LEAVES, parse_key, parse_short


class WeightStore:
    def __init__(self, model_dir, *, device: str = "cpu", cache_dir=None, max_cached: int = 8):
        import torch  # noqa: F401  (fail early with a clear error when torch is missing)

        self.model_dir = Path(model_dir)
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._lru: OrderedDict = OrderedDict()
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
        self._shapes = {}
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

    # ------------------------------------------------------------------ tensors
    def weight(self, short: str, *, device=None, dtype=None):
        import torch
        from safetensors import safe_open

        dtype = dtype or torch.float32
        device = device or self.device
        ck = (short, str(device), str(dtype))
        if ck in self._lru:
            self._lru.move_to_end(ck)
            return self._lru[ck]
        key = self.key(short)
        with safe_open(str(self.model_dir / self.weight_map[key]), "pt", device="cpu") as f:
            w = f.get_tensor(key)
        w = w.to(device=device, dtype=dtype)
        self._lru[ck] = w
        while len(self._lru) > self._max_cached:
            self._lru.popitem(last=False)
        return w

    def norm(self, short: str) -> float:
        return float(self.weight(short).norm())

    def svd(self, short: str, k: int = 16):
        """Top-k singular triplets of W (U [out,k], S [k], V [in,k]); exact for small matrices,
        randomized (torch.svd_lowrank, 6 power iterations) otherwise; cached on disk per model."""
        import torch

        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / f"svd_{self.fingerprint}" / f"{short}.k{k}.pt"
            if cache.exists():
                d = torch.load(cache, map_location="cpu", weights_only=True)
                return d["U"], d["S"], d["V"]
        w = self.weight(short)
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
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"U": U, "S": S, "V": V}, cache)
        return U, S, V
