"""Export a candidate as a FAKE-QUANT BF16 checkpoint loadable by vLLM / transformers.

Same architecture, config and tokenizer files (copied), same shard layout and tensor names; every
quantized linear weight (and the embedding / LM head when not "bf16") is replaced by its
dequantized value in the original dtype. No `quantization_config` is added, so vLLM loads it as a
normal BF16 model. `fake_quant.json` records the resolved config and its analytic packed size.
A real packed export (compressed-tensors) for speed measurement is future work (README).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import quantizers as Q
from .cache import QuantCache
from .schema import module_assignment
from .sizes import EMBED_GROUP, config_bytes, special_precisions
from .spec import ModelSpec, shard_files

SKIP_COPY_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".tmp")


# safetensors header dtype -> torch dtype, for the weights an export replaces (other dtypes: read the tensor)
_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def copy_model_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in Path(src).iterdir():
        if p.is_file() and not p.name.endswith(SKIP_COPY_SUFFIXES) and p.name != "fake_quant.json":
            shutil.copy2(p, dst / p.name)


def export_fake_quant(src_dir: Path, out_dir: Path, norm: dict, spec: ModelSpec, cache: QuantCache, *,
                      meta: dict | None = None, log=print) -> dict:
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    t0 = time.time()
    copy_model_files(src_dir, tmp_dir)
    assign = module_assignment(norm, spec)
    special = special_precisions(spec, norm["embed"], norm["lm_head"])
    idx = src_dir / "model.safetensors.index.json"
    n_q = 0
    for fn in shard_files(src_dir):
        tensors = {}
        with safe_open(str(src_dir / fn), framework="pt") as f:
            smeta = f.metadata() or {}
            for k in f.keys():
                if k in assign and assign[k]["bits"] < 16:
                    # replaced: dtype from the shard header, the stored weight is not read (2026-11-25)
                    dtype = _ST_DTYPES.get(f.get_slice(k).get_dtype())
                    if dtype is None:
                        dtype = f.get_tensor(k).dtype
                    t = cache.dequant(k, assign[k], dtype=dtype)
                    n_q += 1
                else:
                    t = f.get_tensor(k)
                    if k in special:
                        q = Q.rtn(t.to(cache.device), special[k], EMBED_GROUP)
                        t = Q.dequant(q, dtype=t.dtype).cpu()
                        n_q += 1
                tensors[k] = t.contiguous()
        save_file(tensors, str(tmp_dir / fn), metadata=dict(smeta, format=smeta.get("format", "pt")))
        del tensors
    if idx.exists():
        shutil.copy2(idx, tmp_dir / idx.name)
    info = {"config": norm, "bytes": config_bytes(norm, spec), "quantized_tensors": n_q, "source": str(src_dir),
            "seconds": time.time() - t0, **(meta or {})}
    with open(tmp_dir / "fake_quant.json", "w") as f:
        json.dump(info, f, indent=1)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)
    log(f"[export] {n_q} tensors fake-quantized -> {out_dir} ({info['seconds']:.1f}s, "
        f"{info['bytes']['gib']:.2f} GiB packed)")
    return info
