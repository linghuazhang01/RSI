"""Layer-level GPU check of the real packed format (2026-11-28): which Humming GEMMs compute the wrong numbers?

packed_check.py on the H20 (2026-11-28) gave KL(M_0 || dequant) 0.0136 but KL(dequant || packed) 4.03, although the
packed tensors decode (on the CPU) to exactly the dequant weights. This script narrows that down without serving a
model: each case is one vLLM linear module (fused shards, e.g. gate_up_proj = gate + up) at one bit width / group
size, run on one GPU for several batch sizes M, against y_ref = x @ W'^T with W' = bf16((code - zero) * scale), the
weight both exports mean. Per case:
    fmt     humming's own GPU unpack + dequant (humming.utils.weight.dequantize_weight) of the tensors after
            GPTQWeightSchema.convert_humming, compared bit for bit with W': does humming READ our tensors as meant?
    vllm    vLLM 0.24's HummingLinearMethod (the per-layer config resolved by HummingConfig from the
            quantization_config, process_weights_after_loading, apply): the code path of `vllm serve
            --quantization humming`. Relative error vs y_ref, next to a plain BF16 cuBLAS GEMM of the same W'.
    humming the same steps called on humming directly (no vLLM): tells a vLLM integration problem from a kernel one.
    native  humming's own quantizer (uint<b>, zero points, bf16 scales) on the same weights, its own GEMM vs its own
            dequantized weight: if this fails too, the kernel is broken for this shape whatever the format.
    loader  (opt-in, --impls ...,loader) vLLM's real layer classes (QKVParallelLinear / MergedColumnParallelLinear /
            RowParallelLinear, TP disabled) filled shard by shard through their weight_loader_v2, as the model
            loader does (checkpoint shard ids "q"/"k"/"v", (0, 1, 2) for in_proj_qkv, ...): also covers vLLM's shard
            offsets of the packed qzeros. The other impls concatenate the shards themselves.
A case passes when the output is finite and rel_err <= 2 x rel_err(cuBLAS) + 2e-3 (a wrong bit layout, zero point or
tile config gives rel_err ~ 0.1 to 1). Every case runs in a worker process: an illegal memory access kills the CUDA
context, so the parent records the crash (case and M) and restarts a worker at the next case.

Sources of cases:
    synthetic (default)  random weights (RTN, bf16 scales) in the shapes of the model's fused modules (--model: read
                         from the checkpoint; else the Qwen3.8-27B shapes), every bit width x group size the packed
                         format can express (--bits, --group-sizes, --modules)
    --ckpt DIR           a written packed checkpoint (packed_check's <out>/packed): one module per distinct
                         (module kind, bits, group size, shard sizes) of its config, with its own quantization_config
                         and vLLM prefix; --dequant DIR (packed_check's <out>/dequant) also checks that the packed
                         tensors decode to that checkpoint's BF16 weights

    python -I edit_code/packed_kernel_check.py --out <dir> --gpu 0 [--ckpt <check>/packed --dequant <check>/dequant]
        [--model <M_0>] [--quick] [--impls vllm,humming,native,loader] [--m 1,16,256,2048,8192] [--blocking]

Writes <out>/kernel_check.json (every case and M, worst first) and <out>/results.jsonl (raw events);
exit code 0 = every case passed. Needs the eval environment (vLLM 0.24.0 + humming-kernels 0.1.6) and one free GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# vLLM fused module -> its checkpoint projections in vLLM's shard order (vllm 0.24.0 models/qwen3_5.py
# stacked_params_mapping, gdn/qwen_gdn_linear_attn.py); the order only permutes output columns
FUSED_ORDER = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "o_proj": ["o_proj"],
               "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"],
               "out_proj": ["out_proj"], "gate_up_proj": ["gate_proj", "up_proj"], "down_proj": ["down_proj"]}
BLOCK = {"qkv_proj": "self_attn", "o_proj": "self_attn", "in_proj_qkvz": "linear_attn", "in_proj_ba": "linear_attn",
         "out_proj": "linear_attn", "gate_up_proj": "mlp", "down_proj": "mlp"}
DEFAULT_M = [1, 3, 16, 64, 256, 1024, 2048, 4096, 8192]
QUICK_M = [1, 16, 256, 2048, 8192]
PAD_N, PAD_K = 256, 128                      # vLLM 0.24 HummingLinearMethod.process_weights_after_loading
CUDA_DEAD = re.compile(r"CUDA error|illegal memory access|cudaError|CUBLAS_STATUS|device-side assert|"
                       r"unspecified launch failure|misaligned address", re.I)
_QRX = re.compile(r"^(?P<base>(?:.+\.)?layers\.(?P<layer>\d+)\.(?P<block>self_attn|linear_attn|mlp)"
                  r"\.(?P<proj>[a-z_]+))\.qweight$")


# ------------------------------------------------------------------------------------------------ cases
def expressible(shards: list[int], k: int, bits: int, gs: int) -> str | None:
    """Why the packed format cannot hold this module (None: it can): schema.setting_packable's shape rules."""
    if k % gs:
        return f"K={k} is not a multiple of the group size {gs}"
    if (k * bits) % 32:
        return f"K*bits={k * bits} is not a multiple of 32"
    for n in shards:
        if (n * bits) % 32:
            return f"a shard of N={n} at {bits} bit does not fill whole int32 words (qzeros)"
    return None


def model_shapes(model: str | None) -> dict[str, dict]:
    """fused vLLM module -> {"shards": [N per projection], "k", "layer", "prefix"} from the first full-attention and
    first linear-attention layer of the model (--model) or of the Qwen3.8-27B-like synthetic spec."""
    from edit_code.quant.spec import QWEN27B_LIKE, spec_from_checkpoint, synthetic_spec
    spec = spec_from_checkpoint(Path(model)) if model else synthetic_spec()
    head_dim = text_head_dim(Path(model)) if model else QWEN27B_LIKE["head_dim"]
    first = {}
    for kind in ("full_attention", "linear_attention"):
        first[kind] = next(i for i, t in enumerate(spec.layer_types) if t == kind)
    out: dict[str, dict] = {}
    for fused, projs in FUSED_ORDER.items():
        kind = "full_attention" if BLOCK[fused] == "self_attn" else "linear_attention"
        layer = first[kind]
        mods = {m.proj: m for m in spec.modules if m.layer == layer}
        if not all(p in mods for p in projs):
            continue
        ks = {int(mods[p].shape[1]) for p in projs}
        if len(ks) != 1:
            raise ValueError(f"{fused}: shards with different input sizes {ks}")
        out[fused] = {"shards": [int(mods[p].shape[0]) for p in projs], "k": ks.pop(), "layer": layer,
                      "prefix": f"language_model.model.layers.{layer}.{BLOCK[fused]}.{fused}", "head_dim": head_dim}
    return out


def text_head_dim(model_dir: Path) -> int | None:
    """head_dim of the decoder's full attention (config.json, text_config for Qwen3.5); the loader impl needs it."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    t = cfg.get("text_config") or cfg
    hd = t.get("head_dim")
    if hd is None and t.get("hidden_size") and t.get("num_attention_heads"):
        hd = t["hidden_size"] // t["num_attention_heads"]
    return int(hd) if hd else None


def synthetic_cases(a) -> list[dict]:
    shapes = model_shapes(a.model)
    mods = a.modules.split(",") if a.modules else list(shapes)
    cases, skipped = [], []
    for fused in mods:
        if fused not in shapes:
            raise SystemExit(f"unknown module {fused!r}; choose from {', '.join(shapes)}")
        sh = shapes[fused]
        for bits in a.bits:
            for gs in a.group_sizes:
                why = expressible(sh["shards"], sh["k"], bits, gs)
                if why:
                    skipped.append({"module": fused, "bits": bits, "group_size": gs, "why": why})
                    continue
                qcfg = {"quant_method": "gptq", "bits": bits, "group_size": gs, "sym": False, "desc_act": False,
                        "checkpoint_format": "gptq_v2"}
                cases.append({"source": "synthetic", "module": fused, "prefix": sh["prefix"], "shards": sh["shards"],
                              "k": sh["k"], "bits": bits, "group_size": gs, "qcfg": qcfg, "head_dim": sh["head_dim"],
                              "seed": 1000 * bits + gs + 17 * len(cases)})
    return cases, skipped


def ckpt_cases(a) -> tuple[list[dict], list[dict]]:
    """One case per distinct (fused module, bits, group size, shard sizes) of a written packed checkpoint."""
    from edit_code.quant.export import resolve_humming
    from edit_code.quant.spec import checkpoint_tensors, vllm_module_prefix
    ck = Path(a.ckpt).resolve()
    qcfg = json.loads((ck / "config.json").read_text()).get("quantization_config")
    if not qcfg:
        raise SystemExit(f"{ck}/config.json has no quantization_config (not a packed checkpoint)")
    have = checkpoint_tensors(ck)
    fused: dict[str, dict] = {}
    for name, t in have.items():
        m = _QRX.match(name)
        if not m or "visual" in name or name.startswith("mtp."):
            continue
        base, proj = m.group("base"), m.group("proj")
        prefix = vllm_module_prefix(base)
        f = fused.setdefault(prefix, {"layer": int(m.group("layer")), "module": prefix.rsplit(".", 1)[1],
                                      "parts": {}})
        f["parts"][proj] = {"base": base, "qweight": t, "qzeros": have.get(base + ".qzeros"),
                            "scales": have.get(base + ".scales")}
    cases, skipped, seen = [], [], {}
    for prefix, f in sorted(fused.items(), key=lambda kv: (kv[1]["layer"], kv[0])):
        order = FUSED_ORDER.get(f["module"])
        if order is None:
            skipped.append({"prefix": prefix, "why": f"unknown fused module {f['module']}"})
            continue
        missing = [p for p in order if p not in f["parts"]]
        if missing:
            skipped.append({"prefix": prefix, "why": f"shards {missing} are not quantized (vLLM cannot load this)"})
            continue
        lack = [f["parts"][p]["base"] + "." + s for p in order for s in ("qzeros", "scales")
                if not f["parts"][p][s]]
        if lack:
            skipped.append({"prefix": prefix, "why": f"missing tensors {lack}"})
            continue
        got = resolve_humming(qcfg, prefix)
        if got is None:
            skipped.append({"prefix": prefix, "why": "vLLM would keep it BF16 (quantization_config) but it has qweight"})
            continue
        b, gs = got["bits"], got["group_size"]
        qws = [f["parts"][p]["qweight"]["shape"] for p in order]
        ks = {int(s[0]) * 32 // b for s in qws}
        if len(ks) != 1 or any(int(s[0]) * 32 % b for s in qws):
            skipped.append({"prefix": prefix, "why": f"qweight shapes {qws} do not fit {b} bit"})
            continue
        k = ks.pop()
        shards = [int(s[1]) for s in qws]
        key = (f["module"], b, gs, tuple(shards), k)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > a.per_kind:
            continue
        cases.append({"source": "ckpt", "module": f["module"], "prefix": prefix, "layer": f["layer"],
                      "shards": shards, "k": k, "bits": b, "group_size": gs, "qcfg": qcfg, "ckpt": str(ck),
                      "head_dim": text_head_dim(ck),
                      "dequant": str(Path(a.dequant).resolve()) if a.dequant else None,
                      "bases": [f["parts"][p]["base"] for p in order],
                      "files": {f["parts"][p]["base"]: {s: f["parts"][p][s]["file"] for s in
                                                        ("qweight", "qzeros", "scales") if f["parts"][p][s]}
                                for p in order},
                      "seed": 7 + len(cases)})
    return cases, skipped


# ------------------------------------------------------------------------------------------------ worker
def _load(ckpt: str, file: str, name: str):
    from safetensors import safe_open
    with safe_open(str(Path(ckpt) / file), framework="pt") as f:
        return f.get_tensor(name)


def case_tensors(case: dict, dev):
    """(gptq tensors of the fused module, W' bf16 [N, K] on dev, extra checks, per-shard tensors): concatenated along
    N in shard order, the way vLLM's loader fills the fused params (qweight / scales by output column, qzeros by
    output word; every shard's N * bits is a multiple of 32, so the shards start on word boundaries)."""
    import torch
    from edit_code.quant import quantizers as Q
    from edit_code.quant.export import decode_rows, gptq_tensors
    b, gs, k = case["bits"], case["group_size"], case["k"]
    parts, refs, extra = [], [], {}
    if case["source"] == "synthetic":
        g = torch.Generator(device=dev).manual_seed(case["seed"])
        for n in case["shards"]:
            w = torch.randn(n, k, generator=g, device=dev) * 0.02
            w = w + 0.01 * torch.randn(n, k // gs, generator=g, device=dev).repeat_interleave(gs, 1)
            w[: min(8, n), :gs] = w[: min(8, n), :gs].abs() + 1e-3          # groups with zero point 0 ...
            w[min(8, n): min(16, n), :gs] = -w[min(8, n): min(16, n), :gs].abs() - 1e-3   # ... and 2^b - 1
            q = Q.rtn(w, b, gs, scale_dtype=torch.bfloat16)
            parts.append(gptq_tensors(q.codes, q.zero, q.scale, b))
            refs.append(Q.dequant(q, dtype=torch.bfloat16, device=dev))
            del w, q
    else:
        bad = 0
        for base in case["bases"]:
            fl = case["files"][base]
            t = {s: _load(case["ckpt"], fl[s], f"{base}.{s}").to(dev) for s in ("qweight", "qzeros", "scales")}
            n = t["qweight"].shape[1]
            codes = decode_rows(t["qweight"].t().contiguous(), b, k)                       # [N, K]
            zero = decode_rows(t["qzeros"], b, n).t()                                      # [N, K/g]
            col = torch.arange(k, device=dev) // gs
            w = ((codes.float() - zero.float()[:, col]) * t["scales"].t().float()[:, col]).to(torch.bfloat16)
            parts.append(t)
            refs.append(w)
            if case.get("dequant"):
                from edit_code.quant.spec import checkpoint_tensors
                dq = Path(case["dequant"])
                meta = checkpoint_tensors(dq).get(base + ".weight")
                if meta is None:
                    extra.setdefault("export_missing", []).append(base)
                else:
                    ref = _load(str(dq), meta["file"], base + ".weight").to(dev)
                    bad += int((ref != w).sum())
            del codes, zero
        if case.get("dequant"):
            extra["export_mismatch_elems"] = bad
    gptq = {s: torch.cat([p[s] for p in parts], dim=1) for s in ("qweight", "qzeros", "scales")}
    return gptq, torch.cat(refs, dim=0), extra, parts


def fmt_check(case, gptq, wref) -> dict:
    """Humming's reading of our tensors: GPTQWeightSchema.convert_humming + humming's GPU dequantize_weight."""
    import torch
    from humming import dtypes
    from humming.schema import BaseWeightSchema
    from humming.utils.weight import dequantize_weight
    ws = BaseWeightSchema.from_config({"quant_method": "gptq", "bits": case["bits"], "group_size": case["group_size"],
                                       "sym": False, "desc_act": False})
    hs, t = ws.convert_humming(tensors={k: v.clone() for k, v in gptq.items()}, shape_n_stacks=case["shards"],
                               shape_k_stacks=[case["k"]], param_dtype=torch.bfloat16)
    # humming's pack / unpack helpers launch nelement // 1024 full blocks (kernel/unpack_weight.py:32) and leave the
    # tail of torch.empty unwritten: pad N to a multiple of 1024 for this reader only (in_proj_ba: N = 96), then slice
    # back. The serving path pads N to 256 itself before it unpacks (layer.py check_and_pad_tensors).
    b, n = case["bits"], wref.shape[0]
    pad = (-n) % 1024
    F = torch.nn.functional
    w = dequantize_weight(F.pad(t["weight"], (0, 0, 0, pad)), F.pad(t["weight_scale"], (0, 0, 0, pad)),
                          F.pad(t["zero_point"], (0, 0, 0, pad * b // 32)), None,
                          dtypes.DataType.from_str(f"uint{b}"), packed=True)
    w = w.to(torch.bfloat16)[:n].reshape(wref.shape)
    diff = (w != wref)
    out = {"mismatch_elems": int(diff.sum()), "elems": int(wref.numel())}
    if out["mismatch_elems"]:
        rows = diff.any(dim=1).nonzero().flatten()
        out["first_bad_rows"] = rows[:8].tolist()
        out["bad_row_count"] = int(rows.numel())
        out["max_abs"] = float((w.float() - wref.float()).abs().max())
    return out


def _layer_attrs(layer, case, dev):
    import torch
    layer.is_fallback = False
    layer.param_dtype = torch.bfloat16
    layer.input_size = case["k"]
    layer.output_size = sum(case["shards"])
    layer.input_size_per_partition = case["k"]
    layer.output_partition_sizes = list(case["shards"])
    layer.output_partition_sizes_sum = sum(case["shards"])
    layer.has_bias = False
    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device=dev))


def build_vllm(case, gptq, dev):
    """vLLM 0.24's own code path: HummingConfig(full quantization_config).get_quant_config_for_layer(prefix) ->
    HummingLinearMethod.process_weights_after_loading -> apply."""
    import torch
    from vllm.model_executor.layers.quantization.humming import HummingConfig, HummingLinearMethod
    lq = HummingConfig(full_config=case["qcfg"]).get_quant_config_for_layer(case["prefix"], "linear")
    if lq is None:
        raise RuntimeError(f"vLLM keeps {case['prefix']} BF16 under this quantization_config")
    ws = lq.weight_schema
    got = (getattr(ws, "bits", None), getattr(ws, "group_size", None), getattr(ws, "sym", None))
    if got != (case["bits"], case["group_size"], False):
        raise RuntimeError(f"vLLM resolves {case['prefix']} to bits/group/sym {got}, the case expects "
                           f"{(case['bits'], case['group_size'], False)}")
    meth = HummingLinearMethod(lq)
    layer = torch.nn.Module()
    _layer_attrs(layer, case, dev)
    for k, v in gptq.items():
        setattr(layer, k, torch.nn.Parameter(v.clone(), requires_grad=False))
    layer.g_idx = torch.nn.Parameter((torch.arange(case["k"], device=dev) // case["group_size"]).to(torch.int32),
                                     requires_grad=False)
    meth.process_weights_after_loading(layer)
    return (lambda x: meth.apply(layer, x)), layer


def build_humming(case, gptq, dev):
    """The same steps on humming directly (no vLLM classes), with vLLM's padding and compute config."""
    import torch
    from humming.layer import HummingMethod
    from humming.schema import BaseWeightSchema, HummingInputSchema
    ws = BaseWeightSchema.from_config({"quant_method": "gptq", "bits": case["bits"],
                                       "group_size": case["group_size"], "sym": False, "desc_act": False})
    hs, t = ws.convert_humming(tensors={k: v.clone() for k, v in gptq.items()}, shape_n_stacks=case["shards"],
                               shape_k_stacks=[case["k"]], param_dtype=torch.bfloat16)
    layer = torch.nn.Module()
    _layer_attrs(layer, case, dev)
    for k, v in t.items():
        setattr(layer, k, torch.nn.Parameter(v, requires_grad=False))
    HummingMethod.prepare_layer_meta(layer=layer, shape_n=sum(case["shards"]), shape_k=case["k"], weight_schema=hs,
                                     input_schema=HummingInputSchema(), pad_n_to_multiple=PAD_N,
                                     pad_k_to_multiple=PAD_K, has_bias=False, torch_dtype=torch.bfloat16)
    HummingMethod.transform_humming_layer(layer)
    cc = json.dumps({"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "dense"})
    return (lambda x: HummingMethod.forward_layer(layer=layer, inputs=x, compute_config=cc)), layer


def build_native(case, wref, dev):
    """Humming's own uint<b> + zero point + bf16 group scale quantization of W', its GEMM and its own reference."""
    import torch
    from humming import dtypes
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.utils.weight import dequantize_weight, quantize_weight
    b = case["bits"]
    bdt = dtypes.DataType.from_str(f"uint{b}")
    n, k = wref.shape
    np_ = -(-n // 1024) * 1024        # humming's pack_weight asserts nelement % 1024 == 0 (ops/weight.py:42): pad N
    wp = wref if np_ == n else wref.repeat(-(-np_ // n), 1)[:np_].contiguous()   # with real rows (no zero-range groups)
    w, s, z, _ = quantize_weight(wp, dtype=bdt, scale_dtype=dtypes.bfloat16, group_size=case["group_size"],
                                 has_zero_point=True, pack=True)
    ref = dequantize_weight(w, s, z, None, bdt, packed=True).to(torch.bfloat16).reshape(np_, k)[:n].contiguous()
    w, s, z = w[:n].contiguous(), s[:n].contiguous(), z[: n * b // 32].contiguous()   # z: [N*b/32, G], N*b % 32 == 0
    hs = HummingWeightSchema(b_dtype=bdt, weight_scale_group_size=case["group_size"], has_zero_point=True)
    layer = torch.nn.Module()
    _layer_attrs(layer, case, dev)
    for k, v in (("weight", w), ("weight_scale", s), ("zero_point", z)):
        setattr(layer, k, torch.nn.Parameter(v, requires_grad=False))
    HummingMethod.prepare_layer_meta(layer=layer, shape_n=sum(case["shards"]), shape_k=case["k"], weight_schema=hs,
                                     input_schema=HummingInputSchema(), pad_n_to_multiple=PAD_N,
                                     pad_k_to_multiple=PAD_K, has_bias=False, torch_dtype=torch.bfloat16)
    HummingMethod.transform_humming_layer(layer)
    cc = json.dumps({"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "dense"})
    return (lambda x: HummingMethod.forward_layer(layer=layer, inputs=x, compute_config=cc)), layer, ref


def build_loader(case, parts, dev, gptq=None):
    """vLLM's real layer class for this module, TP disabled, params created on the GPU as in model init, every
    checkpoint shard loaded through param.weight_loader (HummingLinearMethod wraps weight_loader_v2) with the shard
    id vLLM's Qwen3.5 loader uses, then process_weights_after_loading and apply."""
    import contextlib
    import torch
    import vllm.model_executor.parameter as vparam
    from vllm.model_executor.layers import linear as vlin
    from vllm.model_executor.layers.quantization.humming import HummingConfig
    vparam.get_tensor_model_parallel_rank = lambda: 0           # one process, no distributed init: TP rank 0 of 1
    vparam.get_tensor_model_parallel_world_size = lambda: 1
    ctx = contextlib.nullcontext()
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        ctx = set_current_vllm_config(VllmConfig())
    except Exception:  # noqa: BLE001 - older / newer vLLM: try without a config context
        pass
    qc = HummingConfig(full_config=case["qcfg"])
    mod, k, shards, pre = case["module"], case["k"], case["shards"], case["prefix"]
    common = dict(bias=False, params_dtype=torch.bfloat16, quant_config=qc, prefix=pre, disable_tp=True)
    with ctx, torch.device(dev):
        if mod == "qkv_proj":
            hd = case.get("head_dim")
            if not hd or shards[0] % hd or shards[1] % hd or shards[1] != shards[2]:
                raise RuntimeError(f"qkv shards {shards} do not fit head_dim {hd}")
            layer = vlin.QKVParallelLinear(k, hd, shards[0] // hd, shards[1] // hd, **common)
            ids = ["q", "k", "v"]
        elif mod == "in_proj_qkvz":
            value = shards[1]
            key = (shards[0] - value) // 2
            layer = vlin.MergedColumnParallelLinear(k, [key, key, value, value], **common)
            ids = [(0, 1, 2), 3]
        elif mod in ("gate_up_proj", "in_proj_ba"):
            layer = vlin.MergedColumnParallelLinear(k, list(shards), **common)
            ids = list(range(len(shards)))
        else:
            layer = vlin.RowParallelLinear(k, shards[0], input_is_parallel=True, **common)
            ids = [None]
    if type(layer.quant_method).__name__ != "HummingLinearMethod":
        raise RuntimeError(f"vLLM gave {pre} {type(layer.quant_method).__name__}, not HummingLinearMethod")
    for sid, part in zip(ids, parts):
        for name in ("qweight", "qzeros", "scales"):
            param = getattr(layer, name)
            if sid is None:
                param.weight_loader(param, part[name].to(dev))
            else:
                param.weight_loader(param, part[name].to(dev), sid)
    if gptq is not None:                   # what vLLM's loader put into the fused params vs our concatenation
        mism = {}
        for name in ("qweight", "qzeros", "scales"):
            have, want = getattr(layer, name).data, gptq[name].to(dev)
            mism[name] = (int((have != want).sum()) if have.shape == want.shape
                          else f"shape {tuple(have.shape)} vs {tuple(want.shape)}")
        layer._load_mismatch = mism
    layer.quant_method.process_weights_after_loading(layer)
    return (lambda x: layer.quant_method.apply(layer, x)), layer


def compare(y, yref32, ycub, shards) -> dict:
    """Relative Frobenius errors vs the fp32 reference, per shard and the worst 256-column block."""
    import torch
    yf = y.float()
    den = float(yref32.norm()) or 1.0
    err = yf - yref32
    out = {"finite": bool(torch.isfinite(yf).all()), "rel_err": float(err.norm()) / den,
           "rel_err_cublas": float((ycub.float() - yref32).norm()) / den,
           "max_abs_rel": float(err.abs().max()) / (float(yref32.abs().max()) or 1.0)}
    per, c0 = [], 0
    for n in shards:
        d = float(yref32[:, c0:c0 + n].norm()) or 1.0
        per.append(round(float(err[:, c0:c0 + n].norm()) / d, 6))
        c0 += n
    out["rel_err_by_shard"] = per
    cols = yref32.shape[1]
    worst, wb = -1.0, 0
    for b0 in range(0, cols, 256):
        d = float(yref32[:, b0:b0 + 256].norm()) or 1.0
        e = float(err[:, b0:b0 + 256].norm()) / d
        if e > worst:
            worst, wb = e, b0
    out["worst_block"] = {"cols": [wb, min(wb + 256, cols)], "rel_err": worst}
    out["ok"] = out["finite"] and out["rel_err"] <= 2 * out["rel_err_cublas"] + 2e-3
    return out


def worker(a) -> int:
    import torch
    cases = json.loads(Path(a.cases).read_text())
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    res = open(a.results, "a", buffering=1)

    def emit(ev):
        res.write(json.dumps(ev) + "\n")
        res.flush()
        os.fsync(res.fileno())

    emit({"event": "worker", "start": a.start, "pid": os.getpid(), "device": torch.cuda.get_device_name(dev)})
    for ci in range(a.start, len(cases)):
        case = cases[ci]
        stage = "tensors"
        try:
            emit({"event": "begin", "case": ci})
            gptq, wref, extra, parts = case_tensors(case, dev)
            if extra:
                emit({"event": "export", "case": ci, **extra})
            stage = "fmt"
            emit({"event": "stage", "case": ci, "stage": stage})
            emit({"event": "fmt", "case": ci, **fmt_check(case, gptq, wref)})
            for impl in a.impls:
                stage = f"build:{impl}"
                emit({"event": "stage", "case": ci, "stage": stage})
                ref_w = wref
                try:
                    if impl == "vllm":
                        fwd, layer = build_vllm(case, gptq, dev)
                    elif impl == "humming":
                        fwd, layer = build_humming(case, gptq, dev)
                    elif impl == "loader":
                        fwd, layer = build_loader(case, parts, dev, gptq)
                    else:
                        fwd, layer, ref_w = build_native(case, wref, dev)
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001 - recorded; a CUDA error ends the worker below
                    if CUDA_DEAD.search(str(e)):
                        raise
                    emit({"event": "error", "case": ci, "impl": impl, "stage": stage,
                          "error": f"{type(e).__name__}: {e}"[:2000], "trace": traceback.format_exc()[-3000:]})
                    continue
                meta = getattr(layer, "humming_metas", {}).get("")
                emit({"event": "built", "case": ci, "impl": impl, "meta": meta.to_str() if meta is not None else None,
                      "load_mismatch": getattr(layer, "_load_mismatch", None)})
                wt = ref_w.t().contiguous()
                wt32 = wt.float()
                for m in a.m:
                    stage = f"gemm:{impl}:M={m}"
                    emit({"event": "stage", "case": ci, "stage": stage, "impl": impl, "m": m})
                    g = torch.Generator(device=dev).manual_seed(case["seed"] * 7919 + m)
                    x = torch.randn(m, case["k"], generator=g, device=dev).to(torch.bfloat16)
                    yref = x.float() @ wt32
                    ycub = x @ wt
                    try:
                        y = fwd(x)
                        torch.cuda.synchronize()
                    except Exception as e:  # noqa: BLE001
                        if CUDA_DEAD.search(str(e)):
                            raise
                        emit({"event": "error", "case": ci, "impl": impl, "m": m, "stage": stage,
                              "error": f"{type(e).__name__}: {e}"[:2000], "trace": traceback.format_exc()[-3000:]})
                        continue
                    r = compare(y, yref, ycub, case["shards"])
                    emit({"event": "result", "case": ci, "impl": impl, "m": m, **r})
                    del x, yref, ycub, y
                del fwd, layer, wt, wt32
                torch.cuda.empty_cache()
            del gptq, wref, parts
            torch.cuda.empty_cache()
            emit({"event": "end", "case": ci})
        except Exception as e:  # noqa: BLE001
            dead = bool(CUDA_DEAD.search(str(e)))
            emit({"event": "crash" if dead else "error", "case": ci, "stage": stage,
                  "error": f"{type(e).__name__}: {e}"[:2000], "trace": traceback.format_exc()[-3000:]})
            if dead:
                return 3                      # the CUDA context is gone: the parent restarts at the next case
    emit({"event": "done"})
    return 0


# ------------------------------------------------------------------------------------------------ parent
def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass                               # a line cut by a dying worker
    return out


def run_workers(a, cases_path: Path, results: Path, n_cases: int, script: str | None = None) -> None:
    """Run workers until every case has an outcome. A worker that dies (CUDA error, segfault, OOM kill) or writes
    no event for a.stall_timeout seconds (a hung kernel; killed) gets a "crash" event for the case it was in and the
    next worker starts at the case after it; two workers in a row that die before reaching any case end the run
    (the GPU or environment is broken), and the report is written from what ran. script: tests only."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
    if a.blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    start, restarts, startup_fail = 0, 0, 0
    while start < n_cases:
        cmd = [sys.executable, "-I", script or str(Path(__file__).resolve()), "--worker", "--cases", str(cases_path),
               "--results", str(results), "--start", str(start), "--m", ",".join(map(str, a.m)),
               "--impls", ",".join(a.impls)]
        t0 = time.time()
        log = results.parent / f"worker_{start:03d}.log"
        seen = len(read_events(results))
        stalled = False
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
            last_t = time.time()
            while proc.poll() is None:
                time.sleep(a.poll)
                evs = read_events(results)
                if len(evs) > seen:
                    for e in evs[seen:]:
                        progress(e, n_cases)
                    seen, last_t = len(evs), time.time()
                elif time.time() - last_t > a.stall_timeout:
                    stalled = True
                    proc.kill()
                    proc.wait()
            rc = proc.returncode
        evs = read_events(results)
        for e in evs[seen:]:
            progress(e, n_cases)
        if any(e.get("event") == "done" for e in evs[-3:]):
            if rc != 0:
                print(f"[kernel-check] warning: worker exited {rc} after finishing (log {log})", flush=True)
            return
        mine = [e for e in evs if isinstance(e.get("case"), int) and e["case"] >= start]
        last_case = max((e["case"] for e in mine), default=None)
        if last_case is None:                       # died before reaching any case (CUDA init, import, OOM kill)
            startup_fail += 1
            last_case, stage = start, "worker-start"
        else:
            startup_fail = 0
            stage = next((e.get("stage") for e in reversed(mine) if e.get("event") == "stage"
                          and e.get("case") == last_case), "begin")
        finished = any(e.get("event") == "end" and e.get("case") == last_case for e in mine)
        if not finished and not any(e.get("event") == "crash" and e.get("case") == last_case for e in mine):
            with open(results, "a") as f:           # killed without a Python exception (segfault, abort, watchdog)
                f.write(json.dumps({"event": "crash", "case": last_case, "stage": stage, "returncode": rc,
                                    "stalled_s": a.stall_timeout if stalled else None,
                                    "error": ("no event for %ds: killed\n" % a.stall_timeout if stalled else "")
                                    + log.read_text(errors="replace")[-3000:], "log": str(log)}) + "\n")
        restarts += 1
        print(f"[kernel-check] worker {'stalled' if stalled else 'died'} in case {last_case + 1}/{n_cases} ({stage}) "
              f"after {time.time() - t0:.0f}s (exit {rc}, log {log}); restarting at case {last_case + 2}", flush=True)
        if startup_fail >= 2 or restarts > n_cases + 2:
            print("[kernel-check] workers keep dying before any case: the GPU / environment is broken (see "
                  f"{log}); writing the report from what ran", flush=True)
            return
        start = last_case + 1


def progress(e: dict, n_cases: int) -> None:
    """One terminal line per finished case / crash / error (the sweep can take an hour)."""
    ev = e.get("event")
    if ev == "end":
        print(f"[kernel-check] case {e['case'] + 1}/{n_cases} done", flush=True)
    elif ev == "crash":
        print(f"[kernel-check] case {e['case'] + 1}/{n_cases} CRASH at {e.get('stage')}: "
              f"{(e.get('error') or '')[:160]}", flush=True)
    elif ev == "error":
        print(f"[kernel-check] case {e['case'] + 1}/{n_cases} error in {e.get('stage')}: "
              f"{(e.get('error') or '')[:160]}", flush=True)
    elif ev == "result" and not e.get("ok"):
        print(f"[kernel-check] case {e['case'] + 1}/{n_cases} {e['impl']} M={e['m']}: rel_err {e['rel_err']:.3g} "
              f"(cuBLAS {e['rel_err_cublas']:.3g}){'' if e.get('finite') else ' NON-FINITE'}", flush=True)
    elif ev == "worker":
        print(f"[kernel-check] worker on {e.get('device')} from case {e.get('start', 0) + 1}", flush=True)


def summarize(cases: list[dict], skipped: list[dict], evs: list[dict], a) -> dict:
    rows = []
    for ci, c in enumerate(cases):
        ce = [e for e in evs if e.get("case") == ci]
        fmt = next((e for e in ce if e["event"] == "fmt"), None)
        exp = next((e for e in ce if e["event"] == "export"), None)
        res = [e for e in ce if e["event"] == "result"]
        errs = [{k: e.get(k) for k in ("impl", "m", "stage", "error")} for e in ce if e["event"] == "error"]
        crash = [{k: e.get(k) for k in ("stage", "error", "returncode")} for e in ce if e["event"] == "crash"]
        by_impl = {}
        for e in res:
            by_impl.setdefault(e["impl"], []).append(
                {"m": e["m"], "ok": e["ok"], "rel_err": round(e["rel_err"], 6),
                 "rel_err_cublas": round(e["rel_err_cublas"], 6), "finite": e["finite"],
                 "rel_err_by_shard": e["rel_err_by_shard"], "worst_block": e["worst_block"]})
        built = {e["impl"]: e.get("meta") for e in ce if e["event"] == "built"}
        load_mm = next((e.get("load_mismatch") for e in ce if e["event"] == "built" and e.get("load_mismatch")), None)
        fmt_ok = fmt is not None and fmt.get("mismatch_elems") == 0
        export_ok = exp is None or (not exp.get("export_mismatch_elems") and not exp.get("export_missing"))
        impl_ok = {i: bool(v) and all(x["ok"] for x in v) for i, v in by_impl.items()}
        complete = all(len(by_impl.get(i, [])) == len(a.m) for i in a.impls)
        ok = fmt_ok and export_ok and not errs and not crash and complete and all(impl_ok.values())
        worst = max((x["rel_err"] for v in by_impl.values() for x in v), default=None)
        rows.append({"case": ci, "ok": ok, "module": c["module"], "prefix": c["prefix"], "source": c["source"],
                     "bits": c["bits"], "group_size": c["group_size"], "shards": c["shards"], "k": c["k"],
                     "fmt": fmt, "export": exp, "impl_ok": impl_ok, "worst_rel_err": worst, "results": by_impl,
                     "errors": errs, "crash": crash, "kernel_meta": built, "load_mismatch": load_mm})
    rows.sort(key=lambda r: (r["ok"], -(r["worst_rel_err"] or (1e9 if not r["ok"] else 0))))
    return {"cases": len(cases), "failed": sum(not r["ok"] for r in rows), "skipped": skipped,
            "m": a.m, "impls": a.impls, "rows": rows}


def diagnosis(rep: dict) -> list[str]:
    """Plain-language reading of the failures (which layer of the stack is wrong)."""
    bad = [r for r in rep["rows"] if not r["ok"]]
    if not bad:
        return ["every case passed: the kernels compute W' for these shapes / settings; the full-model KL gap is "
                "elsewhere (module mapping, the config, a shape or M not covered here)"]
    out = []
    for r in bad:
        tag = f"{r['prefix']} {r['bits']}b g{r['group_size']} shards {r['shards']}"
        if r["export"] and (r["export"].get("export_mismatch_elems") or r["export"].get("export_missing")):
            out.append(f"{tag}: the packed tensors do not decode to the dequant checkpoint (export bug)")
        if r["fmt"] and r["fmt"].get("mismatch_elems"):
            out.append(f"{tag}: humming READS the tensors differently from what we meant (format bug): "
                       f"{r['fmt']['mismatch_elems']} of {r['fmt']['elems']} weights differ")
        if r["crash"]:
            out.append(f"{tag}: CUDA crash at {r['crash'][0].get('stage')}")
        fails = {i: [x["m"] for x in v if not x["ok"]] for i, v in r["results"].items()}
        fails = {i: ms for i, ms in fails.items() if ms}
        fmt_ok = r["fmt"] is not None and r["fmt"].get("mismatch_elems") == 0
        if fails:
            where = "; ".join(f"{i} wrong at M={ms}" for i, ms in fails.items())
            if fmt_ok and "native" in fails:
                out.append(f"{tag}: {where} -> the humming GEMM itself is wrong here (native quantization too)")
            elif fmt_ok and "vllm" in fails and "humming" in r["results"] and "humming" not in fails:
                out.append(f"{tag}: {where} -> vLLM's integration differs from calling humming directly")
            elif fmt_ok and "loader" in fails and not ({"vllm", "humming", "native"} & fails.keys()):
                out.append(f"{tag}: {where} -> the GEMM is right on the concatenated tensors, but vLLM's real layer "
                           f"(weight_loader_v2 shard offsets) fills the fused params differently: "
                           f"load_mismatch {r.get('load_mismatch')}")
            elif fmt_ok:
                out.append(f"{tag}: {where} -> tensors are read right but the GEMM is wrong (kernel / tile config)")
            else:
                out.append(f"{tag}: {where}")
        for e in r["errors"][:2]:
            out.append(f"{tag}: error in {e.get('stage')}: {(e.get('error') or '')[:200]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output dir (kernel_check.json, results.jsonl)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", help="a packed checkpoint (packed_check's <out>/packed): test its own modules")
    ap.add_argument("--dequant", help="with --ckpt: the dequant checkpoint of the same config (<out>/dequant)")
    ap.add_argument("--per-kind", type=int, default=1, help="with --ckpt: modules per distinct (kind, bits, group)")
    ap.add_argument("--model", help="synthetic cases: read the module shapes from this checkpoint")
    ap.add_argument("--modules", help="synthetic cases: comma list of " + ",".join(FUSED_ORDER))
    ap.add_argument("--bits", default="2,3,4,8")
    ap.add_argument("--group-sizes", default="32,64,128")
    ap.add_argument("--m", help=f"batch sizes (default {','.join(map(str, DEFAULT_M))})")
    ap.add_argument("--impls", default="vllm,humming,native",
                    help="comma list of vllm, humming, native, loader (opt-in: vLLM's real layer + weight loader)")
    ap.add_argument("--quick", action="store_true", help=f"group size 128 only, M {QUICK_M}")
    ap.add_argument("--blocking", action="store_true", help="CUDA_LAUNCH_BLOCKING=1 (exact crash site, slower)")
    ap.add_argument("--stall-timeout", type=float, default=1800,
                    help="kill a worker that writes no event for this many seconds (a hung kernel); default 1800")
    ap.add_argument("--poll", type=float, default=2.0, help=argparse.SUPPRESS)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cases", help=argparse.SUPPRESS)
    ap.add_argument("--results", help=argparse.SUPPRESS)
    ap.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    a.m = [int(x) for x in a.m.split(",")] if a.m else (QUICK_M if a.quick else DEFAULT_M)
    a.impls = [x.strip() for x in a.impls.split(",") if x.strip()]
    for i in a.impls:
        if i not in ("vllm", "humming", "native", "loader"):
            ap.error(f"unknown impl {i!r}")
    if a.worker:
        return worker(a)
    if not a.out:
        ap.error("--out is required")
    a.bits = [int(x) for x in a.bits.split(",")]
    a.group_sizes = [128] if a.quick else [int(x) for x in a.group_sizes.split(",")]
    if a.dequant and not a.ckpt:
        ap.error("--dequant needs --ckpt")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases, skipped = ckpt_cases(a) if a.ckpt else synthetic_cases(a)
    if not cases:
        raise SystemExit(f"no cases (skipped: {skipped})")
    cases_path, results = out / "cases.json", out / "results.jsonl"
    cases_path.write_text(json.dumps(cases, indent=1))
    results.unlink(missing_ok=True)
    print(f"[kernel-check] {len(cases)} cases x impls {a.impls} x M {a.m} on GPU {a.gpu}"
          + (f" ({len(skipped)} not expressible / skipped)" if skipped else ""), flush=True)
    t0 = time.time()
    run_workers(a, cases_path, results, len(cases))
    rep = summarize(cases, skipped, read_events(results), a)
    rep["diagnosis"] = diagnosis(rep)
    rep["seconds"] = round(time.time() - t0, 1)
    rep["args"] = {k: v for k, v in vars(a).items() if k not in ("worker", "cases", "results", "start")}
    (out / "kernel_check.json").write_text(json.dumps(rep, indent=1))
    print(f"{'ok':4} {'module':14} {'bits':>4} {'g':>4} {'shards':24} {'fmt':>8}  worst rel_err per impl (M)")
    for r in rep["rows"]:
        fm = "-" if r["fmt"] is None else ("ok" if r["fmt"].get("mismatch_elems") == 0 else
                                           f"{r['fmt']['mismatch_elems']}")
        per = "  ".join(f"{i}:{max((x['rel_err'] for x in v), default=float('nan')):.3g}"
                        f"@{max(v, key=lambda x: x['rel_err'])['m'] if v else '-'}"
                        for i, v in r["results"].items())
        extra = " CRASH" if r["crash"] else (" ERROR" if r["errors"] else "")
        print(f"{'ok' if r['ok'] else 'FAIL':4} {r['module']:14} {r['bits']:>4} {r['group_size']:>4} "
              f"{str(r['shards']):24} {fm:>8}  {per}{extra}")
    print("\n".join(f"[diagnosis] {x}" for x in rep["diagnosis"]))
    print(f"[kernel-check] {rep['failed']} of {rep['cases']} cases failed; report {out / 'kernel_check.json'}")
    return 0 if rep["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
