"""Module naming for Qwen3.5 / Qwen3.8 (`Qwen3_5ForConditionalGeneration`) and its vLLM LoRA mapping.

Checkpoint keys (verified on a transformers 5.9 tiny model, lora_code/tests/test_primitives.py):
  model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight                  (full attention layers)
  model.language_model.layers.{i}.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}.weight
                                                                                    (Gated DeltaNet layers)
  model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
  model.visual.*  (vision tower, never edited)   mtp.* (multi-token prediction head, never edited)
A text-only checkpoint (Qwen3_5ForCausalLM) uses the prefix `model.` instead; the prefix is detected.

Inside lora_code a module is addressed by its SHORT name `layers.{i}.{block}.{leaf}`, e.g.
`layers.12.mlp.down_proj`; the full name is `<prefix>` + short (prefix `model.language_model.`).

vLLM 0.24.0 (`vllm/model_executor/models/qwen3_5.py`) packs sub-modules for LoRA:
  qkv_proj <- q_proj, k_proj, v_proj ; gate_up_proj <- gate_proj, up_proj
  in_proj_qkvz <- in_proj_qkv, in_proj_z ; in_proj_ba <- in_proj_b, in_proj_a
PEFT adapters must therefore use the UNPACKED HF names (vLLM packs at load time) and the HF key
prefix `base_model.model.model.language_model.layers...`; vLLM maps it with the model's
hf_to_vllm_mapper (model.language_model. -> language_model.model.).
"""
from __future__ import annotations

import re

VLLM_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
# leaf -> (block, group)
LEAVES = {
    "q_proj": ("self_attn", "attn"), "k_proj": ("self_attn", "attn"), "v_proj": ("self_attn", "attn"),
    "o_proj": ("self_attn", "attn"),
    "in_proj_qkv": ("linear_attn", "attn"), "in_proj_z": ("linear_attn", "attn"),
    "in_proj_b": ("linear_attn", "attn"), "in_proj_a": ("linear_attn", "attn"),
    "out_proj": ("linear_attn", "attn"),
    "gate_proj": ("mlp", "mlp"), "up_proj": ("mlp", "mlp"), "down_proj": ("mlp", "mlp"),
}
VLLM_LORA_LEAVES = set(LEAVES)  # every leaf above is LoRA-capable in vLLM's Qwen3.5 class (packed or not)
SHORT_RE = re.compile(r"^layers\.(\d+)\.(self_attn|linear_attn|mlp)\.([a-z_]+)$")
KEY_RE = re.compile(r"^(?P<prefix>.*?)layers\.(?P<layer>\d+)\.(?P<block>self_attn|linear_attn|mlp)\.(?P<leaf>[a-z_]+)\.weight$")
EXCLUDE_PREFIXES = ("model.visual.", "visual.", "mtp.", "model.mtp.")


def parse_short(short: str) -> tuple[int, str, str]:
    m = SHORT_RE.match(short)
    if not m:
        raise ValueError(f"bad module name {short!r}; expected 'layers.<i>.<self_attn|linear_attn|mlp>.<leaf>'")
    layer, block, leaf = int(m.group(1)), m.group(2), m.group(3)
    if leaf not in LEAVES or LEAVES[leaf][0] != block:
        raise ValueError(f"unknown module leaf {block}.{leaf}")
    return layer, block, leaf


def make_short(layer: int, leaf: str) -> str:
    return f"layers.{int(layer)}.{LEAVES[leaf][0]}.{leaf}"


def parse_key(key: str):
    """checkpoint key -> (prefix, short) for editable linear weights, else None."""
    if key.startswith(EXCLUDE_PREFIXES):
        return None
    m = KEY_RE.match(key)
    if not m or m.group("leaf") not in LEAVES or LEAVES[m.group("leaf")][0] != m.group("block"):
        return None
    pre = m.group("prefix")
    if "visual" in pre or "mtp" in pre:
        return None
    return pre, f"layers.{m.group('layer')}.{m.group('block')}.{m.group('leaf')}"


def vllm_packed_name(leaf: str) -> str:
    for packed, subs in VLLM_PACKED.items():
        if leaf in subs:
            return packed
    return leaf


def peft_key(prefix: str, short: str, which: str) -> str:
    """PEFT tensor key, which in {'A','B'}."""
    return f"base_model.model.{prefix}{short}.lora_{which}.weight"


PEFT_KEY_RE = re.compile(r"^base_model\.model\.(?P<full>.+)\.lora_(?P<which>[AB])(?:\.default)?\.weight$")


def parse_peft_key(key: str):
    m = PEFT_KEY_RE.match(key)
    if not m:
        return None
    full = m.group("full")
    i = full.find("layers.")
    if i < 0:
        return None
    return full[:i], full[i:], m.group("which")
