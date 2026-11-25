"""Model description for quantization: which linear modules exist, how they are grouped, shapes.

Read from the checkpoint's safetensors headers (no tensors are loaded, torch not needed), so the
same code works for the real Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`, tensor names
`model.language_model.layers.<i>.{self_attn,linear_attn,mlp}.<proj>.weight`), for a text-only
`Qwen3_5ForCausalLM` (`model.layers.<i>...`) and for the tiny test models. A synthetic spec with
Qwen3.8-27B-like shapes is used in mock mode.

Module groups (METHOD_DESIGN §2.1, plus the Gated-DeltaNet projections of the linear-attention
layers):
  full-attention layers : qk = {q_proj, k_proj}, v = {v_proj}, o = {o_proj}
  linear-attention (GDN): gdn_qkv = {in_proj_qkv}, gdn_z = {in_proj_z}, gdn_ba = {in_proj_b, in_proj_a},
                          gdn_out = {out_proj}
  every layer (MLP)     : gate_up = {gate_proj, up_proj}, down = {down_proj}
Group key = "<layer>.<group>", e.g. "17.down". Embedding / LM head are separate settings.
vLLM fuses q/k/v, gate/up, in_proj_qkv/z and in_proj_b/a at load time; for the fake-quant BF16
checkpoints used here that does not matter (a real packed export would need equal settings inside
each fused module -- see README, future work).
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

GROUPS = {
    "full_attention": {"qk": ["q_proj", "k_proj"], "v": ["v_proj"], "o": ["o_proj"]},
    "linear_attention": {"gdn_qkv": ["in_proj_qkv"], "gdn_z": ["in_proj_z"],
                         "gdn_ba": ["in_proj_b", "in_proj_a"], "gdn_out": ["out_proj"]},
    "mlp": {"gate_up": ["gate_proj", "up_proj"], "down": ["down_proj"]},
}
PROJ_GROUP = {p: g for kind in GROUPS.values() for g, ps in kind.items() for p in ps}
GROUP_NAMES = [g for kind in GROUPS.values() for g in kind]
# modules that see the same input activations share one Hessian / activation statistic
INPUT_KEY = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_in",
             "in_proj_qkv": "gdn_in", "in_proj_z": "gdn_in", "in_proj_b": "gdn_in", "in_proj_a": "gdn_in",
             "out_proj": "gdn_out_in", "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in"}
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
_LAYER_RE = re.compile(r"^(?P<prefix>(?:.+\.)?layers)\.(?P<layer>\d+)\.(?P<block>self_attn|linear_attn|mlp)"
                       r"\.(?P<proj>[a-z_]+)\.weight$")


@dataclass
class ModuleInfo:
    name: str            # full tensor name, e.g. model.language_model.layers.3.mlp.down_proj.weight
    layer: int
    group: str
    proj: str
    shape: list[int]     # [out_features, in_features]
    input_key: str

    @property
    def numel(self) -> int:
        return int(self.shape[0]) * int(self.shape[1])

    @property
    def module_path(self) -> str:
        return self.name[: -len(".weight")]


@dataclass
class ModelSpec:
    layer_types: list[str]
    layers_prefix: str
    modules: list[ModuleInfo]
    embed: dict | None             # {"name", "shape", "dtype_bytes"}
    lm_head: dict | None           # None if tied to the embedding
    other_bytes: int               # all other counted tensors (norms, conv1d, A_log, visual, ...)
    excluded_bytes: int = 0        # tensors not counted (mtp.* -- vLLM skips them)
    weight_dtype_bytes: int = 2
    source: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self._by_name = {m.name: m for m in self.modules}

    # groups -----------------------------------------------------------------------------
    def groups(self) -> dict[str, list[ModuleInfo]]:
        out: dict[str, list[ModuleInfo]] = {}
        for m in self.modules:
            out.setdefault(f"{m.layer}.{m.group}", []).append(m)
        return out

    def group_keys(self) -> list[str]:
        return list(self.groups().keys())

    def group_params(self) -> dict[str, int]:
        return {k: sum(m.numel for m in ms) for k, ms in self.groups().items()}

    def module(self, name: str) -> ModuleInfo:
        return self._by_name[name]

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "ModelSpec":
        d = dict(d)
        d["modules"] = [ModuleInfo(**m) for m in d["modules"]]
        return cls(**d)

    def summary(self) -> dict:
        gp = self.group_params()
        per_group: dict[str, int] = {}
        for k, n in gp.items():
            g = k.split(".", 1)[1]
            per_group[g] = per_group.get(g, 0) + n
        lin = sum(gp.values())
        return {"num_layers": self.num_layers,
                "layer_types": {t: self.layer_types.count(t) for t in sorted(set(self.layer_types))},
                "linear_params": lin, "params_by_group": per_group,
                "embed_params": _numel(self.embed), "lm_head_params": _numel(self.lm_head),
                "lm_head_tied": self.lm_head is None, "other_bytes": self.other_bytes}


def _numel(t: dict | None) -> int:
    if not t:
        return 0
    n = 1
    for s in t["shape"]:
        n *= int(s)
    return n


# ------------------------------------------------------------------------------- readers
def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def shard_files(model_dir: Path) -> list[str]:
    """Shard file names of a safetensors checkpoint (the index's weight_map, else every *.safetensors), sorted."""
    model_dir = Path(model_dir)
    idx = model_dir / "model.safetensors.index.json"
    return sorted(set(json.load(open(idx))["weight_map"].values())) if idx.exists() else \
        sorted(p.name for p in model_dir.glob("*.safetensors"))


def checkpoint_tensors(model_dir: Path) -> dict[str, dict]:
    """name -> {"dtype", "shape", "file"} for every tensor of a safetensors checkpoint."""
    model_dir = Path(model_dir)
    files = shard_files(model_dir)
    if not files:
        raise FileNotFoundError(f"no safetensors files in {model_dir}")
    out = {}
    for fn in files:
        for k, v in read_safetensors_header(model_dir / fn).items():
            out[k] = {"dtype": v["dtype"], "shape": v["shape"], "file": fn}
    return out


def load_model_config(model_dir: Path) -> dict:
    return json.load(open(Path(model_dir) / "config.json"))


def _text_config(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def spec_from_checkpoint(model_dir: Path, *, count_visual: bool = True,
                         exclude_prefixes: tuple[str, ...] = ("mtp.",)) -> ModelSpec:
    model_dir = Path(model_dir)
    cfg = load_model_config(model_dir)
    tcfg = _text_config(cfg)
    tensors = checkpoint_tensors(model_dir)
    modules, prefixes = [], set()
    embed = lm_head = None
    other = excluded = 0
    for name, t in tensors.items():
        nbytes = DTYPE_BYTES.get(t["dtype"], 2) * _numel(t)
        if any(name.startswith(p) for p in exclude_prefixes):
            excluded += nbytes
            continue
        if "visual" in name.split(".")[:3] and not count_visual:
            excluded += nbytes
            continue
        m = _LAYER_RE.match(name)
        if m and m.group("proj") in PROJ_GROUP and len(t["shape"]) == 2 and "visual" not in name:
            proj = m.group("proj")
            prefixes.add(m.group("prefix"))
            modules.append(ModuleInfo(name=name, layer=int(m.group("layer")), group=PROJ_GROUP[proj],
                                      proj=proj, shape=list(t["shape"]), input_key=INPUT_KEY[proj]))
            continue
        if name.endswith("embed_tokens.weight") and "visual" not in name:
            embed = {"name": name, "shape": t["shape"], "dtype_bytes": DTYPE_BYTES.get(t["dtype"], 2)}
            continue
        if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
            lm_head = {"name": name, "shape": t["shape"], "dtype_bytes": DTYPE_BYTES.get(t["dtype"], 2)}
            continue
        other += nbytes
    if len(prefixes) != 1:
        raise ValueError(f"expected one decoder-layer prefix, found {sorted(prefixes)} in {model_dir}")
    n_layers = int(tcfg.get("num_hidden_layers") or (max(m.layer for m in modules) + 1))
    layer_types = tcfg.get("layer_types")
    if not layer_types:
        interval = int(tcfg.get("full_attention_interval", 4))
        layer_types = ["linear_attention" if (i + 1) % interval else "full_attention" for i in range(n_layers)]
    modules.sort(key=lambda m: (m.layer, list(PROJ_GROUP).index(m.proj)))
    # sanity: group membership consistent with the layer type
    for m in modules:
        kind = layer_types[m.layer]
        if m.group in GROUPS["full_attention"] and kind != "full_attention":
            raise ValueError(f"{m.name}: attention module in a {kind} layer")
        if m.group in GROUPS["linear_attention"] and kind != "linear_attention":
            raise ValueError(f"{m.name}: GDN module in a {kind} layer")
    wbytes = DTYPE_BYTES.get(tensors[modules[0].name]["dtype"], 2) if modules else 2
    return ModelSpec(layer_types=list(layer_types), layers_prefix=prefixes.pop(), modules=modules,
                     embed=embed, lm_head=lm_head, other_bytes=int(other), excluded_bytes=int(excluded),
                     weight_dtype_bytes=wbytes, source=str(model_dir),
                     extra={"architectures": cfg.get("architectures"),
                            "tie_word_embeddings": bool(cfg.get("tie_word_embeddings",
                                                                tcfg.get("tie_word_embeddings", False)))})


# Qwen3.8-27B-like shapes for mock mode (about 26.9B parameters; the real numbers are read from
# the checkpoint on the GPU machine).
QWEN27B_LIKE = dict(num_layers=64, hidden=5120, intermediate=17408, heads=24, head_dim=256, kv_heads=4,
                    lin_k_heads=16, lin_v_heads=48, lin_head_dim=128, vocab=248320, interval=4,
                    visual_params=460_000_000)


def synthetic_spec(preset: dict | None = None) -> ModelSpec:
    p = dict(QWEN27B_LIKE, **(preset or {}))
    L, H, I = p["num_layers"], p["hidden"], p["intermediate"]
    pre = "model.language_model.layers"
    lt = ["linear_attention" if (i + 1) % p["interval"] else "full_attention" for i in range(L)]
    shapes = {
        "q_proj": [p["heads"] * p["head_dim"] * 2, H], "k_proj": [p["kv_heads"] * p["head_dim"], H],
        "v_proj": [p["kv_heads"] * p["head_dim"], H], "o_proj": [H, p["heads"] * p["head_dim"]],
        "in_proj_qkv": [2 * p["lin_k_heads"] * p["lin_head_dim"] + p["lin_v_heads"] * p["lin_head_dim"], H],
        "in_proj_z": [p["lin_v_heads"] * p["lin_head_dim"], H],
        "in_proj_b": [p["lin_v_heads"], H], "in_proj_a": [p["lin_v_heads"], H],
        "out_proj": [H, p["lin_v_heads"] * p["lin_head_dim"]],
        "gate_proj": [I, H], "up_proj": [I, H], "down_proj": [H, I]}
    block = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
             "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
    mods = []
    for i, t in enumerate(lt):
        projs = (GROUPS["full_attention"] if t == "full_attention" else GROUPS["linear_attention"]).values()
        for proj in [x for ps in projs for x in ps] + ["gate_proj", "up_proj", "down_proj"]:
            b = block.get(proj, "linear_attn")
            mods.append(ModuleInfo(name=f"{pre}.{i}.{b}.{proj}.weight", layer=i, group=PROJ_GROUP[proj],
                                   proj=proj, shape=shapes[proj], input_key=INPUT_KEY[proj]))
    norms = L * 2 * H * 2 + H * 2
    return ModelSpec(layer_types=lt, layers_prefix=pre, modules=mods,
                     embed={"name": "model.language_model.embed_tokens.weight", "shape": [p["vocab"], H],
                            "dtype_bytes": 2},
                     lm_head={"name": "lm_head.weight", "shape": [p["vocab"], H], "dtype_bytes": 2},
                     other_bytes=int(norms + p["visual_params"] * 2), source="synthetic:qwen3.8-27b-like",
                     extra={"synthetic": True})
