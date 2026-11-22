"""CPU tests of lora_code/primitives on a TINY randomly initialised Qwen3.5 model (transformers 5.9 config:
4 layers, 3 Gated-DeltaNet + 1 full attention, tiny vision tower). Skipped cleanly when torch /
transformers / peft / safetensors are not importable (e.g. eval/.venv).

  <venv with torch transformers==5.9.0 peft safetensors>/bin/python -I -m pytest -q lora_code/tests/test_primitives.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

MISSING = [m for m in ("torch", "transformers", "peft", "safetensors") if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(bool(MISSING), reason=f"missing {MISSING}")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    tc = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
              linear_num_key_heads=2, linear_num_value_heads=4, max_position_embeddings=512)
    vc = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=2,
              spatial_merge_size=1, temporal_patch_size=1)
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=tc, vision_config=vc))
    d = tmp_path_factory.mktemp("tiny_qwen35")
    model.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def store(tiny):
    from lora_code.primitives.weights import WeightStore
    return WeightStore(tiny, cache_dir=tiny / "_cache")


def _examples(n=8):
    out = {}
    for i in range(n):
        out[f"ex/{i}"] = {"messages": [{"role": "user", "content": f"what is {i} plus {i}?"}],
                          "target": f"the answer is {2 * i}. " * 2, "source": "gold"}
    return out


def test_naming_and_store(store):
    from lora_code.primitives.naming import vllm_packed_name
    mods = store.modules()
    assert store.prefix == "model.language_model."
    assert "layers.0.linear_attn.in_proj_qkv" in mods and "layers.3.self_attn.q_proj" in mods
    assert not any("visual" in m for m in mods)
    assert store.shape("layers.0.mlp.down_proj") == (64, 128)
    assert store.info("layers.3.self_attn.q_proj")["layer_type"] == "full_attention"
    assert vllm_packed_name("in_proj_b") == "in_proj_ba" and vllm_packed_name("q_proj") == "qkv_proj"
    U, S, V = store.svd("layers.0.mlp.up_proj", 4)
    W = store.weight("layers.0.mlp.up_proj")
    assert U.shape == (128, 4) and V.shape == (64, 4)
    assert abs(float(S[0]) - float(__import__("torch").linalg.svdvals(W)[0])) < 1e-3


def test_constructors_shapes(store):
    import torch
    from lora_code.primitives import construct
    from lora_code.primitives.lora import LoRA, check_lora
    lo = LoRA()
    m = "layers.1.mlp.down_proj"
    out_f, in_f = store.shape(m)
    construct.rank1(lo, m, torch.randn(out_f), torch.randn(in_f), 0.5, shape=(out_f, in_f))
    key = torch.randn(in_f)
    construct.from_direction(lo, m, torch.randn(out_f), key, 0.3, shape=(out_f, in_f))
    U, S, V = store.svd(m, 4)
    construct.scale_singular(lo, m, U, S, V, [0, 1], [1.1, 0.9])
    assert not hasattr(construct, "from_lowrank_grad")              # no gradient constructor (2026-11-20)
    A, B = lo.factors[m]
    assert A.shape == (1 + 1 + 2, in_f) and B.shape == (out_f, 4)
    ok, rep = check_lora(lo, store, rank_cap=64, tau=10.0)
    assert ok, rep
    # from_direction maps the key onto alpha * d_hat
    lo2 = LoRA()
    d = torch.randn(out_f)
    construct.from_direction(lo2, m, d, key, 0.3)
    assert torch.allclose(lo2.delta(m) @ key, 0.3 * d / d.norm(), atol=1e-5)
    with pytest.raises(ValueError):
        construct.rank1(LoRA(), m, torch.randn(out_f + 1), torch.randn(in_f), 1.0, shape=(out_f, in_f))


def test_checks_reject(store):
    import torch
    from lora_code.primitives.lora import LoRA, check_lora
    m = "layers.0.mlp.gate_proj"
    out_f, in_f = store.shape(m)
    big = LoRA({m: (torch.randn(80, in_f), torch.randn(out_f, 80))})
    ok, rep = check_lora(big, store, rank_cap=64, tau=0.1)
    assert not ok and any("rank" in e for e in rep["errors"]) and any("tau" in e for e in rep["errors"])
    nan = LoRA({m: (torch.full((2, in_f), float("nan")), torch.zeros(out_f, 2))})
    assert not check_lora(nan, store, rank_cap=64, tau=1)[0]
    bad = LoRA({m: (torch.randn(2, in_f + 1), torch.randn(out_f, 2))})
    assert not check_lora(bad, store, rank_cap=64, tau=1)[0]


def _rand_lora(store, mods, r=4, s=0.01):
    import torch
    from lora_code.primitives.lora import LoRA
    g = torch.Generator().manual_seed(1)
    lo = LoRA()
    for m in mods:
        o, i = store.shape(m)
        lo.factors[m] = (torch.randn(r, i, generator=g) * s, torch.randn(o, r, generator=g) * s)
    return lo


def test_peft_roundtrip_and_vllm_layout(store, tmp_path):
    import torch
    from safetensors import safe_open
    from lora_code.primitives.lora import load_peft, save_peft
    mods = ["layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_b", "layers.3.self_attn.k_proj",
            "layers.2.mlp.down_proj"]
    lo = _rand_lora(store, mods, r=3)
    lo.factors["layers.2.mlp.down_proj"] = (lo.factors["layers.2.mlp.down_proj"][0][:2],
                                            lo.factors["layers.2.mlp.down_proj"][1][:, :2])
    p = save_peft(lo, tmp_path / "ad", store, dtype="float32")
    cfg = json.load(open(p / "adapter_config.json"))
    assert cfg["r"] == 8 and cfg["lora_alpha"] == 8          # padded to a vLLM rank, scaling exactly 1
    assert set(cfg["target_modules"]) == {"in_proj_qkv", "in_proj_b", "k_proj", "down_proj"}
    with safe_open(str(p / "adapter_model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert "base_model.model.model.language_model.layers.3.self_attn.k_proj.lora_A.weight" in keys
    back, prefix = load_peft(p)
    assert prefix == "model.language_model."
    for m in mods:
        assert torch.allclose(back.delta(m), lo.delta(m), atol=1e-6)
    # a standard PEFT model loads our adapter and reproduces W + dW
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration
    base = Qwen3_5ForConditionalGeneration.from_pretrained(store.model_dir)
    pm = PeftModel.from_pretrained(base, str(p))
    merged = pm.merge_and_unload()
    W = merged.get_submodule(store.full_name("layers.3.self_attn.k_proj")).weight.detach()
    assert torch.allclose(W, store.weight("layers.3.self_attn.k_proj") + lo.delta("layers.3.self_attn.k_proj"), atol=1e-5)


def test_merge_equals_base_plus_ba(store, tmp_path):
    import torch
    from lora_code.primitives.lora import merge_into_model
    from lora_code.primitives.weights import WeightStore
    mods = ["layers.1.mlp.up_proj", "layers.3.self_attn.o_proj"]
    lo = _rand_lora(store, mods, r=4, s=0.05)
    rep = merge_into_model(lo, store.model_dir, tmp_path / "merged", store=store)
    new = WeightStore(tmp_path / "merged")
    for m in mods:
        assert torch.allclose(new.weight(m), store.weight(m) + lo.delta(m), atol=1e-5)
    untouched = "layers.0.mlp.down_proj"
    assert torch.equal(new.weight(untouched), store.weight(untouched))
    assert (tmp_path / "merged" / "config.json").exists() and rep["max_realised_rel_err"] < 1e-5


def test_operators(store):
    import torch
    from lora_code.primitives import ops
    mods = ["layers.0.mlp.gate_proj", "layers.1.mlp.down_proj"]
    lo = _rand_lora(store, mods, r=8, s=0.02)
    s = ops.scale(lo, 0.5)
    for m in mods:
        assert torch.allclose(s.delta(m), 0.5 * lo.delta(m), atol=1e-7)
    t = ops.truncate_rank(lo, 3)
    for m in mods:
        assert t.rank(m) == 3 and torch.linalg.matrix_rank(t.delta(m)) <= 3
        U, S, V = lo.compact(m)
        assert torch.allclose(t.delta(m), (U[:, :3] * S[:3]) @ V[:, :3].T, atol=1e-6)
    i = ops.interpolate(lo, s, 0.5)
    assert torch.allclose(i.delta(mods[0]), 0.75 * lo.delta(mods[0]), atol=1e-6)
    pr = ops.relchange_prune(lo, store, frac=0.05)
    assert pr.rank(mods[0]) <= lo.rank(mods[0])
    intr = ops.intruder_scale(lo, store, lam=0.0, eps=1.01, k0=4)   # every direction is an "intruder" -> zeroed
    assert all(m not in intr.factors or float(intr.delta(m).norm()) < 1e-6 for m in mods)
    o = ops.orthogonalize(lo, [s])                                   # same column space -> removed
    assert float(o.delta(mods[0]).norm()) < 1e-4 * float(lo.delta(mods[0]).norm()) + 1e-8
    assert not hasattr(ops, "regmean_preserve")                      # removed with the retention domains (2026-11-22)
    ti = ops.ties([lo, lo], density=1.0, cap=64)
    assert torch.allclose(ti.delta(mods[0]), lo.delta(mods[0]), atol=1e-5)


def test_train_restriction_sgd_acts(store):
    import torch
    from lora_code.primitives.train import HFModel, TrainData
    td = TrainData(_examples(8))
    with pytest.raises(PermissionError):
        td.check(["ex/1", "signal/7"])
    hf = HFModel(store.model_dir, store, device="cpu", dtype="float32", max_len=128)
    ex = td.get(td.ids())
    mods = ["layers.0.mlp.down_proj", "layers.3.self_attn.v_proj", "layers.1.linear_attn.out_proj"]
    l0 = hf.loss(ex)
    lo, info = hf.sgd(ex, None, modules=mods, rank=4, steps=30, lr=3e-3, batch_size=4, checkpoints=[0.5, 1.0])
    l1 = hf.loss(ex, lora=lo)
    assert l1 < l0 - 1e-3, (l0, l1)
    assert info["steps_done"] == 30 and len(info["checkpoints"]) == 2 and set(lo.modules()) == set(mods)
    # the model is restored (no peft wrappers left) and hooks give the same loss as a merged copy
    assert "lora" not in "".join(n for n, _ in hf.model.named_modules())
    assert not hasattr(hf, "grad")                                   # sgd is only for the raw LoRA
    a = hf.acts(td.texts(td.ids()[:4]), [mods[0]], k=2)[mods[0]]
    assert a["in_mean"].shape == (128,) and a["pcs"].shape == (2, 128)
    r = hf.act_ratio(td.texts(td.ids()[:2]), lo)
    assert set(r) == set(mods) and all(v >= 0 for v in r.values())
    # warm start: continuing from lo starts at lo's loss
    lo2, info2 = hf.sgd(ex, lo, rank=4, steps=1, lr=1e-9, batch_size=8)
    assert abs(info2["losses"][0] - hf.loss(ex, lora=lo, batch_size=8)) < 1e-3


def test_workspace_refuses_non_train_ids(store, tmp_path):
    """The sandbox API (Workspace tools) refuses ids outside this generation's train set and has no training /
    gradient tools (2026-11-20)."""
    from lora_code.primitives.train import HFModel, TrainData
    from lora_code.tools import Workspace
    hf = HFModel(store.model_dir, store, device="cpu", dtype="float32", max_len=64)
    ws = Workspace(mock=False, store=store, train=TrainData(_examples(4)), cfg={"lora": {"rank_cap": 16, "norm_cap_tau": 5}},
                   out_dir=tmp_path, hf=hf, budget={"evaluate": 2, "gpu_seconds": 600})
    ws.build_api()                                   # gpu seconds are charged by the rsicore ToolAPI
    assert not set(ws.api.names()) & {"sgd", "grad", "from_grad"}
    with pytest.raises(PermissionError):
        ws.api.call("acts", dict(module="layers.0.mlp.down_proj", ids=["heldout/3"]))
    ws.api.call("acts", dict(module="layers.0.mlp.down_proj", ids=ws.t_train_ids()[:2], k=2))
    assert ws.spent["gpu_seconds"] > 0
    s = ws.t_svd(module="layers.0.mlp.down_proj", k=2)
    new = ws.t_new_lora()["lora"]
    r = ws.t_rank1(lora=new, module="layers.0.mlp.down_proj", u=f"{s['handle']}.u0", v=f"{s['handle']}.v1", s=0.1)
    assert ws._get(r["lora"]).rank("layers.0.mlp.down_proj") == 1
    rep = ws.t_inspect(lora=r["lora"])
    assert rep["check_ok"] and rep["max_rank"] == 1


def test_diagnostics_real(store):
    from lora_code import diagnostics as D
    from lora_code.primitives.train import HFModel
    hf = HFModel(store.model_dir, store, device="cpu", dtype="float32", max_len=64)
    lo = _rand_lora(store, ["layers.0.mlp.down_proj", "layers.3.self_attn.q_proj"], r=4, s=0.02)
    diag = D.finalize(D.compute(store, [(0, lo)], hf=hf, target_texts=["a b c", "d e f"], svd_k=4, intruder_k=4))
    rows = {r["module"]: r for r in diag["rows"]}
    e = rows["layers.0.mlp.down_proj"]["stack"][0]
    assert e["rel_norm"] > 0 and 1 <= e["eff_rank"] <= 4 and 0 <= e["intruder"] <= 1
    assert e["relchange_q"]["q99"] >= e["relchange_q"]["q50"] and e["act_ratio_target"] > 0
    assert "act_ratio_retention" not in e and "act_ratio_ratio" not in e      # no retention texts (2026-11-22)
    assert not rows["layers.1.mlp.up_proj"]["stack"] and len(diag["summary"]) == store.num_layers
    assert "most-changed modules" in D.render(diag)
