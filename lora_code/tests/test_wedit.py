"""CPU tests of lora_code/primitives/wedit.py (dense weight edits) on a tiny fake safetensors checkpoint
(no transformers needed). Skipped when torch / safetensors are not importable.

  python3 -I -m pytest -q lora_code/tests/test_wedit.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

MISSING = [m for m in ("torch", "safetensors") if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(bool(MISSING), reason=f"missing {MISSING}")

P = "model.language_model."
DOWN, UP, Q = "layers.0.mlp.down_proj", "layers.0.mlp.up_proj", "layers.1.self_attn.q_proj"
SHAPES = {DOWN: (8, 16), UP: (16, 8), Q: (8, 8)}
CONFIG = {"text_config": {"layer_types": ["full_attention", "full_attention"], "num_hidden_layers": 2}}


def _tensors(seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    t = {P + m + ".weight": torch.randn(*s, generator=g).to(torch.bfloat16) for m, s in SHAPES.items()}
    t[P + "embed_tokens.weight"] = torch.randn(32, 8, generator=g).to(torch.bfloat16)
    return t


def _write_ckpt(d: Path, sharded: bool, seed=0) -> Path:
    from safetensors.torch import save_file
    d.mkdir(parents=True, exist_ok=True)
    t = _tensors(seed)
    if sharded:
        s1 = {k: v for k, v in t.items() if ".layers.0." in k}
        s2 = {k: v for k, v in t.items() if k not in s1}
        names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        save_file(s1, str(d / names[0]), metadata={"format": "pt"})
        save_file(s2, str(d / names[1]), metadata={"format": "pt"})
        wm = {k: names[0] for k in s1} | {k: names[1] for k in s2}
        json.dump({"metadata": {}, "weight_map": wm}, open(d / "model.safetensors.index.json", "w"))
    else:
        save_file(t, str(d / "model.safetensors"), metadata={"format": "pt"})
    json.dump(CONFIG, open(d / "config.json", "w"))
    (d / "tokenizer.json").write_text('{"fake": true}')
    (d / "chat_template.jinja").write_text("{{ messages }}")
    return d


@pytest.fixture
def ckpt(tmp_path):
    return _write_ckpt(tmp_path / "m0", sharded=True)


@pytest.fixture
def store(ckpt):
    from lora_code.primitives.weights import WeightStore
    return WeightStore(ckpt)


def _read(d: Path, key: str):
    from safetensors import safe_open
    idx = d / "model.safetensors.index.json"
    shard = json.load(open(idx))["weight_map"][key] if idx.exists() else "model.safetensors"
    with safe_open(str(d / shard), "pt") as f:
        return f.get_tensor(key)


# ============================================================================ op math
def test_each_op_matches_manual():
    import torch
    from lora_code.primitives.wedit import WeightEdit
    torch.manual_seed(0)
    W, Wb = torch.randn(6, 5), torch.randn(6, 5)
    W0 = W.clone()

    e = WeightEdit({DOWN: [{"op": "scale", "s": 0.5}]})
    assert torch.allclose(e.apply(DOWN, W), 0.5 * W)

    e = WeightEdit({DOWN: [{"op": "rows", "idx": [1, 4], "f": [2.0, -1.0]}]})
    ref = W.clone(); ref[1] *= 2.0; ref[4] *= -1.0
    assert torch.allclose(e.apply(DOWN, W), ref)

    e = WeightEdit({DOWN: [{"op": "rows", "idx": None, "f": 3.0}]})
    assert torch.allclose(e.apply(DOWN, W), 3.0 * W)

    f = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    e = WeightEdit({DOWN: [{"op": "rows", "idx": None, "f": f}]})
    assert torch.allclose(e.apply(DOWN, W), W * torch.tensor(f)[:, None])

    e = WeightEdit({DOWN: [{"op": "cols", "idx": [0, 3], "f": 0.0}]})
    ref = W.clone(); ref[:, [0, 3]] = 0.0
    assert torch.allclose(e.apply(DOWN, W), ref)

    e = WeightEdit({DOWN: [{"op": "rows", "idx": [2, 2], "f": [2.0, 3.0]}]})   # repeated index compounds
    ref = W.clone(); ref[2] *= 6.0
    assert torch.allclose(e.apply(DOWN, W), ref)

    A, B = torch.randn(2, 5), torch.randn(6, 2)
    e = WeightEdit({DOWN: [{"op": "lowrank", "A": A, "B": B}]})
    assert torch.allclose(e.apply(DOWN, W), W + B @ A, atol=1e-6)

    e = WeightEdit({DOWN: [{"op": "toward_base", "alpha": 0.25}]})
    assert torch.allclose(e.apply(DOWN, W, Wb), W + 0.25 * (Wb - W), atol=1e-6)
    with pytest.raises(ValueError):
        e.apply(DOWN, W)

    assert torch.equal(W, W0)                                       # input never modified
    assert WeightEdit().apply(DOWN, W.to(torch.bfloat16)).dtype == torch.float32
    with pytest.raises(ValueError):
        WeightEdit({DOWN: [{"op": "rows", "idx": [9], "f": 1.0}]}).apply(DOWN, W)
    with pytest.raises(ValueError):
        WeightEdit({DOWN: [{"op": "nope"}]})


def test_op_order_matters():
    import torch
    from lora_code.primitives.wedit import WeightEdit
    torch.manual_seed(1)
    W = torch.randn(4, 3)
    A, B = torch.randn(1, 3), torch.randn(4, 1)
    e1 = WeightEdit({DOWN: [{"op": "scale", "s": 2.0}, {"op": "lowrank", "A": A, "B": B}]})
    e2 = WeightEdit({DOWN: [{"op": "lowrank", "A": A, "B": B}, {"op": "scale", "s": 2.0}]})
    assert torch.allclose(e1.apply(DOWN, W), 2 * W + B @ A, atol=1e-6)
    assert torch.allclose(e2.apply(DOWN, W), 2 * (W + B @ A), atol=1e-6)
    assert e1.digest() != e2.digest()


def test_basics_copy_summary_roundtrip(tmp_path):
    import torch
    from lora_code.primitives.wedit import WeightEdit
    A, B = torch.randn(3, 16), torch.randn(8, 3)
    e = WeightEdit({Q: [{"op": "toward_base", "alpha": 0.1}],
                    DOWN: [{"op": "rows", "idx": [1, 2], "f": [1.5, 0.5]}, {"op": "lowrank", "A": A, "B": B},
                           {"op": "cols", "idx": None, "f": 2}],
                    "layers.10.mlp.up_proj": [{"op": "scale", "s": 1}]}, meta={"note": "x"})
    assert e.kind == "weights" and e.max_rank() == 0
    assert e.modules() == [DOWN, Q, "layers.10.mlp.up_proj"]
    s = e.summary()
    json.dumps(s)
    assert s[DOWN][1] == {"op": "lowrank", "rank": 3}
    assert s[DOWN][0] == {"op": "rows", "idx": [1, 2], "f": [1.5, 0.5]}

    d = e.digest()
    assert len(d) == 16 and d == e.digest()
    c = e.copy()
    c.add(DOWN, {"op": "scale", "s": 0.9})
    c.ops[DOWN][1]["A"].add_(1.0)
    assert len(e.ops[DOWN]) == 3 and e.digest() == d and c.digest() != d

    e.save_raw(tmp_path / "e" / "edit.safetensors")
    r = WeightEdit.load_raw(tmp_path / "e" / "edit.safetensors")
    assert r.digest() == d and r.summary() == s and r.meta == {"note": "x"}
    assert torch.equal(r.ops[DOWN][1]["A"], A) and torch.equal(r.ops[DOWN][1]["B"], B)
    W = torch.randn(8, 16)
    assert torch.equal(r.apply(DOWN, W), e.apply(DOWN, W))

    WeightEdit().save_raw(tmp_path / "empty.safetensors")          # no tensors at all
    assert WeightEdit.load_raw(tmp_path / "empty.safetensors").modules() == []


def test_lora_to_edit_equals_lowrank_add():
    import torch
    from lora_code.primitives.lora import LoRA
    from lora_code.primitives.wedit import lora_to_edit
    lo = LoRA()
    lo.add(DOWN, torch.randn(2, 16), torch.randn(8, 2))
    lo.add(Q, torch.randn(1, 8), torch.randn(8, 1))
    e = lora_to_edit(lo)
    assert e.modules() == lo.modules()
    for m in lo.modules():
        W = torch.randn(*SHAPES[m])
        assert torch.allclose(e.apply(m, W), W + lo.delta(m), atol=1e-6)


# ============================================================================ check_wedit
def test_check_wedit_pass_and_failures(store, tmp_path):
    import torch
    from lora_code.primitives.weights import WeightStore
    from lora_code.primitives.wedit import WeightEdit, check_wedit
    base = WeightStore(_write_ckpt(tmp_path / "base", sharded=False, seed=1))

    ok, rep = check_wedit(WeightEdit(), store, tau=0.1)
    assert ok and rep["empty"]

    e = WeightEdit({DOWN: [{"op": "scale", "s": 1.05}], Q: [{"op": "cols", "idx": [0], "f": 1.01}]})
    ok, rep = check_wedit(e, store, tau=0.1)
    assert ok, rep
    assert abs(rep["modules"][DOWN]["rel_norm"] - 0.05) < 1e-4 and rep["modules"][DOWN]["ops"] == 1
    W = store.weight(Q)
    assert abs(rep["modules"][Q]["rel_norm"] - 0.01 * float(W[:, 0].norm() / W.norm())) < 1e-5

    def fails(edit, needle, **kw):
        ok, rep = check_wedit(edit, kw.pop("st", store), tau=kw.pop("tau", 0.1), **kw)
        assert not ok and rep["n_errors"] >= 1 and any(needle in x for x in rep["errors"]), rep
        return rep

    fails(WeightEdit({"layers.7.mlp.down_proj": [{"op": "scale", "s": 1.0}]}), "not in model")
    fails(WeightEdit({DOWN: [{"op": "scale", "s": 1.5}]}), "> tau")
    fails(WeightEdit({DOWN: [{"op": "rows", "idx": [8], "f": 1.0}]}), "out of range")
    fails(WeightEdit({DOWN: [{"op": "cols", "idx": [-1], "f": 1.0}]}), "out of range")
    fails(WeightEdit({DOWN: [{"op": "rows", "idx": [0, 1], "f": [1.0]}]}), "factors for 2 indices")
    fails(WeightEdit({DOWN: [{"op": "rows", "idx": None, "f": [1.0, 1.0]}]}), "factors for 8 channels")
    fails(WeightEdit({DOWN: [{"op": "lowrank", "A": torch.zeros(1, 8), "B": torch.zeros(8, 1)}]}), "bad shapes")
    fails(WeightEdit({DOWN: [{"op": "scale", "s": float("inf")}]}), "non-finite", tau=1e9)
    fails(WeightEdit({DOWN: [{"op": "toward_base", "alpha": 0.01}]}), "base_store")
    fails(WeightEdit({DOWN: [{"op": "scale", "s": 1.0}], Q: [{"op": "scale", "s": 1.0}]}), "max_modules",
          max_modules=1)

    ok, rep = check_wedit(WeightEdit({DOWN: [{"op": "toward_base", "alpha": 0.01}]}), store, base, tau=1.0)
    Wb, W = base.weight(DOWN), store.weight(DOWN)
    assert ok and abs(rep["modules"][DOWN]["rel_norm"] - 0.01 * float((Wb - W).norm() / W.norm())) < 1e-5


# ============================================================================ edit_into_model
def test_edit_into_model_sharded(ckpt, store, tmp_path):
    import torch
    from lora_code.primitives.weights import WeightStore
    from lora_code.primitives.wedit import WeightEdit, edit_into_model
    base = WeightStore(_write_ckpt(tmp_path / "base", sharded=False, seed=1))
    A, B = torch.randn(2, 16) * 0.1, torch.randn(8, 2) * 0.1
    e = WeightEdit({DOWN: [{"op": "rows", "idx": [0, 3], "f": [0.5, 2.0]}, {"op": "lowrank", "A": A, "B": B},
                           {"op": "toward_base", "alpha": 0.2}]})
    out = tmp_path / "m1"
    rep = edit_into_model(e, ckpt, out, store=store, base_store=base)

    k = P + DOWN + ".weight"
    W, Wb = _read(ckpt, k).float(), base.weight(DOWN)
    expect = e.apply(DOWN, W, Wb).to(torch.bfloat16)
    got = _read(out, k)
    assert got.dtype == torch.bfloat16 and torch.equal(got, expect)
    assert torch.equal(_read(out, P + UP + ".weight"), _read(ckpt, P + UP + ".weight"))   # same shard, untouched

    s1, s2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    assert rep["shards_rewritten"] == [s1]
    assert os.stat(out / s2).st_ino == os.stat(ckpt / s2).st_ino                          # hard link
    assert os.stat(out / s1).st_ino != os.stat(ckpt / s1).st_ino
    for name in ("config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors.index.json"):
        assert (out / name).read_bytes() == (ckpt / name).read_bytes()
    assert not out.with_name("m1.partial").exists()

    rel = float((got.float() - W).norm() / W.norm())
    assert abs(rep["modules"][k]["rel_change"] - rel) < 1e-6 and rep["modules"][k]["short"] == DOWN
    assert json.load(open(out / "edit_report.json"))["modules"][k]["rel_change"] == pytest.approx(rel)
    assert rep["edit_digest"] == e.digest()
    WeightStore(out)                                                     # output is a loadable checkpoint

    with pytest.raises(ValueError):
        edit_into_model(e, ckpt, tmp_path / "m2", store=store)             # toward_base without base_store


def test_edit_into_model_single_file_no_link(tmp_path):
    import torch
    from lora_code.primitives.wedit import WeightEdit, edit_into_model
    src = _write_ckpt(tmp_path / "s", sharded=False)
    out = tmp_path / "o"
    (out).mkdir()
    (out / "stale.txt").write_text("old")                                 # existing out_dir is replaced
    e = WeightEdit({Q: [{"op": "scale", "s": 0.9}]})
    rep = edit_into_model(e, src, out, link_unchanged=False)
    k = P + Q + ".weight"
    assert torch.equal(_read(out, k), (0.9 * _read(src, k).float()).to(torch.bfloat16))
    assert torch.equal(_read(out, P + "embed_tokens.weight"), _read(src, P + "embed_tokens.weight"))
    assert rep["shards_rewritten"] == ["model.safetensors"] and not (out / "stale.txt").exists()


def test_merge_into_model_regression(ckpt, store, tmp_path):
    import torch
    from lora_code.primitives.lora import LoRA, merge_into_model
    lo = LoRA()
    lo.add(Q, torch.randn(2, 8) * 0.1, torch.randn(8, 2) * 0.1)
    out = tmp_path / "merged"
    rep = merge_into_model(lo, ckpt, out, store=store)
    k = P + Q + ".weight"
    W = _read(ckpt, k).float()
    assert torch.equal(_read(out, k), (W + lo.delta(Q)).to(torch.bfloat16))
    s1, s2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    assert rep["shards_rewritten"] == [s2]
    assert os.stat(out / s1).st_ino == os.stat(ckpt / s1).st_ino
    assert set(rep) == {"src", "modules", "shards_rewritten", "max_realised_rel_err", "mean_realised_rel_err"}
    assert set(rep["modules"][k]) == {"dw_rel", "realised_rel_err"}
    assert json.load(open(out / "merge_report.json")) == json.loads(json.dumps(rep))
    assert (out / "tokenizer.json").exists() and not (out / "edit_report.json").exists()
