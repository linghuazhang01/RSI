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
# non-linear language-model tensors (2026-11-21, tensor catalogue / delta op): name -> (shape, dtype, mean)
NORM0, DT, ALOG, CONV = ("layers.0.input_layernorm.weight", "layers.0.linear_attn.dt_bias", "layers.0.linear_attn.A_log",
                         "layers.0.linear_attn.conv1d.weight")
EXTRA = {NORM0: ((8,), "bf16", 0.0), DT: ((4,), "bf16", 1.0), ALOG: ((4,), "f32", 1.5), CONV: ((6, 1, 4), "bf16", 0.0),
         "norm.weight": ((8,), "bf16", 0.0)}
CONFIG = {"text_config": {"layer_types": ["full_attention", "full_attention"], "num_hidden_layers": 2}}


def _tensors(seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    t = {P + m + ".weight": torch.randn(*s, generator=g).to(torch.bfloat16) for m, s in SHAPES.items()}
    t[P + "embed_tokens.weight"] = torch.randn(32, 8, generator=g).to(torch.bfloat16)
    for n, (shape, dt, mean) in EXTRA.items():
        x = torch.randn(*shape, generator=g) * 0.1 + mean
        t[P + n] = x.to(torch.bfloat16 if dt == "bf16" else torch.float32)
    t["lm_head.weight"] = torch.randn(32, 8, generator=g).to(torch.bfloat16)
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
    assert c.ops[DOWN][1]["A"] is e.ops[DOWN][1]["A"]          # op tensors are shared, never modified (2026-11-21)
    c.add(DOWN, {"op": "scale", "s": 0.9})
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


# ============================================================================ tensor catalogue / delta op (2026-11-21)
def test_tensor_catalogue(store, ckpt):
    import torch
    names = store.tensor_names()
    assert names[0] == "embed_tokens.weight" and names[-2:] == ["lm_head.weight", "norm.weight"]
    assert {DOWN + ".weight", NORM0, DT, ALOG, CONV} <= set(names) and store.modules() == sorted(SHAPES, key=lambda m: (
        int(m.split(".")[1]), m))                                           # modules() stays linear-only
    assert store.tensor_shape(CONV) == (6, 1, 4) and store.tensor_shape(DOWN) == (8, 16) == store.shape(DOWN)
    assert store.tensor_dtype(ALOG) == "float32" and store.tensor_dtype(DT) == "bfloat16"
    assert store.tensor_key(DOWN) == P + DOWN + ".weight" == store.tensor_key(DOWN + ".weight")
    assert store.tensor_key("lm_head.weight") == "lm_head.weight" and store.tensor_key("norm.weight") == P + "norm.weight"
    assert store.edit_name(DOWN + ".weight") == DOWN and store.edit_name(DT) == DT
    assert store.editable(DT) and store.editable(DOWN)
    assert not any(store.editable(n) for n in ("embed_tokens.weight", "norm.weight", "lm_head.weight"))
    inf = store.tensor_info(NORM0)
    assert inf == {"name": NORM0, "shape": [8], "dtype": "bfloat16", "editable": True, "layer": 0}
    path, key = store.tensor_source(DT)
    assert Path(path).is_file() and key == P + DT
    assert torch.equal(store.tensor(DT), _read(ckpt, P + DT).float())
    assert torch.equal(store.tensor(DOWN), store.weight(DOWN))
    with pytest.raises(KeyError):
        store.tensor_key("layers.0.mlp.nope.weight")


def test_delta_op_math_digest_save_load(tmp_path):
    import torch
    from lora_code.primitives.wedit import WeightEdit
    torch.manual_seed(3)
    D1, D3 = torch.randn(4), torch.randn(6, 1, 4)
    e = WeightEdit({DT: [{"op": "delta", "D": D1}], CONV: [{"op": "delta", "D": D3}],
                    DOWN: [{"op": "scale", "s": 1.01}, {"op": "delta", "D": torch.ones(8, 16) * 0.01}]})
    assert e.modules() == [CONV, DT, DOWN]                               # by layer, then name
    W = torch.randn(4)
    assert torch.allclose(e.apply(DT, W), W + D1) and torch.allclose(e.apply(CONV, torch.zeros(6, 1, 4)), D3)
    Wd = torch.randn(8, 16)
    assert torch.allclose(e.apply(DOWN, Wd), 1.01 * Wd + 0.01, atol=1e-6)
    with pytest.raises(ValueError):
        e.apply(DT, torch.randn(5))
    assert e.summary()[DT] == [{"op": "delta", "shape": [4], "norm": round(float(D1.norm()), 6)}]
    d = e.digest()
    assert d == e.digest() and len(e._hash_cache) >= 3                  # tensor hashes cached
    e.ops[DT][0]["D"].add_(1.0)                                          # an in-place change invalidates the cache
    assert e.digest() != d
    d = e.digest()
    e.save_raw(tmp_path / "d.safetensors")
    r = WeightEdit.load_raw(tmp_path / "d.safetensors")
    assert r.digest() == d and torch.equal(r.ops[CONV][0]["D"], D3) and r.summary() == e.summary()
    c = e.copy()
    assert c.digest() == d and c.ops[DT][0]["D"] is e.ops[DT][0]["D"]   # shared (no ~356 MB clone per copy)
    assert c.ops[DT] is not e.ops[DT] and c.ops[DT][0] is not e.ops[DT][0]
    with pytest.raises(ValueError):
        WeightEdit({DT: [{"op": "delta", "D": [1.0, 2.0]}]})


def test_check_wedit_tensors(store):
    import torch
    from lora_code.primitives.wedit import WeightEdit, check_wedit, rescale_ratio

    def fails(edit, needle, **kw):
        ok, rep = check_wedit(edit, store, tau=kw.pop("tau", 0.05), **kw)
        assert not ok and any(needle in x for x in rep["errors"]), rep
    W2, W1, Wn = store.weight(DOWN), store.tensor(DT), store.tensor(NORM0)
    D2 = torch.zeros_like(W2)
    D2[0] = 0.04 * W2.norm() / W2[0].norm() * W2[0]                      # one row: not a rescale
    ok, rep = check_wedit(WeightEdit({DOWN: [{"op": "delta", "D": D2}],
                                      DT: [{"op": "delta", "D": torch.tensor([0.3, -0.3, 0.2, 0.0])}],
                                      NORM0: [{"op": "delta", "D": torch.linspace(-0.6, 0.6, 8)}]}), store, tau=0.05)
    assert ok, rep
    assert rep["modules"][DOWN]["rel_norm"] == pytest.approx(0.04, rel=1e-3) and "rms_cap" not in rep["modules"][DOWN]
    assert rep["modules"][NORM0]["rms_cap"] == pytest.approx(0.5)        # zero-centred norm: absolute cap 0.5 * 1
    assert rep["modules"][DT]["rms_cap"] == pytest.approx(0.5 * max(float(W1.norm()) / 2, 1.0))
    fails(WeightEdit({DOWN: [{"op": "delta", "D": 2 * D2}]}), "> tau 0.05")
    fails(WeightEdit({NORM0: [{"op": "delta", "D": torch.full((8,), 0.6) * torch.tensor([1, -1] * 4)}]}),
          "(1-D tensor)")
    fails(WeightEdit({DOWN: [{"op": "delta", "D": 0.03 * W2}]}), "uniform rescale")        # rescale refused ...
    fails(WeightEdit({DT: [{"op": "delta", "D": 0.1 * W1}]}), "uniform rescale")
    assert check_wedit(WeightEdit({DOWN: [{"op": "scale", "s": 1.03}]}), store, tau=0.05)[0]   # ... w_scale is not
    assert rescale_ratio(0.5 * Wn, Wn) == pytest.approx(0.0, abs=1e-6) and rescale_ratio(torch.zeros(8), Wn) is None
    for n in ("norm.weight", "lm_head.weight", "embed_tokens.weight"):
        shape = store.tensor_shape(n)
        fails(WeightEdit({n: [{"op": "delta", "D": torch.zeros(shape)}]}), "never editable")
    fails(WeightEdit({DT: [{"op": "delta", "D": torch.zeros(3)}]}), "shape (3,) != (4,)")
    fails(WeightEdit({DT: [{"op": "rows", "idx": [0], "f": 1.1}]}), "only for 2-D weights")
    fails(WeightEdit({"layers.0.linear_attn.nope": [{"op": "delta", "D": torch.zeros(4)}]}), "not in model")
    fails(WeightEdit({DT: [{"op": "delta", "D": torch.tensor([float("nan"), 0, 0, 0])}]}), "non-finite", tau=1e9)


def test_edit_into_model_nonlinear(ckpt, store, tmp_path):
    import torch
    from lora_code.primitives.weights import WeightStore
    from lora_code.primitives.wedit import WeightEdit, edit_into_model
    D = {DT: torch.tensor([0.1, -0.2, 0.3, 0.0]), ALOG: torch.tensor([0.05, 0.0, -0.05, 0.1]),
         NORM0: torch.linspace(-0.1, 0.1, 8), CONV: torch.randn(6, 1, 4) * 0.01}
    e = WeightEdit({n: [{"op": "delta", "D": d}] for n, d in D.items()})
    e.add(DOWN, {"op": "scale", "s": 1.01})
    out = tmp_path / "m_nl"
    rep = edit_into_model(e, ckpt, out, store=store)
    for n, d in D.items():
        src = _read(ckpt, P + n)
        got = _read(out, P + n)
        assert got.dtype == src.dtype and torch.equal(got, (src.float() + d).to(src.dtype)), n
        assert rep["modules"][P + n]["short"] == n
    assert _read(out, P + ALOG).dtype == torch.float32                  # stored dtype kept
    for k in (P + UP + ".weight", P + "norm.weight", "lm_head.weight", P + "embed_tokens.weight"):
        assert torch.equal(_read(out, k), _read(ckpt, k))
    assert torch.equal(_read(out, P + DOWN + ".weight"), (1.01 * _read(ckpt, P + DOWN + ".weight").float()).to(
        torch.bfloat16))
    assert rep["edit_digest"] == e.digest()
    s2 = WeightStore(out)
    assert torch.equal(s2.tensor(DT), (store.tensor(DT) + D[DT]).to(torch.bfloat16).float())


def _has_qwen35():
    try:
        from transformers import Qwen3_5Config  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_qwen35(), reason="transformers without Qwen3_5 (needs transformers >= 5)")
def test_nonlinear_export_and_hf_applied_tiny_qwen35(tmp_path):
    """Real Qwen3.5 tensors (tiny random model): catalogue names, a delta op on dt_bias / a zero-centred norm /
    conv1d written by edit_into_model and loaded back by transformers, and HFModel.applied swapping them in."""
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from lora_code.primitives.train import HFModel
    from lora_code.primitives.weights import WeightStore
    from lora_code.primitives.wedit import WeightEdit, edit_into_model
    tc = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
              linear_num_key_heads=2, linear_num_value_heads=4, max_position_embeddings=512)
    vc = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=2,
              spatial_merge_size=1, temporal_patch_size=1)
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=tc, vision_config=vc))
    src = tmp_path / "tiny"
    model.save_pretrained(src)
    store = WeightStore(src)
    names = set(store.tensor_names())
    for n in ("layers.0.linear_attn.dt_bias", "layers.0.linear_attn.A_log", "layers.0.linear_attn.conv1d.weight",
              "layers.0.linear_attn.norm.weight", "layers.3.self_attn.q_norm.weight", "layers.0.input_layernorm.weight",
              "norm.weight", "embed_tokens.weight", "lm_head.weight", "layers.0.mlp.down_proj.weight"):
        assert n in names, n
    assert not any("visual" in n or "bias" in n.replace("dt_bias", "") for n in names)       # no linear biases
    assert float(store.tensor("layers.0.input_layernorm.weight").abs().max()) == 0.0         # zero-centred RMSNorm
    dt, nm, cv = "layers.0.linear_attn.dt_bias", "layers.0.input_layernorm.weight", "layers.0.linear_attn.conv1d.weight"
    D = {dt: torch.linspace(-0.2, 0.2, 4), nm: torch.linspace(-0.1, 0.1, 64), cv: torch.randn(*store.tensor_shape(cv)) * 0.01}
    e = WeightEdit({n: [{"op": "delta", "D": d}] for n, d in D.items()})
    out = tmp_path / "edited"
    edit_into_model(e, src, out, store=store)
    m2 = Qwen3_5ForConditionalGeneration.from_pretrained(out)
    for n, d in D.items():
        p = dict(m2.named_parameters())[store.tensor_key(n)]
        assert torch.allclose(p.detach().float(), store.tensor(n) + d, atol=1e-6), n
    hf = HFModel(src, store, device="cpu", dtype="float32")
    hf._model = model
    before = {n: hf.param(n).detach().clone() for n in D}
    with hf.applied(e):
        for n, d in D.items():
            assert torch.allclose(hf.param(n).detach(), before[n] + d, atol=1e-6)
    for n in D:
        assert torch.equal(hf.param(n).detach(), before[n])
