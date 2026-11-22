"""merge.mode adapter (2026-11-22): CPU tests of the pieces that keep accepted LoRAs as one cumulative adapter
instead of merging them into bf16 weights. Needs torch + safetensors only (a hand-written 2-layer checkpoint, no
transformers / peft).

  eval/.venv/bin/python -I -m pytest -q lora_code/tests/test_adapter_mode.py

- compose_cumulative: exact rank-concatenation, inputs untouched, SVD re-compression above the budget.
- OverlayStore: weight / norm / svd / tensor of W_0 + B_C A_C, untouched modules and metadata from the base.
- Evaluator: a delta is registered as C (+) delta (one composition point), a full adapter as is; mock quality is the
  delta's relative to the parent and a full adapter keeps its registry entry.
- Workspace: store overlay, composed train-loss adapter, current-model parent; mock unaffected.
- common: merge_mode validation and the main servers' --max-lora-rank.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

MISSING = [m for m in ("torch", "safetensors") if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(bool(MISSING), reason=f"missing {MISSING}")

MODS = ["layers.0.mlp.up_proj", "layers.0.mlp.down_proj", "layers.1.mlp.up_proj", "layers.1.mlp.gate_proj"]
SHAPES = {"up_proj": (48, 32), "gate_proj": (48, 32), "down_proj": (32, 48)}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    import torch
    from safetensors.torch import save_file
    d = tmp_path_factory.mktemp("tiny_ckpt")
    torch.manual_seed(0)
    t = {f"model.language_model.{m}.weight": (0.02 * torch.randn(*SHAPES[m.rsplit('.', 1)[1]])).bfloat16()
         for m in MODS}
    t["model.language_model.norm.weight"] = torch.ones(32, dtype=torch.bfloat16)
    save_file(t, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "hidden_size": 32}))
    return d


@pytest.fixture(scope="module")
def store(ckpt):
    from lora_code.primitives.weights import WeightStore
    return WeightStore(ckpt, cache_dir=ckpt / "_cache")


def _lora(store, mods, r=4, s=0.05, seed=1):
    import torch
    from lora_code.primitives.lora import LoRA
    g = torch.Generator().manual_seed(seed)
    f = {}
    for m in mods:
        o, i = store.shape(m)
        f[m] = (s * torch.randn(r, i, generator=g), s * torch.randn(o, r, generator=g))
    return LoRA(f)


def test_compose_exact_and_pure(store):
    import torch
    from lora_code.primitives.lora import compose_cumulative
    C = _lora(store, MODS[:2], r=4, seed=1)
    D = _lora(store, MODS[1:3], r=3, seed=2)
    c0, d0 = C.digest(), D.digest()
    X = compose_cumulative(C, D, budget=64)
    assert C.digest() == c0 and D.digest() == d0                     # inputs never mutated
    for m in MODS[:3]:
        want = (C.delta(m) if m in C.factors else 0) + (D.delta(m) if m in D.factors else 0)
        assert torch.allclose(X.delta(m), want, atol=1e-7)
    assert X.rank(MODS[1]) == 7 and X.meta["cumulative"] and "recompressed" not in X.meta
    assert len(X.meta["components"]) == 1
    Y = compose_cumulative(None, D)                                  # C_0 = M_0: the delta itself
    assert all(torch.allclose(Y.delta(m), D.delta(m)) for m in D.modules())


def test_compose_recompress_budget(store):
    import torch
    from lora_code.primitives.lora import compose_cumulative
    C = _lora(store, [MODS[0]], r=6, seed=3)
    D = _lora(store, [MODS[0]], r=6, seed=4)
    exact = C.delta(MODS[0]) + D.delta(MODS[0])
    X = compose_cumulative(C, D, budget=8)                           # rank 12 > 8: SVD-truncated to 8
    assert X.rank(MODS[0]) == 8 and X.meta["recompressed"]["modules"] == 1
    U, S, V = torch.linalg.svd(exact)
    best8 = (U[:, :8] * S[:8]) @ V[:8]
    assert torch.allclose(X.delta(MODS[0]), best8, atol=1e-6)        # the optimal rank-8 approximation
    assert 0 < X.meta["recompressed"]["max_dropped_energy"] < 1
    lossless = compose_cumulative(C, ops_scale(C, -1.0), budget=4)   # C - C = 0: every direction dropped (tol)
    assert lossless.rank(MODS[0]) == 0 or float(lossless.delta(MODS[0]).abs().max()) < 1e-6


def ops_scale(lo, s):
    from lora_code.primitives import ops
    return ops.scale(lo, s)


def test_overlay_store(store, ckpt):
    import torch
    from lora_code.primitives.weights import OverlayStore
    C = _lora(store, MODS[:2], r=4, seed=5)
    ov = OverlayStore(store, C)
    for m in MODS[:2]:
        assert torch.allclose(ov.weight(m), store.weight(m) + C.delta(m), atol=1e-6)
        assert abs(ov.norm(m) - float((store.weight(m) + C.delta(m)).norm())) < 1e-5
        U, S, V = ov.svd(m, 4)
        S_ref = torch.linalg.svdvals(store.weight(m) + C.delta(m))[:4]
        assert torch.allclose(S, S_ref, atol=1e-5)
        assert ov.is_overlaid(m) and ov.is_overlaid(m + ".weight")
        assert torch.allclose(ov.tensor(m + ".weight"), ov.weight(m))
    u = MODS[3]
    assert torch.equal(ov.weight(u), store.weight(u)) and not ov.is_overlaid(u)
    assert torch.equal(ov.tensor("norm.weight"), store.tensor("norm.weight"))
    assert ov.model_dir == store.model_dir and ov.prefix == store.prefix and ov.modules() == store.modules()
    assert ov.fingerprint != store.fingerprint
    # the base store's own SVD cache is untouched (W_0's directions stay W_0's)
    _, S0, _ = store.svd(MODS[0], 4)
    assert torch.allclose(S0, torch.linalg.svdvals(store.weight(MODS[0]))[:4], atol=1e-5)


def _ev(tmp_path, store, mock, registry=None, parent_quality=None):
    from lora_code.evaluator import Evaluator

    class Srv:
        endpoint = "http://127.0.0.1:1"
        loras = {}

        def load_lora(self, *a):
            pass

        def unload_lora(self, *a):
            pass

    s = Srv()
    rc = SimpleNamespace(domain="math", seed=42)    # the mock quality's target key = the run's domain
    return Evaluator(rc, {"lora": {"save_dtype": "float32"}}, out_dir=tmp_path, endpoint=s.endpoint, served="m0",
                     server=s, ids={}, target_dataset="simple_math_pool", gate_dataset="simple_math_eval", store=store,
                     mock=mock, registry=registry, parent_quality=parent_quality)


def test_evaluator_composes_deltas_real(store, tmp_path):
    import torch
    from lora_code.primitives.lora import load_peft
    C = _lora(store, MODS[:2], r=4, seed=6)
    D = _lora(store, MODS[1:3], r=2, seed=7)
    ev = _ev(tmp_path, store, mock=False)
    ev.set_parent(C, "C1")
    ev.set_base(C, budget=64)
    pc = ev.register(D, "cand")                                      # a delta: served as C (+) D
    got, _ = load_peft(pc)
    for m in MODS[:3]:
        want = (C.delta(m) if m in C.factors else 0) + (D.delta(m) if m in D.factors else 0)
        assert torch.allclose(got.delta(m), want, atol=1e-6)
    assert ev.compose(D).rank(MODS[1]) == 6 and len(got.modules()) == 3
    par, _ = load_peft(tmp_path / "adapters" / "C1")                 # the parent itself: full, not composed
    assert set(par.modules()) == set(MODS[:2])
    for m in MODS[:2]:
        assert torch.allclose(par.delta(m), C.delta(m), atol=1e-6)
    ev2 = _ev(tmp_path / "no_base", store, mock=False)               # no base (merge mode): registered as is
    p2, _ = load_peft(ev2.register(D, "cand"))
    assert set(p2.modules()) == set(D.modules()) and ev2.compose(D) is D
    ev2.set_base(None, budget=1)                                     # empty C: the budget still applies
    assert ev2.compose(D).rank(MODS[1]) == 1 and ev2.compose(D).meta["recompressed"]["modules"] == 2


def test_evaluator_mock_quality(tmp_path):
    from lora_code.mock import MockLoRA, MockStore
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"m0": 0.6, "C1": 0.65}))
    C = MockLoRA({"layers.0.mlp.up_proj": {"rank": 8, "rel": 0.01}}, gain=0.05)
    D = MockLoRA({"layers.0.mlp.up_proj": {"rank": 4, "rel": 0.01}, "layers.1.mlp.up_proj": {"rank": 4, "rel": 0.01}},
                 gain=0.02, damage=0.001)
    ev = _ev(tmp_path, MockStore(), mock=True, registry=reg, parent_quality=0.65)
    ev.set_parent(C, "C1")                                           # full: the registry entry is kept
    assert json.loads(reg.read_text())["C1"] == 0.65
    ev.set_base(C, budget=256)
    ev.register(D, "cand")
    q = json.loads(reg.read_text())["cand"]
    assert q == D.quality(0.65, salt="cand", target="math")         # the delta's quality vs the current model
    S = ev.compose(D)
    assert S.modules_["layers.0.mlp.up_proj"]["rank"] == 12 and abs(S.gain - 0.07) < 1e-9
    assert json.loads((tmp_path / "adapters" / "cand.json").read_text())["meta"]["cumulative"]


def test_workspace_adapter_views(store, tmp_path):
    import torch
    from lora_code.primitives.train import TrainData
    from lora_code.primitives.weights import OverlayStore
    from lora_code.tools import Workspace
    C = _lora(store, MODS[:2], r=4, seed=8)
    D = _lora(store, [MODS[1]], r=2, seed=9)
    train = TrainData({"t/0": {"messages": [{"role": "user", "content": "q"}], "target": "a", "source": "gold"}})
    ws = Workspace(mock=False, store=store, train=train, cfg={"lora": {"rank_cap": 64, "norm_cap_tau": 0.5}},
                   out_dir=tmp_path, base_lora=C, base_store=store)
    assert isinstance(ws.store, OverlayStore) and ws.store is ws.store          # cached per underlying store
    assert torch.allclose(ws.store.weight(MODS[0]), store.weight(MODS[0]) + C.delta(MODS[0]), atol=1e-6)
    assert ws.base_store is store
    assert ws._parent_lora_for_loss() is C
    L = ws._loss_lora(D)
    assert torch.allclose(L.delta(MODS[1]), C.delta(MODS[1]) + D.delta(MODS[1]), atol=1e-7)
    ok, rep = ws.check(D)                                            # tau relative to W_g = W_0 + B_C A_C
    wg = float((store.weight(MODS[1]) + C.delta(MODS[1])).norm())
    assert ok and abs(rep["modules"][MODS[1]]["rel_norm"] - D.fro_norm(MODS[1]) / wg) < 1e-6
    plain = Workspace(mock=False, store=store, train=train, cfg={"lora": {}}, out_dir=tmp_path / "p")
    assert plain.store is store and plain._parent_lora_for_loss() is None and plain._loss_lora(D) is D


def test_merge_mode_config():
    from lora_code.common import merge_mode, server_lora_rank
    assert merge_mode({}) == "merge" and server_lora_rank({}) == 64
    cfg = {"method": {"mode": "scratch"}, "merge": {"mode": "adapter"}}
    assert merge_mode(cfg) == "adapter" and server_lora_rank(cfg) == 256          # composites truncated to 256
    assert server_lora_rank(dict(cfg, merge={"mode": "adapter", "adapter_rank_budget": 300})) == 320
    assert server_lora_rank(dict(cfg, merge={"mode": "adapter", "adapter_rank_budget": 16})) == 64   # >= rank_cap
    assert server_lora_rank({"method": {"mode": "edit"}, "merge": {"mode": "adapter"}}) == 64   # edit: candidates only
    with pytest.raises(SystemExit):
        merge_mode({"merge": {"mode": "bogus"}})
    with pytest.raises(SystemExit):
        merge_mode({"method": {"mode": "weights"}, "merge": {"mode": "adapter"}})
    with pytest.raises(SystemExit):
        server_lora_rank(dict(cfg, merge={"mode": "adapter", "adapter_rank_budget": 600}))
