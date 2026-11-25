"""Paper B api.tensor_compute / api.tensors (2026-11-21): CPU torch code in a fresh sandbox on copies of the current
model's tensors, with three output tiers (json, vectors -> T handles, lora factors -> L handle) plus weights-mode
writes -> a W edit with dense delta ops. Mock workspaces run the code on seeded synthetic tensors (MockStore shapes);
the real path uses a tiny fake safetensors checkpoint (test_wedit). Needs torch + safetensors.

  python3 -I -m pytest -q lora_code/tests/test_tensor_compute.py
"""
import importlib.util
import math
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

MISSING = [m for m in ("torch", "safetensors") if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(bool(MISSING), reason=f"missing {MISSING}")

B = "layers.0.linear_attn.in_proj_b"          # [48, 5120] in the mock geometry
DT, NORM0, CONV = "layers.0.linear_attn.dt_bias", "layers.0.input_layernorm.weight", "layers.0.linear_attn.conv1d.weight"


def _ws(tmp_path, mode="scratch", cfg=None, store=None, **extra):
    from lora_code.mock import MockStore
    from lora_code.primitives.train import TrainData
    from lora_code.tools import Workspace
    train = TrainData({f"t/{i}": {"messages": [{"role": "user", "content": "q"}], "target": "a", "source": "gold"}
                       for i in range(10)})
    c = {"lora": {"rank_cap": 64, "norm_cap_tau": 0.1}, "weights": {"norm_cap_tau": 0.05}}
    for k, v in (cfg or {}).items():
        c[k] = {**c.get(k, {}), **v} if isinstance(v, dict) else v
    ws = Workspace(mock=store is None, store=store or MockStore(), train=train, cfg=c, out_dir=tmp_path,
                   budget={"gpu_seconds": 500}, mode=mode, **extra)
    ws.build_api()
    return ws


def _tc(ws, code, tensors, **kw):
    return ws.api.call("tensor_compute", {"code": code, "tensors": tensors, **kw})


# ============================================================================ surface / catalogue
def test_surface_and_catalogue(tmp_path):
    from lora_code.tools import TOOL_DOCS, WEIGHT_TOOLS, tools_for_mode
    assert "tensor_compute" not in WEIGHT_TOOLS and {"tensors", "tensor_compute"} <= set(TOOL_DOCS)
    on = {}                                    # absent key: on (2026-11-22; base.yaml sets it explicitly)
    for mode in ("scratch", "edit", "weights"):
        assert {"tensors", "tensor_compute"} <= set(tools_for_mode(mode))                          # no cfg: on
        assert {"tensors", "tensor_compute"} <= set(tools_for_mode(mode, cfg={"tools": {"tensor_compute": True}}))
        t = tools_for_mode(mode, cfg=on)
        assert {"tensors", "tensor_compute"} <= set(t)
        assert ("writes" in t["tensor_compute"]) == (mode == "weights")
        off = tools_for_mode(mode, cfg={"tools": {"tensor_compute": False}})
        assert not {"tensors", "tensor_compute"} & set(off)
    ws = _ws(tmp_path)
    rows = ws.api.call("tensors", {"layer": 0})
    by = {r["name"]: r for r in rows}
    assert by[NORM0] == {"name": NORM0, "shape": [5120], "dtype": "bfloat16", "editable": True}
    assert by[DT]["shape"] == [48] and by[CONV]["shape"] == [10240, 1, 4] and by[B + ".weight"]["shape"] == [48, 5120]
    assert all(r["name"].startswith("layers.0.") for r in rows)
    glob = ws.api.call("tensors", {"pattern": "layers.3.self_attn.*_norm.weight"})
    assert [r["name"] for r in glob] == ["layers.3.self_attn.k_norm.weight", "layers.3.self_attn.q_norm.weight"]
    fixed = {r["name"]: r["editable"] for r in ws.api.call("tensors", {"pattern": "_tokens"})
             + ws.api.call("tensors", {"pattern": "lm_head"})}
    fixed["norm.weight"] = ws.store.tensor_info("norm.weight")["editable"]
    assert fixed == {"embed_tokens.weight": False, "lm_head.weight": False, "norm.weight": False}


# ============================================================================ tier 1: json
def test_json_result_stdout_and_truncation(tmp_path):
    ws = _ws(tmp_path, cfg={"tensor_compute": {"result_chars": 200}})
    code = ("W = T['layers.0.linear_attn.in_proj_b']\nprint('shape', tuple(W.shape))\n"
            "result = {'fro': W.norm(), 'dt': T[%r][:3], 'nan': float('nan'), 'mod_alias': B in T}\n" % DT)
    r = _tc(ws, code.replace("B in T", repr(B) + " in T"), [B, DT])
    assert set(r["json"]) == {"fro", "dt", "nan", "mod_alias"} and r["json"]["mod_alias"] is True
    assert abs(r["json"]["fro"] - ws.store.norm(B)) / ws.store.norm(B) < 0.02          # synthetic ||W|| = norm(m)
    assert len(r["json"]["dt"]) == 3 and r["json"]["nan"] == "nan" and "shape (48, 5120)" in r["stdout"]
    assert r["seconds"] > 0 and "vectors" not in r and "lora" not in r and "edit" not in r
    big = _tc(ws, "result = list(range(500))", [DT])
    assert big["json_truncated"] and len(big["json"]) == 200
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):                        # user errors come back
        _tc(ws, "x = 1 / 0", [DT])
    with pytest.raises(RuntimeError, match="max 10000"):
        _tc(ws, "result = T[%r]" % B, [B])
    with pytest.raises(KeyError):
        _tc(ws, "result = 1", ["layers.0.nope"])
    with pytest.raises(ValueError, match="at most 4"):
        _tc(ws, "result = 1", [DT, NORM0, CONV, B, "layers.1.linear_attn.dt_bias"])
    assert not list((tmp_path / "work" / "tensor_compute").glob("*"))               # workdirs removed


# ============================================================================ tier 2: vectors
def test_vector_handles_in_rank1_and_from_direction(tmp_path):
    from lora_code.mock import MockLoRA
    ws = _ws(tmp_path)
    r = _tc(ws, "U, S, V = torch.svd_lowrank(T[%r], q=4)\nvectors = {'u': U[:, 0], 'v': V[:, 0], 'v1': V[:, 1]}" % B,
            [B])
    assert r["vectors"] == {"u": r["vectors"]["u"], "v": r["vectors"]["v"], "v1": r["vectors"]["v1"]}
    h = r["vectors"]["u"].split(".")[0]
    assert h.startswith("T") and r["vector_sizes"] == {"u": 48, "v": 5120, "v1": 5120}
    lo = ws.api.call("new_lora", {})["lora"]
    a = ws.api.call("rank1", {"lora": lo, "module": B, "u": f"{h}.u", "v": f"{h}.v", "s": 0.5})["lora"]
    b = ws.api.call("from_direction", {"lora": a, "module": B, "direction": f"{h}.u", "key": f"{h}.v1", "alpha": 0.3})
    assert isinstance(ws._get(b["lora"]), MockLoRA) and ws._get(b["lora"]).modules() == [B]
    with pytest.raises(ValueError, match="size 5120, need 48"):                       # wrong side
        ws.api.call("rank1", {"lora": lo, "module": B, "u": f"{h}.v", "v": f"{h}.u", "s": 0.5})
    with pytest.raises(ValueError, match="no vector"):
        ws.api.call("rank1", {"lora": lo, "module": B, "u": f"{h}.w", "v": f"{h}.v", "s": 0.5})
    with pytest.raises(RuntimeError, match="vectors"):
        _tc(ws, "vectors = [1, 2]", [DT])
    with pytest.raises(ValueError, match="1-D"):
        _tc(ws, "vectors = {'m': T[%r][:2]}" % B, [B])


def test_vector_handles_real_store(tmp_path):
    """Real path: the child reads the shard files; T-handle vectors build a real LoRA with rank1."""
    import torch

    from lora_code.primitives.lora import LoRA
    from lora_code.primitives.weights import WeightStore
    from lora_code.tests.test_wedit import DOWN, _write_ckpt
    store = WeightStore(_write_ckpt(tmp_path / "m0", sharded=True))
    ws = _ws(tmp_path / "ws", store=store)
    r = _tc(ws, "U, S, Vh = torch.linalg.svd(T[%r], full_matrices=False)\n"
                "vectors = {'u': U[:, 0], 'v': Vh[0]}\nresult = S[:2]" % DOWN, [DOWN])
    W = store.weight(DOWN)
    assert r["json"][0] == pytest.approx(float(torch.linalg.svdvals(W)[0]), rel=1e-5)
    h = r["vectors"]["u"].split(".")[0]
    lo = ws.api.call("rank1", {"lora": ws.api.call("new_lora", {})["lora"], "module": DOWN, "u": f"{h}.u",
                               "v": f"{h}.v", "s": 0.1})["lora"]
    L = ws._get(lo)
    assert isinstance(L, LoRA) and L.modules() == [DOWN] and L.max_rank() == 1
    assert torch.allclose(ws._h[h]["vectors"]["u"], torch.linalg.svd(W)[0][:, 0], atol=1e-5)


# ============================================================================ tier 3: LoRA factors
def test_lora_factors_registered_and_cap_checked(tmp_path):
    from lora_code.tools import classify, program_build
    ws = _ws(tmp_path)
    small = ("U, S, V = torch.svd_lowrank(T[%r], q=4)\n"
             "lora = {%r: (V[:, :2].T, U[:, :2] * S[:2] * 0.02)}\n") % (B, B)
    r = _tc(ws, small, [B])
    lo = ws._get(r["lora"], "lora")
    assert r["lora"].startswith("L") and r["lora_check"] == {"ok": True, "errors": []}
    assert lo.modules() == [B] and lo.rank(B) == 2 and 0.0 < lo.modules_[B]["rel"] < 0.1 and lo.gain > 0
    big = _tc(ws, small.replace("0.02", "0.5"), [B])
    assert not big["lora_check"]["ok"] and "tau 0.1" in big["lora_check"]["errors"][0]
    with pytest.raises(ValueError, match="rank 65 > rank cap 64"):
        _tc(ws, "lora = {%r: (torch.zeros(65, 5120), torch.zeros(48, 65))}" % B, [B])
    with pytest.raises(ValueError, match="needs A"):
        _tc(ws, "lora = {%r: (torch.zeros(2, 48), torch.zeros(5120, 2))}" % B, [B])
    with pytest.raises(KeyError, match="not a linear module"):
        _tc(ws, "lora = {%r: (torch.zeros(1, 4), torch.zeros(48, 1))}" % DT, [DT])
    prog = ("def main(api):\n    r = api.tensor_compute(code=%r, tensors=[%r])\n    return {'lora': r['lora']}\n"
            % (small, B))
    art, err = program_build(ws, prog, "g00_t0_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w0")
    assert err is None, err
    assert art["strategy"] == "pure_construct" and art["modules"] == 1
    assert classify(["tensor_compute"]) == "keep" and classify(["tensor_compute", "tensor_compute.lora"]) == \
        "pure_construct"


# ============================================================================ weights mode: writes
def test_weights_writes_caps_and_refusals(tmp_path):
    from lora_code.mock import MockWeightEdit
    from lora_code.tools import classify, program_build
    ws = _ws(tmp_path, mode="weights")
    rows = "W = T[%r].clone()\nW[:2] *= 1.1\nwrites = {%r: W}\n" % (B, B)
    r = _tc(ws, rows, [B])
    ed = ws._get(r["edit"], "edit")
    assert isinstance(ed, MockWeightEdit) and ed.modules() == [B] and ed.ops[-1][0] == "w_delta"
    rel = r["writes"][B + ".weight"]["rel_norm"]
    assert 0.01 < rel < 0.05 and abs(ed.modules_[B]["rel"] - rel) < 1e-5 and ws.check(ed)[0]
    vec = ("d = T[%r] + 0.3 * torch.linspace(-1, 1, 48)\nn = T[%r] + 0.4 * torch.linspace(-1, 1, 5120)\n"
           "writes = {%r: d, %r: n}\n") % (DT, NORM0, DT, NORM0)
    r2 = _tc(ws, vec, [DT, NORM0], edit=r["edit"])                    # appended to the given edit
    ed2 = ws._get(r2["edit"], "edit")
    assert set(ed2.modules()) == {B, DT, NORM0} and ed2.modules_[DT]["ndim"] == 1 and ws.check(ed2)[0]
    assert r2["writes"][NORM0]["rms_cap"] == pytest.approx(0.5)          # zero-centred norm: absolute cap
    assert ws.report(ed2)["check_ok"] and len(ed.modules()) == 1           # the input edit is unchanged
    with pytest.raises(ValueError, match="> tau 0.05"):
        _tc(ws, rows.replace("W[:2] *= 1.1", "W[:20] *= 1.2"), [B])
    with pytest.raises(ValueError, match="1-D tensor"):
        _tc(ws, "writes = {%r: T[%r] + 0.9 * torch.linspace(-1, 1, 5120)}" % (NORM0, NORM0), [NORM0])
    for code, t in (("writes = {%r: T[%r] * 1.01}" % (B, B), B), ("writes = {%r: T[%r] * 1.1}" % (DT, DT), DT)):
        with pytest.raises(ValueError, match="uniform rescale"):
            _tc(ws, code, [t])
    ws.store._catalogue()["embed_tokens.weight"] = (16, 5120)              # small stand-ins so the test stays fast
    ws.store._catalogue()["lm_head.weight"] = (16, 5120)
    for t in ("norm.weight", "embed_tokens.weight", "lm_head.weight"):
        with pytest.raises(PermissionError, match="never editable"):
            _tc(ws, "x = T[%r].clone()\nx[0] += 0.01\nwrites = {%r: x}" % (t, t), [t])
    ok, rep = ws.check(MockWeightEdit({"norm.weight": {"rank": 0, "rel": 0.01, "ndim": 1}}))
    assert not ok and "never editable" in rep["errors"][0]
    with pytest.raises(ValueError, match="listed in tensors"):
        _tc(ws, "writes = {%r: torch.zeros(48)}" % DT, [NORM0])
    with pytest.raises(ValueError, match="shape"):
        _tc(ws, "writes = {%r: torch.zeros(47)}" % DT, [DT])
    assert "edit" not in _tc(ws, "writes = {%r: T[%r]}" % (DT, DT), [DT])   # unchanged: nothing written
    prog = "def main(api):\n    return {'edit': api.tensor_compute(code=%r, tensors=[%r])['edit']}\n" % (rows, B)
    art, err = program_build(ws, prog, "g00_t0_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w0")
    assert err is None, err
    assert art["kind"] == "weights" and art["strategy"] == "weights_only"
    assert classify(["tensor_compute", "tensor_compute.writes", "scale_singular"]) == "weights+construct"
    sc = _ws(tmp_path / "s", mode="scratch")
    with pytest.raises(ValueError, match="mode=weights"):
        _tc(sc, "writes = {%r: T[%r] + 0.1}" % (DT, DT), [DT])
    with pytest.raises(ValueError, match="mode=weights"):
        _tc(sc, "result = 1", [DT], edit="W1")


def test_weights_writes_real_store(tmp_path):
    """Real path: writes become 'delta' ops of a WeightEdit (T reflects the given edit's ops), checked by
    check_wedit and written by edit_into_model."""
    import torch

    from lora_code.primitives.wedit import WeightEdit, edit_into_model
    from lora_code.primitives.weights import WeightStore
    from lora_code.tests.test_wedit import DOWN, P, _read, _write_ckpt
    from lora_code.tests.test_wedit import DT as DT0
    from lora_code.tests.test_wedit import NORM0 as N0
    ck = _write_ckpt(tmp_path / "m0", sharded=True)
    store = WeightStore(ck)
    ws = _ws(tmp_path / "ws", mode="weights", store=store)
    e0 = ws.api.call("w_scale_rows", {"edit": ws.api.call("new_edit", {})["edit"], "module": DOWN, "factors": [1.02],
                                      "rows": [0]})["edit"]
    code = ("W = T[%r].clone()\nresult = float(W[0, 0])\nW[1] *= 1.03\n"
            "writes = {%r: W, %r: T[%r] + torch.tensor([0.2, -0.1, 0.0, 0.1])}\n") % (DOWN, DOWN, DT0, DT0)
    r = _tc(ws, code, [DOWN, DT0], edit=e0)
    W = store.weight(DOWN)
    assert r["json"] == pytest.approx(1.02 * float(W[0, 0]), rel=1e-6)     # T = W after the edit's ops
    ed = ws._get(r["edit"], "edit")
    assert isinstance(ed, WeightEdit) and [o["op"] for o in ed.ops[DOWN]] == ["rows", "delta"]
    assert ed.ops[DT0][0]["op"] == "delta" and ws.check(ed)[0]
    expect = W.clone()
    expect[0] *= 1.02
    expect[1] *= 1.03
    assert torch.allclose(ed.apply(DOWN, W), expect, atol=1e-6)
    out = tmp_path / "m1"
    edit_into_model(ed, ck, out, store=store)
    assert torch.equal(_read(out, P + DT0), (store.tensor(DT0) + torch.tensor([0.2, -0.1, 0.0, 0.1])).to(torch.bfloat16))
    with pytest.raises(ValueError, match="uniform rescale"):
        _tc(ws, "writes = {%r: T[%r] * 1.2}" % (N0, N0), [N0])
    with pytest.raises(PermissionError):
        _tc(ws, "writes = {'lm_head.weight': T['lm_head.weight'] + 0.001 * torch.randn(32, 8)}", ["lm_head.weight"])


# ============================================================================ limits / locking / exploration
def test_timeout_and_deadline(tmp_path):
    ws = _ws(tmp_path, cfg={"tensor_compute": {"timeout_s": 6}})
    t0 = time.time()
    with pytest.raises(RuntimeError, match="timeout"):
        _tc(ws, "import time\ntime.sleep(60)", [DT])
    assert time.time() - t0 < 30
    ws2 = _ws(tmp_path / "b")
    ws2.begin_program("g00_t0_s00", timeout_s=8)                       # the program has ~8 s left
    try:
        from rsicore.sandbox import BudgetExceeded
        with pytest.raises(BudgetExceeded, match="not enough time"):
            _tc(ws2, "result = 1", [DT])
    finally:
        ws2.end_program("g00_t0_s00")
    from lora_code.tools import tensor_compute_settings
    assert tensor_compute_settings({}) == {"max_tensors": 4, "timeout_s": 300, "mem_gb": 16, "threads": 4,
                                           "max_concurrent": 2, "result_chars": 4000, "explore": True,
                                           "max_input_mb": 8192}
    assert tensor_compute_settings({"tensor_compute": {"threads": 8, "mem_gb": None}})["threads"] == 8


def test_runs_without_the_workspace_lock(tmp_path):
    ws = _ws(tmp_path)
    out = {}
    th = threading.Thread(target=lambda: out.update(r=_tc(ws, "import time\ntime.sleep(4)\nresult = 7", [DT])))
    th.start()
    time.sleep(2.0)
    t0 = time.time()
    got = ws._lock.acquire(timeout=1.0)                                  # other threads' tool calls proceed
    if got:
        ws._lock.release()
    names = ws.api.call("modules", {"layer": 0})
    assert got and names and time.time() - t0 < 1.5 and th.is_alive()
    th.join(60)
    assert out["r"]["json"] == 7


def test_explore_availability(tmp_path):
    ws = _ws(tmp_path, mode="weights")
    n0, spent0 = len(ws._h), dict(ws.api.spent)
    api, cleanup = ws.explore_api("g00_t0_s00~x01_0")
    try:
        assert {"tensors", "tensor_compute"} <= set(api.names())
        r = api.call("tensor_compute", {"code": "vectors = {'d': T[%r]}\nresult = float(T[%r].mean())" % (DT, DT),
                                        "tensors": [DT]})
        assert r["vectors"]["d"].startswith("T") and math.isfinite(r["json"])
        assert ws._time_left() is not None and ws._time_left() <= 300
    finally:
        cleanup()
    assert len(ws._h) == n0 and ws.api.spent == spent0 and ws.gpu_measured == 0.0      # freed, nothing charged
    off = _ws(tmp_path / "off", cfg={"tensor_compute": {"explore": False}})
    api, cleanup = off.explore_api("g00_t0_s00~x01_0")
    cleanup()
    assert not {"tensors", "tensor_compute"} & set(api.names()) and "tensor_compute" in off.api.names()


# ============================================================================ loras=: factors of LoRA handles (2026-11-25)
A_MOD = "layers.0.linear_attn.in_proj_a"      # [48, 5120] in the mock geometry


def test_lora_inputs_surface_and_prompt():
    from lora_code.prompts import system_prompt
    from lora_code.tools import tools_for_mode
    on, adapter = {"tools": {"tensor_compute": True}}, {"tools": {"tensor_compute": True}, "merge": {"mode": "adapter"}}
    for mode, cfg, names, absent in (("scratch", on, ["'H0'", "already merged"], ["'CUR'", "'RAW'"]),
                                     ("scratch", adapter, ["'H0'", "'CUR'", "SVD-truncated"], ["'RAW'", "merged into"]),
                                     ("edit", on, ["'RAW'", "not in T"], ["'H0'", "'CUR'"]),
                                     ("weights", on, ["weight edits are not LoRAs"], ["'H0'", "'RAW'", "candidate("])):
        t = tools_for_mode(mode, cfg=cfg)
        doc = t["tensor_compute"][t["tensor_compute"].find(" loras=["):]
        assert "loras=None) -> {" in t["tensor_compute"] and "LORA = {handle: {module:" in doc
        assert ". loras=[handles]" in t["tensor_compute"]
        assert all(x in doc for x in names) and not any(x in doc for x in absent), (mode, cfg, doc)
        # stack() is empty in edit / weights mode: only scratch points there for factors
        assert ("read their factors with tensor_compute" in t["stack"]) == (mode == "scratch")
        assert ("'CUR'" in t["stack"]) == (cfg is adapter)
        assert "loras" not in tools_for_mode(mode, cfg={"tools": {"tensor_compute": False}}).get("stack", "")
        sp = system_prompt(cfg, budget={"gpu_seconds": 7200}, mode=mode)
        want = {"weights": None, "edit": "With loras=['RAW', ...]"}.get(
            mode, "With loras=['CUR', 'H0', ...]" if cfg is adapter else "With loras=['H0', ...]")
        assert (want in sp) if want else "With loras=" not in sp
        assert "loras=" not in system_prompt({"tools": {"tensor_compute": False}}, budget={"gpu_seconds": 7200},
                                             mode=mode)
    paper = tools_for_mode("weights", cfg={"tools": {"tensor_compute": True, "paper_tools": True}})["tensor_compute"]
    assert paper.startswith("tensor_compute(code, tensors, vectors=None, edit=None, loras=None) -> {")


def test_lora_inputs_mock(tmp_path):
    from lora_code.mock import MockLoRA
    h0 = MockLoRA()
    h0.touch([B], 4, 0.02)
    h0.touch([A_MOD], 8, 0.01)
    ws = _ws(tmp_path, stack=[(0, h0)])
    assert ws.api.call("stack", {}) == [{"handle": "H0", "gen": 0, "modules": 2, "max_rank": 8}]
    code = ("W = {m: T[m] for m in T}\n"
            "result = {m: [list(A.shape), list(B.shape), float((B @ A).norm())] for m, (A, B) in LORA['H0'].items()}")
    r = ws.api.call("tensor_compute", {"code": code, "loras": ["H0"]})          # tensors= may be empty
    assert set(r["json"]) == {B, A_MOD}
    assert r["json"][B][:2] == [[4, 5120], [48, 4]] and r["json"][A_MOD][:2] == [[8, 5120], [48, 8]]
    for m, rel in ((B, 0.02), (A_MOD, 0.01)):                                     # ||B A|| = rel * ||W|| (mock)
        assert r["json"][m][2] == pytest.approx(rel * ws.store.norm(m), rel=1e-3)
    again = ws.api.call("tensor_compute", {"code": code, "loras": "H0"})
    assert again["json"] == r["json"]                                             # seeded: the same factors
    one = ws.api.call("tensor_compute", {"code": "result = sorted(LORA['H0'])", "loras": {"H0": [B]},
                                         "tensors": [B]})
    assert one["json"] == [B]
    # reverse an earlier change: lora = (A, -B) -> a new L handle with the same size
    rev = ws.api.call("tensor_compute", {"code": "A, Bf = LORA['H0'][%r]\nlora = {%r: (A, -Bf)}" % (B, B),
                                         "loras": {"H0": B}})
    assert rev["lora"].startswith("L") and rev["lora_check"]["ok"]
    assert ws._get(rev["lora"]).modules_[B]["rel"] == pytest.approx(0.02, rel=1e-3)
    # a handle result ({"lora": ...}) and an own L handle work too
    lo = ws.api.call("scale", {"lora": "H0", "s": 0.5})
    both = ws.api.call("tensor_compute", {"code": "result = {h: len(v) for h, v in LORA.items()}",
                                          "loras": [lo, "H0"]})
    assert both["json"] == {lo["lora"]: 2, "H0": 2}
    # whole tool results / stack rows are one handle each (like every other tool's handle arguments)
    code_n = "result = {h: len(v) for h, v in LORA.items()}"
    assert ws.api.call("tensor_compute", {"code": code_n, "loras": rev})["json"] == {rev["lora"]: 1}
    assert ws.api.call("tensor_compute", {"code": code_n, "loras": ws.api.call("stack", {})[0]})["json"] == {"H0": 2}
    assert ws.api.call("tensor_compute", {"code": code_n, "loras": ws.api.call("stack", {})})["json"] == {"H0": 2}
    with pytest.raises(KeyError, match="has no saved LoRA"):
        ws.api.call("tensor_compute", {"code": "result = 1", "loras": "C:nope"})
    with pytest.raises(ValueError, match="list of module names"):
        ws.api.call("tensor_compute", {"code": "result = 1", "loras": {"H0": 3}})
    plain = ws.api.call("tensor_compute", {"code": "result = 1", "tensors": [B]})     # a result without a LoRA
    with pytest.raises(ValueError, match="not a LoRA handle"):
        ws.api.call("tensor_compute", {"code": "result = 1", "loras": plain})
    with pytest.raises(KeyError, match="unknown handle"):
        ws.api.call("tensor_compute", {"code": "result = 1", "loras": ["H7"]})
    with pytest.raises(KeyError, match="no module"):
        ws.api.call("tensor_compute", {"code": "result = 1", "loras": {"H0": ["layers.5.mlp.down_proj"]}})
    with pytest.raises(ValueError, match="at most 8 LoRA handles"):
        ws.api.call("tensor_compute", {"code": "result = 1", "loras": {f"L{i}": None for i in range(9)}})
    with pytest.raises(ValueError, match="not a LoRA handle"):
        ws.api.call("tensor_compute", {"code": "result = 1", "loras": [3]})
    with pytest.raises(ValueError, match="at least one tensor name"):
        ws.api.call("tensor_compute", {"code": "result = 1", "loras": []})
    small = _ws(tmp_path / "small", stack=[(0, h0)], cfg={"tensor_compute": {"max_input_mb": 0.1}})
    with pytest.raises(ValueError, match="pass fewer LoRA modules"):
        small.api.call("tensor_compute", {"code": "result = 1", "loras": ["H0"]})
    # the exploration api sees the same handles, nothing is charged
    api, cleanup = ws.explore_api("g00_t0_s00~x01_0")
    try:
        x = api.call("tensor_compute", {"code": "result = sorted(LORA['H0'])", "loras": ["H0"]})
        assert x["json"] == sorted([B, A_MOD])
    finally:
        cleanup()
    assert not list((tmp_path / "work" / "tensor_compute").glob("*"))


def test_lora_inputs_weights_mode_refuses_edits(tmp_path):
    ws = _ws(tmp_path, mode="weights")
    res = ws.api.call("new_edit", {})
    for arg in ([res["edit"]], res):                            # the handle, or the whole {"edit": ...} result
        with pytest.raises(TypeError, match="weight edit"):
            ws.api.call("tensor_compute", {"code": "result = 1", "loras": arg})


def test_lora_inputs_real_store(tmp_path):
    """Real path: the exact factors (bf16 stored -> fp32 in the code) of a stack adapter, RAW and an own LoRA."""
    import torch

    from lora_code.primitives.lora import LoRA
    from lora_code.primitives.weights import WeightStore
    from lora_code.tests.test_wedit import DOWN, UP, _write_ckpt
    store = WeightStore(_write_ckpt(tmp_path / "m0", sharded=True))
    g = torch.Generator().manual_seed(3)
    A1, B1 = torch.randn(2, 16, generator=g).to(torch.bfloat16), torch.randn(8, 2, generator=g).to(torch.bfloat16)
    A2, B2 = torch.randn(1, 8, generator=g), torch.randn(16, 1, generator=g)
    h0 = LoRA({DOWN: (A1 * 0.01, B1 * 0.01)})
    raw = LoRA({DOWN: (A2[:, :8].repeat(1, 2) * 0.01, B2[:8] * 0.01), UP: (A2 * 0.01, B2 * 0.01)})
    ws = _ws(tmp_path / "ws", store=store, mode="edit", stack=[(0, h0)], raw=raw)
    r = ws.api.call("tensor_compute", {"code": "result = {h: {m: [list(a.shape), str(a.dtype), float((b @ a).sum())] "
                                                "for m, (a, b) in v.items()} for h, v in LORA.items()}",
                                       "loras": ["H0", "RAW"]})
    want = float((h0.factors[DOWN][1].float() @ h0.factors[DOWN][0].float()).sum())
    assert r["json"]["H0"][DOWN][:2] == [[2, 16], "torch.float32"]
    assert r["json"]["H0"][DOWN][2] == pytest.approx(want, rel=1e-5)
    assert set(r["json"]["RAW"]) == {DOWN, UP} and r["json"]["RAW"][UP][0] == [1, 8]
    rev = ws.api.call("tensor_compute", {"code": "a, b = LORA['H0'][%r]\nlora = {%r: (a, -b)}" % (DOWN, DOWN),
                                         "loras": {"H0": [DOWN]}})
    L = ws._get(rev["lora"])
    assert isinstance(L, LoRA) and torch.allclose(L.delta(DOWN), -h0.delta(DOWN).float(), atol=1e-6)


def test_lora_inputs_raw_and_cur_handles(tmp_path):
    """mode=edit: a whole api.raw_lora() result is one handle; merge.mode adapter: 'CUR' = the served C_g."""
    from lora_code.mock import MockLoRA
    raw = MockLoRA()
    raw.touch([B], 4, 0.02)
    ws = _ws(tmp_path, mode="edit", raw=raw)
    r = ws.api.call("raw_lora", {})
    assert set(r) > {"lora"}
    got = ws.api.call("tensor_compute", {"code": "result = {h: sorted(v) for h, v in LORA.items()}", "loras": r})
    assert got["json"] == {"RAW": [B]} and "CUR" not in ws._h
    cum, h0 = MockLoRA(), MockLoRA()
    cum.touch([B, A_MOD], 6, 0.015)
    h0.touch([B], 4, 0.02)
    ad = _ws(tmp_path / "ad", stack=[(0, h0)], base_lora=cum)
    x = ad.api.call("tensor_compute", {"code": "result = {h: {m: list(a.shape) for m, (a, b) in v.items()} "
                                               "for h, v in LORA.items()}", "loras": ["CUR", "H0"]})
    assert x["json"] == {"CUR": {B: [6, 5120], A_MOD: [6, 5120]}, "H0": {B: [4, 5120]}}
    assert "CUR" not in _ws(tmp_path / "none", base_lora=MockLoRA())._h          # empty C_g: no handle


def test_lora_inputs_write_time_counts_toward_the_deadline(tmp_path, monkeypatch):
    """Writing large inputs (LoRA factors) is charged to the call's time: with the caller's deadline close, a slow
    write ends the call before the child starts instead of overrunning the deadline."""
    import safetensors.torch as st

    from lora_code.mock import MockLoRA
    h0 = MockLoRA()
    h0.touch([B], 4, 0.02)
    ws = _ws(tmp_path, stack=[(0, h0)])
    real = st.save_file
    monkeypatch.setattr(st, "save_file", lambda *a, **k: (time.sleep(4.0), real(*a, **k))[1])
    ws._local.deadline = time.time() + 13.0                    # timeout = 13 - 5 = 8 s; after the 4 s write < 5 s
    t0 = time.time()
    with pytest.raises(Exception, match="after writing the inputs"):
        ws.api.call("tensor_compute", {"code": "result = 1", "loras": ["H0"]})
    assert time.time() - t0 < 8.0
    ws._local.deadline = None
    assert ws.api.call("tensor_compute", {"code": "result = 1", "loras": ["H0"]})["json"] == 1
