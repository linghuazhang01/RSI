"""Paper B method.mode=weights (structured, any-rank edits of the dense weights; programs return {"edit": h}),
mock only (no weights, no GPU):

  eval/.venv/bin/python -I -m pytest -q lora_code/tests/test_weights_tools.py
"""
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

try:
    import rsicore.sandbox  # noqa: F401
    HAVE_CORE = True
except Exception:  # noqa: BLE001
    HAVE_CORE = False
pytestmark = pytest.mark.skipif(not HAVE_CORE, reason="eval/rsicore not available")

W_TOOLS = {"new_edit", "w_scale", "w_scale_rows", "w_scale_cols", "w_add_lora", "w_toward_original", "inspect_edit"}


class _Ev:
    """Fake evaluator: deterministic score = gain - damage of the registered (mock) object."""

    def __init__(self):
        self.reg, self.calls = {}, []

    def known(self, name):
        return name in self.reg

    def register(self, lo, name):
        self.reg[name] = lo

    def feedback(self, name, level, lora_report=None, cost=None):
        self.calls.append((name, level))
        lo = self.reg.get(name)
        g, d = (lo.gain, lo.damage) if lo is not None else (0.0, 0.0)
        sc = round(g - d, 6)
        return sc, {"score": sc, "target": {"vs_parent": {"delta": round(g, 6)}},
                    "lora_stats": lora_report, "cost": cost}

    def cached(self, name, level):
        return None


def _ws(tmp_path, mode="weights", budget=None, cfg=None, stack=None, evaluator=None):
    from lora_code.mock import MockStore
    from lora_code.primitives.train import TrainData
    from lora_code.tools import Workspace
    train = TrainData({f"t/{i}": {"messages": [{"role": "user", "content": "q"}], "target": "a", "source": "gold"}
                       for i in range(10)})
    c = {"lora": {"rank_cap": 64, "norm_cap_tau": 0.1}, "weights": {"norm_cap_tau": 0.05}}
    c.update(cfg or {})
    ws = Workspace(mock=True, store=MockStore(), train=train, cfg=c, out_dir=tmp_path,
                   budget=budget or {"evaluate": 4, "gpu_seconds": 500}, mode=mode, stack=stack,
                   evaluator=evaluator or _Ev())
    ws.build_api()
    return ws


def _down(ws, n=4):
    return [m for m in ws.store.modules() if m.endswith("down_proj")][:n]


def test_tool_surface_per_mode(tmp_path):
    from lora_code.tools import TOOL_DOCS, WEIGHT_TOOLS, tools_for_mode
    assert set(WEIGHT_TOOLS) == W_TOOLS and W_TOOLS <= set(TOOL_DOCS)
    w = tools_for_mode("weights")
    assert W_TOOLS <= set(w) and "raw_lora" not in w
    assert {"new_lora", "scale_singular", "svd", "acts", "evaluate", "train_loss", "compare"} <= set(w)
    assert "edit" in w["evaluate"] and "edit" in w["train_loss"]
    for mode in ("scratch", "edit"):
        t = tools_for_mode(mode)
        assert not (W_TOOLS & set(t))
        assert ("raw_lora" in t) == (mode == "edit")
        assert "edit=" not in t["evaluate"]
    off = tools_for_mode("weights", cfg={"tools": {"w_toward_original": False, "train_loss": False}})
    assert "w_toward_original" not in off and "train_loss" not in off and "regmean_preserve" not in off
    ws = _ws(tmp_path)
    names = set(ws.api.names())
    assert W_TOOLS <= names and "raw_lora" not in names
    sc = _ws(tmp_path / "s", mode="scratch")
    assert not (W_TOOLS & set(sc.api.names()))
    # tensor tools (2026-11-21): every mode when enabled (absent key: on, 2026-11-22), not a WEIGHT_TOOL; the
    # weights doc documents writes / caps / the edit arg
    assert {"tensors", "tensor_compute"} <= names                      # this workspace's cfg has no tools block
    on = {"tools": {"tensor_compute": True}}
    w = tools_for_mode("weights", cfg=on)
    assert {"tensors", "tensor_compute"} <= set(w) and {"tensors", "tensor_compute"} <= set(tools_for_mode("scratch", cfg=on))
    assert "writes" in w["tensor_compute"] and "edit=None" in w["tensor_compute"] and "vector_cap_tau" in w["tensor_compute"]
    assert "writes" not in tools_for_mode("scratch", cfg=on)["tensor_compute"]
    assert not {"tensors", "tensor_compute"} & set(tools_for_mode("weights", cfg={"tools": {"tensor_compute": False}}))


def test_weight_tools_change_gain_damage_rel(tmp_path):
    from lora_code.mock import MockWeightEdit, m_construct
    ws = _ws(tmp_path)
    mods = _down(ws)
    e0 = ws.api.call("new_edit", {})["edit"]
    assert e0.startswith("W") and isinstance(ws._get(e0, "edit"), MockWeightEdit)
    # module scaling near 1: a little gain, rel = |s-1|
    e1 = ws.api.call("w_scale", {"edit": e0, "modules": mods[:2], "s": 1.01})["edit"]
    E1 = ws._get(e1, "edit")
    assert E1.gain > 0 and E1.damage > 0 and E1.gain > E1.damage
    assert all(abs(E1.modules_[m]["rel"] - 0.01) < 1e-9 and E1.modules_[m]["rank"] == 0 for m in mods[:2])
    assert E1.ops[-1][0] == "w_scale" and E1.max_rank() == 0 and not ws._get(e0).modules()   # inputs unchanged
    # large scaling: damage grows quadratically, net loss
    big = ws._get(ws.api.call("w_scale", {"edit": e0, "modules": mods[:2], "s": 1.04})["edit"])
    assert big.damage > 10 * E1.damage and big.damage > big.gain
    # rows / cols: relative change of scaling k of dim channels
    out_f, in_f = ws.store.shape(mods[0])
    r = ws._get(ws.api.call("w_scale_rows", {"edit": e1, "module": mods[0], "factors": [1.5] * 16,
                                             "rows": list(range(16))})["edit"])
    d = math.sqrt(16 * 0.25 / out_f)
    assert abs(r.modules_[mods[0]]["rel"] - math.sqrt(0.01 ** 2 + d ** 2)) < 1e-9 and r.ops[-1][0] == "w_rows"
    assert r.gain > E1.gain
    c = ws._get(ws.api.call("w_scale_cols", {"edit": e0, "module": mods[1], "factors": 1.02})["edit"])
    assert abs(c.modules_[mods[1]]["rel"] - 0.02) < 1e-9 and c.ops[-1][0] == "w_cols"
    # add a LoRA: its gain / damage scaled by alpha (damage by alpha^2)
    lo = m_construct(ws._empty(), mods[2], 0.5, "rank1")
    hl = ws._new("L", lo)
    a = ws._get(ws.api.call("w_add_lora", {"edit": e1, "lora": hl, "alpha": 0.5})["edit"])
    assert abs(a.gain - (E1.gain + 0.5 * lo.gain)) < 1e-12 and abs(a.damage - (E1.damage + 0.25 * lo.damage)) < 1e-12
    assert abs(a.modules_[mods[2]]["rel"] - 0.5 * lo.modules_[mods[2]]["rel"]) < 1e-12 and a.ops[-1][0] == "w_add_lora"
    # toward the original: shrinks the edit on those modules (alpha=1 removes it there)
    t = ws._get(ws.api.call("w_toward_original", {"edit": e1, "modules": mods[:1], "alpha": 0.5})["edit"])
    assert t.gain < E1.gain and t.damage < E1.damage and abs(t.modules_[mods[0]]["rel"] - 0.005) < 1e-9
    t1 = ws._get(ws.api.call("w_toward_original", {"edit": e1, "modules": mods[:2], "alpha": 1.0})["edit"])
    assert not t1.modules() and t1.ops[-1][0] == "w_toward_original"
    info = ws.api.call("inspect_edit", {"edit": e1})
    assert set(info) >= {"modules", "check_ok", "check_errors", "top_rel_norm", "total_rel_norm", "ops"}
    assert info["modules"] == 2 and info["check_ok"] and info["max_rank"] == 0
    assert abs(info["total_rel_norm"] - math.sqrt(2) * 0.01) < 1e-5
    # bad arguments
    with pytest.raises(TypeError):
        ws.api.call("w_scale", {"edit": hl, "modules": mods[:1], "s": 1.01})          # a LoRA is not an edit
    with pytest.raises(TypeError):
        ws.api.call("scale", {"lora": e1, "s": 0.5})                                 # an edit is not a LoRA
    with pytest.raises(ValueError):
        ws.api.call("w_scale_rows", {"edit": e0, "module": mods[0], "factors": [1.1], "rows": [out_f]})
    with pytest.raises(ValueError):
        ws.api.call("w_scale_cols", {"edit": e0, "module": mods[0], "factors": [1.1, 1.2], "cols": [0]})
    with pytest.raises(ValueError):
        ws.api.call("w_toward_original", {"edit": e0, "modules": mods[:1], "alpha": 0})
    with pytest.raises(KeyError):
        ws.api.call("w_scale_rows", {"edit": e0, "module": "down_proj", "factors": 1.1})   # one exact module


def test_toward_original_undoes_merges(tmp_path):
    from lora_code.mock import MockLoRA, m_construct
    mod = "layers.10.mlp.down_proj"
    merged = m_construct(MockLoRA(), mod, 1.0, "rank1")
    ws = _ws(tmp_path, stack=[(0, merged)])
    e = ws.api.call("new_edit", {})["edit"]
    t = ws._get(ws.api.call("w_toward_original", {"edit": e, "modules": [mod], "alpha": 0.5})["edit"])
    assert abs(t.modules_[mod]["rel"] - 0.5 * merged.modules_[mod]["rel"]) < 1e-12
    assert t.gain < 0                                          # half of the merged gain is undone


def test_check_rejects(tmp_path):
    from lora_code.mock import MockWeightEdit
    ws = _ws(tmp_path, cfg={"weights": {"norm_cap_tau": 0.05, "max_modules": 2}})
    mods = _down(ws)
    e0 = ws.api.call("new_edit", {})["edit"]
    ok, rep = ws.check(ws._get(e0))
    assert ok and rep["modules"] == {}
    big = ws._get(ws.api.call("w_scale", {"edit": e0, "modules": mods[:1], "s": 1.2})["edit"])
    ok, rep = ws.check(big)
    assert not ok and "tau 0.05" in rep["errors"][0]
    ok, rep = ws.check(MockWeightEdit({"layers.999.mlp.down_proj": {"rank": 0, "rel": 0.01}}))
    assert not ok and "not in model" in rep["errors"][0]
    many = ws._get(ws.api.call("w_scale", {"edit": e0, "modules": mods[:3], "s": 1.01})["edit"])
    ok, rep = ws.check(many)
    assert not ok and "max_modules" in rep["errors"][0]
    assert ws.check(ws._get(ws.api.call("w_scale", {"edit": e0, "modules": mods[:2], "s": 1.01})["edit"]))[0]
    rep = ws.report(big)
    assert rep["check_ok"] is False and rep["max_rank"] == 0 and rep["ops"][-1][0] == "w_scale"
    assert rep["top_rel_norm"][0]["ops"] == 1


def test_check_nonlinear_tensor_entries(tmp_path):
    """Mock edits of non-linear tensors (tensor_compute writes, 2026-11-21): 1-D entries use weights.vector_cap_tau,
    >= 2-D ones norm_cap_tau; embed_tokens / final norm / lm_head are refused."""
    from lora_code.mock import MockStore, MockWeightEdit, m_w_delta
    ws = _ws(tmp_path, cfg={"weights": {"norm_cap_tau": 0.05, "vector_cap_tau": 0.4}})
    dt, conv = "layers.0.linear_attn.dt_bias", "layers.0.linear_attn.conv1d.weight"
    e = m_w_delta(MockWeightEdit(), dt, 0.3, 1, tau_ratio=0.05 / 0.4)
    e = m_w_delta(e, conv, 0.04, 3)
    assert e.modules_[dt] == {"rank": 0, "rel": 0.3, "ndim": 1} and e.modules_[conv]["ndim"] == 3
    assert e.gain > 0 and e.ops[-1] == ["w_delta", [conv], 0.04]
    ok, rep = ws.check(e)
    assert ok, rep
    assert rep["modules"][dt] == {"rel_norm": 0.3, "ops": 1}
    ok, rep = ws.check(m_w_delta(e, dt, 0.3, 1))                       # sqrt(0.3^2 + 0.3^2) = 0.42 > 0.4
    assert not ok and "(1-D tensor)" in rep["errors"][0]
    ok, rep = ws.check(m_w_delta(MockWeightEdit(), conv, 0.06, 3))
    assert not ok and "tau 0.05" in rep["errors"][0]
    for bad, needle in (("norm.weight", "never editable"), ("lm_head.weight", "never editable"),
                        ("layers.0.linear_attn.nope", "not in model")):
        ok, rep = ws.check(MockWeightEdit({bad: {"rank": 0, "rel": 0.01, "ndim": 1}}))
        assert not ok and needle in rep["errors"][0], rep
    st = MockStore()
    assert st.edit_name("layers.0.mlp.down_proj.weight") == "layers.0.mlp.down_proj" and st.editable(dt)
    assert st.tensor_shape(conv) == (10240, 1, 4) and st.tensor_key("lm_head.weight") == "lm_head.weight"
    assert st.synth_spec("layers.0.input_layernorm.weight")["mean"] == 0.0               # zero-centred RMSNorm


def test_resolve_result(tmp_path):
    from lora_code.mock import MockLoRA, MockWeightEdit, m_construct
    ws = _ws(tmp_path)
    mod = _down(ws)[0]
    e = ws.api.call("w_scale", {"edit": ws.api.call("new_edit", {})["edit"], "modules": [mod], "s": 1.01})["edit"]
    assert ws.resolve_result({"edit": e}) is ws._get(e)
    lo = m_construct(MockLoRA(), mod, 0.5, "rank1")
    conv = ws.resolve_result({"lora": ws._new("L", lo)})
    assert isinstance(conv, MockWeightEdit) and conv.modules() == [mod] and abs(conv.gain - lo.gain) < 1e-12
    assert conv.ops[-1][0] == "w_add_lora"
    for keep in ({"edit": None}, {"lora": None}):
        k = ws.resolve_result(keep)
        assert isinstance(k, MockWeightEdit) and not k.modules()
    with pytest.raises(ValueError):
        ws.resolve_result(None)
    with pytest.raises(ValueError):
        ws.resolve_result({"foo": 1})
    with pytest.raises(KeyError):
        ws.resolve_result({"edit": "W999"})
    sc = _ws(tmp_path / "s", mode="scratch")          # other modes unchanged
    assert isinstance(sc.resolve_result({"lora": None}), MockLoRA)
    assert not isinstance(sc.resolve_result({"lora": None}), MockWeightEdit)


def test_save_load_candidate_edit(tmp_path):
    from lora_code.mock import MockLoRA, MockWeightEdit, m_construct
    ws = _ws(tmp_path)
    mods = _down(ws)
    e = ws._get(ws.api.call("w_scale_rows", {"edit": ws.api.call("new_edit", {})["edit"], "module": mods[0],
                                             "factors": [1.2] * 4, "rows": [0, 1, 2, 3]})["edit"])
    p = ws.save_candidate("g00_t0_s00", e)
    assert p.name == "g00_t0_s00.wedit.json" and p == ws.cand_path("g00_t0_s00", "weights")
    back = ws.load_candidate("g00_t0_s00")
    assert isinstance(back, MockWeightEdit) and back.digest() == e.digest() and back.to_json() == e.to_json()
    assert MockWeightEdit.from_json(json.loads(p.read_text())).kind == "weights"
    assert ws._get("C:g00_t0_s00", "edit").digest() == e.digest()
    h = ws.api.call("candidate", {"cid": "g00_t0_s00"})
    assert "edit" in h and h["edit"].startswith("W")
    lo = m_construct(MockLoRA(), mods[1], 0.5, "rank1")         # a LoRA candidate is still found
    ws.save_candidate("g00_t1_s00", lo)
    assert type(ws.load_candidate("g00_t1_s00")) is MockLoRA and ws.load_candidate("nope") is None
    assert MockWeightEdit().digest() != MockLoRA().digest()      # an empty edit and an empty LoRA differ


def test_program_build_weights(tmp_path):
    from lora_code.mock import make_mock_policy
    from lora_code.tools import classify, program_build
    ws = _ws(tmp_path, budget={"evaluate": 4, "gpu_seconds": 5000})
    pol = make_mock_policy(42, skill=1.0, mode="weights")
    progs = {pol([{"role": "user", "content": f"[thread={t} step={s}]"}]) for t in range(3) for s in range(3)}
    assert len(progs) > 3 and all('return {"edit": e}' in p for p in progs)
    recipes = set()
    for i, prog in enumerate(sorted(progs)[:5]):
        art, err = program_build(ws, prog, f"g00_t{i}_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / f"w{i}")
        assert err is None, err
        assert art["strategy"].startswith("weights") and art["max_rank"] == 0 and art["kind"] == "weights"
        assert art["path"].endswith(".wedit.json") and not art["keep"]
        recipes.add(prog.splitlines()[0])
    assert len(recipes) > 1
    lora_prog = ("def main(api):\n    m = [x['module'] for x in api.modules() if x['leaf'] == 'down_proj'][0]\n"
                 "    s = api.svd(module=m, k=4)\n    lo = api.new_lora()['lora']\n"
                 "    lo = api.scale_singular(lora=lo, module=m, svd=s['handle'], idx=[0], factors=[1.05])['lora']\n"
                 "    return {'lora': lo}\n")
    art, err = program_build(ws, lora_prog, "g00_t9_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "wl")
    assert err is None, err
    assert art["kind"] == "weights" and art["modules"] == 1 and art["strategy"] == "pure_construct"
    assert ws.load_candidate("g00_t9_s00").ops[-1][0] == "w_add_lora"
    keep = "def main(api):\n    return {'edit': None}\n"
    art, err = program_build(ws, keep, "g00_t8_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "wk")
    assert err is None and art["keep"] and art["modules"] == 0
    over = ("def main(api):\n    e = api.new_edit()['edit']\n    m = [x['module'] for x in api.modules()][0]\n"
            "    return {'edit': api.w_scale(edit=e, modules=[m], s=1.5)['edit']}\n")
    art, err = program_build(ws, over, "g00_t7_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "wo")
    assert art is None and "sandbox checks failed" in err
    assert classify(["new_edit", "w_scale"]) == "weights_only"
    assert classify(["new_lora", "scale_singular", "new_edit", "w_add_lora"]) == "weights+construct"
    assert classify(["new_edit", "w_scale_rows", "scale"]) == "weights+ops"
    assert classify(["scale_singular", "truncate_rank", "w_add_lora"]) == "weights+construct+ops"
    assert classify(["scale_singular"]) == "pure_construct"
    assert classify(["tensor_compute", "tensor_compute.writes"]) == "weights_only"
    assert classify(["tensor_compute", "tensor_compute.lora", "tensor_compute.writes"]) == "weights+construct"


def test_evaluate_tool_with_edits(tmp_path):
    from lora_code.mock import m_construct
    ev = _Ev()
    ws = _ws(tmp_path, evaluator=ev, budget={"evaluate": 2, "gpu_seconds": 500})
    mod = _down(ws)[0]
    e = ws.api.call("w_scale", {"edit": ws.api.call("new_edit", {})["edit"], "modules": [mod], "s": 1.01})["edit"]
    fb = ws.api.call("evaluate", {"edit": e})
    assert fb["score"] is not None and ws.spent["evaluate"] == 1
    ws.api.call("evaluate", {"edit": e})                          # identical edit: free
    assert ws.spent["evaluate"] == 1
    ws.api.call("evaluate", {"edit": ws.api.call("new_edit", {})["edit"]})   # keep: free
    assert ws.spent["evaluate"] == 1
    lo = m_construct(ws._empty(), mod, 0.5, "rank1")
    hl = ws._new("L", lo)
    ws.api.call("evaluate", {"lora": hl})                         # a LoRA is evaluated as the edit that adds it
    assert ws.spent["evaluate"] == 2
    assert ws.lora_name(ws.resolve_result({"lora": hl})) in ev.reg
    sc, fb2 = ws.evaluate_lora(ws._get(e), "screen", charge=True)   # charged path dedupes with the tool
    assert sc == fb["score"] and ws.spent["evaluate"] == 2
    r = ws.api.call("train_loss", {"edit": e, "n": 8})
    assert r["n"] == 8 and r["parent_loss"] == 1.2
    big = ws._get(ws.api.call("w_scale", {"edit": e, "modules": [mod], "s": 1.3})["edit"])
    sc, fb3 = ws.evaluate_lora(big, "full", charge=False)
    assert sc is None and "edit failed the sandbox checks" in fb3["error"]


def test_system_prompt_modes():
    from lora_code.prompts import MODE_TEXT, system_prompt
    for mode in ("edit", "scratch", "weights"):
        assert {"action", "goal", "strategies", "what", "return_line", "limits", "objects", "unit"} <= set(MODE_TEXT[mode])
    cfg = {"lora": {}, "sandbox": {}, "weights": {"norm_cap_tau": 0.03}}
    b = {"evaluate": 4, "gpu_seconds": 10}
    pw = system_prompt(cfg, budget=b, mode="weights")
    ps = system_prompt(cfg, budget=b, mode="scratch")
    pe = system_prompt(cfg, budget=b, mode="edit")
    assert "w_scale_rows" in pw and '{"edit"' in pw and "- new_edit(" in pw and "0.03 per module" in pw
    assert "- raw_lora(" not in pw and "dW = B A per linear module" not in pw
    assert "w_scale_rows" not in ps and '{"edit"' not in ps and '{"lora": handle, "note"' in ps
    assert "dW = B A per linear module" in ps and "rank <= 64" in ps
    assert "w_scale" not in pe and '{"lora"' in pe and "- raw_lora(" in pe
    plain = system_prompt(dict(cfg, tools={"paper_tools": False, "tensor_compute": False}), budget=b, mode="weights")
    assert "{" not in plain.split("Tools")[0].replace('{"edit": handle, "note": "..."}', "").replace(
        '{"edit": None, "note": "..."}', "")                          # no unfilled placeholder


def test_summarize_candidate_edit():
    from types import SimpleNamespace
    from lora_code.prompts import summarize_candidate
    c = SimpleNamespace(cid="g00_t0_s00", thread=0, step=0, screen={"score": 0.1, "feedback": {}}, full=None,
                        artifact={"strategy": "weights_only", "kind": "weights", "modules": 3}, error=None)
    s = summarize_candidate(c)
    assert s["strategy"] == "weights_only" and s["kind"] == "weights" and s["modules"] == 3
    c.artifact = {"strategy": "pure_construct"}
    assert "kind" not in summarize_candidate(c)


def test_random_weight_search(tmp_path):
    from lora_code import baselines as BL
    from lora_code.mock import MockWeightEdit
    ev = _Ev()
    ws = _ws(tmp_path, evaluator=ev)
    best, log = BL.random_weight_search(ws, evals=6, seed=42, top_full=2)
    assert len(log["points"]) == 6 and isinstance(best, MockWeightEdit) and best.modules()
    scored = [p for p in log["points"] if p.get("score") is not None]
    assert scored and log["best"]["score"] == max(p["score"] for p in scored)
    assert all(p["kind"] in ("rows", "cols", "scale") for p in log["points"])
    assert any(lvl == "full" for _, lvl in ev.calls)
    assert ws.check(best)[0] and ws.spent.get("evaluate", 0) == 0     # not charged to the editor budget
    best2, log2 = BL.random_weight_search(_ws(tmp_path / "b", evaluator=_Ev()), evals=6, seed=42, top_full=2)
    assert best2.digest() == best.digest()                             # deterministic
