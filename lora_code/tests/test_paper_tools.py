"""Paper B simple v2 (2026-11-21): the FINAL gate-per-candidate flow settings, the screen tools over the current
model's own answers, the paper-style tools (save_edit / load_edit / list_edits / drop_edit, output_matrices, unit,
direction, project_out, acts(source='own'), tensor_compute(vectors=...)), the persistent exploration api, the
prompt text and the end-of-generation acceptance. Quick (mock workspaces; one real-store project_out check).

  python3 -I -m pytest -q lora_code/tests/test_paper_tools.py
"""
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

MISSING = [m for m in ("torch", "safetensors", "yaml") if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(bool(MISSING), reason=f"missing {MISSING}")

DS = "simple_math_pool"
SIMPLE = {"lora": {"rank_cap": 64, "norm_cap_tau": 0.1}, "weights": {"norm_cap_tau": 0.05},
          "search": {"threads": 8, "steps": 1, "charge_screens": False, "program_evaluate_calls": 0,
                     "screen_candidates": False, "gate_every_candidate": True, "top_full": 0},
          "tools": {"tensor_compute": True}, "editor": {"tool_rounds": 20, "explore": {"persistent": True}}}


def _screen():
    def rec(i, scores):
        n = len(scores)
        return {"id": f"s/{i}", "score": sum(scores) / n, "scores": list(scores), "sample_scores": list(scores),
                "raw_output": [f"answer {i}.{k} \\boxed{{{k}}}" for k in range(n)],
                "reasoning": [f"thinking {i}.{k} " * 400 for k in range(n)],
                "finish_reason": ["stop"] * (n - 1) + ["length" if i == 0 else "stop"],
                "parsed_answer": [str(k) for k in range(n)]}
    recs = [rec(0, [0, 0, 0, 0]), rec(1, [1, 1, 1, 1]), rec(2, [1, 0, 1, 0]), rec(3, [0, 0, 0, 0])]
    items = {f"s/{i}": {"prompt": f"What is {i}+{i}?", "reference": str(2 * i),
                        "messages": [{"role": "user", "content": f"What is {i}+{i}?"}]} for i in range(4)}
    return {"records": {DS: recs}, "items": {DS: items}}


def _ws(tmp_path, mode="scratch", cfg=None, store=None, screen=True):
    from lora_code.mock import MockStore
    from lora_code.primitives.train import TrainData
    from lora_code.tools import Workspace
    train = TrainData({f"t/{i}": {"messages": [{"role": "user", "content": "q"}], "target": "a", "source": "gold"}
                       for i in range(10)})
    c = json.loads(json.dumps(SIMPLE))
    c.update(cfg or {})
    ws = Workspace(mock=store is None, store=store or MockStore(), train=train, cfg=c, out_dir=tmp_path,
                   budget={"evaluate": 0, "gpu_seconds": 500}, mode=mode)
    ws.screen = _screen() if screen else None
    ws.build_api()
    return ws


def _call(ws, tool, **kw):
    return ws.api.call(tool, kw)


def test_settings_and_tool_surface():
    """base.yaml (the simple flow, folded 2026-11-22) resolves to the FINAL flow; the code defaults of the search
    knobs (screen_candidates true, gate_every_candidate false) keep the screen / submit tool surface."""
    import yaml
    from lora_code.tools import search_settings, tools_for_mode
    simple = yaml.safe_load(open(REPO / "lora_code" / "configs" / "base.yaml"))
    ss = search_settings(simple)
    assert (ss["threads"], ss["steps"], ss["max_candidates"]) == (8, 1, 8)
    assert not ss["screen_candidates"] and ss["gate_every_candidate"] and not ss["evaluate_tool"] and ss["pool"] == 0
    assert simple["decoding"]["screen"]["n"] == 4 and simple["decoding"]["gate"]["n"] == 4
    assert simple["editor"]["tool_rounds"] == 20 and simple["editor"]["explore"]["persistent"] is True
    new = {"screen_results", "screen_output", "save_edit", "load_edit", "list_edits", "drop_edit", "output_matrices",
           "unit", "direction", "project_out"}
    for mode in ("scratch", "weights", "edit"):
        t = tools_for_mode(mode, cfg=simple)
        assert new <= set(t) and "evaluate" not in t and "source='own'" in t["acts"]
        assert "vectors=None" in t["tensor_compute"] and "level='gate'" in t["compare"]
    knobs = dict(simple, search={}, tools={"paper_tools": False})        # code defaults of the search knobs
    old = search_settings(knobs)
    assert old["screen_candidates"] and not old["gate_every_candidate"] and old["evaluate_tool"]
    t = tools_for_mode("scratch", cfg=knobs)
    assert not new & set(t) and "evaluate" in t and "source=" not in t["acts"]
    off = tools_for_mode("scratch", cfg=dict(simple, tools={"paper_tools": False, "screen_results": False}))
    assert not new & set(off)


def test_screen_tools(tmp_path):
    ws = _ws(tmp_path)
    rows = _call(ws, "screen_results")
    assert [r["id"] for r in rows] == ["s/0", "s/1", "s/2", "s/3"] and rows[0]["gold"] == "0"
    assert rows[0]["prompt"] == "What is 0+0?" and rows[0]["scores"] == [0, 0, 0, 0] and len(rows[0]["answers"]) == 4
    assert [r["id"] for r in _call(ws, "screen_results", filter="wrong")] == ["s/0", "s/3"]
    assert [r["id"] for r in _call(ws, "screen_results", filter="unstable")] == ["s/2"]
    assert [r["id"] for r in _call(ws, "screen_results", ids=["s/1", "s/2"], filter="right")] == ["s/1"]
    out = _call(ws, "screen_output", id="s/2", sample=1)
    assert out["has_thinking"] and out["text"].startswith("<think>\nthinking 2.1") and out["next_offset"] == 4000
    page = _call(ws, "screen_output", id="s/2", sample=1, offset=out["next_offset"])
    assert "answer 2.1" in page["text"] and "next_offset" not in page
    with pytest.raises(PermissionError):
        _call(ws, "screen_output", id="t/1")
    ws2 = _ws(tmp_path / "b", screen=False)
    with pytest.raises(RuntimeError, match="no screen results"):
        _call(ws2, "screen_results")


def test_acts_own_and_screen_prompts(tmp_path):
    ws = _ws(tmp_path)
    mod = "layers.40.mlp.down_proj"
    a = _call(ws, "acts", module=mod, ids=["s/0", "s/3"], source="own")
    b = _call(ws, "acts", module=mod, ids=["s/1"], source="own", sample="all")
    p = _call(ws, "acts", module=mod, ids=["s/1"])                 # source 'prompt' on a screen id
    t = _call(ws, "acts", module=mod, ids=["t/1"])                 # train ids as before
    assert a["tokens"] == 1600 and b["tokens"] == 3200 and p["tokens"] == 200 and t["tokens"] == 200
    texts = ws._screen_texts(["s/2"], own=True, sample="all")
    assert len(texts) == 4 and texts[1].startswith("What is 2+2?\n\n<think>\nthinking 2.1") and "answer 2.1" in texts[1]
    with pytest.raises(PermissionError):
        _call(ws, "acts", module=mod, ids=["t/1"], source="own")   # own: screen ids only
    with pytest.raises(PermissionError):
        _call(ws, "acts", module=mod, ids=["x/9"])
    with pytest.raises(ValueError, match="sample"):
        _call(ws, "acts", module=mod, ids=["s/1"], source="own", sample=7)
    assert ws.spent["gpu_seconds"] > 0
    d = _call(ws, "direction", a=a["handle"] + ".out_mean", b=b["handle"] + ".out_mean")
    assert d["size"] == ws.store.shape(mod)[0] and d["vector"].endswith(".dir")
    u = _call(ws, "unit", ref=a["handle"] + ".in_mean")
    assert u["size"] == ws.store.shape(mod)[1]
    with pytest.raises(ValueError, match="equal"):
        _call(ws, "direction", a=a["handle"] + ".out_mean", b=a["handle"] + ".in_mean")
    with pytest.raises(ValueError, match="identical"):
        _call(ws, "direction", a=d["vector"], b=d["vector"])


def test_output_matrices_project_out_and_classify(tmp_path):
    from lora_code.tools import classify
    ws = _ws(tmp_path)
    mats = _call(ws, "output_matrices", layer=3)
    assert {m["kind"] for m in mats} == {"o_proj", "down_proj"} and all(m["shape"][0] == 5120 for m in mats)
    assert {m["kind"] for m in _call(ws, "output_matrices", layer=0)} == {"out_proj", "down_proj"}
    mod = "layers.3.mlp.down_proj"
    s = _call(ws, "svd", module=mod, k=2)
    lo = _call(ws, "project_out", target=_call(ws, "new_lora")["lora"], module=mod, u=s["handle"] + ".u0", alpha=0.5)
    assert ws._get(lo["lora"]).modules() == [mod]
    with pytest.raises(ValueError, match="OUTPUT side"):
        _call(ws, "project_out", target=lo["lora"], module=mod, u=s["handle"] + ".v0")
    assert classify(["new_lora", "svd", "project_out"]) == "pure_construct"
    wsw = _ws(tmp_path / "w", mode="weights")
    sw = _call(wsw, "svd", module=mod, k=2)
    e = _call(wsw, "project_out", target=_call(wsw, "new_edit")["edit"], module=mod, u=sw["handle"] + ".u0", alpha=0.2)
    assert wsw._get(e["edit"]).modules() == [mod]
    assert classify(wsw.tool_log["_"]) == "weights_only"


def test_project_out_real_store(tmp_path):
    """W' = W - alpha u u^T W exactly: a LoRA target gets B = -alpha u, A = u^T W (with the target's own change on
    the module); a weights-mode edit target gets the same as a low-rank op after its earlier ops."""
    import torch
    from lora_code.primitives.weights import WeightStore
    from lora_code.tests.test_wedit import DOWN, _write_ckpt
    store = WeightStore(_write_ckpt(tmp_path / "m0", sharded=True))
    ws = _ws(tmp_path / "ws", store=store, screen=False)
    W = store.weight(DOWN).float()
    out_f = W.shape[0]
    u = torch.randn(out_f, generator=torch.Generator().manual_seed(1))
    un = u / u.norm()
    lo = _call(ws, "project_out", target=_call(ws, "new_lora")["lora"], module=DOWN, u=[float(x) for x in u], alpha=0.5)
    L = ws._get(lo["lora"])
    assert torch.allclose(W + L.delta(DOWN), W - 0.5 * torch.outer(un, un) @ W, atol=1e-4)
    lo2 = _call(ws, "project_out", target=lo["lora"], module=DOWN, u=[float(x) for x in u], alpha=1.0)
    W1 = W + L.delta(DOWN)
    assert torch.allclose(W + ws._get(lo2["lora"]).delta(DOWN), W1 - torch.outer(un, un) @ W1, atol=1e-4)
    wsw = _ws(tmp_path / "wsw", mode="weights", store=store, screen=False)
    e0 = _call(wsw, "w_scale", edit=_call(wsw, "new_edit")["edit"], modules=[DOWN], s=1.01)["edit"]
    e1 = _call(wsw, "project_out", target=e0, module=DOWN, u=[float(x) for x in u], alpha=0.3)["edit"]
    ed = wsw._get(e1, "edit")
    assert [o["op"] for o in ed.ops[DOWN]] == ["scale", "lowrank"]
    Ws = 1.01 * W
    assert torch.allclose(ed.apply(DOWN, W), Ws - 0.3 * torch.outer(un, un) @ Ws, atol=1e-4)


def test_save_load_scoped_to_candidate(tmp_path):
    from lora_code.tools import classify
    ws = _ws(tmp_path)
    mod = "layers.40.mlp.down_proj"
    ws.begin_program("g00_t0_s00~x")                    # exploration of candidate g00_t0_s00
    s = _call(ws, "svd", module=mod, k=2)
    lo = _call(ws, "scale_singular", lora=_call(ws, "new_lora")["lora"], module=mod, svd=s["handle"], idx=[0],
               factors=[1.05])["lora"]
    meta = _call(ws, "save_edit", handle=lo, name="best")
    assert meta == {"name": "best", "kind": "lora", "modules": 1, "max_rank": 1}
    _call(ws, "save_edit", handle=lo, name="tmp")
    assert [e["name"] for e in _call(ws, "list_edits")] == ["best", "tmp"]
    assert _call(ws, "drop_edit", name="tmp") == {"dropped": "tmp"}
    with pytest.raises(ValueError):
        _call(ws, "save_edit", handle=lo, name="../x")
    ws.end_program("g00_t0_s00~x")
    assert (tmp_path / "saved" / "g00_t0_s00" / "best.json").exists()
    ws.begin_program("g00_t0_s00")                       # its final program (a later thread / process step)
    got = _call(ws, "load_edit", name="best")
    assert ws._get(got["lora"]).modules() == [mod] and ws._get(got["lora"]).max_rank() == 1
    assert classify(ws.tool_log["g00_t0_s00"]) == "pure_construct"       # the tools that built it count
    ws.end_program("g00_t0_s00")
    ws.begin_program("g00_t1_s00")                       # another candidate cannot see it
    assert _call(ws, "list_edits") == []
    with pytest.raises(KeyError):
        _call(ws, "load_edit", name="best")
    ws.end_program("g00_t1_s00")
    with pytest.raises(RuntimeError):                    # outside any candidate
        _call(ws, "list_edits")


def test_tensor_compute_vectors(tmp_path):
    ws = _ws(tmp_path)
    s = _call(ws, "svd", module="layers.0.linear_attn.in_proj_b", k=2)
    u = _call(ws, "unit", ref=s["handle"] + ".v0")
    r = _call(ws, "tensor_compute", code="result = [float(V[%r].norm()), len(V)]" % u["vector"],
              vectors=[u["vector"], s["handle"] + ".u0"])
    assert r["json"][0] == pytest.approx(1.0, abs=1e-5) and r["json"][1] == 2
    off = _ws(tmp_path / "off", cfg={"tools": {"tensor_compute": True, "paper_tools": False}})
    with pytest.raises(ValueError, match="vectors"):
        _call(off, "tensor_compute", code="result = 1", vectors=[u["vector"]])


def test_persistent_explore_api(tmp_path):
    from lora_code.tools import editor_chat_tools, explore_doc
    ws = _ws(tmp_path)
    made = ws.explore_api("g00_t0_s00~x", persistent=True)
    assert len(made) == 3
    api, cleanup, before = made
    assert "evaluate" not in api.names() and {"screen_results", "save_edit", "project_out"} <= set(api.names())
    before(30)
    h = api.call("new_lora", {})["lora"]
    ws._local.cid = None                                 # another thread / state: each tool call binds the tag
    before(30)
    assert ws._time_left() == pytest.approx(30, abs=2)
    assert api.call("inspect", {"lora": h})["modules"] == 0          # the handle persists across calls
    api.call("save_edit", {"handle": h, "name": "keep"})
    assert (tmp_path / "saved" / "g00_t0_s00" / "keep.json").exists()
    cleanup()
    with pytest.raises(KeyError):
        ws._get(h)
    assert len(ws.explore_api("g00_t0_s00~x01_0")) == 2                # non-persistent: (api, cleanup)
    assert "persist across your run_python calls" in explore_doc(ws)
    assert editor_chat_tools(ws, ws.cfg) is not None


def test_prompts_gate_mode_and_context():
    from lora_code.prompts import ContextRenderer, screen_text, system_prompt
    cfg = json.loads(json.dumps(SIMPLE))
    cfg["decoding"] = {"screen": {"n": 4, "thinking": True}, "gate": {"n": 4, "thinking": True}}
    cfg["gate"] = {"min_gain": 0.01, "require_significant": True}
    sysmsg = system_prompt(cfg, budget={"evaluate": 0, "gpu_seconds": 300}, mode="scratch", gate_n=500)
    assert "there is no api.evaluate" in sysmsg and "evaluated ONCE on a fixed hidden set of 500 items" in sysmsg
    assert "LARGEST gain" in sysmsg and "- evaluate(" not in sysmsg and "api.evaluate draws" not in sysmsg
    assert "source='own'" in sysmsg and "api.project_out" in sysmsg and "api.load_edit(name='best')['lora']" in sysmsg
    assert "answered by YOU (the current model) 4 time(s)" in sysmsg and "this generation's screen ids" in sysmsg
    assert "up to 20 rounds" in sysmsg and "share ONE Python session" in sysmsg
    cfg["editor"]["tool_rounds"] = 7                      # the round count is configurable (prompt follows it)
    assert "up to 7 rounds" in system_prompt(cfg, budget={"evaluate": 0, "gpu_seconds": 300})
    summ = {"n_items": 4, "samples_per_item": 4, "accuracy": 0.375, "correct_counts": {"0": 2, "2": 1, "4": 1},
            "truncation_rate": 0.0625, "truncated_thinking": 0, "failing": [{"id": "s/0", "gold": "0"}]}
    txt = screen_text(summ)
    assert "accuracy avg@4 0.375" in txt and '"s/0"' in txt and screen_text({}) == ""
    ctx = ContextRenderer(task_text="T", diag_text="D", gate_mode=True, screen_text=txt, sessions=8,
                          budget_fn=lambda: {"evaluate_tool": False, "evaluate": 0, "evaluate_left": 0,
                                             "program_evaluate_calls": 0, "gpu_seconds": 300, "gpu_seconds_left": 300})
    c = SimpleNamespace(cid="g00_t0_s00", thread=0, step=0, program="p", artifact={"strategy": "keep"}, error=None,
                        score=0.02, cost={}, full=None,
                        screen={"score": 0.02, "feedback": {"gate": {"passed": True, "gain": 0.02, "ci": [0.001, 0.04],
                                                                     "accuracy": 0.6, "broken": 3, "fixed": 13,
                                                                     "broken_ids": ["g/1"]}}})
    msg = ctx([c], 1)
    assert "session 2 of 8" in msg and "No api.evaluate this generation" in msg and "at most 0" not in msg
    assert "paired gain in gate accuracy" in msg and '"gate": {"passed": true, "gain": 0.02' in msg
    assert "accuracy avg@4 0.375" in msg


def test_accept_best_gate(tmp_path):
    from lora_code.run import Run
    r = Run.__new__(Run)
    import threading
    r._gate_lock = threading.Lock()

    def cand(cid, gain, passed):
        fb = {"gate": {"passed": passed, "gain": gain}}
        return SimpleNamespace(cid=cid, screen={"score": gain, "feedback": fb}, score=gain)
    cs = [cand("g00_t0_s00", 0.03, True), cand("g00_t1_s00", 0.05, True), cand("g00_t2_s00", 0.08, False),
          cand("g00_t3_s00", 0.05, True)]
    for c in cs:
        r.record_gate(tmp_path, c.cid, c.screen["feedback"]["gate"])
    r.record_gate(tmp_path, "g00_t0_s00", cs[0].screen["feedback"]["gate"])        # a re-run replaces, no duplicate
    best = r.accept_best_gate(tmp_path, cs)
    assert best.cid == "g00_t1_s00"                       # largest passing gain; tie -> the earlier cid
    log = json.loads((tmp_path / "gate.json").read_text())
    assert [s["cid"] for s in log["submissions"]] == [c.cid for c in cs] and log["accepted"] == "g00_t1_s00"
    assert [s["accepted"] for s in log["submissions"]] == [False, True, False, False]
    assert Run.accept_best_gate(r, tmp_path / "x", cs[2:3]) is None
    fb = Run.gate_feedback(True, {"scores": {"g": 0.6}, "paired_gain": 0.02, "paired_ci": [0.0, 0.1], "gate_n": 5,
                                  "broken": 1, "fixed": 2, "broken_ids": ["a"], "fixed_ids": ["b", "c"],
                                  "reasons": [], "reference_score": 0.58}, "g")
    assert fb["accuracy"] == 0.6 and fb["reference_accuracy"] == 0.58 and fb["fixed_ids"] == ["b", "c"]
    assert time.time() > 0
