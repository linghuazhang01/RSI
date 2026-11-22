"""Paper A FINAL simple flow (2026-11-21): the incumbent's screen run + one gate evaluation per candidate.

    python3 -I -m pytest -q edit_code/tests/test_gate_flow.py

- screen_results / screen_output show the incumbent's screen records (filters, pages, thinking text); evaluate and
  compare are not registered (program and exploration API);
- gate_stats: accuracy, paired gain, CI, broken / fixed counts and ids (no item content);
- screen_fn of the gate flow: no screen, ONE gate evaluation, gen_XX/gate.json entry, over-budget refused unevaluated;
- accept_best_gate: the passing candidate with the largest gain; failed candidates get a 'not evaluated' entry;
  none passes -> keep;
- prompts: no api.evaluate text, configured tool rounds / threads / steps, persistent session text, gate table,
  archive with gate statistics and the reference screen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_simple_v2 as V  # noqa: E402

GATE_SCFG = {"threads": 2, "steps": 1, "max_evaluate": 0, "charge_screens": False, "program_evaluate_calls": 0,
             "screen_candidates": False, "gate_every_candidate": True, "top_full": 0}


def _jl(recs):
    return "".join(json.dumps(r) + "\n" for r in recs)


def test_search_settings_gate_flow():
    from edit_code.common import search_settings
    ss = search_settings(GATE_SCFG)
    assert (ss["screen_candidates"], ss["gate_every_candidate"], ss["evaluate_tool"], ss["eval_budget"]) == \
        (False, True, False, 0)
    with pytest.raises(ValueError):
        search_settings(dict(GATE_SCFG, gate_every_candidate=False))


def test_screen_tools_and_no_evaluate(tmp_path, monkeypatch):
    import edit_code.problem as P
    from edit_code.tools import make_tool_api
    prob = V._problem(tmp_path, budget=0, cfg={"search": GATE_SCFG})
    prob.datasets = ["simple_math_pool"]
    d = prob.out_dir / "evals" / "incumbent" / "screen"
    d.mkdir(parents=True)
    (d / "simple_math_pool.jsonl").write_text(_jl([
        {"id": "a", "score": 0.75, "scores": [1, 1, 0, 1], "parsed_answer": ["4", "4", None, "4"],
         "raw_output": ["ans 4"] * 4, "reasoning": ["think " * 1000] * 4,
         "finish_reason": ["stop", "stop", "length", "stop"], "truncated_thinking": 1},
        {"id": "b", "score": 0.0, "scores": [0, 0, 0, 0], "parsed_answer": ["1", "2", "1", "3"],
         "raw_output": ["x"] * 4, "reasoning": [None] * 4, "finish_reason": ["stop"] * 4}]))
    monkeypatch.setattr(P, "load_processed", lambda ds, ids=None: {i: {"prompt": f"Q{i}", "reference": "4"}
                                                                   for i in (ids or [])})
    assert prob.screen_tools
    for api in (make_tool_api(prob, {}, cid="p", per_program_evaluate=0, free_repeats=True, evaluate_tool=False),
                make_tool_api(prob, {}, cid="p~x", per_program_evaluate=0, explore=True)):
        names = api.names()
        assert {"screen_results", "screen_output"} <= set(names)
        assert "evaluate" not in names and "compare" not in names
        assert "evaluate" not in api.describe()
    rows = api.call("screen_results", {"filter": "wrong"})
    assert [r["id"] for r in rows] == ["b"] and rows[0]["prompt"] == "Qb" and rows[0]["gold"] == "4"
    assert [r["id"] for r in api.call("screen_results", {"filter": "unstable"})] == ["a"]
    out = api.call("screen_output", {"id": "a", "sample": 0})
    assert out["has_thinking"] and out["text"].startswith("<think>") and out["total_chars"] > len(out["text"])
    nxt = api.call("screen_output", {"id": "a", "sample": 0, "offset": out["next_offset"]})
    assert nxt["offset"] == out["next_offset"]
    s = prob.screen_summary()
    assert s["n_items"] == 2 and s["samples_per_item"] == 4 and s["correct_counts"]["0"] == 1
    # the program API of the old flow keeps evaluate and compare
    old = V._problem(tmp_path / "old", budget=2)
    assert {"evaluate", "compare"} <= set(make_tool_api(old, {}, cid="q", per_program_evaluate=2).names())


def test_gate_stats(tmp_path):
    prob = V._problem(tmp_path, budget=0, cfg={"search": GATE_SCFG})
    inc_p, cand_p = tmp_path / "inc.jsonl", tmp_path / "cand.jsonl"
    inc_p.write_text(_jl([{"id": "a", "score": 1.0}, {"id": "b", "score": 0.0}, {"id": "c", "score": 0.75}]))
    cand_p.write_text(_jl([{"id": "a", "score": 0.25}, {"id": "b", "score": 1.0}, {"id": "c", "score": 1.0}]))
    prob.incumbent = {"gate": {"simple_math_eval": {"score": 0.583, "records": str(inc_p)}}}
    r = {"feedback": {"gate": {"passed": False, "reason": "no significant gain"},
                      "probes": {"math": {"dataset": "simple_math_eval", "score": 0.75, "incumbent": 0.5833,
                                          "vs_incumbent": 0.1667, "delta_ci": [-0.1, 0.4], "records": str(cand_p)}}}}
    g = prob.gate_stats({"bytes_gib": 9.5}, r)
    assert g == {"passed": False, "reason": "no significant gain", "accuracy": 0.75, "reference_accuracy": 0.5833,
                 "gain": 0.1667, "ci": [-0.1, 0.4], "broken": 1, "fixed": 1, "broken_ids": ["a"],
                 "fixed_ids": ["b"], "bytes_gib": 9.5}


def _runner_fns(tmp_path, monkeypatch, prob, seen_apis):
    from edit_code import run as RUN

    class Res:
        ok, result, stdout, stderr, error, seconds, info = True, {"default": {"bits": 4}}, "", "", None, 0.1, {}

    def fake_run(program, api, **k):
        seen_apis.append(api)
        return Res()
    monkeypatch.setattr(RUN, "run_program", fake_run)
    monkeypatch.setattr(RUN, "system_prompt", lambda *a, **k: "SYS")
    monkeypatch.setattr(RUN, "make_render_context", lambda *a, **k: (lambda h, t: "CTX"))
    cfg = {"sandbox": {"timeout_s": 10, "mem_gb": 1}, "editor": {"tool_rounds": 0}, "feedback": {}}
    runner = SimpleNamespace(rc=SimpleNamespace(mock=True, seed=42), cfg=cfg, chat_tools=None,
                             search_history=lambda g, t: None)
    return RUN.Runner.search_fns(runner, prob, tmp_path / "gen", None, task="main", archive="", scfg=GATE_SCFG)


def test_screen_fn_gates_once(tmp_path, monkeypatch):
    prob = V._problem(tmp_path, budget=0, cfg={"search": GATE_SCFG})
    calls = []

    def fake(art, stage, cid, key, charged, origin):
        calls.append((art["config_hash"], stage, origin))
        rec = {"hash": art["config_hash"], "stage": stage, "cid": cid, "score": 1.0, "feedback": {
            "stage": stage, "gate": {"passed": True, "reason": "beats the incumbent"},
            "probes": {"math": {"dataset": "dapo_math", "score": 0.6, "incumbent": 0.5, "vs_incumbent": 0.1,
                                "delta_ci": [0.05, 0.15], "records": "/nonexistent.jsonl"}}}}
        with prob._lock:
            prob._evals[key] = rec
        return dict(rec)
    monkeypatch.setattr(prob, "_evaluate", fake)
    monkeypatch.setattr(prob, "build", lambda result, cid: {"config_hash": f"h{cid}", "bytes_gib": 1.0,
                                                            "budget_gib": 10.0, "linear_avg_bits": 4.0})
    apis = []
    fns = _runner_fns(tmp_path, monkeypatch, prob, apis)
    art, err = fns["build_fn"]("def main(api): pass", "g01_t0_s00")
    assert err is None and "evaluate" not in apis[-1].names()
    sc, fb = fns["screen_fn"](art, "g01_t0_s00")
    assert calls == [("hg01_t0_s00", "gate", "gate")]                       # no screen, one gate evaluation
    assert sc == 0.1 and fb["stage"] == "gate" and fb["gate"]["passed"] and fb["gate"]["gain"] == 0.1
    assert "records" not in json.dumps(fb["probes"])                          # no record paths for the editor
    big = dict(art, config_hash="hbig", bytes_gib=99.0)                       # over the budget: refused, unevaluated
    sc2, fb2 = fns["screen_fn"](big, "g01_t1_s00")
    assert len(calls) == 1 and sc2 is None and not fb2["gate"]["passed"] and "refused" in fb2["gate"]["reason"]
    subs = json.loads((tmp_path / "gen" / "gate.json").read_text())["submissions"]
    assert [(s["cid"], s["passed"]) for s in subs] == [("g01_t0_s00", True), ("g01_t1_s00", False)]
    assert subs[0]["info"]["gain"] == 0.1 and subs[0]["info"]["ci"] == [0.05, 0.15]


def test_accept_best_gate(tmp_path, monkeypatch):
    from rsicore.search import Candidate
    from edit_code import run as RUN
    prob = V._problem(tmp_path, budget=0, cfg={"search": GATE_SCFG})
    monkeypatch.setattr(prob, "evaluate", lambda *a, **k: {"feedback": {"probes": {}}})
    gd = tmp_path / "gen_01"
    gd.mkdir()

    def cand(cid, h, err=None):
        return Candidate(cid=cid, thread=int(cid[5]), step=0, program="", artifact=None if err else
                         {"config_hash": h, "config": {}, "bytes_gib": 1.0}, error=err,
                         screen={"score": 0.0, "feedback": {"stage": "gate"}}, full=None, score=0.0)
    ranked = [cand("g01_t0_s00", "h0"), cand("g01_t1_s00", "h1"), cand("g01_t2_s00", "h2"),
              cand("g01_t3_s00", None, err="program failed: boom")]
    subs = [("g01_t0_s00", True, 0.02), ("g01_t1_s00", True, 0.05), ("g01_t2_s00", False, 0.1)]
    (gd / "gate.json").write_text(json.dumps({"submissions": [
        {"cid": c, "passed": p, "info": {"reason": "r", "gain": x}} for c, p, x in subs]}))
    marks = {}
    runner = SimpleNamespace(st=SimpleNamespace(mark=lambda k, v: marks.update({k: v})), log=lambda *a: None,
                             incumbent_record=lambda g, cid, art, fb, md, accepted: {"cid": cid})
    RUN.Runner.accept_best_gate(runner, prob, 1, gd, ranked, {"config_hash": "hinc"}, 10.0)
    acc = json.loads((gd / "accepted.json").read_text())
    assert acc["accepted"] and acc["cid"] == "g01_t1_s00" and acc["gain"] == 0.05      # largest PASSING gain
    assert json.loads((gd / "incumbent.json").read_text()) == {"cid": "g01_t1_s00"}
    log = {s["cid"]: s for s in json.loads((gd / "gate.json").read_text())["submissions"]}
    assert len(log) == 4 and "not evaluated: program failed" in log["g01_t3_s00"]["info"]["reason"]
    # none passes -> the incumbent stays
    (gd / "gate.json").write_text(json.dumps({"submissions": [
        {"cid": c, "passed": False, "info": {"reason": "small", "gain": x}} for c, _, x in subs]}))
    RUN.Runner.accept_best_gate(runner, prob, 1, gd, ranked, {"config_hash": "hinc", "cid": "q0"}, 10.0)
    acc = json.loads((gd / "accepted.json").read_text())
    assert not acc["accepted"] and len(acc["gate_reason"]) == 4
    assert json.loads((gd / "incumbent.json").read_text()) == {"config_hash": "hinc", "cid": "q0", "accepted": False}


def _fake_problem():
    return SimpleNamespace(space={"bits": [2, 4, 16], "group_sizes": [128], "quantizers": ["rtn", "gptq"]},
                           data={"simple_math_pool": {"screen": ["i"] * 20}}, datasets=["simple_math_pool"],
                           gate_datasets=["simple_math_eval"], _ids=lambda ds, st: ["x"] * 500,
                           decoding={"screen": {"n": 4}, "gate": {"n": 4}}, budget_gib=10.0, gen=2, evals_left=0,
                           noise={}, incumbent=None, domains=["math"],
                           screen_summary=lambda examples=3: {"n_items": 20, "samples_per_item": 4, "accuracy": 0.6,
                                                              "correct_counts": {"0": 5, "4": 15},
                                                              "truncation_rate": 0.1, "truncated_thinking": 2,
                                                              "failing": [{"id": "i7", "gold": "4"}]})


@pytest.mark.parametrize("rounds", [20, 7])
def test_system_prompt_gate_flow(rounds):
    from edit_code.common import search_settings
    from edit_code.prompts import system_prompt
    cfg = {"gate": {"min_delta": 0.005, "require_significant": True}, "tools": {},
           "editor": {"tool_rounds": rounds, "explore": {"persistent": True}}}
    text = system_prompt(_fake_problem(), "- bytes(config)", cfg, history=True, search=search_settings(GATE_SCFG))
    assert "api.evaluate(" not in text and "evaluate(config)" not in text
    assert "There is no api.evaluate." in text and "compare(cid)" not in text
    assert f"for up to {rounds} rounds" in text
    assert "20 items x 4 sampled answers" in text and "500 items x 4 sampled answers" in text
    assert "LARGEST gain" in text and "persist between your run_python calls" in text
    assert "so you do not repeat a config" in text
    # the old flow keeps its text (evaluate, compare, nothing persists)
    old = system_prompt(_fake_problem(), "- bytes(config)", dict(cfg, editor={"tool_rounds": rounds}), history=True,
                        search=search_settings({"threads": 4, "steps": 6, "max_evaluate": 24}))
    assert "api.evaluate(config=cfg)" in old and "compare(cid)" in old and "Nothing persists" in old


def test_render_gate_table_and_archive():
    from rsicore.search import Candidate
    from edit_code.common import search_settings
    from edit_code.prompts import archive_text, make_render_context
    cfg = {"feedback": {"examples": 2}}
    g = {"passed": False, "reason": "no significant gain over the incumbent (math ...)", "accuracy": 0.61,
         "gain": 0.004, "ci": [-0.01, 0.02], "broken": 9, "fixed": 11, "broken_ids": ["e1"], "fixed_ids": ["e2"]}
    c = Candidate(cid="g02_t0_s00", thread=0, step=0, program="", artifact={"bytes_gib": 9.9, "linear_avg_bits": 4.1,
                  "config": {"default": {"bits": 4}}}, error=None,
                  screen={"score": 0.004, "feedback": {"stage": "gate", "gate": g}}, full=None, score=0.004)
    render = make_render_context(_fake_problem(), cfg, search=search_settings(dict(GATE_SCFG, steps=3)))
    text = render([c], 0)
    assert "failed gain +0.0040 [-0.0100, +0.0200], broken 9 / fixed 11" in text
    assert "accuracy avg@4 0.6" in text and "screen_results(filter='wrong'" in text
    assert "improve this thread's best config" in text and "2 session(s) x 3 step(s)" in text
    from edit_code.prompts import ARCHIVE_HEAD, ARCHIVE_READ, archive_row
    bad = Candidate(cid="g01_t1_s00", thread=1, step=0, program="", artifact=None, error="program failed: boom\nmore",
                    screen=None, full=None, score=None)
    c1 = Candidate(cid="g01_t0_s00", thread=0, step=0, program="", artifact={
        "bytes_gib": 9.9, "linear_avg_bits": 4.1, "note": "8 bit down 0-3: largest herr",
        "config": {"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"},
                   "groups": {"0.down": {"bits": 8}, "1.down": {"bits": 8}, "3.down": {"bits": 8}}}},
        error=None, screen=None, full=None, score=0.004)
    ent = {"gen": 1, "accepted": False, "gate_reason": ["x"], "n": 2, "gate_every": True, "best": [(c1.cid, 0.004)],
           "reference_screen": {"accuracy": 0.5, "correct_counts": {"0": 1}, "truncation_rate": 0.0, "n_items": 500,
                                "samples_per_item": 4},
           "rows": [archive_row(c1, g), archive_row(bad, {"passed": False, "reason": "not evaluated: program failed"})],
           "configs": [(c1.cid, c1.artifact["config"])]}
    arch = archive_text([ent])
    assert arch.startswith(ARCHIVE_HEAD) and arch.endswith(ARCHIVE_READ)
    assert "screen_results(gen=1)" in arch and "over 500 items x 4 samples" in arch
    assert '"gain": 0.004' in arch and '"ci": [-0.01, 0.02]' in arch and "8 bit down 0-3: largest herr" in arch
    assert "down L0-1,3 x3" in arch                                    # config_brief with layer ranges
    assert '"error": "program failed: boom"' in arch and "not evaluated" not in arch   # failures listed, one line
    assert "full config of" not in arch and "full config of g01_t0_s00" in archive_text([ent], read_tools=False)
    many = archive_text([dict(ent, gen=t) for t in (1, 2, 3)], max_chars=len(arch) + 400)
    assert "rows omitted for length; search_history(gen=1)" in many and "round 3" in many
