"""Paper A simple v2 (2026-11-21): scoring decoding, free screens + the api.evaluate pool, the 8 x 1 profile.

    python3 -I -m pytest -q edit_code/tests/test_simple_v2.py

- search_settings: the code defaults reproduce the old accounting; program_evaluate_calls null does not crash;
- the real base.yaml (the folded simple profile, 2026-11-22) resolves to 8 threads x 1 step, 8 candidates, unscreened candidates with one gate
  evaluation each, no api.evaluate, top_full 0, thinking decoding (screen / gate / held-out n 4), candidate servers at
  32768 tokens and a persistent exploration session (FINAL flow, 2026-11-21);
- make_tool_api(free_repeats=True): an already-evaluated config is free, the per-program cap and the pool hold,
  failed calls cost nothing; without it every call costs one (as before);
- QuantProblem passes the stage decoding to every score() call, reuses records only under the same decoding, and
  samples the incumbent's screen records once for all candidates.
"""
from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "thinking": True, "max_tokens": 16384}


def _load(tmp_path, sets, mock_effective=False):
    """(paper_cfg, effective cfg) of a Paper A --mock invocation (paper_cfg = no mock overrides)."""
    from rsicore import config as C
    from edit_code import run as RUN
    ap = C.build_arg_parser("A")
    ap.add_argument("--budget-gib", type=float, default=None)
    args = ap.parse_args(["--domain", "math", "--mock", "--runs-dir", str(tmp_path),
                          "--run-id", "cfg_" + str(abs(hash(tuple(sets))))] + [x for s in sets for x in ("--set", s)])
    rc = C.load_run_config("A", args)
    return rc.paper_cfg, RUN.effective_config(rc, args)


def _mml(args):
    return int(args[args.index("--max-model-len") + 1])


def test_search_settings_defaults():
    from edit_code.common import search_settings
    old = search_settings({"threads": 4, "steps": 6, "max_evaluate": 24, "program_evaluate_calls": 2, "top_full": 3})
    assert old == {"threads": 4, "steps": 6, "max_candidates": 24, "charge_screens": True, "per_program": 2,
                   "eval_budget": 24, "top_full": 3, "rerun_seeds": 0, "screen_candidates": True,
                   "gate_every_candidate": False, "evaluate_tool": True}
    # charged screens: max_evaluate still caps the candidates (the old max_candidates default)
    assert search_settings({"threads": 4, "steps": 6, "max_evaluate": 12})["max_candidates"] == 12
    assert search_settings({"threads": 2, "steps": 2, "max_evaluate": 4, "max_candidates": None})["max_candidates"] == 4
    free = search_settings({"threads": 8, "steps": 1, "max_evaluate": 24, "charge_screens": False,
                            "program_evaluate_calls": 2})
    assert (free["max_candidates"], free["eval_budget"], free["per_program"]) == (8, 16, 2)   # 8 x 2
    nocap = search_settings({"threads": 8, "steps": 1, "max_evaluate": 24, "charge_screens": False,
                             "program_evaluate_calls": None})
    assert (nocap["per_program"], nocap["eval_budget"]) == (None, 24)                          # no crash
    assert search_settings({"threads": 3, "steps": 1, "max_evaluate": 4, "charge_screens": False,
                            "program_evaluate_calls": None, "program_evaluate_total": 5})["eval_budget"] == 5


def test_simple_profile_resolution(tmp_path):
    """The real base.yaml (no mock overrides): 8 x 1, decoding, servers; the search knobs still switch flows."""
    from rsicore.config import stage_decoding
    from edit_code.common import search_settings
    from edit_code.run import server_args
    cfg, eff = _load(tmp_path, [])
    # FINAL flow (2026-11-21): candidates not screened, one gate evaluation each, no api.evaluate
    assert search_settings(cfg["search"]) == {"threads": 8, "steps": 1, "max_candidates": 8, "charge_screens": False,
                                              "per_program": 0, "eval_budget": 0, "top_full": 0, "rerun_seeds": 0,
                                              "screen_candidates": False, "gate_every_candidate": True,
                                              "evaluate_tool": False}
    assert cfg["editor"]["explore"]["persistent"] is True
    assert stage_decoding(cfg, "screen", 42) == dict(THINK, n=4, seed=42)                   # the incumbent's screen
    assert stage_decoding(cfg, "full", 42) == dict(THINK, n=4, seed=42)                      # full = screen
    assert stage_decoding(cfg, "gate", 42) == dict(THINK, n=4, seed=42)
    hd = cfg["heldout"]["decoding"]
    assert {k: hd[k] for k in ("temperature", "top_p", "top_k", "thinking", "max_tokens", "n", "seed")} == \
        dict(THINK, n=4, seed=42)
    assert not hd.get("n_by_task") and cfg["heldout"]["enabled"] is True
    assert "data_roles" not in cfg and cfg["editor"]["tool_rounds"] == 20
    assert _mml(server_args(cfg, "candidate_args")) == 32768
    assert _mml(server_args(cfg, "editor_args")) == 131072
    for hw in ("h20", "h20_80"):                                   # the hardware profiles' candidate args too
        c, _ = _load(tmp_path / hw, [f"hardware={hw}"])
        args = server_args(c, "candidate_args")
        assert _mml(args) == 32768 and args.count("--max-model-len") == 1
        assert _mml(server_args(c, "editor_args")) >= 32768
    # --mock: mock_overrides (the small layout, 3 x 1) win over the profile
    ms = search_settings(eff["search"])
    assert (ms["threads"], ms["steps"], ms["max_candidates"], ms["charge_screens"], ms["per_program"],
            ms["eval_budget"], ms["top_full"], ms["screen_candidates"], ms["gate_every_candidate"],
            ms["evaluate_tool"]) == (3, 1, 3, False, 0, 0, 0, False, True, False)
    # the search knobs (kept, 2026-11-22): charged screens cap the candidates at max_evaluate
    tg, _ = _load(tmp_path / "tg", ["search.charge_screens=true", "search.screen_candidates=true",
                                    "search.max_candidates=null", "search.threads=4", "search.steps=6",
                                    "search.program_evaluate_calls=2", "decoding.screen=null"])
    assert stage_decoding(tg, "screen", 42) is None
    assert search_settings(tg["search"])["max_candidates"] == search_settings(tg["search"])["eval_budget"] == 24


# ---------------------------------------------------------------------------------- free screens + program pool
def _problem(tmp, budget, cfg=None, n_gpus=2):
    from edit_code.problem import QuantProblem
    rc = SimpleNamespace(mock=True, run_dir=tmp, run_id="t", seed=42, gpus=list(range(n_gpus)))
    base = {"gpu_layout": {"candidate_group_size": 1}, "candidates": {"keep_checkpoints": 1}}
    return QuantProblem(rc, dict(base, **(cfg or {})), tag="t", gen=1, out_dir=tmp / "p", prep_dir=tmp / "prep",
                        source_dir=None, source_state=None, budget_gib=10.0, datasets=["dapo_math"], data={},
                        kl_texts=[], refs_dir=tmp / "refs", eval_budget=budget)


def _fake_evaluate(prob, calls):
    def fake(art, stage, cid, key, charged, origin):
        calls.append((art["config_hash"], stage, charged))
        rec = {"hash": art["config_hash"], "stage": stage, "cid": cid, "score": 0.5, "feedback": {},
               "charged": charged}
        with prob._lock:
            prob._evals[key] = rec
        return dict(rec)
    return fake


def test_free_screens_program_pool_and_cap(tmp_path, monkeypatch):
    from rsicore.sandbox import BudgetExceeded
    from edit_code.tools import ToolError, make_tool_api
    prob = _problem(tmp_path, budget=3)                            # the generation's api.evaluate pool: 3
    calls = []
    monkeypatch.setattr(prob, "_evaluate", _fake_evaluate(prob, calls))

    def ev(api, h):
        return api.call("evaluate", {"config": {"config_hash": h}})
    late = make_tool_api(prob, {}, cid="p4", per_program_evaluate=2, free_repeats=True)   # cap 2, made early
    # automatic screens (charge=False) never spend the pool
    for h in ("a", "b"):
        prob.evaluate({"config_hash": h}, "screen", f"c{h}", charge=False)
    assert prob.evals_used == 0 and prob.evals_left == 3
    p1 = make_tool_api(prob, {}, cid="p1", per_program_evaluate=2, free_repeats=True)
    assert ev(p1, "a")["cached"] is True and p1.spent["evaluate"] == 0      # already screened: free
    assert ev(p1, "c")["cached"] is False and p1.spent["evaluate"] == 1
    assert ev(p1, "c")["cached"] is True and p1.spent["evaluate"] == 1      # the same config again: free
    ev(p1, "d")
    with pytest.raises(BudgetExceeded):                                      # per-program cap 2
        ev(p1, "e")
    assert ("e", "screen", True) not in calls and prob.evals_used == 2
    p2 = make_tool_api(prob, {}, cid="p2", per_program_evaluate=2, free_repeats=True)
    assert p2.budget["evaluate"] == 1                                        # min(cap, pool left)
    ev(p2, "e")
    assert prob.evals_used == 3 and prob.evals_left == 0
    p3 = make_tool_api(prob, {}, cid="p3", per_program_evaluate=2, free_repeats=True)
    assert ev(p3, "a")["cached"] is True                                     # cached stays free with an empty pool
    with pytest.raises(BudgetExceeded):
        ev(p3, "f")
    with pytest.raises(ToolError, match="exhausted"):                        # the pool is empty: refused, no charge
        ev(late, "g")
    assert late.spent["evaluate"] == 0 and prob.evals_used == 3
    assert {h for h, st, ch in calls if ch} == {"c", "d", "e"}
    # without free_repeats (charged screens, the old behaviour) every call costs one, cached or not
    old = _problem(tmp_path / "old", budget=5)
    monkeypatch.setattr(old, "_evaluate", _fake_evaluate(old, []))
    api = make_tool_api(old, {}, cid="q", per_program_evaluate=2)
    ev(api, "a")
    ev(api, "a")
    assert api.spent["evaluate"] == 2
    with pytest.raises(BudgetExceeded):
        ev(api, "a")
    nocap = make_tool_api(old, {}, cid="r", per_program_evaluate=None, free_repeats=True)
    assert nocap.budget["evaluate"] == old.evals_left                         # null per-program cap: the pool


def test_programs_run_when_screens_are_free(tmp_path, monkeypatch):
    """build_fn: with charged screens an empty budget refuses to run the program (as before); with free screens
    the program always runs and its returned config is screened without charge."""
    from edit_code import run as RUN
    prob = _problem(tmp_path, budget=0)
    calls = []
    monkeypatch.setattr(prob, "_evaluate", _fake_evaluate(prob, calls))
    monkeypatch.setattr(prob, "build", lambda result, cid: {"config_hash": f"h{cid}", "bytes_gib": 1.0})

    class Res:
        ok, result, stdout, stderr, error, seconds, info = True, {"default": {"bits": 4}}, "", "", None, 0.1, {}
    monkeypatch.setattr(RUN, "run_program", lambda *a, **k: Res())
    monkeypatch.setattr(RUN, "system_prompt", lambda *a, **k: "SYS")
    monkeypatch.setattr(RUN, "make_render_context", lambda *a, **k: (lambda h, t: "CTX"))
    cfg = {"sandbox": {"timeout_s": 10, "mem_gb": 1}, "editor": {"tool_rounds": 0}, "feedback": {}}
    runner = SimpleNamespace(rc=SimpleNamespace(mock=True, seed=42), cfg=cfg, chat_tools=None,
                             search_history=lambda g: None)
    for charge, want_run in ((True, False), (False, True)):
        cfg["search"] = {"threads": 2, "steps": 1, "max_evaluate": 0, "charge_screens": charge,
                         "program_evaluate_calls": 2}
        fns = RUN.Runner.search_fns(runner, prob, tmp_path / f"g{charge}", None, archive="")
        art, err = fns["build_fn"]("def main(api): pass", f"x{charge}")
        assert (art is not None) == want_run, err
        if want_run:
            sc, fb = fns["screen_fn"](art, f"x{charge}")
            assert sc == 0.5 and calls[-1] == (f"hx{charge}", "screen", False)
        else:
            assert "program not run" in err


# ---------------------------------------------------------------------------------- decoding + shared incumbent
def _real_problem(tmp_path, monkeypatch, cfg, captured):
    """A prepared mock QuantProblem on two screen items; score() is faked (records written like run_eval's)."""
    import rsicore.evaluate as E
    from edit_code import mock as M
    from edit_code.problem import QuantProblem
    rc = SimpleNamespace(mock=True, run_dir=tmp_path, run_id="t", seed=7, gpus=[0, 1])
    prob = QuantProblem(rc, cfg, tag="g01", gen=1, out_dir=tmp_path / "gen", prep_dir=tmp_path / "prep",
                        source_dir=None, source_state=M.fresh_state(0.8), budget_gib=float("inf"),
                        datasets=["simple_math_pool"], gate_datasets=["simple_math_eval"],
                        data={"simple_math_pool": {"screen": ["a", "b"], "signal": ["a", "b"]}}, kl_texts=[],
                        refs_dir=tmp_path / "refs", eval_budget=0, show_examples=0)
    prob.prepare()
    monkeypatch.setattr(prob, "_ids", lambda ds, stage: ["a", "b"])

    @contextmanager
    def serve(art, name):
        yield SimpleNamespace(endpoint="http://mock.invalid/v1", served_name=name)
    monkeypatch.setattr(prob, "serve", serve)
    lock = threading.Lock()

    def fake_score(endpoint, served, task, *, pool, ids, out_dir, mock=False, decoding=None, **kw):
        with lock:
            captured.append((served.rsplit("-", 1)[-1], task, decoding))
        recs = [{"id": i, "score": 1.0 if i == "a" else 0.0} for i in ids]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        return SimpleNamespace(records=recs)
    monkeypatch.setattr(E, "score", fake_score)
    return prob


def test_decoding_passed_and_incumbent_shared(tmp_path, monkeypatch):
    _, cfg = _load(tmp_path / "cfg", [])
    captured = []
    prob = _real_problem(tmp_path, monkeypatch, cfg, captured)
    screen, gate = dict(THINK, n=4, seed=42), dict(THINK, n=4, seed=42)
    assert prob.decoding == {"screen": screen, "full": screen, "gate": gate}
    ref = cfg["budget"]["reference"]
    prob.incumbent = {"gen": 0, "cid": "q0", "config": ref, "gate": {}}
    inc_hash = prob.build(ref, "incumbent")["config_hash"]
    prob.evaluate_incumbent()                                       # screen scored once, full reuses it
    assert captured == [(inc_hash, "simple_math_pool", screen)]
    cands = [dict(ref, lm_head=4), dict(ref, embed=4), dict(ref, embed="bf16")]   # ref: embed 8, lm_head 8
    scores = []
    for i, c in enumerate(cands):
        r = prob.evaluate(c, "screen", f"g01_t{i}_s00", charge=False)
        scores.append(r["feedback"]["probes"]["math"]["incumbent"])
    assert len(captured) == 1 + len(cands)                         # the incumbent was not re-sampled
    assert [h for h, _, _ in captured].count(inc_hash) == 1
    assert all(d == screen for _, _, d in captured)
    inc_recs = (tmp_path / "gen" / "evals" / "incumbent" / "screen" / "simple_math_pool.jsonl").read_text()
    assert len(inc_recs.splitlines()) == 2 and set(scores) == {0.5}  # every candidate paired with the same records
    # gate: another dataset (simple_math_eval) -> scored, not reused; full -> reused
    prob.evaluate(cands[0], "full", "g01_t0_s00", charge=False)
    assert len(captured) == 1 + len(cands)
    prob.evaluate(cands[0], "gate", "g01_t0_s00", charge=False, origin="gate")
    assert captured[-1][1:] == ("simple_math_eval", gate)
    recs = [json.loads(x) for x in (tmp_path / "gen" / "evals.jsonl").read_text().splitlines()]
    assert {json.loads(r["decoding"])["n"] for r in recs if r["stage"] == "gate"} == {4}
    assert {json.loads(r["decoding"])["n"] for r in recs if r["stage"] != "gate"} == {4}
    # a reopened problem reuses the cached records only under the same decoding
    again = _real_problem(tmp_path, monkeypatch, cfg, [])
    assert again.evaluated(cands[0], "screen") and again.evaluated(cands[0], "gate")
    other = json.loads(json.dumps(cfg))
    other["decoding"]["screen"]["n"] = 2
    changed = _real_problem(tmp_path, monkeypatch, other, [])
    assert not changed.evaluated(cands[0], "screen") and changed.evaluated(cands[0], "gate")


def test_greedy_records_unchanged(tmp_path, monkeypatch):
    """decoding null (decoding.<stage>: null): score() gets decoding=None and records carry no "decoding"."""
    _, cfg = _load(tmp_path / "cfg", ["decoding.screen=null", "decoding.gate=null"])
    captured = []
    prob = _real_problem(tmp_path, monkeypatch, cfg, captured)
    assert prob.decoding == {"screen": None, "full": None, "gate": None}
    prob.evaluate(cfg["budget"]["reference"], "screen", "c0", charge=False)
    assert captured[0][2] is None
    rec = json.loads((tmp_path / "gen" / "evals.jsonl").read_text().splitlines()[0])
    assert "decoding" not in rec
