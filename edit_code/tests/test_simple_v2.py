"""Paper A simple v2 (2026-11-21): scoring decoding, the 8 x 1 profile, the hard-wired flow (2026-11-23).

    python3 -I -m pytest -q edit_code/tests/test_simple_v2.py

- search_settings: only the search shape (threads, steps, max_candidates; null = threads x steps);
- the real base.yaml (the folded simple profile, 2026-11-22) resolves to 8 threads x 1 step, 8 candidates, thinking
  decoding (screen / gate / held-out n 4), candidate servers at 32768 tokens and a persistent exploration session;
  the removed search / gate / noise / KL knobs are absent and refused on --set (their hard-wired value is a no-op);
- build_fn always runs the program; the tool API has no evaluate;
- QuantProblem passes the stage decoding to every score() call, reuses records only under the same decoding, and
  samples the incumbent's screen records once for all evaluations.
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


REMOVED_A = {"search": ("screen_candidates", "gate_every_candidate", "charge_screens", "program_evaluate_calls",
                         "program_evaluate_total", "max_evaluate", "top_full", "rerun_seeds", "kl_weight"),
             "gate": ("max_submissions",), "kl": ("eval_texts", "top_k")}


def test_search_settings_defaults():
    from edit_code.common import search_settings
    assert search_settings({"threads": 4, "steps": 6}) == {"threads": 4, "steps": 6, "max_candidates": 24}
    assert search_settings({"threads": 2, "steps": 2, "max_candidates": None})["max_candidates"] == 4
    assert search_settings({"threads": 8, "steps": 1, "max_candidates": 5})["max_candidates"] == 5
    # a stale key of a removed knob is ignored here (rsicore.config refuses it on --set / --config)
    assert search_settings({"threads": 3, "steps": 1, "max_evaluate": 1}) == {"threads": 3, "steps": 1,
                                                                             "max_candidates": 3}


def test_simple_profile_resolution(tmp_path):
    """The real base.yaml (no mock overrides): 8 x 1, decoding, servers; the removed knobs are gone."""
    from rsicore.config import stage_decoding
    from edit_code.common import search_settings
    from edit_code.run import server_args
    cfg, eff = _load(tmp_path, [])
    # the hard-wired flow (2026-11-23): only the search shape is configured
    assert search_settings(cfg["search"]) == {"threads": 8, "steps": 1, "max_candidates": 8}
    for block, keys in REMOVED_A.items():
        assert not set(keys) & set(cfg[block]), block
    assert "noise" not in cfg and "noise" not in cfg["mock_overrides"]
    assert cfg["editor"]["explore"]["persistent"] is True
    assert stage_decoding(cfg, "screen", 42) == dict(THINK, n=4, seed=42)                   # the incumbent's screen
    assert stage_decoding(cfg, "full", 42) == dict(THINK, n=4, seed=42)                      # full = screen
    assert stage_decoding(cfg, "gate", 42) == dict(THINK, n=4, seed=42)
    hd = cfg["heldout"]["decoding"]
    # held-out n 8 (2026-11-24) = baselines.heldout_n, so the end-of-run comparison rows reuse the generations' runs
    assert {k: hd[k] for k in ("temperature", "top_p", "top_k", "thinking", "max_tokens", "n", "seed")} == \
        dict(THINK, n=8, seed=42)
    assert cfg["baselines"]["heldout_n"] == hd["n"]
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
    assert search_settings(eff["search"]) == {"threads": 3, "steps": 1, "max_candidates": 3}
    tg, _ = _load(tmp_path / "tg", ["search.max_candidates=null", "search.threads=4", "search.steps=6",
                                    "decoding.screen=null"])
    assert stage_decoding(tg, "screen", 42) is None
    assert search_settings(tg["search"])["max_candidates"] == 24


@pytest.mark.parametrize("sets", [["search.screen_candidates=true"], ["search.gate_every_candidate=false"],
                                  ["search.program_evaluate_calls=2"], ["search.max_evaluate=12"],
                                  ["search.top_full=3"], ["search.kl_weight=0.1"], ["gate.max_submissions=3"],
                                  ["noise.temperature=0.6"], ["noise.fixed=0.03"], ["kl.eval_texts=64"]])
def test_removed_knobs_refused(tmp_path, sets):
    with pytest.raises(SystemExit, match="removed"):
        _load(tmp_path, sets)


def test_removed_knobs_hardwired_value_is_noop(tmp_path):
    from edit_code.common import search_settings
    cfg, _ = _load(tmp_path, ["search.screen_candidates=false", "search.gate_every_candidate=true",
                              "search.program_evaluate_calls=0", "search.top_full=0", "search.kl_weight=0",
                              "noise.temperature=0"])
    assert search_settings(cfg["search"]) == {"threads": 8, "steps": 1, "max_candidates": 8}
    assert "noise" not in cfg and not set(REMOVED_A["search"]) & set(cfg["search"])


# ---------------------------------------------------------------------------------- build_fn, tool API
def _problem(tmp, budget=0, cfg=None, n_gpus=2):
    from edit_code.problem import QuantProblem
    rc = SimpleNamespace(mock=True, run_dir=tmp, run_id="t", seed=42, gpus=list(range(n_gpus)))
    base = {"gpu_layout": {"candidate_group_size": 1}, "candidates": {"keep_checkpoints": 1}}
    return QuantProblem(rc, dict(base, **(cfg or {})), tag="t", gen=1, out_dir=tmp / "p", prep_dir=tmp / "prep",
                        source_dir=None, source_state=None, budget_gib=10.0, datasets=["dapo_math"], data={},
                        kl_texts=[])


def test_programs_always_run_no_evaluate_tool(tmp_path, monkeypatch):
    """build_fn always runs the program (there is no evaluate budget) with an API that has no evaluate / compare; the
    artifact's cost has no evaluate count."""
    from edit_code import run as RUN
    prob = _problem(tmp_path)
    monkeypatch.setattr(prob, "build", lambda result, cid: {"config_hash": f"h{cid}", "bytes_gib": 1.0})
    apis = []

    class Res:
        ok, result, stdout, stderr, error, seconds, info = True, {"default": {"bits": 4}}, "", "", None, 0.1, {}
    monkeypatch.setattr(RUN, "run_program", lambda program, api, **k: apis.append(api) or Res())
    monkeypatch.setattr(RUN, "system_prompt", lambda *a, **k: "SYS")
    monkeypatch.setattr(RUN, "make_render_context", lambda *a, **k: (lambda h, t: "CTX"))
    cfg = {"sandbox": {"timeout_s": 10, "mem_gb": 1}, "editor": {"tool_rounds": 0}, "feedback": {},
           "search": {"threads": 2, "steps": 1}}
    runner = SimpleNamespace(rc=SimpleNamespace(mock=True, seed=42), cfg=cfg, chat_tools=None,
                             search_history=lambda g: None)
    fns = RUN.Runner.search_fns(runner, prob, tmp_path / "g", None, archive="")
    assert set(fns) == {"editor_factory", "render_context", "build_fn", "screen_fn"}     # no full_fn
    art, err = fns["build_fn"]("def main(api): pass", "x")
    assert err is None and art is not None and "tool_evaluate" not in art["cost"]
    assert not {"evaluate", "compare"} & set(apis[-1].names()) and apis[-1].budget == {}
    assert not hasattr(prob, "evals_left") and not hasattr(prob, "screen_tools")


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
                        show_examples=0)
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
        r = prob.evaluate(c, "screen", f"base_{i}", origin="baseline")
        scores.append(r["feedback"]["probes"]["math"]["incumbent"])
    assert len(captured) == 1 + len(cands)                         # the incumbent was not re-sampled
    assert [h for h, _, _ in captured].count(inc_hash) == 1
    assert all(d == screen for _, _, d in captured)
    inc_recs = (tmp_path / "gen" / "evals" / "incumbent" / "screen" / "simple_math_pool.jsonl").read_text()
    assert len(inc_recs.splitlines()) == 2 and set(scores) == {0.5}  # every candidate paired with the same records
    # gate: another dataset (simple_math_eval) -> scored, not reused; full -> reused
    prob.evaluate(cands[0], "full", "base_0", origin="baseline")
    assert len(captured) == 1 + len(cands)
    prob.evaluate(cands[0], "gate", "g01_t0_s00", origin="gate")
    assert captured[-1][1:] == ("simple_math_eval", gate)
    recs = [json.loads(x) for x in (tmp_path / "gen" / "evals.jsonl").read_text().splitlines()]
    assert {json.loads(r["decoding"])["n"] for r in recs if r["stage"] == "gate"} == {4}
    assert {json.loads(r["decoding"])["n"] for r in recs if r["stage"] != "gate"} == {4}
    assert not [r for r in recs if "charged" in r or "kl" in r["feedback"]]       # no evaluate budget, no KL
    # a reopened problem reuses the cached records only under the same decoding
    h0 = prob.build(cands[0], "x")["config_hash"]
    again = _real_problem(tmp_path, monkeypatch, cfg, [])
    assert (h0, "screen") in again._evals and (h0, "gate") in again._evals
    other = json.loads(json.dumps(cfg))
    other["decoding"]["screen"]["n"] = 2
    changed = _real_problem(tmp_path, monkeypatch, other, [])
    assert (h0, "screen") not in changed._evals and (h0, "gate") in changed._evals


def test_greedy_records_unchanged(tmp_path, monkeypatch):
    """decoding null (decoding.<stage>: null): score() gets decoding=None and records carry no "decoding"."""
    _, cfg = _load(tmp_path / "cfg", ["decoding.screen=null", "decoding.gate=null"])
    captured = []
    prob = _real_problem(tmp_path, monkeypatch, cfg, captured)
    assert prob.decoding == {"screen": None, "full": None, "gate": None}
    prob.evaluate(cfg["budget"]["reference"], "screen", "c0", origin="baseline")
    assert captured[0][2] is None
    rec = json.loads((tmp_path / "gen" / "evals.jsonl").read_text().splitlines()[0])
    assert "decoding" not in rec
