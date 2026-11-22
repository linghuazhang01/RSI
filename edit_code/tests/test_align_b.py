"""Paper A aligned with Paper B's flow (2026-11-22).

    python3 -I -m pytest -q edit_code/tests/test_align_b.py

- the incumbent's screen run is served on every free candidate slot (QuantProblem.serve_all) and split over them;
  candidate gates keep one slot each;
- a program's "note" (in the returned config, or {"config", "note"}) is stored in the artifact (cut to 600 chars;
  solve_allocation's automatic note dropped) and shown in search_history rows;
- a candidate identical to the incumbent is not gate-evaluated (gain exactly 0, rejected, as Paper B's 'keep');
- Runner.kl_used: no per-generation KL work when the score does not use KL and no baselines run; compute_refs
  writes M_0's KL reference only with kl_weight > 0;
- base.yaml: kl_weight 0, patience 3; the mock keeps the real gate rule and no baselines.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gate_flow as G  # noqa: E402
import test_simple_v2 as V  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ incumbent screen over every slot
def test_serve_all_takes_every_free_slot(tmp_path, monkeypatch):
    import rsicore.servers as SV
    prob = V._problem(tmp_path, budget=0, n_gpus=4)
    started, stopped = [], []

    class FakeServer:
        def __init__(self, path, name, gpus, **k):
            self.gpus, self.endpoint = list(gpus), f"http://fake/{gpus[0]}/v1"

        def stop(self):
            stopped.append(self.gpus)
    monkeypatch.setattr(SV, "VLLMServer", FakeServer)
    monkeypatch.setattr(SV, "start_all", lambda srvs: started.append([s.gpus for s in srvs]) or srvs)
    n = prob.n_slots
    assert n >= 2
    busy = prob._pool.get()                                  # one slot is in use elsewhere: not waited for
    with prob.serve_all({"config_hash": "h", "mock_quality": 0.5}, "inc") as srvs:
        assert len(srvs) == n - 1 and prob._pool.empty()
    prob._pool.put(busy)
    assert len(started) == 1 and len(started[0]) == n - 1 and len(stopped) == n - 1
    assert prob._pool.qsize() == n                           # every slot back in the pool


def test_incumbent_screen_uses_serve_all_and_gate_one_slot(tmp_path, monkeypatch):
    prob = V._problem(tmp_path, budget=0)
    used = []

    class Stop(Exception):
        pass

    def fake(kind):
        @contextmanager
        def cm(art, name):
            used.append(kind)
            raise Stop
            yield  # pragma: no cover
        return cm
    monkeypatch.setattr(prob, "serve_all", fake("all"))
    monkeypatch.setattr(prob, "serve", fake("one"))
    art = {"config_hash": "h"}
    for stage, origin in (("screen", "incumbent"), ("screen", "candidate"), ("gate", "incumbent"), ("gate", "gate")):
        with pytest.raises(Stop):
            prob._evaluate(art, stage, "c", ("h", stage), False, origin)
    assert used == ["all", "one", "one", "one"]


# ------------------------------------------------------------------ note
def _build_fns(tmp_path, monkeypatch, result):
    from edit_code import run as RUN

    class Res:
        ok, stdout, stderr, error, seconds, info = True, "", "", None, 0.1, {}
    Res.result = result
    prob = V._problem(tmp_path, budget=0, cfg={"search": G.GATE_SCFG})
    seen = []
    monkeypatch.setattr(prob, "build", lambda r, cid: seen.append(r) or {"config_hash": f"h{cid}", "bytes_gib": 1.0,
                                                                          "budget_gib": 10.0, "linear_avg_bits": 4.0})
    monkeypatch.setattr(RUN, "run_program", lambda program, api, **k: Res())
    monkeypatch.setattr(RUN, "system_prompt", lambda *a, **k: "SYS")
    monkeypatch.setattr(RUN, "make_render_context", lambda *a, **k: (lambda h, t: "CTX"))
    cfg = {"sandbox": {"timeout_s": 10, "mem_gb": 1}, "editor": {"tool_rounds": 0}, "feedback": {}}
    cfg["search"] = G.GATE_SCFG
    runner = SimpleNamespace(rc=SimpleNamespace(mock=True, seed=42), cfg=cfg, chat_tools=None,
                             search_history=lambda g: None)
    fns = RUN.Runner.search_fns(runner, prob, tmp_path / "gen", None, archive="")
    return fns, seen


@pytest.mark.parametrize("result,note", [
    ({"default": {"bits": 4}, "note": "  8 bit down 0-3:\n largest herr  "}, "8 bit down 0-3: largest herr"),
    ({"config": {"default": {"bits": 4}}, "note": "wrapped"}, "wrapped"),
    ({"default": {"bits": 4}, "note": "solve_allocation: 9.900 GiB of 10.000; predicted weighted damage 0.01234"}, None),
    ({"default": {"bits": 4}, "note": "solve_allocation: 9.900 GiB of 10.000; predicted weighted damage 0.01234; "
                                       "8 bit down 0-3"}, "8 bit down 0-3"),             # the editor's addition kept
    ({"default": {"bits": 4}, "note": "solve_allocation with *.down weighted 2x"}, "solve_allocation with *.down weighted 2x"),
    ({"default": {"bits": 4}}, None),
    ({"default": {"bits": 4}, "note": "x" * 900}, "x" * 597 + "..."),
])
def test_program_note_stored(tmp_path, monkeypatch, result, note):
    fns, seen = _build_fns(tmp_path, monkeypatch, result)
    art, err = fns["build_fn"]("def main(api): ...", "g01_t0_s00")
    assert err is None and art.get("note") == note
    assert "default" in seen[-1]                              # {"config", "note"} is unwrapped before build


def test_history_row_shows_note():
    from edit_code.prompts import history_row
    row = history_row({"artifact": {"config": {"default": {"bits": 4}}, "bytes_gib": 9.9, "note": "why"}})
    assert row["note"] == "why"
    assert history_row({"artifact": {"config": {"default": {"bits": 4}}}})["note"] is None


def test_prompt_states_note_time_and_calls():
    from edit_code.prompts import system_prompt
    text = system_prompt(G._fake_problem(), "- bytes(config)",
                         {"gate": {"min_delta": 0.0}, "sandbox": {"timeout_s": 900},
                          "editor": {"tool_rounds": 30, "max_tool_calls": 60}},
                         history=True, search=None)
    assert '"note" key of the returned config' in text and "killed after 900 s" in text
    assert "for up to 30 rounds" in text and "at most 60 calls in all" in text


# ------------------------------------------------------------------ identical to the incumbent
def test_identical_to_incumbent_not_gated(tmp_path, monkeypatch):
    from edit_code import run as RUN
    prob = V._problem(tmp_path, budget=0, cfg={"search": G.GATE_SCFG})
    prob.incumbent = {"config_hash": "hinc", "gate_scores": {"simple_math_eval": 0.61}}
    monkeypatch.setattr(prob, "evaluate", lambda *a, **k: pytest.fail("the incumbent's config must not be re-gated"))
    log = RUN.GateLog(tmp_path / "gate.json")
    g = RUN.gate_candidate(prob, {"config_hash": "hinc", "bytes_gib": 1.0}, "g02_t0_s00", log)
    assert not g["passed"] and g["gain"] == 0.0 and g["accuracy"] == g["reference_accuracy"] == 0.61
    assert "identical to the incumbent" in g["reason"]
    sub = json.loads((tmp_path / "gate.json").read_text())["submissions"][0]
    assert sub["cid"] == "g02_t0_s00" and sub["passed"] is False and sub["info"]["gain"] == 0.0


# ------------------------------------------------------------------ KL work and the simple profile
@pytest.mark.parametrize("klw,methods,used", [(0.0, [], False), (0.5, [], True), (0.0, ["uniform"], True)])
def test_kl_used(klw, methods, used):
    from edit_code import run as RUN
    r = SimpleNamespace(cfg={"search": {"kl_weight": klw}, "baselines": {"methods": methods}})
    assert RUN.Runner.kl_used(r) is used


def test_base_config_and_mock():
    import yaml
    cfg = yaml.safe_load((REPO / "edit_code" / "configs" / "base.yaml").read_text())
    assert cfg["search"]["kl_weight"] == 0.0 and cfg["loop"]["patience"] == 3
    ms = cfg["mock_overrides"]
    assert ms["gate"]["min_delta"] == 0.0 and ms["baselines"]["methods"] == []
    assert (ms["search"]["threads"], ms["search"]["steps"], ms["search"]["max_candidates"]) == (3, 1, 3)
    assert not {"objective", "objective_profiles", "scope", "scope_profiles", "mock_objective_overrides", "data",
                "e_task", "draw_config"} & set(cfg)
    assert cfg["archive"]["max_chars"] == 30000


# ------------------------------------------------------------------ review fixes (2026-11-22)
def test_error_line_and_dict_ci():
    from rsicore.search import Candidate
    from edit_code.common import error_line
    from edit_code.prompts import archive_row
    err = "invalid quantization config:\n  - default: 5 not in [2, 3, 4, 8, 16]\n  - embed: 3 not allowed"
    assert error_line(err) == "invalid quantization config: - default: 5 not in [2, 3, 4, 8, 16]"
    assert error_line("program failed: boom\nstderr tail:\nx") == "program failed: boom"
    c = Candidate(cid="g01_t0_s00", thread=0, step=0, program="", artifact=None, error=err, screen=None, full=None,
                  score=None)
    assert archive_row(c)["error"].startswith("invalid quantization config: - default: 5")
    ok = Candidate(cid="g01_t1_s00", thread=1, step=0, program="", artifact={"config": {"default": {"bits": 4}}},
                   error=None, screen=None, full=None, score=0.01)
    row = archive_row(ok, {"passed": True, "gain": 0.01, "ci": {"a": [0.0, 0.02], "b": [0.001, 0.03]}})
    assert row["gate"]["ci"] == {"a": [0.0, 0.02], "b": [0.001, 0.03]}


def test_refs_only_with_kl_weight(tmp_path, monkeypatch):
    import rsicore.evaluate as E
    prob = V._problem(tmp_path, budget=0, cfg={"search": {"kl_weight": 0.0}, "kl": {"top_k": 5}})
    prob.mock = False
    prob.kl_texts = [("math", "t1")]
    wrote = []
    monkeypatch.setattr(E, "reference_logprobs", lambda ep, name, texts, path, k: wrote.append(Path(path).name))
    prob.compute_refs("http://x/v1", "m0")
    assert wrote == []                              # kl_ref only with kl_weight > 0
    prob.cfg["search"]["kl_weight"] = 0.5
    prob.compute_refs("http://x/v1", "m0")
    assert wrote == ["kl_ref.jsonl"]
