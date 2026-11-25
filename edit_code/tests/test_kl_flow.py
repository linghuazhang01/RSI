"""Paper A KL-only flow (2026-11-25): the fixed KL set, one KL evaluation per candidate, the lowest KL wins.

    python3 -I -m pytest -q edit_code/tests/test_kl_flow.py

- base.yaml: kl_eval (256 traces), 8 x 1 search, no screen / gate / decoding / held-out / reuse / baselines blocks;
- tool API: no evaluate, no screen tools; group_error refuses uncached settings while exploring;
- QuantProblem: kl_slots, serve_all over free slots, thread-safe evaluate (one KL run per config), kl_stats;
- run: kl_candidate (over budget refused, the incumbent's config not evaluated), accept_lowest_kl (first
  quantization, strictly below the incumbent, otherwise keep), notes stored, run_search sets kl_parallel;
- prompts: KL text only (no screen / gate / accuracy), KL table, archive rows with KL.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

SCFG = {"threads": 2, "steps": 1}


def _load(tmp_path, sets):
    """(paper_cfg, effective cfg) of a Paper A --mock invocation (paper_cfg = no mock overrides)."""
    from rsicore import config as C
    from edit_code import run as RUN
    ap = C.build_arg_parser("A")
    ap.add_argument("--budget-gib", type=float, default=None)
    args = ap.parse_args(["--domain", "math", "--mock", "--runs-dir", str(tmp_path),
                          "--run-id", "cfg_" + str(abs(hash(tuple(sets))))] + [x for s in sets for x in ("--set", s)])
    rc = C.load_run_config("A", args)
    return rc.paper_cfg, RUN.effective_config(rc, args)


def _problem(tmp, cfg=None, n_gpus=2, budget=10.0):
    from edit_code.problem import QuantProblem
    rc = SimpleNamespace(mock=True, run_dir=tmp, run_id="t", seed=42, gpus=list(range(n_gpus)))
    base = {"gpu_layout": {"candidate_group_size": 1}, "candidates": {"keep_checkpoints": 1}}
    return QuantProblem(rc, dict(base, **(cfg or {})), tag="t", gen=1, out_dir=tmp / "p", prep_dir=tmp / "prep",
                        source_dir=None, source_state=None, budget_gib=budget, datasets=["simple_math_pool"],
                        kl_texts=[])


def _mml(args):
    return int(args[args.index("--max-model-len") + 1])


# ------------------------------------------------------------------ config
def test_base_config(tmp_path):
    import yaml
    from edit_code.common import kl_eval_cfg, search_settings
    from edit_code.run import server_args
    raw = yaml.safe_load((REPO / "edit_code" / "configs" / "base.yaml").read_text())
    assert not {"simple", "decoding", "max_tokens", "gate", "feedback", "scoring", "heldout", "reuse",
                "baselines", "noise"} & set(raw)
    assert raw["kl_eval"]["n"] == 256 and kl_eval_cfg(raw)["top_k"] == 20
    assert raw["loop"] == {"editor": "frozen", "start": "bf16", "patience": 3}
    assert raw["mock_overrides"]["kl_eval"]["n"] == 24
    assert "parallel_heldout" not in raw["gpu_layout"] and raw["gpu_layout"]["gate_split"] is True
    cfg, eff = _load(tmp_path, [])
    assert search_settings(cfg["search"]) == {"threads": 8, "steps": 1, "max_candidates": 8}
    assert search_settings(eff["search"]) == {"threads": 3, "steps": 1, "max_candidates": 3}
    assert _mml(server_args(cfg, "candidate_args")) == 32768 and _mml(server_args(cfg, "editor_args")) == 131072
    assert "--max-num-batched-tokens" not in server_args(cfg, "candidate_args")
    c80, _ = _load(tmp_path / "h80", ["hardware=h20_80"])
    for k in ("candidate_args", "editor_args"):        # prompt logprobs over long traces: chunk cap on h20_80
        a = server_args(c80, k)
        assert a[a.index("--max-num-batched-tokens") + 1] == "1024"


@pytest.mark.parametrize("sets", [["search.screen_candidates=true"], ["search.max_evaluate=12"],
                                  ["search.kl_weight=0.1"], ["noise.temperature=0.6"], ["kl.eval_texts=64"]])
def test_removed_knobs_refused(tmp_path, sets):
    with pytest.raises(SystemExit, match="removed"):
        _load(tmp_path, sets)


def test_lock_ignores_gate_split():
    from rsicore.config import lock_view
    assert lock_view({"gpu_layout": {"editor_sleep": True, "gate_split": True}}, "A") == \
        lock_view({"gpu_layout": {"editor_sleep": True}}, "A")


# ------------------------------------------------------------------ tool API
def test_tool_api_no_evaluate_no_screen():
    from edit_code.tools import ToolError, make_tool_api
    prob = SimpleNamespace(spec=None, herr_missing=lambda c: ["0.down.down_proj|gptq|8|128"],
                           group_error=lambda c, top=10: {"total_per_token": 0.0})
    full = make_tool_api(prob, {})
    ex = make_tool_api(prob, {}, explore=True)
    for api in (full, ex):
        assert not {"evaluate", "compare", "screen_results", "screen_output"} & set(api.names())
        assert api.budget == {} and "evaluate(" not in api.describe()
    assert set(full.names()) == set(ex.names())
    with pytest.raises(ToolError, match="uncached setting"):
        ex.call("group_error", {"config": {"default": {"bits": 8}}})
    assert full.call("group_error", {"config": {"default": {"bits": 8}}}) == {"total_per_token": 0.0}
    gx = make_tool_api(prob, {"editor": {"explore": {"gpu_tools": True}}}, explore=True)
    assert gx.call("group_error", {"config": {"default": {"bits": 8}}}) == {"total_per_token": 0.0}


# ------------------------------------------------------------------ QuantProblem
@pytest.mark.parametrize("parallel,want", [(None, 1), (1, 8), (2, 4), (3, 3), (4, 2), (8, 1), (16, 1)])
def test_kl_slots(tmp_path, parallel, want):
    prob = _problem(tmp_path, n_gpus=9)                      # editor 1 GPU + 8 candidate slots
    assert prob.n_slots == 8
    prob.kl_parallel = parallel
    assert prob.kl_slots() == want
    off = _problem(tmp_path / "off", cfg={"gpu_layout": {"candidate_group_size": 1, "gate_split": False}}, n_gpus=9)
    off.kl_parallel = 2
    assert off.kl_slots() == 1


def test_serve_all_takes_every_free_slot(tmp_path, monkeypatch):
    import rsicore.servers as SV
    prob = _problem(tmp_path, n_gpus=4)
    started, stopped = [], []

    class FakeServer:
        def __init__(self, path, name, gpus, **k):
            self.gpus, self.endpoint = list(gpus), f"http://fake/{gpus[0]}/v1"

        def stop(self):
            stopped.append(self.gpus)
    monkeypatch.setattr(SV, "VLLMServer", FakeServer)
    monkeypatch.setattr(SV, "start_all", lambda srvs: started.append([s.gpus for s in srvs]) or srvs)
    n = prob.n_slots
    busy = prob._pool.get()                                  # one slot is in use elsewhere: not waited for
    with prob.serve_all({"config_hash": "h", "mock_quality": 0.5}, "x") as srvs:
        assert len(srvs) == n - 1 and prob._pool.empty()
    prob._pool.put(busy)
    assert len(started[0]) == n - 1 and len(stopped) == n - 1 and prob._pool.qsize() == n
    with prob.serve_all({"config_hash": "h", "mock_quality": 0.5}, "x", most=2) as srvs:
        assert len(srvs) == 2


def test_evaluate_serving_by_origin(tmp_path, monkeypatch):
    """Q_0 (origin reference) on every free slot; a candidate on kl_slots() slots (shared when > 1, else serve())."""
    prob = _problem(tmp_path, n_gpus=9)
    used = []

    class Stop(Exception):
        pass

    def fake(kind):
        @contextmanager
        def cm(art, name, **k):
            used.append((kind, k.get("most")))
            raise Stop
            yield  # pragma: no cover
        return cm
    monkeypatch.setattr(prob, "serve_all", fake("all"))
    monkeypatch.setattr(prob, "serve", fake("one"))
    monkeypatch.setattr(prob, "kl_items", lambda: [])
    for origin, par in (("reference", None), ("candidate", None), ("candidate", 4)):
        prob.kl_parallel = par
        with pytest.raises(Stop):
            prob._evaluate({"config_hash": "h"}, "c", origin)
    assert used == [("all", None), ("one", None), ("all", 2)]


def test_evaluate_thread_safe_and_cached(tmp_path, monkeypatch):
    prob = _problem(tmp_path, n_gpus=4)
    calls, lock = [], threading.Lock()

    def fake(art, cid, origin):
        with lock:
            calls.append(art["config_hash"])
        time.sleep(0.2)
        rec = {"hash": art["config_hash"], "stage": "kl", "cid": cid, "kl": 0.01, "feedback": {}}
        with prob._lock:
            prob._evals[art["config_hash"]] = rec
        return dict(rec)
    monkeypatch.setattr(prob, "_evaluate", fake)
    res = []
    ts = [threading.Thread(target=lambda h=h: res.append(prob.evaluate({"config_hash": h}, f"c{h}")))
          for h in ["a", "a", "a", "b", "c"]]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert calls.count("a") == 1 and sorted(set(calls)) == ["a", "b", "c"]
    assert sum(1 for r in res if r.get("cached")) == 2


def test_evaluate_records_kl(tmp_path, monkeypatch):
    """_evaluate: per-trace records, mean KL, quantiles, evals.jsonl (reloaded by a new problem)."""
    import rsicore.kl as K
    prob = _problem(tmp_path)
    items = [{"ds": "simple_math_pool", "id": f"i{k}", "text": f"t{k}"} for k in range(4)]
    monkeypatch.setattr(prob, "kl_items", lambda: items)

    @contextmanager
    def fake_serve(art, name):
        yield SimpleNamespace(endpoint="http://fake/v1")
    monkeypatch.setattr(prob, "serve", fake_serve)
    monkeypatch.setattr(prob, "group_error", lambda *a, **k: {"top": [("0.down", 1.0)]})
    monkeypatch.setattr(prob, "norm_of", lambda art: {})
    monkeypatch.setattr(K, "token_kl", lambda ref, eps, name, texts, top_k=20: {
        "mean": 0.025, "token_mean": 0.02, "per_text": [0.01, 0.02, 0.03, 0.04], "n_positions": [5, 5, 5, 5],
        "n_texts": 4, "n_tokens": 20, "top_k": top_k})
    art = {"config_hash": "h1", "bytes_gib": 9.0, "budget_gib": 10.0, "linear_avg_bits": 4.0}
    r = prob.evaluate(art, "g01_t0_s00")
    assert r["kl"] == 0.025 and r["feedback"]["n_traces"] == 4 and r["feedback"]["error_top_groups"]
    assert r["feedback"]["kl_quantiles"]["max"] == 0.04
    recs = [json.loads(x) for x in (prob.out_dir / "evals" / "g01_t0_s00" / "kl.jsonl").read_text().splitlines()]
    assert [x["kl"] for x in recs] == [0.01, 0.02, 0.03, 0.04]
    again = _problem(tmp_path)
    assert again.evaluate(art, "other")["cached"] is True             # reloaded from evals.jsonl
    prob.incumbent = {"config_hash": "hi", "kl": 0.03}
    st = prob.kl_stats(art, r)
    assert st["passed"] and st["kl"] == 0.025 and st["reference_kl"] == 0.03 and st["gain"] == 0.005
    prob.incumbent = {"original": True}
    assert prob.kl_stats(art, r)["reference_kl"] == 0.0
    assert not prob.kl_stats(dict(art, bytes_gib=11.0), r)["passed"]  # over the budget


# ------------------------------------------------------------------ run: candidates and acceptance
def test_kl_candidate_budget_and_identical(tmp_path, monkeypatch):
    from edit_code import run as RUN
    prob = _problem(tmp_path)
    prob.incumbent = {"config_hash": "hinc", "kl": 0.02}
    monkeypatch.setattr(prob, "evaluate", lambda *a, **k: pytest.fail("must not be evaluated"))
    log = RUN.GateLog(tmp_path / "gate.json")
    g = RUN.kl_candidate(prob, {"config_hash": "hinc", "bytes_gib": 1.0}, "g02_t0_s00", log)
    assert not g["passed"] and g["gain"] == 0.0 and g["kl"] == g["reference_kl"] == 0.02
    g = RUN.kl_candidate(prob, {"config_hash": "big", "bytes_gib": 11.0}, "g02_t1_s00", log)
    assert not g["passed"] and "> budget" in g["reason"]
    subs = json.loads((tmp_path / "gate.json").read_text())["submissions"]
    assert [s["cid"] for s in subs] == ["g02_t0_s00", "g02_t1_s00"] and not any(s["passed"] for s in subs)


def _cand(cid, h, kl, error=None):
    from rsicore.search import Candidate
    art = None if error else {"config_hash": h, "config": {"default": {"bits": 4}}, "bytes_gib": 9.0,
                              "budget_gib": 10.0, "linear_avg_bits": 4.0}
    return Candidate(cid=cid, thread=int(cid[5]), step=0, program="", artifact=art, error=error,
                     screen=None if error else {"score": -kl, "feedback": {}}, full=None,
                     score=None if error else -kl)


def _accept(tmp_path, monkeypatch, inc, kls):
    """accept_lowest_kl with candidates g01_t<i>_s00 of KL kls[i] (None = failed to build)."""
    from edit_code import run as RUN
    from rsicore.state import RunState
    gd = tmp_path / "gen_01"
    gd.mkdir(parents=True)
    log = RUN.GateLog(gd / "gate.json")
    cands = []
    for i, k in enumerate(kls):
        c = _cand(f"g01_t{i}_s00", f"h{i}", k if k is not None else 0.0, error=None if k is not None else "boom")
        cands.append(c)
        if k is not None:
            log.add(c.cid, True, {"metric": "kl", "kl": k, "reference_kl": inc.get("kl"), "gain": 0.0})
    prob = SimpleNamespace(budget_gib=10.0, evaluate=lambda art, cid: {"kl": dict(zip([f"h{i}" for i in range(9)],
                                                                                     kls))[art["config_hash"]],
                                                                      "feedback": {"records": "r"}})
    runner = SimpleNamespace(st=RunState(tmp_path), log=lambda *a: None,
                             incumbent_record=lambda g, cid, art, r, md: RUN.Runner.incumbent_record(None, g, cid,
                                                                                                    art, r, md))
    RUN.Runner.accept_lowest_kl(runner, prob, 1, gd, cands, inc)
    return json.loads((gd / "accepted.json").read_text()), json.loads((gd / "incumbent.json").read_text())


def test_accept_first_quantization_takes_lowest(tmp_path, monkeypatch):
    acc, inc = _accept(tmp_path, monkeypatch, {"original": True, "gen": 0, "cid": "m0", "config_hash": None},
                       [0.05, 0.02, None, 0.03])
    assert acc["accepted"] and acc["cid"] == "g01_t1_s00" and acc["first_quantization"] and acc["kl"] == 0.02
    assert inc["cid"] == "g01_t1_s00" and inc["kl"] == 0.02 and inc["gen"] == 1
    gate = json.loads((tmp_path / "gen_01" / "gate.json").read_text())["submissions"]
    assert any(s["cid"] == "g01_t2_s00" and "not evaluated" in s["info"]["reason"] for s in gate)


def test_accept_only_below_incumbent(tmp_path, monkeypatch):
    base = {"gen": 0, "cid": "q0", "config_hash": "hq", "kl": 0.03}
    acc, inc = _accept(tmp_path / "a", monkeypatch, base, [0.05, 0.029, 0.04])
    assert acc["accepted"] and acc["cid"] == "g01_t1_s00" and acc["gain"] == 0.001 and "first_quantization" not in acc
    acc, inc = _accept(tmp_path / "b", monkeypatch, base, [0.05, 0.03, 0.04])      # a tie is not an improvement
    assert not acc["accepted"] and "not below" in acc["reason"] and inc["cid"] == "q0" and inc["accepted"] is False


def test_notes_stored_and_screen_fn_scores_minus_kl(tmp_path, monkeypatch):
    from edit_code import run as RUN

    class Res:
        ok, stdout, stderr, error, seconds, info = True, "", "", None, 0.1, {}
    Res.result = {"config": {"default": {"bits": 4}}, "note": "  8 bit down 0-3:\n why  "}
    prob = _problem(tmp_path)
    monkeypatch.setattr(prob, "build", lambda r, cid: {"config_hash": f"h{cid}", "bytes_gib": 1.0,
                                                       "budget_gib": 10.0, "linear_avg_bits": 4.0})
    monkeypatch.setattr(RUN, "run_program", lambda program, api, **k: Res())
    monkeypatch.setattr(RUN, "system_prompt", lambda *a, **k: "SYS")
    monkeypatch.setattr(RUN, "make_render_context", lambda *a, **k: (lambda h, t: "CTX"))
    monkeypatch.setattr(prob, "evaluate", lambda art, cid, **k: {"kl": 0.0123, "feedback": {}})
    monkeypatch.setattr(prob, "group_error", lambda *a, **k: {"top": []})
    monkeypatch.setattr(prob, "norm_of", lambda art: {})
    cfg = {"sandbox": {"timeout_s": 10, "mem_gb": 1}, "editor": {"tool_rounds": 0}, "search": SCFG}
    runner = SimpleNamespace(rc=SimpleNamespace(mock=True, seed=42), cfg=cfg, chat_tools=None,
                             search_history=lambda g: None)
    fns = RUN.Runner.search_fns(runner, prob, tmp_path / "gen", None, archive="")
    art, err = fns["build_fn"]("def main(api): ...", "g01_t0_s00")
    assert err is None and art["note"] == "8 bit down 0-3: why"
    score, fb = fns["screen_fn"](art, "g01_t0_s00")
    assert score == -0.0123 and fb["kl_eval"]["kl"] == 0.0123 and fb["kl_eval"]["metric"] == "kl"
    sub = json.loads((tmp_path / "gen" / "gate.json").read_text())["submissions"][0]
    assert sub["passed"] and sub["info"]["kl"] == 0.0123


def test_run_search_sets_kl_parallel(monkeypatch):
    from edit_code import run as RUN
    monkeypatch.setattr(RUN.S, "inner_search", lambda rc, gen, **k: [])
    for (th, mc), want in (((4, 4), 4), ((8, 8), 8), ((1, 8), 1), ((8, 3), 3)):
        prob = SimpleNamespace(n_slots=8, kl_parallel=None)
        runner = SimpleNamespace(cfg={"search": {"threads": th, "steps": 1, "max_candidates": mc}}, rc=None,
                                 gpu_editor=[0], editor_sleep=True, log=print, search_fns=lambda *a, **k: {})
        RUN.Runner.run_search(runner, prob, 1, Path("."), None, archive="")
        assert prob.kl_parallel == want


# ------------------------------------------------------------------ prompts
def _fake_problem(first=False):
    from edit_code.quant.spec import synthetic_spec
    inc = ({"original": True, "bytes_gib": 50.9} if first else
           {"gen": 1, "cid": "g01_t0_s00", "kl": 0.0123, "bytes_gib": 9.9, "linear_avg_bits": 3.9,
            "kl_quantiles": {"p50": 0.01}, "config": {"default": {"bits": 4}}})
    return SimpleNamespace(space={"bits": [2, 3, 4, 8, 16], "group_sizes": [32, 64, 128],
                                  "quantizers": ["rtn", "gptq", "awq"]},
                           incumbent=inc, incumbent_is_original=first, budget_gib=10.0, gen=2,
                           spec=synthetic_spec(None), kl_items=lambda: [{}] * 256,
                           tables={"profile": [], "activation": []})


@pytest.mark.parametrize("first", [False, True])
def test_system_prompt_kl_only(first):
    from edit_code import prompts as P
    cfg = {"sandbox": {"timeout_s": 900}, "editor": {"tool_rounds": 30, "max_tool_calls": 60}}
    t = P.system_prompt(_fake_problem(first), "- bytes(config)", cfg, history=True)
    assert "256 fixed reference traces" in t and "KL(original || quantized)" in t and "top-20" in t
    assert "There is no api.evaluate and no accuracy score" in t
    for stale in ("screen", "gate evaluation", "gate set", "gate result", "accuracy gain", "paired per item",
                  "significant gain", "held-out test"):
        assert stale not in t, stale
    assert ("LOWEST KL" in t) and (("first incumbent" in t) == first)
    assert '"note" key of the returned config' in t and "killed after 900 s" in t
    assert "for up to 30 rounds" in t and "at most 60 calls in all" in t and "search_history(gen" in t
    plain = P.system_prompt(_fake_problem(first), "DOC", {"editor": {"tool_rounds": 0}})
    assert plain.endswith(P.FINAL_RULE) and "run_python" not in plain


def test_render_context_and_archive():
    from rsicore.search import Candidate
    from edit_code import prompts as P
    prob = _fake_problem()
    render = P.make_render_context(prob, {}, search={"threads": 2, "steps": 1}, archive="ARCH", history_tool=True)
    t0 = render([], 1)
    assert "You are session 2 of 2" in t0 and "KL to the original model 0.012300" in t0 and "ARCH" in t0
    assert "screen" not in t0 and "accuracy" not in t0
    fb = {"kl_eval": {"passed": True, "kl": 0.0111, "reference_kl": 0.0123, "gain": 0.0012, "metric": "kl"}}
    c = Candidate(cid="g02_t0_s00", thread=0, step=0, program="", error=None, full=None, score=-0.0111,
                  artifact={"config": {"default": {"bits": 4}}, "bytes_gib": 9.9, "linear_avg_bits": 3.9},
                  screen={"score": -0.0111, "feedback": fb})
    t1 = render([c], 0)
    assert "| g02_t0_s00 | 0 | 0 | 9.900 | 3.90 | 0.011100 | +0.001200 |" in t1
    assert "Lowest KL so far: g02_t0_s00 (0.011100)" in t1
    row = P.archive_row(c, {"passed": True, "kl": 0.0111, "gain": 0.0012})
    assert row["kl"] == 0.0111 and row["config"].startswith("default 4b") and "gate" not in row
    bad = Candidate(cid="g02_t1_s00", thread=1, step=0, program="", error="program failed: boom\nx",
                    full=None, score=None, artifact=None, screen=None)
    assert P.archive_row(bad, {"passed": False, "reason": "not evaluated"})["error"] == "program failed: boom"
    text = P.archive_text([{"gen": 1, "accepted": True, "cid": "g01_t0_s00", "incumbent_kl": 0.0,
                            "first_quantization": True, "n": 2, "best": [("g01_t0_s00", 0.0123)],
                            "rows": [row]}])
    assert "round 1: new incumbent g01_t0_s00" in text and "lowest KLs g01_t0_s00=0.012300" in text
    assert "first-quantization round" in text and "kl" in text
    assert P.history_row({"artifact": {"config": {"default": {"bits": 4}}, "note": "why"},
                          "screen": {"feedback": fb}}) == {"config": "default 4b", "GiB": None, "avg_bits": None,
                                                          "kl": 0.0111, "note": "why"}
