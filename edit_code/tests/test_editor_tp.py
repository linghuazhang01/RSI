"""Editor sessions bounded by rounds only, the editor served tensor parallel over every GPU (2026-11-26).

    <venv>/bin/python -I -m pytest -q edit_code/tests/test_editor_tp.py

- editor_tp_gpus: editor.tensor_parallel all / N / 1 -> the editor's TP group (power of two, starting at the editor
  role's GPU, continuing over the candidate GPUs);
- Runner: the frozen editor (and the current-mode editor) runs on that group with sleep mode, generation 0's M_0
  stays one replica per GPU, and a TP editor without editor_sleep is refused;
- the editor loop: max_tool_calls / proposal_tokens / proposal_wall_s null = no cap, tool_rounds still applies; the
  system prompt drops the tool-call clause without a cap.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]


def _tp(n_gpus, value, editor=None):
    from edit_code.common import editor_tp_gpus
    rc = SimpleNamespace(gpus=[10 + i for i in range(n_gpus)])
    cfg = {} if value is ... else {"editor": {"tensor_parallel": value}}
    return editor_tp_gpus(rc, cfg, editor or [10])


def test_editor_tp_gpus():
    assert _tp(8, "all") == [10, 11, 12, 13, 14, 15, 16, 17]
    assert _tp(8, None) == list(range(10, 18))
    assert _tp(8, 4) == [10, 11, 12, 13]
    assert _tp(8, 6) == [10, 11, 12, 13]                     # rounded down to a power of two
    assert _tp(6, "all") == [10, 11, 12, 13]
    assert _tp(8, 1) == [10] and _tp(8, ...) == [10]          # absent = the editor role's GPU only (old layout)
    assert _tp(1, "all") == [10]
    assert _tp(8, "all", editor=[13]) == [13, 10, 11, 12, 14, 15, 16, 17]   # starts at the editor role's GPU
    assert _tp(5, 2, editor=[10, 11]) == [10, 11]             # not above the editor role's GPUs: unchanged
    for bad in (0, -2, "x"):
        with pytest.raises(ValueError):
            _tp(8, bad)


def _runner(tmp_path, gpus="0,1,2,3,4,5,6,7", *sets):
    from rsicore.config import build_arg_parser, load_run_config
    from edit_code.run import Runner, effective_config
    ap = build_arg_parser("A")
    ap.add_argument("--budget-gib", type=float, default=None)
    args = ap.parse_args(["--domain", "math", "--mock", "--gpus", gpus, "--runs-dir", str(tmp_path / "runs"),
                          "--run-id", "tp", *[x for s in sets for x in ("--set", s)]])
    rc = load_run_config("A", args)
    return Runner(rc, effective_config(rc, args))


def test_runner_editor_on_tp_group_with_sleep(tmp_path):
    r = _runner(tmp_path)
    assert r.gpu_editor == [0] and r.editor_gpus == list(range(8))
    assert r.cand_slots == [[g] for g in range(1, 8)]          # candidates keep one GPU each
    ed = r.editor_server(1, r.m0_server())
    assert ed.kw["gpu_ids"] == list(range(8)) and ed.kw["sleep_mode"] is True
    m0 = r.m0_server()                                         # generation 0's M_0: data parallel, one GPU each
    assert m0.kw["gpu_ids"] == [0] and [s.kw["gpu_ids"] for s in r.m0_replicas_for(m0)[1:]] == r.cand_slots


def test_h20_80_keeps_editor_on_its_own_gpu(tmp_path):
    r = _runner(tmp_path, "0,1,2,3,4,5,6,7", "hardware=h20_80")
    assert r.editor_gpus == [0]
    assert _runner(tmp_path / "b", "0,1,2,3,4,5,6,7", "hardware=h20").editor_gpus == list(range(8))


def test_editor_phase_frees_cuda_before_waking(tmp_path, monkeypatch):
    from edit_code import run as RUN
    seen = []
    monkeypatch.setattr(RUN, "free_cuda", lambda: seen.append("free"))
    ed = type("Ed", (), {"get": lambda self: seen.append("get"), "srv": None})()
    RUN.EditorPhases(ed, None, [0], True).editor_phase()
    assert seen == ["free", "get"]


def test_runner_tp1_keeps_old_layout(tmp_path):
    r = _runner(tmp_path, "0,1,2,3", "editor.tensor_parallel=1")
    m0 = r.m0_server()
    assert r.editor_gpus == [0] and r.editor_server(1, m0) is m0


def test_runner_refuses_tp_editor_that_cannot_sleep(tmp_path):
    with pytest.raises(ValueError, match="tensor_parallel"):
        _runner(tmp_path, "0,1,2,3", "gpu_layout.editor_sleep=false")
    r = _runner(tmp_path / "b", "0,1,2,3", "gpu_layout.editor_sleep=false", "editor.tensor_parallel=1")
    assert r.editor_gpus == [0]


def test_current_mode_editor_sleeps(tmp_path):
    """loop.editor current: the incumbent's server also spans the TP group and sleeps while its GPUs are lent."""
    from edit_code import run as RUN
    r = _runner(tmp_path, "0,1,2,3", "loop.editor=current")
    mdir = tmp_path / "q"
    mdir.mkdir()
    RUN.M.save_state(mdir / "mock_model.json", RUN.M.fresh_state(0.8))
    r.incumbent = lambda g: {"gen": 1, "model_dir": str(mdir)}
    ed = r.editor_server(2, r.m0_server())
    assert ed.kw["gpu_ids"] == [0, 1, 2, 3] and ed.kw["sleep_mode"] is True


def test_editor_caps_null_rounds_only(tmp_path):
    from rsicore.editor import Editor, editor_kwargs
    kw = editor_kwargs({"editor": {"max_tool_calls": None, "proposal_tokens": None, "proposal_wall_s": None,
                                   "tool_rounds": 50}})
    assert kw["max_tool_calls"] is None and kw["proposal_tokens"] is None and kw["proposal_wall_s"] is None
    assert kw["tool_rounds"] == 50
    ed = Editor("http://mock.invalid/v1", "m", system_prompt="S", seed=1, transcript=tmp_path / "t.jsonl", **kw)
    assert ed.max_tool_calls is None and ed.proposal_tokens is None and ed.proposal_wall_s is None
    assert editor_kwargs({"editor": {"max_tool_calls": 0}})["max_tool_calls"] is None     # 0 = no cap, as the prompt
    # the loop itself (only tool_rounds forces the final turn): eval/tests/rsicore/test_editor_loop.py


def test_prompt_without_tool_call_cap():
    from edit_code.prompts import TOOL_PROTOCOL, _tool_rounds
    rounds, _, _, calls = _tool_rounds({"editor": {"tool_rounds": 50, "max_tool_calls": None}})
    assert (rounds, calls) == (50, 0)
    assert "{calls}" in TOOL_PROTOCOL and "at most {calls} calls" not in TOOL_PROTOCOL


# ---------------------------------------------------------------------------- exploration GPU tools (2026-11-26)
def _ex(n_gpus, tp, tools="rest", gpu_tools=True, editor=None):
    from edit_code.common import explore_layout
    rc = SimpleNamespace(gpus=list(range(n_gpus)))
    cfg = {"editor": {"tensor_parallel": tp, "explore": {"gpu_tools": gpu_tools}},
           "gpu_layout": {"explore_tool_gpus": tools}}
    return explore_layout(rc, cfg, editor or [0])


def test_explore_layout_split():
    assert _ex(8, 4) == ([0, 1, 2, 3], [4, 5, 6, 7])                   # rest = outside the TP group
    with pytest.raises(ValueError, match="covers every GPU"):           # TP over everything: no tool GPU left
        _ex(8, "all")
    assert _ex(8, "all", tools=0) == (list(range(8)), [])               # 0 = accept the shared first candidate GPU
    assert _ex(8, 1) == ([0], list(range(1, 8)))                        # TP 1: every candidate GPU for the tools
    assert _ex(8, "all", tools=4) == ([0, 1, 2, 3], [4, 5, 6, 7])      # k tool GPUs, TP from the others
    assert _ex(8, "all", tools=2) == ([0, 1, 2, 3], [6, 7])            # 6 left -> TP 4; GPUs 4-5 idle meanwhile
    assert _ex(8, 2, tools=2) == ([0, 1], [6, 7])
    assert _ex(8, 4, gpu_tools=False) == ([0, 1, 2, 3], [])             # tools off: nothing reserved
    assert _ex(8, 4, tools=0) == ([0, 1, 2, 3], [])
    with pytest.raises(ValueError, match="does not fit"):
        _ex(8, 8, tools=2)
    for bad in (-1, 8, "x"):
        with pytest.raises(ValueError):
            _ex(8, 4, tools=bad)


def test_runner_explore_tool_gpus(tmp_path):
    r = _runner(tmp_path, "0,1,2,3,4,5,6,7", "editor.tensor_parallel=4", "editor.explore.gpu_tools=true")
    assert r.editor_gpus == [0, 1, 2, 3] and r.tool_gpus == [4, 5, 6, 7]
    assert r.cand_slots == [[g] for g in range(1, 8)]          # evaluation still uses every GPU
    r = _runner(tmp_path / "b", "0,1,2,3,4,5,6,7")
    assert r.tool_gpus == []


def test_tool_device_pool_spreads_and_waits():
    import threading
    from edit_code.problem import QuantProblem
    opened = []
    prob = SimpleNamespace(tool_gpus=[4, 5], mock=False, _lock=threading.Lock(), _tool_pool=None, _tool_caches={},
                           _open_cache=lambda device: opened.append(device) or f"cache@{device}",
                           _device=lambda: "cuda")
    td = QuantProblem.tool_device.__get__(prob)
    with td() as (c1, l1), td() as (c2, l2):                  # two sessions at once: two different GPUs
        assert {c1, c2} == {"cache@cuda:4", "cache@cuda:5"} and l1 is not l2
        got = []
        t = threading.Thread(target=lambda: got.append(td().__enter__()))
        t.start()
        t.join(0.3)
        assert t.is_alive() and not got                        # a third waits for a free tool GPU
    t.join(5)
    assert got and got[0][0] in ("cache@cuda:4", "cache@cuda:5")
    assert sorted(opened) == ["cuda:4", "cuda:5"]              # one layer cache per GPU, reused
    prob.tool_gpus = []
    with QuantProblem.tool_device.__get__(SimpleNamespace(tool_gpus=[], mock=False))() as got2:
        assert got2 == (None, None)                            # no tool GPU: self.cache, as before


def test_explore_group_error_uses_tool_device(tmp_path):
    """editor.explore.gpu_tools: an uncached group_error while exploring runs on a tool GPU's cache."""
    import contextlib
    from edit_code.tools import make_tool_api
    r = _runner(tmp_path, "0,1,2,3", "editor.tensor_parallel=2", "editor.explore.gpu_tools=true")
    prob = r.problem(tag="t", gen=1, out_dir=tmp_path / "g", budget=10.0)
    prob.prepare()
    seen = []

    @contextlib.contextmanager
    def dev(deadline=None):
        seen.append("dev")
        yield "C", "L"
    prob.tool_device = dev
    calls = []
    real = prob.group_error
    prob.group_error = lambda config, top=None, cache=None, lock=None, deadline=None: (
        calls.append((cache, lock)), real(config, top=top))[1]
    prob.herr_missing = lambda config: ["m|gptq|3|64"]
    api = make_tool_api(prob, r.cfg, explore=True)
    cfg = {"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"}}
    api.call("group_error", {"config": cfg})
    assert seen == ["dev"] and calls == [("C", "L")]
    prob.herr_missing = lambda config: []                       # cached: no tool GPU needed
    api.call("group_error", {"config": cfg})
    assert seen == ["dev"] and calls[-1] == (None, None)


def test_herr_deadline_stops_and_keeps_progress(tmp_path):
    """An exploration group_error stops at its deadline with a ToolError that counts what was computed; the computed
    settings stay cached, so the next call continues (mock problem: herr is synthetic but goes through herr_module)."""
    import time
    from edit_code.problem import HerrDeadline
    from edit_code.tools import ToolError, make_tool_api
    r = _runner(tmp_path, "0,1,2,3", "editor.tensor_parallel=2", "editor.explore.gpu_tools=true")
    prob = r.problem(tag="t", gen=1, out_dir=tmp_path / "g", budget=10.0)
    prob.prepare()
    cfg = {"default": {"bits": 3, "group_size": 64, "quantizer": "gptq"}}
    total = len(prob.herr_missing(cfg))
    assert total > 10
    calls = {"n": 0}
    real = prob.herr_module

    def slow(module, setting, cache=None, lock=None, deadline=None):
        calls["n"] += 1
        if calls["n"] > 5:                                    # the 6th uncached module hits the deadline
            deadline = time.monotonic() - 1
        return real(module, setting, cache=cache, lock=lock, deadline=deadline)
    prob.herr_module = slow
    dl = {"t": time.monotonic() + 3600}
    api = make_tool_api(prob, r.cfg, explore=True, deadline=dl)
    with pytest.raises(Exception) as ei:
        api.call("group_error", {"config": cfg})
    assert "ran out of time" in str(ei.value) and f"of {total} uncached" in str(ei.value)
    left = len(prob.herr_missing(cfg))
    assert 0 < left < total                                   # progress kept (flushed to the herr cache)
    prob.herr_module = real
    out = api.call("group_error", {"config": cfg})            # the retry finishes
    assert "total_per_token" in out and prob.herr_missing(cfg) == []
    with pytest.raises(HerrDeadline):
        prob.herr_module("x", {"bits": 3, "group_size": 64, "quantizer": "gptq"}, deadline=time.monotonic() - 1)
    _ = ToolError


def test_tool_device_wait_times_out():
    import threading
    import time
    from edit_code.problem import HerrDeadline, QuantProblem
    prob = SimpleNamespace(tool_gpus=[4], mock=False, _lock=threading.Lock(), _tool_pool=None, _tool_caches={},
                           _open_cache=lambda device: f"cache@{device}", _device=lambda: "cuda")
    td = QuantProblem.tool_device.__get__(prob)
    with td():
        with pytest.raises(HerrDeadline):
            with td(deadline=time.monotonic() + 0.2):
                pass
    cpu = SimpleNamespace(tool_gpus=[4], mock=False, _lock=threading.Lock(), _tool_pool=None, _tool_caches={},
                          _open_cache=lambda device: f"cache@{device}", _device=lambda: "cpu")
    with QuantProblem.tool_device.__get__(cpu)() as (c, _):
        assert c == "cache@cpu"                               # cache.device cpu: kept on the CPU


def test_prompt_tells_editor_about_gpu_tools():
    import types
    from edit_code import prompts as P
    prob = types.SimpleNamespace(space={"bits": [2, 3, 4, 8, 16], "group_sizes": [32, 64, 128],
                                        "quantizers": ["rtn", "gptq", "awq"]}, kl_items=lambda: [{}] * 256)
    on = P.system_prompt(prob, "DOC", {"editor": {"tool_rounds": 50, "explore": {"gpu_tools": True}}})
    off = P.system_prompt(prob, "DOC", {"editor": {"tool_rounds": 50}})
    assert "computes settings that are not cached yet, on a GPU" in on and "calling again continues" in on
    assert "only accepts settings already cached" in off and "on a GPU (seconds" not in off
