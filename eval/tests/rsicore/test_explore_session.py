"""Persistent exploration session (2026-11-21, editor.explore.persistent): rsicore.sandbox.Kernel and its use in
ChatTools (one kernel + one explore api per proposal, timeout restart, isolation, end / begin, Editor calls end),
the per-call mode unchanged; screen records keep the thinking text per sample (screen_view).

  python -I -m pytest -q eval/tests/rsicore/test_explore_session.py -p no:cacheprovider
"""
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from rsicore import sandbox, screen_view  # noqa: E402
from rsicore.chat_tools import (RUN_PYTHON_DOC, RUN_PYTHON_DOC_PERSISTENT, ChatTools,  # noqa: E402
                                chat_tools_kwargs, with_mock_exploration)
from rsicore.editor import Editor, editor_settings  # noqa: E402


class Factory:
    """Explore api factory: counts calls / cleanups / before_call; api.bump() counts per api object."""

    def __init__(self, form="triple"):
        self.made, self.cleaned, self.before, self.form = [], [], [], form

    def __call__(self, tag, ctx):
        api = sandbox.ToolAPI({})
        state = {"n": 0}

        def bump():
            state["n"] += 1
            return state["n"]
        api.register("bump", bump)
        api.register("tag", lambda: tag)
        self.made.append(tag)
        if self.form == "api":
            return api
        cleanup = (lambda: self.cleaned.append(tag))
        if self.form == "pair":
            return api, cleanup
        return api, cleanup, (lambda t: self.before.append((tag, t)))


def run(t, code, ctx, r=1, i=0):
    res = t.call("run_python", json.dumps({"code": code}), ctx, r, i)
    return json.loads(res["content"])


def test_persistent_state_api_and_end(tmp_path):
    f = Factory()
    t = ChatTools(f, timeout_s=20, mem_gb=2, persistent=True)
    assert t.run_python_doc == RUN_PYTHON_DOC_PERSISTENT
    ctx = {"cid": "g01_t0_s01", "gen_dir": str(tmp_path)}
    out = run(t, "import math\nx = 41\ndef f(y):\n    return y + 1\nprint('hello')\nresult = api.bump()", ctx, 1)
    assert out["ok"] and out["result"] == 1 and "hello" in out["stdout"]
    out = run(t, "result = [f(x), round(math.pi, 2), api.bump(), api.tag()]", ctx, 2)
    assert out["result"] == [42, 3.14, 2, "g01_t0_s01~x"]          # vars / imports / functions / api persist
    out = run(t, "y = 5", ctx, 3)
    assert out["ok"] and "result" not in out                          # result cleared before every call
    assert f.made == ["g01_t0_s01~x"] and len(f.before) == 3 and f.before[0][1] == 20
    sess = tmp_path / "explore" / "g01_t0_s01" / "session"
    assert (sess / "r01_0" / "program.py").exists() and "hello" in (sess / "r01_0" / "stdout.txt").read_text()
    assert json.loads((sess / "r02_0" / "result.json").read_text())["ok"]
    pid = t._sessions["g01_t0_s01"]["kernel"].proc.pid
    t.end(ctx)
    assert f.cleaned == ["g01_t0_s01~x"] and not t._sessions
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive
    t.end(ctx)                                                        # idempotent
    out = run(t, "result = 'x' in globals()", ctx, 4)                 # a new session after end
    assert out["result"] is False and len(f.made) == 2
    t.begin(ctx)                                                      # begin discards the stale kernel
    assert not t._sessions and len(f.cleaned) == 2


def test_timeout_restarts_kernel_api_persists(tmp_path):
    f = Factory()
    t = ChatTools(f, timeout_s=3, mem_gb=2, persistent=True)
    ctx = {"cid": "c1", "gen_dir": str(tmp_path)}
    assert run(t, "x = 1\nresult = api.bump()", ctx, 1)["result"] == 1
    out = run(t, "while True:\n    pass", ctx, 2)
    assert not out["ok"] and out["restarted"] and "session restarted: variables lost" in out["error"]
    out = run(t, "result = ['x' in globals(), api.bump()]", ctx, 3)
    assert out["ok"] and out["result"] == [False, 2]                  # variables lost, same api object
    assert len(f.made) == 1
    out = run(t, "import os\nos._exit(3)", ctx, 4)                    # crash -> restart too
    assert out["restarted"] and "variables lost" in out["error"]
    assert run(t, "result = 7", ctx, 5)["result"] == 7
    out = run(t, "import subprocess\nsubprocess.run(['ls'])", ctx, 6)  # same restrictions as run_program
    assert not out["ok"] and "not allowed" in out["error"] and not out.get("restarted")
    t.end_all()


def test_concurrent_proposals_isolated(tmp_path):
    f = Factory(form="pair")
    t = ChatTools(f, timeout_s=20, mem_gb=2, persistent=True)
    errs, outs = [], {}

    def worker(cid, val):
        try:
            ctx = {"cid": cid, "gen_dir": str(tmp_path)}
            run(t, f"v = {val}", ctx, 1)
            outs[cid] = run(t, "result = [v, api.bump(), api.bump()]", ctx, 2)["result"]
            t.end(ctx)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    th = [threading.Thread(target=worker, args=(f"c{k}", k * 10)) for k in range(3)]
    [x.start() for x in th]
    [x.join() for x in th]
    assert not errs and outs == {"c0": [0, 1, 2], "c1": [10, 1, 2], "c2": [20, 1, 2]}
    assert sorted(f.cleaned) == ["c0~x", "c1~x", "c2~x"] and not t._sessions


def test_per_call_mode_unchanged(tmp_path):
    f = Factory(form="api")
    t = ChatTools(f, timeout_s=20, mem_gb=2)
    assert t.run_python_doc == RUN_PYTHON_DOC and not t.persistent
    ctx = {"cid": "c1", "gen_dir": str(tmp_path)}
    assert run(t, "x = 1\nresult = api.bump()", ctx, 1)["result"] == 1
    out = run(t, "result = ['x' in globals(), api.bump()]", ctx, 2)
    assert out["result"] == [False, 1] and f.made == ["c1~x01_0", "c1~x02_0"]
    assert (tmp_path / "explore" / "c1" / "r01_0" / "program.py").exists()
    t.end(ctx)                                                        # no-op
    # a paper doc built on RUN_PYTHON_DOC gets the persistent prefix in persistent mode
    tp = ChatTools(f, run_python_doc=RUN_PYTHON_DOC + " Extra.", persistent=True)
    assert tp.run_python_doc == RUN_PYTHON_DOC_PERSISTENT + " Extra."


def test_settings_and_editor_ends_session(tmp_path):
    assert editor_settings({})["explore"]["persistent"] is False
    assert chat_tools_kwargs({})["persistent"] is False
    assert chat_tools_kwargs({"editor": {"explore": {"persistent": True}}})["persistent"] is True
    f = Factory()
    tools = ChatTools(f, timeout_s=20, mem_gb=2, persistent=True)
    pol = with_mock_exploration(lambda msgs: "def main(api):\n    return 1\n", 2, "n = globals().get('n', 0) + 1\nresult = n")
    ed = Editor("http://unused", "m", system_prompt="S", seed=1, transcript=tmp_path / "m.jsonl", tools=tools,
                mock_policy=pol)
    ed.propose("msg", ctx={"cid": "c0", "gen_dir": str(tmp_path), "visible_cids": []})
    assert f.made == ["c0~x"] and f.cleaned == ["c0~x"] and not tools._sessions
    tl = [json.loads(l["content"]) for l in map(json.loads, (tmp_path / "m.jsonl").read_text().splitlines())
          if l.get("role") == "tool"]
    assert [x["result"] for x in tl] == [1, 2]                         # state persisted across the two rounds


def test_screen_records_keep_reasoning():
    from evallib.tasks import Task
    task = Task.__new__(Task)
    task.name, task.dataset = "math", "math"
    choices = [{"content": f"The answer is {k}", "reasoning": f"think {k}", "finish_reason": "stop"}
               for k in range(4)]
    rec = task.make_record({"id": "q1", "split_pool": "train"}, "q1", choices, [1.0, 0.0, 1.0, 0.0],
                           ["0", "1", "2", "3"], [None] * 4)
    assert rec["reasoning"] == ["think 0", "think 1", "think 2", "think 3"]
    recs = {"math": [rec]}
    items = {"math": {"q1": {"prompt": "What?", "reference": "0"}}}
    s = screen_view.summary(recs, items)
    assert s["accuracy"] == 0.5 and s["correct_counts"] == {"0": 0, "1": 0, "2": 1, "3": 0, "4": 0}
    assert screen_view.rows(recs, items, filter="unstable")[0]["scores"] == [1.0, 0.0, 1.0, 0.0]
    assert screen_view.rows(recs, items, filter="wrong") == []
    o = screen_view.output(rec, sample=1)
    assert o["has_thinking"] and o["text"].startswith("<think>\nthink 1\n</think>") and "answer is 1" in o["text"]
