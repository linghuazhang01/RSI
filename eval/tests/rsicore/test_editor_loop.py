"""Editor tool loop (2026-11-21): rounds, forced final turn, guards, nudges, fallback, compaction, HTTP 400,
transcript v2 / v1 / incomplete proposals / replay, history_keep, context clamp, chat tools, search wiring,
server helpers (set_max_model_len, tool-call parser auto, max_position_embeddings clamp).

  python -I -m pytest -q eval/tests/rsicore/test_editor_loop.py -p no:cacheprovider

A tiny fake OpenAI-compatible server (threading http.server) returns scripted replies with tool_calls,
reasoning_content and usage, and records every request payload.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from rsicore import chat_tools, editor, records, sandbox, search, servers  # noqa: E402
from rsicore.chat_tools import ChatTools, with_mock_exploration  # noqa: E402
from rsicore.editor import Editor, EditorReplyError  # noqa: E402

CODE = "def main(api):\n    return 1\n"
NEXT = "def main(api):\n    return 2\n"
ANSWER = f"Plan: return one.\n```python\n{CODE}```"


# ----------------------------------------------------------------------------- fake server
class FakeServer:
    """script: list of entries, consumed one per /chat/completions request:
       dict {content, reasoning, tool_calls: [(name, args)], finish, usage} | ("http", status, body) |
       callable(payload) -> entry. When the script is empty: `default` (None -> HTTP 599 = test failure)."""

    def __init__(self, script=(), default=None, max_model_len=None, models=None):
        self.script, self.default, self.max_model_len = list(script), default, max_model_len
        self.models = models
        self.requests = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, obj):
                b = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if fake.models is not None:
                    return self._send(200, {"data": fake.models})
                d = {"id": "m", "object": "model"}
                if fake.max_model_len:
                    d["max_model_len"] = fake.max_model_len
                self._send(200, {"data": [d]})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n))
                fake.requests.append(payload)
                e = fake.script.pop(0) if fake.script else fake.default
                if callable(e):
                    e = e(payload)
                if e is None:
                    return self._send(599, {"error": "script exhausted"})
                if isinstance(e, tuple):
                    return self._send(e[1], {"error": {"message": e[2]}})
                calls = [{"id": f"c{len(fake.requests)}_{i}", "type": "function",
                          "function": {"name": nm, "arguments": a if isinstance(a, str) else json.dumps(a)}}
                         for i, (nm, a) in enumerate(e.get("tool_calls") or [])]
                msg = {"role": "assistant", "content": e.get("content"), "reasoning_content": e.get("reasoning")}
                if calls:
                    msg["tool_calls"] = calls
                body = {"choices": [{"message": msg, "finish_reason": e.get("finish") or ("tool_calls" if calls else "stop")}]}
                if e.get("usage") is not False:
                    body["usage"] = e.get("usage") or {"prompt_tokens": 0, "completion_tokens": 10}
                self._send(200, body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.endpoint = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake():
    made = []

    def make(*a, **k):
        s = FakeServer(*a, **k)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


class FakeTools:
    """Minimal tools object: schemas() + call(); records calls; content = 'out:<code>' (or a fixed size)."""

    def __init__(self, size=None):
        self.calls, self.size = [], size

    def schemas(self):
        return [{"type": "function", "function": {"name": "run_python", "description": "run",
                                                  "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}]

    def call(self, name, arguments, ctx, r, i, deadline_s=None):
        self.calls.append((name, arguments, dict(ctx or {}), r, i))
        content = ("x" * self.size) if self.size else f"out:{arguments}"
        return {"content": content, "ok": True, "seconds": 0.01, "args": json.loads(arguments or "{}")}


def tool_turn(code="print(1)", reasoning="thinking", n=1, content="", usage=None):
    return {"content": content, "reasoning": reasoning, "tool_calls": [("run_python", {"code": code})] * n,
            "usage": usage}


def mk(tmp, srv, tools=None, name="t.jsonl", **kw):
    kw.setdefault("context_tokens", None)
    return Editor(srv.endpoint if srv else "http://unused/v1", "m", system_prompt="SYS", seed=7,
                  transcript=Path(tmp) / name, tools=tools, **kw)


def lines(path):
    return records.read_jsonl(path)


# ----------------------------------------------------------------------------- loop
def test_two_rounds_then_answer(tmp_path, fake):
    srv = fake([tool_turn("a=1", usage={"prompt_tokens": 100, "completion_tokens": 20}),
                tool_turn("b=2", reasoning="more", usage={"prompt_tokens": 150, "completion_tokens": 30}),
                {"content": ANSWER, "reasoning": "done", "usage": {"prompt_tokens": 200, "completion_tokens": 40}}])
    tools = FakeTools()
    ed = mk(tmp_path, srv, tools, **editor.editor_kwargs({"editor": {"context_tokens": None}}))
    code = ed.propose("feedback [thread=0 step=0]", ctx={"cid": "g01_t0_s00", "gen_dir": str(tmp_path), "visible_cids": []})
    assert code == CODE
    st = ed.last_stats
    assert (st["rounds"], st["tool_calls"], st["stop"], st["model_calls"]) == (2, 2, "answer", 3)
    assert (st["prompt_tokens"], st["completion_tokens"]) == (450, 90)
    assert ed.n_calls == 3 and len(tools.calls) == 2 and tools.calls[0][2]["cid"] == "g01_t0_s00"
    r0, r1, r2 = srv.requests
    assert r0["tool_choice"] == "auto" and r2["tool_choice"] == "auto" and r0["tools"][0]["function"]["name"] == "run_python"
    assert r0["chat_template_kwargs"] == {"enable_thinking": True} and r0["top_k"] == 20 and r0["min_p"] == 0.0
    assert r0["max_tokens"] == 8192 and r0["temperature"] == 0.6 and r0["top_p"] == 0.95
    m2 = r2["messages"]
    assert [m["role"] for m in m2] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert m2[2]["content"].startswith("<think>\nthinking\n</think>") and m2[2]["tool_calls"][0]["function"]["name"] == "run_python"
    assert m2[3]["tool_call_id"] == "c1_0" and m2[3]["content"].startswith("out:")
    tr = lines(tmp_path / "t.jsonl")
    assert tr[0]["format"] == 2
    kinds = [(l["role"], l.get("r"), bool(l.get("final"))) for l in tr[1:]]
    assert kinds == [("user", None, False), ("assistant", 1, False), ("tool", 1, False), ("assistant", 2, False),
                     ("tool", 2, False), ("assistant", None, True)]
    assert tr[-1]["code"] == CODE and tr[-1]["stats"]["rounds"] == 2 and tr[2]["reasoning"] == "thinking"
    # cross-step history is collapsed: no tool turns, no reasoning
    assert [m["role"] for m in ed._prompt_history()] == ["system", "user", "assistant"]
    assert ed._prompt_history()[2]["content"] == ANSWER


def test_cap_forces_tool_choice_none(tmp_path, fake):
    srv = fake(default=lambda p: (tool_turn(content="```python\nx=0\n```") if p["tool_choice"] == "auto"
                                  else {"content": ANSWER}))
    ed = mk(tmp_path, srv, FakeTools(), tool_rounds=50)
    code = ed.propose("feedback [thread=1 step=2]", ctx={"cid": "c"})
    assert code == CODE and ed.last_stats["stop"] == "forced:rounds" and ed.last_stats["rounds"] == 50
    assert len(srv.requests) == 51
    assert [r["tool_choice"] for r in srv.requests] == ["auto"] * 50 + ["none"]
    msgs = srv.requests[-1]["messages"]
    assert [m["role"] for m in msgs].count("user") == 1 and msgs[1]["content"] == "feedback [thread=1 step=2]"
    assert msgs[-1]["role"] == "tool" and chat_tools.FORCED_MARK in msgs[-1]["content"]
    assert "[5 tool rounds left]" in msgs[2 + 2 * 44 + 1]["content"]       # round 45
    assert "tool rounds left" not in msgs[2 + 2 * 43 + 1]["content"]       # round 44
    assert srv.requests[-1]["max_tokens"] == ed.max_tokens


def test_tool_rounds_zero_payload_is_unchanged(tmp_path, fake):
    srv = fake([{"content": ANSWER, "tool_calls": []}])
    kw = editor.editor_kwargs({"editor": {"tool_rounds": 0, "extra_body": {}, "max_tokens": 16384}})
    ed = mk(tmp_path, srv, FakeTools(), **kw)
    assert ed.propose("hello") == CODE
    assert len(srv.requests) == 1
    p = srv.requests[0]
    assert list(p) == ["model", "messages", "temperature", "top_p", "max_tokens", "seed"]
    assert p["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]
    assert p["max_tokens"] == 16384 and ed.last_stats["rounds"] == 0
    ed2 = mk(tmp_path, fake([{"content": ANSWER}]), None, name="u.jsonl")    # no tools at all: same payload
    ed2.propose("hello")


def test_guards(tmp_path, fake):
    # wall clock: a fake clock advancing 1000 s per read
    t = {"now": 0.0}

    def clock():
        t["now"] += 1000.0
        return t["now"]
    srv = fake(default=lambda p: tool_turn() if p["tool_choice"] == "auto" else {"content": ANSWER})
    ed = mk(tmp_path, srv, FakeTools(), name="w.jsonl", proposal_wall_s=2400, clock=clock)
    assert ed.propose("m") == CODE and ed.last_stats["stop"] == "forced:wall"
    # completion tokens
    srv = fake(default=lambda p: (tool_turn(usage={"prompt_tokens": 5, "completion_tokens": 100})
                                  if p["tool_choice"] == "auto" else {"content": ANSWER}))
    ed = mk(tmp_path, srv, FakeTools(), name="k.jsonl", proposal_tokens=250)
    assert ed.propose("m") == CODE and ed.last_stats["stop"] == "forced:tokens" and ed.last_stats["rounds"] == 3
    # tool calls: 3 parallel calls per turn, cap 4 -> second turn runs 1, skips 2
    srv = fake(default=lambda p: tool_turn(n=3) if p["tool_choice"] == "auto" else {"content": ANSWER})
    tools = FakeTools()
    ed = mk(tmp_path, srv, tools, name="c.jsonl", max_tool_calls=4)
    assert ed.propose("m") == CODE and ed.last_stats["stop"] == "forced:tool_calls"
    assert len(tools.calls) == 4 and ed.last_stats["tool_calls"] == 4
    tl = [l for l in lines(tmp_path / "c.jsonl") if l["role"] == "tool"]
    assert len(tl) == 6 and sum("tool-call cap" in l["content"] for l in tl) == 2


def test_nudges_then_fallback(tmp_path, fake):
    srv = fake([tool_turn(content="draft\n```python\nDRAFT = 1\n```"),
                {"content": ""},
                {"content": "<tool_call>{\"name\": \"run_python\"}</tool_call>"},
                {"content": "I am done thinking."}])
    ed = mk(tmp_path, srv, FakeTools())
    code = ed.propose("m")
    assert code == "DRAFT = 1\n" and ed.last_stats["stop"] == "fallback" and ed.last_stats["nudges"] == 2
    msgs = srv.requests[-1]["messages"]
    assert [m["role"] for m in msgs][-4:] == ["assistant", "user", "assistant", "user"]
    assert msgs[-1]["content"] == editor.NUDGE["tool_text"] and msgs[-3]["content"] == editor.NUDGE["empty"]
    # a reply cut at max_tokens is nudged, then answers
    srv = fake([{"content": "", "finish": "length"}, {"content": ANSWER}])
    ed = mk(tmp_path, srv, FakeTools(), name="l.jsonl")
    assert ed.propose("m") == CODE and ed.last_stats["nudges"] == 1
    assert srv.requests[1]["messages"][-1]["content"] == editor.NUDGE["length"]
    # forced final cut at length without code -> stop "length", empty program, search explains it
    srv = fake([{"content": "", "finish": "length"}])
    ed = mk(tmp_path, srv, FakeTools(), name="f.jsonl", tool_rounds=0)
    assert ed.propose("m") == "" and ed.last_stats["stop"] == "length"
    assert "max_tokens" in editor.describe_stop(ed.last_stats)


# ----------------------------------------------------------------------------- chat tools
def _api_factory(log):
    def factory(tag, ctx):
        api = sandbox.ToolAPI({})
        api.register("double", lambda x: 2 * x, doc="double(x)")
        log.append(tag)
        return api, (lambda: log.append("cleanup:" + tag))
    return factory


def test_chat_tools_errors_and_truncation(tmp_path, fake):
    log = []
    tools = ChatTools(_api_factory(log), timeout_s=3, mem_gb=2, result_chars=6000)
    srv = fake([{"content": "", "tool_calls": [
        ("run_python", {"code": "raise ValueError('boom')"}),
        ("no_such_tool", {}),
        ("run_python", "{not json"),
        ("run_python", {"code": "while True:\n    pass"})]},
        {"content": "", "tool_calls": [
            ("run_python", {"code": "print('y' * 80000)\nresult = api.double(x=21)"}),
            ("run_python", {"code": "def main(api):\n    return api.double(x=5)"}),
            ("run_python", {"code": "hasattr(api, 'evaluate') and print('HAS')\nresult = sorted(api.tools())"})]},
        {"content": ANSWER}])
    gd = tmp_path / "gen"
    ed = mk(tmp_path, srv, tools)
    assert ed.propose("m", ctx={"cid": "g01_t0_s00", "gen_dir": str(gd), "visible_cids": []}) == CODE
    tl = [l for l in lines(tmp_path / "t.jsonl") if l["role"] == "tool"]
    assert len(tl) == 7 and ed.last_stats["rounds"] == 2
    assert not tl[0]["ok"] and "ValueError: boom" in tl[0]["content"]
    assert not tl[1]["ok"] and "unknown tool" in tl[1]["content"]
    assert not tl[2]["ok"] and "bad arguments" in tl[2]["content"]
    assert not tl[3]["ok"] and "timeout" in tl[3]["content"]       # wall clock or SIGXCPU: both read "timeout"
    big = json.loads(tl[4]["content"])
    assert big["ok"] and big["result"] == 42 and "chars cut" in big["stdout"] and len(tl[4]["content"]) <= 6100
    full = (gd / "explore" / "g01_t0_s00" / "r02_0" / "stdout.txt").read_text()
    assert len(full) >= 80000
    assert json.loads(tl[5]["content"])["result"] == 10
    assert json.loads(tl[6]["content"])["result"] == ["double"] and "HAS" not in tl[6]["content"]
    assert "g01_t0_s00~x01_0" in log and "cleanup:g01_t0_s00~x02_2" in log


def test_read_record_visibility(tmp_path):
    gd = tmp_path / "gen"
    (gd / "programs").mkdir(parents=True)
    (gd / "feedback").mkdir()
    (gd / "programs" / "g01_t0_s00.py").write_text(CODE)
    (gd / "feedback" / "g01_t0_s00.json").write_text(json.dumps({"x": "z" * 9000}))
    t = ChatTools(lambda tag, ctx: sandbox.ToolAPI({}), result_chars=2000)
    ctx = {"gen_dir": str(gd), "visible_cids": ["g01_t0_s00"]}
    r = t.call("read_record", {"cid": "g01_t1_s00", "part": "program"}, ctx, 1, 0)
    assert not r["ok"] and "not visible" in r["content"]
    r = json.loads(t.call("read_record", {"cid": "g01_t0_s00", "part": "program"}, ctx, 1, 0)["content"])
    assert r["text"] == CODE and "next_offset" not in r
    r = json.loads(t.call("read_record", {"cid": "g01_t0_s00", "part": "feedback"}, ctx, 1, 0)["content"])
    r2 = json.loads(t.call("read_record", {"cid": "g01_t0_s00", "part": "feedback", "offset": r["next_offset"]},
                           ctx, 1, 0)["content"])
    assert r2["offset"] == r["next_offset"] and r["total_chars"] > 9000
    assert [s["function"]["name"] for s in t.schemas()] == ["run_python", "read_record"]


def _gen(d: Path, recs: list, gate=None):
    (d / "programs").mkdir(parents=True)
    (d / "feedback").mkdir()
    with open(d / "candidates.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(dict(r, program=f"programs/{r['cid']}.py")) + "\n")
            (d / "programs" / f"{r['cid']}.py").write_text(f"# {r['cid']}\n" + CODE)
            (d / "feedback" / f"{r['cid']}.json").write_text(json.dumps({"cid": r["cid"], "artifact": r.get("artifact")}))
    if gate is not None:
        (d / "gate.json").write_text(json.dumps({"submissions": gate}))


def test_search_history(tmp_path):
    """search_history (2026-11-21): earlier generations' candidates + this step's visible ones, the paper's row /
    key / generation functions, same_as, gate outcomes, sorting, paging; read_record of an earlier generation."""
    from rsicore.chat_tools import SearchHistory
    g1, g2 = tmp_path / "gen_01", tmp_path / "gen_02"
    _gen(g1, [{"cid": "g01_t0_s00", "score": 0.1, "artifact": {"h": "A"}},
              {"cid": "g01_t1_s00", "score": 0.3, "artifact": {"h": "B"}},
              {"cid": "g01_t2_s00", "score": None, "error": "program failed: boom\nmore", "artifact": None}],
         gate=[{"cid": "g01_t1_s00", "passed": False, "info": {"reason": "delta below min_delta"}}])
    _gen(g2, [{"cid": "g02_t0_s00", "score": 0.2, "artifact": {"h": "B"}},
              {"cid": "g02_t1_s00", "score": 0.5, "artifact": {"h": "C"}}])      # not visible yet (same step)
    h = SearchHistory({1: g1}, row_fn=lambda r: {"what": (r.get("artifact") or {}).get("h")},
                      key_fn=lambda r: (r.get("artifact") or {}).get("h"), gen_fn=lambda g, d: {"accepted": False})
    t = ChatTools(lambda tag, ctx: sandbox.ToolAPI({}), result_chars=6000, history=h)
    assert [s["function"]["name"] for s in t.schemas()] == ["run_python", "read_record", "search_history"]
    ctx = {"cid": "g02_t2_s01", "gen_dir": str(g2), "visible_cids": ["g02_t0_s00"]}
    r = json.loads(t.call("search_history", {}, ctx, 1, 0)["content"])
    assert r["ok"] and r["total"] == 4 and r["generations"] == [{"gen": 1, "candidates": 3, "accepted": False}]
    assert [x["cid"] for x in r["rows"]] == ["g02_t0_s00", "g01_t1_s00", "g01_t0_s00", "g01_t2_s00"]
    rows = {x["cid"]: x for x in r["rows"]}
    assert rows["g02_t0_s00"]["same_as"] == "g01_t1_s00" and rows["g02_t0_s00"]["gen"] == 2
    assert rows["g01_t1_s00"]["gate"] == {"passed": False, "reason": "delta below min_delta"}
    assert rows["g01_t2_s00"]["error"] == "program failed: boom" and "what" not in rows["g01_t2_s00"]
    assert rows["g01_t0_s00"]["what"] == "A"
    r = json.loads(t.call("search_history", {"gen": 1, "sort": "order"}, ctx, 1, 0)["content"])
    assert [x["cid"] for x in r["rows"]] == ["g01_t0_s00", "g01_t1_s00", "g01_t2_s00"]
    small = ChatTools(lambda tag, ctx: sandbox.ToolAPI({}), result_chars=400, history=h)
    seen, off = [], 0
    while off is not None:
        r = json.loads(small.call("search_history", {"offset": off}, ctx, 1, 0)["content"])
        seen += [x["cid"] for x in r["rows"]]
        off = r.get("next_offset")
    assert len(seen) == 4 and len(set(seen)) == 4
    r = json.loads(t.call("read_record", {"cid": "g01_t1_s00", "part": "program"}, ctx, 1, 0)["content"])
    assert r["ok"] and r["text"].startswith("# g01_t1_s00")
    r = t.call("read_record", {"cid": "g02_t1_s00", "part": "program"}, ctx, 1, 0)      # same step: not visible
    assert not r["ok"] and "generation 1" in r["content"]
    r = t.call("read_record", {"cid": "g01_t9_s00", "part": "program"}, ctx, 1, 0)      # no such candidate
    assert not r["ok"]
    bad = t.call("search_history", {"sort": "x"}, ctx, 1, 0)
    assert not bad["ok"]
    t0 = ChatTools(lambda tag, ctx: sandbox.ToolAPI({}), history=SearchHistory({}))     # generation 1: this step only
    r = json.loads(t0.call("search_history", {}, ctx, 1, 0)["content"])
    assert [x["cid"] for x in r["rows"]] == ["g02_t0_s00"] and "generations" not in r


# ----------------------------------------------------------------------------- transcript / replay
def _script_two_props():
    return [tool_turn("a"), tool_turn("b"), {"content": ANSWER},
            tool_turn("c"), tool_turn("d"), tool_turn("e"), {"content": ANSWER.replace("1", "2")}]


def test_replay_after_restart(tmp_path, fake):
    srv = fake(_script_two_props())
    tools = FakeTools()
    ed = mk(tmp_path, srv, tools)
    c0, c1 = ed.propose("p0", ctx={"cid": "a"}), ed.propose("p1", ctx={"cid": "b"})
    hist, stats = ed._prompt_history(), ed.last_stats
    srv2 = fake([])                                       # any request fails the test
    tools2 = FakeTools()
    ed2 = mk(tmp_path, srv2, tools2)
    assert ed2.propose("p0") == c0 and ed2.propose("p1") == c1
    assert ed2.n_calls == 0 and srv2.requests == [] and tools2.calls == []
    assert ed2._prompt_history() == hist and ed2.history == ed.history
    assert ed2.last_stats["rounds"] == stats["rounds"] == 3 and ed2.last_stats["replayed"]


def test_incomplete_proposal_is_rerun(tmp_path, fake):
    srv = fake(_script_two_props())
    ed = mk(tmp_path, srv, FakeTools())
    c0 = ed.propose("p0")
    ed.propose("p1")
    tr = tmp_path / "t.jsonl"
    ls = tr.read_text().splitlines()
    cut_at = max(i for i, l in enumerate(ls) if json.loads(l).get("r") == 2 and json.loads(l).get("p") == 1)
    tr.write_text("\n".join(ls[:cut_at + 1]) + "\n{\"torn")       # crash after round 2 of proposal 1
    srv2 = fake([tool_turn("z"), {"content": ANSWER.replace("1", "3")}])
    ed2 = mk(tmp_path, srv2, FakeTools())
    assert ed2.propose("p0") == c0 and ed2.n_calls == 0
    assert "3" in ed2.propose("p1") and ed2.n_calls == 2
    l2 = lines(tr)
    p1 = [l for l in l2 if l.get("p") == 1]
    assert sum(l["role"] == "user" for l in p1) == 1 and sum(bool(l.get("final")) for l in p1) == 1
    assert [l["r"] for l in p1 if l["role"] == "tool"] == [1]       # orphaned rounds are gone
    ed3 = mk(tmp_path, fake([]), FakeTools())
    assert ed3.propose("p0") == c0 and "3" in ed3.propose("p1") and ed3.n_calls == 0


def test_v1_transcript_loads_and_replays(tmp_path):
    tr = tmp_path / "v1.jsonl"
    v1 = [{"role": "system", "content": "SYS", "served_name": "m", "seed": 7},
          {"role": "user", "content": "first"}, {"role": "assistant", "content": ANSWER, "reasoning": "r", "code": CODE},
          {"role": "user", "content": "second"}, {"role": "assistant", "content": "```python\nB = 2\n```", "code": "B = 2\n"}]
    tr.write_text("".join(json.dumps(l) + "\n" for l in v1))

    def pol(msgs):
        return f"N = {sum(m['role'] == 'assistant' for m in msgs)}"
    ed = Editor("http://unused", "m", system_prompt="SYS", seed=7, transcript=tr, mock_policy=pol)
    assert ed.propose("first") == CODE and ed.propose("second") == "B = 2\n" and ed.n_calls == 0
    assert ed.last_stats["rounds"] == 0
    assert ed.propose("third") == "N = 2\n" and ed.n_calls == 1
    body = lines(tr)
    assert body[0]["format"] == 2 and all("p" in l for l in body[1:])
    ed2 = Editor("http://unused", "m", system_prompt="SYS", seed=7, transcript=tr, mock_policy=pol)
    assert [ed2.propose(m) for m in ("first", "second", "third")] == [CODE, "B = 2\n", "N = 2\n"] and ed2.n_calls == 0


def test_history_keep_collapses(tmp_path):
    seen = []

    def pol(msgs):
        seen.append([dict(m) for m in msgs])
        return f"STEP = {len(seen) - 1}"
    ed = Editor("http://unused", "m", system_prompt="SYS", seed=1, transcript=tmp_path / "h.jsonl",
                mock_policy=pol, history_keep=2)
    for k in range(6):
        ed.propose(f"msg {k}")
    last = seen[-1]
    users = [m["content"] for m in last if m["role"] == "user"]
    assert users == ["msg 0", editor.HISTORY_STUB, editor.HISTORY_STUB, "msg 3", "msg 4", "msg 5"]
    asst = [m["content"] for m in last if m["role"] == "assistant"]
    assert asst[1] == "```python\nSTEP = 1\n```"
    ed2 = Editor("http://unused", "m", system_prompt="SYS", seed=1, transcript=tmp_path / "h.jsonl",
                 mock_policy=pol, history_keep=2)
    for k in range(6):
        ed2.propose(f"msg {k}")
    assert ed2.n_calls == 0 and ed2._prompt_history() == ed._prompt_history()
    full = Editor("http://unused", "m", system_prompt="SYS", seed=1, transcript=tmp_path / "f.jsonl",
                  mock_policy=pol, history_keep=None)
    for k in range(6):
        full.propose(f"msg {k}")
    assert editor.HISTORY_STUB not in json.dumps(seen[-1])


# ----------------------------------------------------------------------------- context
def test_context_clamp_and_server_max_len(tmp_path, fake):
    srv = fake([{"content": ANSWER}], max_model_len=16000)
    ed = Editor(srv.endpoint, "m", system_prompt="S" * 30000, seed=1, transcript=tmp_path / "c.jsonl",
                tools=FakeTools(), context_tokens=131072, turn_max_tokens=8192)
    ed.propose("m")
    assert ed.context_tokens == 16000                      # from /v1/models
    mt = srv.requests[0]["max_tokens"]
    est = ed._estimate(srv.requests[0]["messages"])
    assert mt < 8192 and mt <= 16000 - est - 512 + 1


def _compact_script():
    return [tool_turn(f"c{k}", reasoning=f"R{k}", usage=False) for k in range(10)] + [{"content": ANSWER, "usage": False}]


def test_compaction_is_deterministic(tmp_path, fake):
    runs = []
    for k in range(2):
        srv = fake(_compact_script())
        ed = mk(tmp_path, srv, FakeTools(size=3000), name=f"cp{k}.jsonl", context_tokens=12000,
                keep_tool_rounds=2, keep_reasoning_rounds=1)
        assert ed.propose("m") == CODE
        runs.append([r["messages"] for r in srv.requests])
        assert ed.last_stats["compactions"] >= 1
        ev = [l for l in lines(tmp_path / f"cp{k}.jsonl") if l["role"] == "event" and l["kind"] == "compact"]
        assert ev and ev[0]["detail"]["tools_elided"] > 0
    assert runs[0] == runs[1]
    last = runs[0][-1]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("[elided: run_python output r1")
    assert not tool_msgs[-1]["content"].startswith("[elided")
    asst = [m for m in last if m["role"] == "assistant"]
    assert not asst[0]["content"].startswith("<think>") and asst[-1]["content"].startswith("<think>\nR9")


def test_http400_compaction_and_forced_final(tmp_path, fake):
    ctx400 = ("http", 400, "This model's maximum context length is 1000 tokens. However, you requested 2000")
    srv = fake([tool_turn("a"), ctx400, {"content": ANSWER}])
    ed = mk(tmp_path, srv, FakeTools())
    assert ed.propose("m") == CODE and ed.last_stats["stop"] == "answer"
    ev = [l["kind"] for l in lines(tmp_path / "t.jsonl") if l["role"] == "event"]
    assert ev[:1] == ["http400"] and "compact" in ev
    assert srv.requests[2]["tool_choice"] == "auto"         # retried once, still a tool turn
    srv = fake([tool_turn("a"), ctx400, ctx400, {"content": ANSWER}])
    ed = mk(tmp_path, srv, FakeTools(), name="b.jsonl")
    assert ed.propose("m") == CODE and ed.last_stats["stop"] == "forced:http400"
    assert srv.requests[-1]["tool_choice"] == "none"
    srv = fake([tool_turn("a")], default=ctx400)
    ed = mk(tmp_path, srv, FakeTools(), name="c.jsonl")
    with pytest.raises(EditorReplyError):
        ed.propose("m")


def test_http_5xx_is_fatal_and_other_400_is_fatal(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(editor.time, "sleep", lambda s: None)
    srv = fake(default=("http", 500, "boom"))
    ed = mk(tmp_path, srv, FakeTools())
    with pytest.raises(RuntimeError) as e:
        ed.propose("m")
    assert not isinstance(e.value, EditorReplyError) and len(srv.requests) == 5
    srv = fake(default=("http", 404, "model not found"))
    ed = mk(tmp_path, srv, FakeTools(), name="n.jsonl")
    with pytest.raises(RuntimeError) as e:
        ed.propose("m")
    assert not isinstance(e.value, EditorReplyError)


def test_search_records_editor_reply_error(tmp_path, fake, monkeypatch):
    from types import SimpleNamespace
    ctx400 = ("http", 400, "maximum context length exceeded")
    srv = fake(default=lambda p: ctx400 if "thread 0" in p["messages"][1]["content"] else {"content": ANSWER})
    rc = SimpleNamespace(run_dir=tmp_path / "run")
    monkeypatch.setattr(search, "gen_dir", lambda rc, g: Path(rc.run_dir) / f"gen_{g:02d}")
    (tmp_path / "run" / "gen_01").mkdir(parents=True)
    seen_ctx = []

    class Ed(Editor):
        def propose(self, user_msg, ctx=None):
            seen_ctx.append(ctx)
            return super().propose(user_msg, ctx=ctx)

    def factory(th):
        return Ed(srv.endpoint, "m", system_prompt="S", seed=th, tools=FakeTools(), context_tokens=None,
                  transcript=tmp_path / "run" / "gen_01" / "editor" / f"thread_{th}.jsonl")
    out = search.inner_search(rc, 1, editor_factory=factory, render_context=lambda h, th: f"thread {th}",
                              build_fn=lambda prog, cid: ({"q": 0.5}, None), screen_fn=lambda a, cid: (0.5, {}),
                              full_fn=lambda a, cid: (0.5, {}), threads=2, steps=2, max_candidates=4, top_full=0)
    by = {c.cid: c for c in out}
    assert by["g01_t0_s00"].error.startswith("editor failed") and by["g01_t0_s00"].program == ""
    assert by["g01_t1_s00"].error is None and by["g01_t1_s00"].cost["editor_stop"] == "answer"
    assert by["g01_t1_s01"].cost["editor_rounds"] == 0 and "editor_completion_tokens" in by["g01_t1_s01"].cost
    assert seen_ctx[0]["cid"] == "g01_t0_s00" and seen_ctx[-1]["visible_cids"] == ["g01_t0_s00", "g01_t1_s00", "g01_t0_s01"]


# ----------------------------------------------------------------------------- mock path
def test_mock_dict_replies_and_exploration(tmp_path):
    def inner(msgs):
        last = [m for m in msgs if m["role"] == "user"][-1]["content"]
        return f"def main(api):\n    return {len(msgs)}  # SEEN {last}\n"
    log = []
    tools = ChatTools(_api_factory(log), timeout_s=30, mem_gb=2)
    pol = with_mock_exploration(inner, 1, "result = api.double(x=4)")
    ed = Editor("http://unused", "m", system_prompt="S", seed=1, transcript=tmp_path / "m.jsonl", tools=tools,
                mock_policy=pol, history_keep=2)
    plain = Editor("http://unused", "m", system_prompt="S", seed=1, transcript=tmp_path / "p.jsonl", tools=None,
                   mock_policy=inner, history_keep=2)
    gd = tmp_path / "gen"
    for k in range(3):
        c = ed.propose(f"msg {k}", ctx={"cid": f"c{k}", "gen_dir": str(gd), "visible_cids": []})
        assert c == plain.propose(f"msg {k}")
        assert ed.last_stats["rounds"] == 1 and ed.last_stats["stop"] == "answer"
    tl = [l for l in lines(tmp_path / "m.jsonl") if l["role"] == "tool"]
    assert len(tl) == 3 and json.loads(tl[0]["content"])["result"] == 8
    assert [l["tool_call_id"] for l in tl] == ["call_0_1_0", "call_1_1_0", "call_2_1_0"]
    # forced final applies to the mock: tool_rounds=1 while the policy wants 3 rounds
    pol3 = with_mock_exploration(inner, 3, "result = 1")
    ed3 = Editor("http://unused", "m", system_prompt="S", seed=1, transcript=tmp_path / "m3.jsonl", tools=tools,
                 mock_policy=pol3, tool_rounds=1)
    assert "# SEEN" in ed3.propose("x", ctx={"cid": "z", "gen_dir": str(gd)})
    assert ed3.last_stats["stop"] == "forced:rounds"


# ----------------------------------------------------------------------------- config / servers
def test_editor_kwargs_and_context_helpers():
    kw = editor.editor_kwargs({})
    assert kw["tool_rounds"] == 50 and kw["context_tokens"] == 131072 and kw["max_tokens"] == 16384
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20, "min_p": 0.0}
    assert kw["history_keep"] == 2 and kw["reasoning_passback"] == "think_tag" and kw["turn_max_tokens"] == 8192
    assert editor.editor_kwargs({"editor": {"tool_rounds": 0, "history_keep": 2}})["history_keep"] is None
    kw = editor.editor_kwargs({"editor": {"context_tokens": None, "extra_body": {}}}, server_args=["--max-model-len", "32768"])
    assert kw["context_tokens"] == 32768 and kw["extra_body"] == {}
    assert editor.editor_kwargs({"editor": {"context_tokens": None}})["context_tokens"] is None
    s = editor.editor_settings({"editor": {"explore": {"gpu_tools": True}}})
    assert s["explore"] == {"timeout_s": 300, "mem_gb": 8, "gpu_tools": True, "gpu_seconds": 300, "persistent": False}
    assert chat_tools.chat_tools_kwargs({}) == {"timeout_s": 300, "mem_gb": 8, "result_chars": 6000,
                                                "persistent": False}
    args = ["--gpu-memory-utilization", "0.9", "--max-model-len", "32768", "--max-logprobs", "20"]
    assert editor.apply_context_tokens(args, {"editor": {"context_tokens": 49152}}) == \
        ["--gpu-memory-utilization", "0.9", "--max-model-len", "49152", "--max-logprobs", "20"]
    assert editor.apply_context_tokens(args, {"editor": {"context_tokens": None}}) == args
    assert servers.set_max_model_len(["--x"], 131072) == ["--x", "--max-model-len", "131072"]
    assert servers.set_max_model_len(["--max-model-len=8", "--y", "--max-model-len", "9"], 10) == \
        ["--max-model-len", "10", "--y"]
    assert servers.max_model_len([]) == 40960 and servers.max_model_len(["--max-model-len", "128k"]) == 131072


def test_tool_call_parser_auto_and_len_clamp(tmp_path):
    xml = tmp_path / "xml"
    xml.mkdir()
    (xml / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ '<tool_call>\\n<function=' }}"}))
    (xml / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 32768}}))
    herm = tmp_path / "hermes"
    herm.mkdir()
    (herm / "chat_template.jinja").write_text('<tool_call>\n{"name": "{{ name }}"}</tool_call>')
    (herm / "config.json").write_text(json.dumps({"max_position_embeddings": 262144}))
    assert servers.detect_tool_call_parser(xml) == "qwen3_coder" and servers.detect_tool_call_parser(herm) == "hermes"
    assert servers.model_max_positions(xml) == 32768 and servers.model_max_positions(tmp_path / "none") is None
    s = " ".join(servers.VLLMServer(xml, "e", [0], port=1, extra_args=["--max-model-len", "131072"]).command())
    assert "--tool-call-parser qwen3_coder" in s and "--max-model-len 32768" in s and "131072" not in s
    s = " ".join(servers.VLLMServer(herm, "e", [0], port=1, extra_args=["--max-model-len", "131072"]).command())
    assert "--tool-call-parser hermes" in s and "--max-model-len 131072" in s
    s = " ".join(servers.VLLMServer(xml, "e", [0], port=1, extra_args=["--tool-call-parser", "pythonic"]).command())
    assert s.count("--tool-call-parser") == 1 and "pythonic" in s and "--max-model-len 32768" in s
    m = servers.VLLMServer(xml, "e", [0], port=1, mock=True, extra_args=["--tasks", "a", "--max-model-len", "9"])
    assert m.command()[-2:] == ["--tasks", "a"] and "--tool-call-parser" not in m.command()


# ----------------------------------------------------------------------------- review fixes (2026-11-21)
def _timeout_post(monkeypatch, plan):
    """Wrap _net.post_json: each request pops plan[0]; True -> raise TimeoutError (never reaches the server).
    Returns the list of (tool_choice, timeout) of every request."""
    monkeypatch.setattr(editor.time, "sleep", lambda s: None)
    real, seen = editor._net.post_json, []

    def post(url, obj, timeout=600, headers=None):
        seen.append((obj.get("tool_choice"), timeout))
        if plan and plan.pop(0):
            raise TimeoutError("timed out")
        return real(url, obj, timeout=timeout)
    monkeypatch.setattr(editor._net, "post_json", post)
    return seen


def test_reply_timeout_forces_final_turn_without_retry(tmp_path, fake, monkeypatch):
    draft = tool_turn(content="draft\n```python\nDRAFT = 1\n```")
    # a tool-round turn times out -> no retry, the final turn is forced and gets the full timeout_s
    seen = _timeout_post(monkeypatch, [False, True, False])
    srv = fake([draft, {"content": ANSWER}])
    ed = mk(tmp_path, srv, FakeTools(), timeout_s=3600, proposal_wall_s=2400)
    assert ed.propose("m") == CODE and ed.last_stats["stop"] == "forced:timeout"
    assert [c for c, _ in seen] == ["auto", "auto", "none"] and len(srv.requests) == 2
    assert 2900 < seen[1][1] <= 3000 and seen[2][1] == 3600
    ev = [l for l in lines(tmp_path / "t.jsonl") if l["role"] == "event"]
    assert [e["kind"] for e in ev][:2] == ["timeout", "forced_final"]
    # wall-forced final turn: full timeout_s (not wall_left + 600); its timeout -> the fallback block
    t = {"now": 0.0}

    def clock():
        t["now"] += 1000.0
        return t["now"]
    seen = _timeout_post(monkeypatch, [False, True])
    srv = fake([draft])
    ed = mk(tmp_path, srv, FakeTools(), name="w.jsonl", timeout_s=3600, proposal_wall_s=2400, clock=clock)
    assert ed.propose("m") == "DRAFT = 1\n" and ed.last_stats["stop"] == "fallback"
    assert seen[-1] == ("none", 3600) and "timed out" in ed.last_stats["error"]
    # final turn times out without a fallback -> EditorReplyError (not a fatal RuntimeError), recorded as failed
    seen = _timeout_post(monkeypatch, [False, True])
    srv = fake([tool_turn(), {"content": f"```python\n{NEXT}```"}])
    ed = mk(tmp_path, srv, FakeTools(), name="f.jsonl", tool_rounds=1)
    with pytest.raises(EditorReplyError, match="timed out"):
        ed.propose("m0", ctx={"cid": "s0"})
    assert len(seen) == 2 and ed.propose("m1", ctx={"cid": "s1"}) == NEXT
    tr = lines(tmp_path / "f.jsonl")
    assert [(l["p"], bool(l.get("failed"))) for l in tr if l.get("final")] == [(0, True), (1, False)]
    assert [m["content"] for m in ed._prompt_history() if m["role"] == "user"] == ["m1"]
    ed2 = mk(tmp_path, fake([]), FakeTools(), name="f.jsonl", tool_rounds=1)     # restart: replays both
    with pytest.raises(EditorReplyError, match="timed out"):
        ed2.propose("m0", ctx={"cid": "s0"})
    assert ed2.propose("m1", ctx={"cid": "s1"}) == NEXT and ed2.n_calls == 0


def test_final_turn_context_error_uses_fallback(tmp_path, fake):
    ctx400 = ("http", 400, "This model's maximum context length is 1000 tokens. However, you requested 2000")
    srv = fake([tool_turn(content="draft\n```python\nDRAFT = 1\n```")], default=ctx400)
    ed = mk(tmp_path, srv, FakeTools())
    assert ed.propose("m") == "DRAFT = 1\n"
    st = ed.last_stats
    assert st["stop"] == "fallback" and st["forced_by"] == "http400" and "rejected" in st["error"]
    assert srv.requests[-1]["tool_choice"] == "none"


def test_forced_final_turn_gets_one_nudge(tmp_path, fake):
    finals = [{"content": "<tool_call>{\"name\": \"run_python\"}</tool_call>"}, {"content": ANSWER}]
    srv = fake(default=lambda p: tool_turn() if p["tool_choice"] == "auto" else finals.pop(0))
    ed = mk(tmp_path, srv, FakeTools(), tool_rounds=2)
    assert ed.propose("m") == CODE
    assert ed.last_stats["stop"] == "forced:rounds" and ed.last_stats["nudges"] == 1
    assert [r["tool_choice"] for r in srv.requests] == ["auto", "auto", "none", "none"]
    assert srv.requests[-1]["messages"][-1]["content"] == editor.NUDGE["tool_text_final"]
    # cut at length twice on the final turn: one nudge only, then stop "length"
    srv = fake(default=lambda p: tool_turn() if p["tool_choice"] == "auto" else {"content": "", "finish": "length"})
    ed = mk(tmp_path, srv, FakeTools(), name="l.jsonl", tool_rounds=1)
    assert ed.propose("m") == "" and ed.last_stats["stop"] == "length"
    assert [r["tool_choice"] for r in srv.requests] == ["auto", "none", "none"]
    assert srv.requests[-1]["messages"][-1]["content"] == editor.NUDGE["length"]


def test_tool_rounds_zero_keeps_full_history(tmp_path):
    seen = []

    def pol(msgs):
        seen.append(msgs)
        return f"STEP = {len(seen)}"
    kw = editor.editor_kwargs({"editor": {"tool_rounds": 0, "history_keep": 2}})
    ed = Editor("http://unused", "m", system_prompt="S", seed=1, transcript=tmp_path / "z.jsonl",
                mock_policy=pol, tools=FakeTools(), **kw)
    for k in range(5):
        ed.propose(f"msg {k}")
    assert [m["content"] for m in seen[-1] if m["role"] == "user"] == [f"msg {k}" for k in range(5)]
    assert editor.HISTORY_STUB not in json.dumps(seen[-1]) and seen[-1] == ed.history[:-1]


def test_search_resume_keeps_editor_history(tmp_path, monkeypatch):
    from types import SimpleNamespace
    rc = SimpleNamespace(run_dir=tmp_path / "run")
    monkeypatch.setattr(search, "gen_dir", lambda rc, g: Path(rc.run_dir) / f"gen_{g:02d}")
    (tmp_path / "run" / "gen_01").mkdir(parents=True)
    tr = tmp_path / "run" / "gen_01" / "editor" / "thread_0.jsonl"

    class Crash(Exception):
        pass
    seen, crash, made = [], {"screen": "g01_t0_s02", "propose": None}, []

    def pol(msgs):
        users = [m["content"] for m in msgs if m["role"] == "user"]
        if crash["propose"] and users[-1] == crash["propose"]:
            raise Crash("editor died")
        seen.append(users)
        return f"STEP = {len(users) - 1}"

    def factory(th):
        ed = Editor("http://unused", "m", system_prompt="S", seed=th, mock_policy=pol, history_keep=None, transcript=tr)
        made.append(ed)
        return ed

    def screen(a, cid):
        if cid == crash["screen"]:
            raise Crash("screen died")
        return 0.5, {}

    def run(steps):
        return search.inner_search(rc, 1, editor_factory=factory, build_fn=lambda prog, cid: ({"q": 0.5}, None),
                                   render_context=lambda h, th: f"step {sum(c.thread == th for c in h)}",
                                   screen_fn=screen, full_fn=lambda a, cid: (0.5, {}), threads=1, steps=steps,
                                   max_candidates=10, top_full=0)
    with pytest.raises(Crash):           # proposal 2 is in the transcript, its candidate is not recorded
        run(4)
    crash.update(screen=None, propose="step 3")
    with pytest.raises(Crash):           # resume: 0, 1 skipped, 2 replayed; proposal 3 dies inside the editor
        run(4)
    assert made[-1].n_calls == 0
    crash["propose"] = None
    out = run(5)                          # resume again: 3 runs live with the full history
    assert made[-1].n_calls == 2 and len(out) == 5
    assert seen[-2] == ["step 0", "step 1", "step 2", "step 3"] and seen[-1][-1] == "step 4"
    finals = [l for l in lines(tr) if l.get("final")]
    assert [l["p"] for l in finals] == [0, 1, 2, 3, 4] and not any(l.get("failed") for l in finals)
    users = [l for l in lines(tr) if l.get("role") == "user"]
    assert [l["cid"] for l in users] == [f"g01_t0_s0{k}" for k in range(5)]
    assert [c.program for c in out] == [f"STEP = {k}\n" for k in range(5)]


def test_explore_dir_of_discarded_attempt_is_cleared(tmp_path):
    gd = tmp_path / "gen"
    stale = gd / "explore" / "c0" / "r05_0"
    stale.mkdir(parents=True)
    (stale / "old.txt").write_text("x")
    keep = gd / "explore" / "other" / "r01_0"
    keep.mkdir(parents=True)
    log = []
    tools = ChatTools(_api_factory(log), timeout_s=30, mem_gb=2)
    ed = Editor("http://unused", "m", system_prompt="S", seed=1, transcript=tmp_path / "e.jsonl", tools=tools,
                mock_policy=with_mock_exploration(lambda m: "X = 1", 1, "result = 1"))
    ed.propose("m", ctx={"cid": "c0", "gen_dir": str(gd)})
    assert not stale.exists() and (gd / "explore" / "c0" / "r01_0" / "program.py").exists() and keep.exists()
    tools.begin({"cid": "../x", "gen_dir": str(gd)})          # never leaves explore/
    assert keep.exists()


def test_sigxcpu_exit_reads_as_timeout(tmp_path, monkeypatch):
    import signal
    if not hasattr(signal, "SIGXCPU"):
        pytest.skip("no SIGXCPU")
    res = sandbox.ProgramResult(False, None, "", "", f"child exited with code {-signal.SIGXCPU} before returning "
                                "(CPU-time or memory limit?)", 8.1, {"returncode": -signal.SIGXCPU})
    monkeypatch.setattr(chat_tools, "run_program", lambda *a, **k: res)
    t = ChatTools(lambda tag, ctx: sandbox.ToolAPI({}), timeout_s=3)
    out = t.call("run_python", {"code": "while True: pass"}, {"cid": "c", "gen_dir": str(tmp_path)}, 1, 0)
    assert not out["ok"] and json.loads(out["content"])["error"].startswith("timeout: CPU-time limit")
    res.info["returncode"] = 1
    out = t.call("run_python", {"code": "x"}, {"cid": "c", "gen_dir": str(tmp_path)}, 1, 0)
    assert "timeout" not in out["content"]


def test_lora_editor_context_from_parent_card(tmp_path, fake):
    srv = fake([{"content": ANSWER}], models=[{"id": "base", "max_model_len": 16000},
                                              {"id": "m", "root": "/ad", "parent": "base", "max_model_len": None}])
    ed = Editor(srv.endpoint, "m", system_prompt="S", seed=1, transcript=tmp_path / "c.jsonl",
                tools=FakeTools(), context_tokens=131072)
    ed.propose("m")
    assert ed.context_tokens == 16000


# ----------------------------------------------------------------------------- program check (2026-11-22)
def test_check_program():
    ok = editor.check_program
    assert ok(CODE) is None
    assert ok("def main(api):\n    '''doc'''\n    h = api.f(x=1)\n    return {'lora': h}\n") is None
    assert ok("def main(api, n=3):\n    return n\n") is None
    assert ok("import math\ndef build(api, m):\n    return math.pi\ndef main(api):\n"
              "    try:\n        return build(api, 1)\n    except (ToolError, BudgetExceeded) as e:\n        return str(e)\n") is None
    assert ok("def main(api):\n    f = lambda k: k + 1\n    return [f(i) for i in range(3)]\n") is None
    assert ok("from x import *\ndef main(api):\n    return y\n") is None          # star import: names unknown
    assert ok("def main(api)\n    return 1\n").startswith("SyntaxError on line 1")
    assert ok("x = 1\n") == "there is no top-level def main(api)"
    assert "exactly one parameter" in ok("def main(api, ws):\n    return 1\n")
    assert "exactly one parameter" in ok("def main():\n    return 1\n")
    assert "skeleton" in ok("def main(api):\n    # build the LoRA here\n    pass\n")
    assert "skeleton" in ok("def main(api):\n    '''Plan: scale layer 12.'''\n    ...\n")
    assert "skeleton" in ok("def main(api):\n    raise NotImplementedError\n")
    assert "placeholders (line 3)" in ok("def main(api):\n    for m in api.modules():\n        ...\n    return {'lora': None}\n")
    assert ok("def main(api):\n    api.f(x=1)\n") == "main never returns a value"
    assert ok("def main(api):\n    def g():\n        return 1\n    g()\n") == "main never returns a value"
    assert "never defined in the program: best_lora" in ok("def main(api):\n    return {'lora': best_lora}\n")
    assert "read `api` as a global" in ok("def b(m):\n    return api.svd(module=m)\ndef main(api):\n    return b(1)\n")
    # names=False: only the structural checks
    assert ok("def main(api):\n    return {'lora': best_lora}\n", names=False) is None


SKELETON = "Plan.\n```python\ndef main(api):\n    # TODO\n    pass\n```"
PROG = "def main(api):\n    h = api.new(k=1)\n    return {'lora': h}\n"


def test_final_program_gets_one_repair_turn(tmp_path, fake):
    srv = fake([{"content": SKELETON}, {"content": ANSWER}])
    ed = mk(tmp_path, srv, FakeTools())
    assert ed.propose("m") == CODE
    st = ed.last_stats
    assert st["stop"] == "answer" and st["repairs"] == 1 and st["check"] is None and st["model_calls"] == 2
    last = srv.requests[-1]["messages"]
    assert last[-1]["role"] == "user" and last[-1]["content"].startswith("Your program cannot run as written: main is only")
    ev = [l for l in lines(tmp_path / "t.jsonl") if l["role"] == "event"]
    assert [e["kind"] for e in ev] == ["check"] and "skeleton" in ev[0]["detail"]["problem"]
    # still broken after the one repair turn and nothing better: submitted as is, the problem recorded
    srv = fake([{"content": SKELETON}, {"content": SKELETON}])
    ed = mk(tmp_path, srv, FakeTools(), name="b.jsonl")
    assert "pass" in ed.propose("m") and ed.last_stats["stop"] == "answer" and "skeleton" in ed.last_stats["check"]
    assert len(srv.requests) == 2


def test_fallback_takes_checked_program_from_run_python(tmp_path, fake):
    # final reply still a skeleton after the repair turn -> the complete program of a run_python call wins
    srv = fake([tool_turn(code=PROG, content="```python\nDRAFT = 1\n```"), {"content": SKELETON}, {"content": SKELETON}])
    ed = mk(tmp_path, srv, FakeTools())
    assert ed.propose("m") == PROG and ed.last_stats["stop"] == "fallback" and ed.last_stats["check"] is None
    # forced final turn cut at max_tokens twice (no code at all) -> the run_python program, not the older block
    srv = fake([tool_turn(content="```python\nDRAFT = 1\n```"), tool_turn(code=PROG),
                {"content": "", "finish": "length"}, {"content": "", "finish": "length"}])
    ed = mk(tmp_path, srv, FakeTools(), name="f.jsonl", tool_rounds=2)
    assert ed.propose("m") == PROG and ed.last_stats["stop"] == "fallback"
    # a run_python program that leans on the session's globals is no fallback: the last block is used
    srv = fake([tool_turn(content="```python\nDRAFT = 1\n```"),
                tool_turn(code="def main(api):\n    return {'lora': built}\n"),
                {"content": "", "finish": "length"}, {"content": "", "finish": "length"}])
    ed = mk(tmp_path, srv, FakeTools(), name="g.jsonl", tool_rounds=2)
    assert ed.propose("m") == "DRAFT = 1\n" and ed.last_stats["stop"] == "fallback"
    assert ed.last_stats["check"] == "there is no top-level def main(api)"
