"""Chat tools of the editor's tool loop (2026-11-21): what the editor may call WHILE it writes a candidate.

    tools = ChatTools(explore_api_factory, timeout_s=300, mem_gb=8, result_chars=6000)
    ed = Editor(endpoint, name, system_prompt=SYS, seed=s, transcript=tr, tools=tools, **editor_kwargs(cfg))
    code = ed.propose(msg, ctx={"cid": cid, "gen_dir": gen_dir, "visible_cids": [...]})

Two functions (OpenAI tool schemas from schemas()):
  run_python(code)            runs `code` in rsicore.sandbox.run_program (new process, no network) with the
                              paper's EXPLORE ToolAPI as `api` (no evaluate). Top-level code is fine; set
                              `result = ...` to return a value, or define main(api). Nothing persists between
                              calls. Workdir gen_dir/explore/<cid>/r{r:02d}_{i}/ (program.py, stdout.txt,
                              stderr.txt, result.json: the full output, for auditing).
  read_record(cid, part, offset=0)
                              the full program ("program": gen_dir/programs/<cid>.py) or L1 feedback
                              ("feedback": gen_dir/feedback/<cid>.json) of a candidate in ctx["visible_cids"]
                              (the history snapshot the proposal was rendered from), in pages of the result size.

explore_api_factory(tag: str, ctx: dict) is called once per run_python call, in the proposing thread, with
tag = f"{cid}~x{r:02d}_{i}" (r = round, i = index of the call in that turn). It returns one of
  * a ToolAPI,
  * (ToolAPI, cleanup) where cleanup() is called after the run (e.g. Paper B: begin_program(tag) inside the
    factory, end_program in cleanup, so exploration handles are freed and the tool log lives under "~x" keys),
  * a context manager whose __enter__ returns the ToolAPI.
The factory must not register evaluate (or anything that spends a generation budget).

call(name, arguments, ctx, r, i, deadline_s=None) never raises: every failure (bad JSON arguments, unknown
tool, wrong parameters, factory / sandbox errors, timeout) becomes '{"ok": false, "error": "<Type>: msg"}'.
It returns {"content": str (JSON, about result_chars at most), "ok": bool, "seconds": float, "args": dict}.
A child killed by the sandbox CPU-time limit (SIGXCPU; RLIMIT_CPU = timeout + 5 s) is reported as a timeout,
like the wall-clock kill. begin(ctx) (called by the editor when a proposal starts live) removes
gen_dir/explore/<cid>/ left by a discarded earlier attempt of the same proposal.

with_mock_exploration(policy, rounds, code) wraps a mock editor policy: it answers with one run_python call
per round until `rounds` tool messages follow the last user message, then calls `policy` on the messages
WITHOUT the in-loop turns (so the program equals the one of a run without tools).
"""
import json
import shutil
import signal
import tempfile
import time
from pathlib import Path

from .sandbox import ToolAPI, run_program

RUN_PYTHON_DOC = (
    "Run exploratory Python in the program sandbox (new process, no network, no files outside its own "
    "workdir). The same `api` object as in your final program is available as a global (call tools by "
    "keyword: api.<tool>(...)); api.evaluate and other budget-spending tools are NOT available here. Write "
    "top-level code: print() what you need, or set `result = ...` to get a JSON value back (defining "
    "main(api) also works). Filter big tables inside the code instead of printing them whole. Output is "
    "cut to a few thousand characters. Nothing persists between calls or into your final program.")
READ_RECORD_DOC = (
    "Read the full program or the full feedback of an earlier candidate listed in the latest message "
    "(the message shows them shortened). Long texts come in pages: call again with offset=next_offset.")

FORCED_MARK = "[No more function calls"     # start of the editor's forced-final notice (see editor.py)

_WRAP = '''def main(api):
    g = {{"api": api, "__name__": "__explore__", "BudgetExceeded": api.BudgetExceeded, "ToolError": api.ToolError}}
    exec(compile({src}, "<explore>", "exec"), g)
    if "result" in g:
        r = g["result"]
    elif callable(g.get("main")):
        r = g["main"](api)
    else:
        r = None
    import json as _json
    try:
        _json.dumps(r, allow_nan=False)
        return r
    except (TypeError, ValueError):
        return {{"__repr__": repr(r)[:20000]}}
'''


def cut(text: str, n: int, head: float = 0.25) -> str:
    """text if len <= n, else its first ~head*n and last chars with a "[... N chars cut ...]" marker."""
    text = "" if text is None else str(text)
    n = int(n)
    if n <= 0 or len(text) <= n:
        return text
    marker = f"\n[... {len(text) - n:,} chars cut ...]\n"
    h = max(0, int(n * head))
    t = max(0, n - h)
    return text[:h] + marker + (text[-t:] if t else "")


def error_json(e) -> str:
    msg = e if isinstance(e, str) else f"{type(e).__name__}: {e}"
    return json.dumps({"ok": False, "error": str(msg)[:2000]}, ensure_ascii=False)


def _schema(name, doc, props, required):
    return {"type": "function", "function": {"name": name, "description": doc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


class ChatTools:
    def __init__(self, explore_api_factory, *, timeout_s: float = 300, mem_gb: float = 8, result_chars: int = 6000,
                 run_python_doc: str | None = None, read_record: bool = True):
        self.factory = explore_api_factory
        self.timeout_s, self.mem_gb, self.result_chars = float(timeout_s), mem_gb, int(result_chars)
        self.run_python_doc = run_python_doc or RUN_PYTHON_DOC
        self.read_record_enabled = bool(read_record)

    # ---------------------------------------------------------------- schemas
    def names(self) -> list[str]:
        return ["run_python"] + (["read_record"] if self.read_record_enabled else [])

    def schemas(self) -> list[dict]:
        out = [_schema("run_python", self.run_python_doc,
                       {"code": {"type": "string", "description": "Python source; uses the global `api`."}}, ["code"])]
        if self.read_record_enabled:
            out.append(_schema("read_record", READ_RECORD_DOC, {
                "cid": {"type": "string", "description": "candidate id, e.g. g01_t2_s03"},
                "part": {"type": "string", "enum": ["program", "feedback"]},
                "offset": {"type": "integer", "description": "character offset of the page (default 0)"}},
                ["cid", "part"]))
        return out

    # ---------------------------------------------------------------- dispatch
    def begin(self, ctx: dict | None):
        """A proposal starts live: remove gen_dir/explore/<cid>/ of a discarded earlier attempt."""
        ctx = ctx or {}
        cid, gd = str(ctx.get("cid") or ""), ctx.get("gen_dir")
        if not gd or not cid or "/" in cid or "\\" in cid or cid in (".", ".."):
            return
        d = Path(gd) / "explore" / cid
        if d.is_dir() and not d.is_symlink():
            shutil.rmtree(d, ignore_errors=True)

    def call(self, name: str, arguments, ctx: dict | None, r: int, i: int, deadline_s: float | None = None) -> dict:
        t0 = time.time()
        ctx = dict(ctx or {})
        args = {}
        try:
            if isinstance(arguments, dict):
                args = dict(arguments)
            else:
                args = json.loads(arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
        except (TypeError, ValueError) as e:
            return {"content": error_json(f"bad arguments ({type(e).__name__}: {e})"), "ok": False,
                    "seconds": round(time.time() - t0, 3), "args": {"_raw": str(arguments)[:2000]}}
        try:
            if name == "run_python":
                content, ok = self.run_python(ctx, r, i, deadline_s=deadline_s, **args)
            elif name == "read_record" and self.read_record_enabled:
                content, ok = self.read_record(ctx, **args)
            else:
                content, ok = error_json(f"unknown tool {name!r}; available: {self.names()}"), False
        except Exception as e:  # noqa: BLE001 - a failing tool must not stop the proposal
            content, ok = error_json(e), False
        return {"content": content, "ok": ok, "seconds": round(time.time() - t0, 3), "args": args}

    # ---------------------------------------------------------------- run_python
    def _enter(self, tag: str, ctx: dict):
        made = self.factory(tag, ctx)
        if isinstance(made, ToolAPI):
            return made, None
        if isinstance(made, tuple) and len(made) == 2:
            return made[0], made[1]
        if hasattr(made, "__enter__") and hasattr(made, "__exit__"):
            api = made.__enter__()
            return api, lambda: made.__exit__(None, None, None)
        raise TypeError(f"explore_api_factory returned {type(made).__name__}; expected ToolAPI, (api, cleanup) "
                        "or a context manager")

    def run_python(self, ctx: dict, r: int, i: int, *, code: str, deadline_s: float | None = None) -> tuple[str, bool]:
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        cid = str(ctx.get("cid") or "explore")
        tag = f"{cid}~x{int(r):02d}_{int(i)}"
        gd = ctx.get("gen_dir")
        base = Path(gd) if gd else Path(tempfile.mkdtemp(prefix="rsi_explore_"))
        wd = base / "explore" / cid / f"r{int(r):02d}_{int(i)}"
        timeout = self.timeout_s if deadline_s is None else min(self.timeout_s, max(5.0, float(deadline_s)))
        api, cleanup = self._enter(tag, ctx)
        try:
            before = dict(getattr(api, "calls", {}) or {})
            res = run_program(_WRAP.format(src=repr(code)), api, timeout_s=int(round(timeout)), mem_gb=self.mem_gb,
                              workdir=wd, save_output=True)
            after = dict(getattr(api, "calls", {}) or {})
        finally:
            if cleanup is not None:
                cleanup()
        try:
            wd.mkdir(parents=True, exist_ok=True)
            (wd / "result.json").write_text(json.dumps({"ok": res.ok, "result": res.result, "error": res.error,
                                                        "seconds": res.seconds, "info": res.info}, default=str))
        except OSError:
            pass
        calls = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) - before.get(k, 0)}
        out = {"ok": bool(res.ok), "seconds": round(res.seconds, 2), "tool_calls": calls}
        error = res.error
        xcpu = getattr(signal, "SIGXCPU", None)
        if not res.ok and xcpu is not None and (res.info or {}).get("returncode") == -xcpu:
            error = f"timeout: CPU-time limit reached ({error})"   # the CPU limit can fire before the wall clock
        if error:
            out["error"] = cut(error, 2000, head=0.3)
        if not res.ok and res.stderr:
            out["stderr"] = cut(res.stderr, 1500, head=0.0)
        room = max(400, self.result_chars - len(json.dumps(out, ensure_ascii=False)) - 80)
        rtxt = None if res.result is None else json.dumps(res.result, ensure_ascii=False)
        stdout = res.stdout or ""
        if rtxt is not None:
            r_room = room - min(len(stdout), room // 2)
            out["result"] = res.result if len(rtxt) <= r_room else cut(rtxt, r_room)
            room -= min(len(rtxt), r_room)
        if stdout:
            out["stdout"] = cut(stdout, max(200, room))
        return json.dumps(out, ensure_ascii=False), bool(res.ok)

    # ---------------------------------------------------------------- read_record
    def read_record(self, ctx: dict, *, cid: str, part: str = "feedback", offset: int = 0) -> tuple[str, bool]:
        visible = [str(c) for c in (ctx.get("visible_cids") or [])]
        if str(cid) not in visible:
            shown = ", ".join(visible[-12:]) or "(none yet)"
            return error_json(f"cid {cid!r} is not visible in this step; visible: {shown}"), False
        if part not in ("program", "feedback"):
            return error_json(f"part must be 'program' or 'feedback', not {part!r}"), False
        gd = ctx.get("gen_dir")
        if not gd:
            return error_json("no records directory for this proposal"), False
        p = Path(gd) / ("programs" if part == "program" else "feedback") / f"{cid}.{'py' if part == 'program' else 'json'}"
        try:
            text = p.read_text()
        except OSError:
            return error_json(f"no {part} recorded for {cid}"), False
        offset = max(0, int(offset or 0))
        page = max(500, self.result_chars - 300)
        chunk = text[offset:offset + page]
        out = {"ok": True, "cid": cid, "part": part, "offset": offset, "total_chars": len(text), "text": chunk}
        if offset + page < len(text):
            out["next_offset"] = offset + page
        return json.dumps(out, ensure_ascii=False), True


def chat_tools_kwargs(cfg: dict | None) -> dict:
    """ChatTools keyword arguments from a paper config: editor.explore.timeout_s / mem_gb and
    editor.tool_result_chars (defaults 300 / 8 / 6000)."""
    from .editor import editor_settings
    s = editor_settings(cfg)
    return {"timeout_s": s["explore"]["timeout_s"], "mem_gb": s["explore"]["mem_gb"],
            "result_chars": s["tool_result_chars"]}


def strip_loop(messages: list[dict]) -> list[dict]:
    """messages without the in-loop turns after the last user message (assistant turns with tool_calls and
    tool results), i.e. what a run without tools would have sent."""
    msgs = list(messages or [])
    last = max([k for k, m in enumerate(msgs) if m.get("role") == "user"], default=-1)
    return msgs[:last + 1] + [m for m in msgs[last + 1:] if m.get("role") != "tool" and not m.get("tool_calls")]


def with_mock_exploration(policy, rounds: int, code: str, tool: str = "run_python"):
    """Mock editor policy that first makes `rounds` run_python(code) calls, then returns policy(...) on the
    messages without the loop turns. rounds <= 0 -> policy unchanged."""
    rounds = int(rounds or 0)
    if rounds <= 0 or policy is None:
        return policy

    def wrapped(messages):
        msgs = list(messages or [])
        last = max([k for k, m in enumerate(msgs) if m.get("role") == "user"], default=-1)
        tool_msgs = [m for m in msgs[last + 1:] if m.get("role") == "tool"]
        forced = bool(tool_msgs) and FORCED_MARK in str(tool_msgs[-1].get("content") or "")
        if len(tool_msgs) < rounds and not forced:
            return {"content": "", "tool_calls": [{"name": tool, "arguments": {"code": code}}]}
        return policy(strip_loop(msgs))

    wrapped.inner = policy
    return wrapped
