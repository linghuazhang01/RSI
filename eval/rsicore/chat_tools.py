"""Chat tools of the editor's tool loop (2026-11-21): what the editor may call WHILE it writes a candidate.

    tools = ChatTools(explore_api_factory, timeout_s=300, mem_gb=8, result_chars=6000)
    ed = Editor(endpoint, name, system_prompt=SYS, seed=s, transcript=tr, tools=tools, **editor_kwargs(cfg))
    code = ed.propose(msg, ctx={"cid": cid, "gen_dir": gen_dir, "visible_cids": [...]})

Three functions (OpenAI tool schemas from schemas()):
  run_python(code)            runs `code` in rsicore.sandbox.run_program (new process, no network) with the
                              paper's EXPLORE ToolAPI as `api` (no evaluate). Top-level code is fine; set
                              `result = ...` to return a value, or define main(api). Nothing persists between
                              calls, except in the persistent session (below). Workdir
                              gen_dir/explore/<cid>/r{r:02d}_{i}/ (program.py, stdout.txt, stderr.txt,
                              result.json: the full output, for auditing).
  read_record(cid, part, offset=0)
                              the full program ("program": gen_dir/programs/<cid>.py) or feedback
                              ("feedback": gen_dir/feedback/<cid>.json) of a candidate in ctx["visible_cids"]
                              (the history snapshot the proposal was rendered from), in pages of the result size.

  search_history(gen=None, sort="score", offset=0)        (2026-11-21; only with history=SearchHistory(...))
                              every candidate already tried: this step's visible candidates and all candidates
                              of the earlier generations in history.past, one compact row each (cid, gen,
                              scores, error, gate outcome, the paper's summary of what the candidate is via
                              history.row_fn, same_as = an earlier cid with the same history.key_fn key, i.e.
                              an identical result), newest generation first; at offset 0 also one line per
                              earlier generation (history.gen_fn: accepted, gate reason). Paged by result size.
                              With history, read_record also accepts any cid of an earlier generation in
                              history.past (its programs/ and feedback/ there).

explore_api_factory(tag: str, ctx: dict) is called once per run_python call, in the proposing thread, with
tag = f"{cid}~x{r:02d}_{i}" (r = round, i = index of the call in that turn). It returns one of
  * a ToolAPI,
  * (ToolAPI, cleanup) where cleanup() is called after the run (e.g. Paper B: begin_program(tag) inside the
    factory, end_program in cleanup, so exploration handles are freed and the tool log lives under "~x" keys),
  * (ToolAPI, cleanup, before_call) (2026-11-21): before_call(timeout_s) is called before every run (cleanup may
    be None),
  * a context manager whose __enter__ returns the ToolAPI.
The factory must not register anything that spends a generation budget.

Persistent session (2026-11-21, persistent=True <- editor.explore.persistent; default False = the per-call mode
above): ONE rsicore.sandbox.Kernel per proposal (keyed by ctx["cid"]), started lazily by the first run_python call;
every call execs its code in the same globals (variables, imports, functions persist; `api` is a persistent global;
`result` is cleared before each call). The explore api is made ONCE per proposal: factory(f"{cid}~x", ctx) (same
return forms; before_call(timeout_s) runs before every exec, e.g. Paper B's tensor_compute deadline). A call that
times out (editor.explore.timeout_s) or crashes kills the kernel and reports "session restarted: variables lost"
(error + "restarted": true); the next call starts a fresh kernel with the SAME api object (workspace handles
persist). end(ctx) (the Editor calls it when a proposal finishes, on every path) shuts the kernel down and runs
cleanup; begin(ctx) also discards a stale kernel of the same cid. Workdir gen_dir/explore/<cid>/session/ with
r{r:02d}_{i}/ (program.py, stdout.txt, stderr.txt, result.json) per call. Thread-safe across proposals (each
proposal's calls are sequential). A run_python_doc that starts with RUN_PYTHON_DOC gets that prefix replaced by
RUN_PYTHON_DOC_PERSISTENT.

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
import re
import shutil
import signal
import tempfile
import threading
import time
from pathlib import Path

from .records import read_json, read_jsonl
from .sandbox import RESTARTED, Kernel, ToolAPI, run_program

RUN_PYTHON_DOC = (
    "Run exploratory Python in the program sandbox (new process, no network, no files outside its own "
    "workdir). The same `api` object as in your final program is available as a global (call tools by "
    "keyword: api.<tool>(...)); budget-spending tools are NOT available here. Write "
    "top-level code: print() what you need, or set `result = ...` to get a JSON value back (defining "
    "main(api) also works). Filter big tables inside the code instead of printing them whole. Output is "
    "cut to a few thousand characters. Nothing persists between calls or into your final program.")
RUN_PYTHON_DOC_PERSISTENT = (
    "Run exploratory Python in a persistent sandbox session (no network, no files outside its own workdir), like a "
    "notebook kernel: variables, imports and functions you define stay available in your later run_python calls of "
    "this candidate. The same `api` object as in your final program is a global (call tools by keyword: "
    "api.<tool>(...)); budget-spending tools are NOT available here. Write top-level code: "
    "print() what you need, or set `result = ...` to get a JSON value back (`result` is cleared before every call). "
    "Filter big tables inside the code instead of printing them whole. Output is cut to a few thousand characters. "
    "A call that runs past the time limit or crashes restarts the session (the reply says 'session restarted: "
    "variables lost'; recompute what you need). Nothing carries into your final program: it must build everything "
    "it returns itself.")


def run_python_doc(persistent: bool = False) -> str:
    """The rsicore run_python description for the per-call (default) or persistent session mode (2026-11-21)."""
    return RUN_PYTHON_DOC_PERSISTENT if persistent else RUN_PYTHON_DOC
READ_RECORD_DOC = (
    "Read the full program or the full feedback of an earlier candidate listed in the latest message "
    "(the message shows them shortened). Long texts come in pages: call again with offset=next_offset.")

SEARCH_HISTORY_DOC = (
    "List every candidate already tried, so you do not repeat one: this generation's candidates (those in the "
    "latest message) and all candidates of earlier generations, one compact row each (what it is, its scores, "
    "error, gate outcome; same_as = an earlier candidate with an identical result, which would score the same). "
    "Newest generation first, best score first within a generation (sort='order': in the order they were tried). "
    "Scores are relative to the reference model of their own generation: compare within a generation. "
    "gate = the candidate's gate result (passed, reason and, when the gate scored it, statistics such as the paired "
    "gain, its CI, accuracy and the counts / ids of items broken or fixed; never item content). "
    "Under a KL gate (metric 'kl'): kl / reference_kl = mean token KL to the original model (lower is closer), "
    "gain = the KL reduction vs that round's incumbent. "
    "read_record(cid, part) then shows any listed candidate's full program or full feedback.")

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


_CID_GEN = re.compile(r"^g(\d+)_")


def _cid_gen(cid) -> int | None:
    m = _CID_GEN.match(str(cid or ""))
    return int(m.group(1)) if m else None


def _num(x, nd=4):
    try:
        return None if x is None else round(float(x), nd)
    except (TypeError, ValueError):
        return None


class SearchHistory:
    """Earlier generations for the search_history / read_record chat tools (2026-11-21).
    past: {gen: directory} of the earlier generations' searches (candidates.jsonl, programs/, feedback/,
    gate.json); row_fn(record) -> dict: the paper's short summary of one candidates.jsonl record (artifact,
    screen, score, error); key_fn(record) -> str | None: identity of the result (e.g. a config hash),
    used for same_as; gen_fn(gen, directory) -> dict: the paper's outcome of an earlier generation.
    SearchHistory({}) = no earlier generations (e.g. Paper A's generation 1): search_history then lists this
    step's candidates only."""

    def __init__(self, past: dict | None = None, *, row_fn=None, key_fn=None, gen_fn=None):
        self.past = {int(g): Path(d) for g, d in (past or {}).items()}
        self.row_fn, self.key_fn, self.gen_fn = row_fn, key_fn, gen_fn


# gate statistics copied from a gate.json submission's info into search_history rows (2026-11-21, gate per candidate:
# statistics only, never item content; absent keys are skipped, so older gate.json files give passed / reason only)
GATE_STATS = ("gain", "ci", "accuracy", "reference_accuracy", "broken", "fixed", "broken_ids", "fixed_ids",
              "metric", "kl", "reference_kl")
# Paper A's KL gate (2026-11-24): accuracy fields hold -KL there and broken / fixed do not exist, so they are left out
_KL_HIDDEN = ("accuracy", "reference_accuracy", "broken", "fixed", "broken_ids", "fixed_ids")


def _gate_outcomes(d: Path, cids=None) -> dict:
    """cid -> {"passed": bool, "reason": str, + GATE_STATS present in info} from <d>/gate.json
    (a paper's gate-per-candidate records); cids: only these."""
    out = {}
    for sub in (read_json(Path(d) / "gate.json", None) or {}).get("submissions") or []:
        if cids is not None and str(sub.get("cid")) not in cids:
            continue
        info = sub.get("info") if isinstance(sub.get("info"), dict) else {}
        why = info.get("reason") or info.get("reasons")
        if isinstance(why, (list, tuple)):
            why = "; ".join(map(str, why))
        row = {"passed": bool(sub.get("passed")), **({"reason": cut(str(why), 200)} if why else {})}
        for k in GATE_STATS:
            if info.get("metric") == "kl" and k in _KL_HIDDEN:
                continue
            if info.get(k) is not None:
                v = info[k]
                row[k] = v[:20] if isinstance(v, list) and k.endswith("_ids") else v
        out[str(sub.get("cid"))] = row
    return out


def _schema(name, doc, props, required):
    return {"type": "function", "function": {"name": name, "description": doc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


class ChatTools:
    def __init__(self, explore_api_factory, *, timeout_s: float = 300, mem_gb: float = 8, result_chars: int = 6000,
                 run_python_doc: str | None = None, read_record: bool = True, history: SearchHistory | None = None,
                 persistent: bool = False):
        self.factory = explore_api_factory
        self.history = history
        self.timeout_s, self.mem_gb, self.result_chars = float(timeout_s), mem_gb, int(result_chars)
        self.persistent = bool(persistent)
        doc = run_python_doc or RUN_PYTHON_DOC
        if self.persistent and doc.startswith(RUN_PYTHON_DOC):
            doc = RUN_PYTHON_DOC_PERSISTENT + doc[len(RUN_PYTHON_DOC):]
        self.run_python_doc = doc
        self.read_record_enabled = bool(read_record)
        self._sessions: dict[str, dict] = {}      # persistent: cid -> {kernel, api, cleanup, before_call, dir}
        self._slock = threading.Lock()

    # ---------------------------------------------------------------- schemas
    def names(self) -> list[str]:
        return (["run_python"] + (["read_record"] if self.read_record_enabled else [])
                + (["search_history"] if self.history is not None else []))

    def schemas(self) -> list[dict]:
        out = [_schema("run_python", self.run_python_doc,
                       {"code": {"type": "string", "description": "Python source; uses the global `api`."}}, ["code"])]
        if self.read_record_enabled:
            out.append(_schema("read_record", READ_RECORD_DOC, {
                "cid": {"type": "string", "description": "candidate id, e.g. g01_t2_s03"},
                "part": {"type": "string", "enum": ["program", "feedback"]},
                "offset": {"type": "integer", "description": "character offset of the page (default 0)"}},
                ["cid", "part"]))
        if self.history is not None:
            out.append(_schema("search_history", SEARCH_HISTORY_DOC, {
                "gen": {"type": "integer", "description": "only this generation (default: all)"},
                "sort": {"type": "string", "enum": ["score", "order"]},
                "offset": {"type": "integer", "description": "row offset of the page (default 0)"}}, []))
        return out

    # ---------------------------------------------------------------- dispatch
    def begin(self, ctx: dict | None):
        """A proposal starts live: discard a stale persistent session of the same cid, remove
        gen_dir/explore/<cid>/ of a discarded earlier attempt."""
        ctx = ctx or {}
        self.end(ctx)
        cid, gd = str(ctx.get("cid") or ""), ctx.get("gen_dir")
        if not gd or not cid or "/" in cid or "\\" in cid or cid in (".", ".."):
            return
        d = Path(gd) / "explore" / cid
        if d.is_dir() and not d.is_symlink():
            shutil.rmtree(d, ignore_errors=True)

    def end(self, ctx: dict | None):
        """The proposal is finished (2026-11-21): shut its persistent kernel down and run the explore api's
        cleanup. Never raises; a no-op in the per-call mode or when no session exists."""
        cid = str((ctx or {}).get("cid") or "explore")
        with self._slock:
            sess = self._sessions.pop(cid, None)
        if sess is not None:
            self._close(sess)

    def end_all(self):
        """Shut every open persistent session down (e.g. at the end of a search)."""
        with self._slock:
            sessions, self._sessions = list(self._sessions.values()), {}
        for sess in sessions:
            self._close(sess)

    @staticmethod
    def _close(sess: dict):
        try:
            sess["kernel"].close()
        except Exception:  # noqa: BLE001 - housekeeping only
            pass
        if sess.get("cleanup") is not None:
            try:
                sess["cleanup"]()
            except Exception:  # noqa: BLE001
                pass

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
            elif name == "search_history" and self.history is not None:
                content, ok = self.search_history(ctx, **args)
            else:
                content, ok = error_json(f"unknown tool {name!r}; available: {self.names()}"), False
        except Exception as e:  # noqa: BLE001 - a failing tool must not stop the proposal
            content, ok = error_json(e), False
        return {"content": content, "ok": ok, "seconds": round(time.time() - t0, 3), "args": args}

    # ---------------------------------------------------------------- run_python
    def _enter(self, tag: str, ctx: dict):
        """factory(tag, ctx) -> (api, cleanup | None, before_call | None)."""
        made = self.factory(tag, ctx)
        if isinstance(made, ToolAPI):
            return made, None, None
        if isinstance(made, tuple) and len(made) in (2, 3):
            return made[0], made[1], (made[2] if len(made) == 3 else None)
        if hasattr(made, "__enter__") and hasattr(made, "__exit__"):
            api = made.__enter__()
            return api, lambda: made.__exit__(None, None, None), None
        raise TypeError(f"explore_api_factory returned {type(made).__name__}; expected ToolAPI, (api, cleanup), "
                        "(api, cleanup, before_call) or a context manager")

    def _session(self, ctx: dict, cid: str) -> dict:
        """The proposal's persistent session (explore api made once; kernel started lazily by Kernel.run)."""
        with self._slock:
            sess = self._sessions.get(cid)
        if sess is not None:
            return sess
        gd = ctx.get("gen_dir")
        base = Path(gd) if gd else Path(tempfile.mkdtemp(prefix="rsi_explore_"))
        api, cleanup, before = self._enter(f"{cid}~x", ctx)
        d = base / "explore" / cid / "session"
        sess = {"kernel": Kernel(d, mem_gb=self.mem_gb), "api": api, "cleanup": cleanup, "before_call": before,
                "dir": d}
        with self._slock:
            old = self._sessions.get(cid)
            if old is None:
                self._sessions[cid] = sess
        if old is not None:             # another thread made it first (same cid; should not happen)
            self._close(sess)
            return old
        return sess

    def run_python(self, ctx: dict, r: int, i: int, *, code: str, deadline_s: float | None = None) -> tuple[str, bool]:
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        cid = str(ctx.get("cid") or "explore")
        sub = f"r{int(r):02d}_{int(i)}"
        timeout = self.timeout_s if deadline_s is None else min(self.timeout_s, max(5.0, float(deadline_s)))
        if self.persistent:
            sess = self._session(ctx, cid)
            api, wd = sess["api"], sess["dir"] / sub
            if sess["before_call"] is not None:
                sess["before_call"](timeout)
            before = dict(getattr(api, "calls", {}) or {})
            res = sess["kernel"].run(code, api, timeout_s=int(round(timeout)), subdir=sub)
            after = dict(getattr(api, "calls", {}) or {})
        else:
            gd = ctx.get("gen_dir")
            base = Path(gd) if gd else Path(tempfile.mkdtemp(prefix="rsi_explore_"))
            wd = base / "explore" / cid / sub
            api, cleanup, before_call = self._enter(f"{cid}~x{sub[1:]}", ctx)
            try:
                if before_call is not None:
                    before_call(timeout)
                before = dict(getattr(api, "calls", {}) or {})
                res = run_program(_WRAP.format(src=repr(code)), api, timeout_s=int(round(timeout)),
                                  mem_gb=self.mem_gb, workdir=wd, save_output=True)
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
        if (res.info or {}).get("restarted"):
            out["restarted"] = True
        error = res.error
        xcpu = getattr(signal, "SIGXCPU", None)
        if not res.ok and xcpu is not None and (res.info or {}).get("returncode") == -xcpu:
            error = f"timeout: CPU-time limit reached ({error})"   # the CPU limit can fire before the wall clock
        if error:
            out["error"] = cut(error, 2000, head=0.3)
            if out.get("restarted") and RESTARTED not in out["error"]:
                out["error"] += f" ({RESTARTED})"
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
    def _past_dir(self, cid) -> Path | None:
        """Directory of an earlier generation's candidate `cid` (history.past), or None."""
        if self.history is None:
            return None
        g = _cid_gen(cid)
        d = self.history.past.get(g) if g is not None else None
        if d is None or not any(r.get("cid") == str(cid) for r in read_jsonl(d / "candidates.jsonl")):
            return None
        return d

    def read_record(self, ctx: dict, *, cid: str, part: str = "feedback", offset: int = 0) -> tuple[str, bool]:
        visible = [str(c) for c in (ctx.get("visible_cids") or [])]
        gd = ctx.get("gen_dir") if str(cid) in visible else self._past_dir(cid)
        if str(cid) not in visible and gd is None:
            shown = ", ".join(visible[-12:]) or "(none yet)"
            past = (f"; or a cid of generation {', '.join(map(str, sorted(self.history.past)))} (search_history lists them)"
                    if self.history is not None and self.history.past else "")
            return error_json(f"cid {cid!r} is not visible in this step; visible: {shown}{past}"), False
        if part not in ("program", "feedback"):
            return error_json(f"part must be 'program' or 'feedback', not {part!r}"), False
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


    # ---------------------------------------------------------------- search_history
    def _history_rows(self, ctx: dict) -> tuple[list[dict], list[dict]]:
        """(rows of every candidate tried so far, newest generation first; one entry per earlier generation)."""
        h = self.history
        visible = [str(c) for c in (ctx.get("visible_cids") or [])]
        cur = _cid_gen(ctx.get("cid"))
        groups = []
        for g in sorted(h.past):
            groups.append((g, read_jsonl(h.past[g] / "candidates.jsonl"), _gate_outcomes(h.past[g])))
        gd = ctx.get("gen_dir")
        if gd and visible:
            recs = {r.get("cid"): r for r in read_jsonl(Path(gd) / "candidates.jsonl")}
            # gate results of this generation's visible (earlier-step) candidates: the papers gate every
            # candidate during the search (2026-11-21; the only flow since 2026-11-23)
            groups.append((cur, [recs[c] for c in visible if c in recs], _gate_outcomes(Path(gd), set(visible))))
        seen, rows = {}, []
        for g, recs, gate in groups:           # oldest first, so same_as names the first one tried
            for r in recs:
                cid = str(r.get("cid"))
                sc = r.get("screen") or {}      # (2026-11-23) no "full" column: the full / rerun stage was removed
                row = {"cid": cid, "gen": g, "score": _num(r.get("score")), "screen": _num(sc.get("score"))}
                err = (r.get("error") or "").strip()
                if err:
                    row["error"] = cut(err.splitlines()[0], 140, head=1.0)
                if cid in gate:
                    row["gate"] = gate[cid]
                for fn, name in ((h.row_fn, "summary"), (h.key_fn, "key")):
                    if fn is None:
                        continue
                    try:
                        v = fn(r)
                    except Exception:  # noqa: BLE001 - a bad record must not hide the others
                        v = None
                    if name == "summary" and isinstance(v, dict):
                        row.update({k: x for k, x in v.items() if x is not None})
                    elif name == "key" and v and not err:
                        if v in seen:
                            row["same_as"] = seen[v]
                        else:
                            seen[v] = cid
                rows.append(row)
        gens = []
        for g in sorted(h.past):
            e = {"gen": g, "candidates": sum(1 for r in rows if r["gen"] == g)}
            if h.gen_fn is not None:
                try:
                    e.update(h.gen_fn(g, h.past[g]) or {})
                except Exception:  # noqa: BLE001
                    pass
            gens.append(e)
        return rows, gens

    def search_history(self, ctx: dict, *, gen=None, sort: str = "score", offset: int = 0) -> tuple[str, bool]:
        if sort not in ("score", "order"):
            return error_json(f"sort must be 'score' or 'order', not {sort!r}"), False
        rows, gens = self._history_rows(ctx)
        if gen is not None:
            rows = [r for r in rows if r["gen"] == int(gen)]
        order = {g: i for i, g in enumerate(dict.fromkeys(r["gen"] for r in rows))}

        def key(r):
            s = r["score"] if r["score"] is not None else float("-inf")
            return (-order[r["gen"]], -s if sort == "score" else 0)
        rows = sorted(rows, key=key)           # stable: 'order' keeps the order tried within a generation
        offset = max(0, int(offset or 0))
        out = {"ok": True, "total": len(rows), "offset": offset}
        if offset == 0:
            out["note"] = ("score: higher is better, relative to the reference model of that generation (compare "
                           "within a generation); gate: submitted to the gate (passed or not); same_as: identical "
                           "result to that earlier candidate")
            if gens:
                out["generations"] = gens
        room = self.result_chars - len(json.dumps(out, ensure_ascii=False)) - 120
        page, i = [], offset
        while i < len(rows):
            t = len(json.dumps(rows[i], ensure_ascii=False)) + 2
            if page and t > room:
                break
            page.append(rows[i])
            room -= t
            i += 1
        out["rows"] = page
        if i < len(rows):
            out["next_offset"] = i
        return json.dumps(out, ensure_ascii=False), True


def chat_tools_kwargs(cfg: dict | None) -> dict:
    """ChatTools keyword arguments from a paper config: editor.explore.timeout_s / mem_gb / persistent and
    editor.tool_result_chars (defaults 300 / 8 / false / 6000)."""
    from .editor import editor_settings
    s = editor_settings(cfg)
    return {"timeout_s": s["explore"]["timeout_s"], "mem_gb": s["explore"]["mem_gb"],
            "result_chars": s["tool_result_chars"], "persistent": bool(s["explore"].get("persistent", False))}


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
