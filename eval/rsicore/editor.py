"""The editor (改造者): a multi-turn chat with the current model M_t that returns programs.

    ed = Editor(endpoint, served_name, system_prompt=SYS, seed=rc.seed + thread,
                transcript=gen_dir / "editor" / f"thread_{thread}.jsonl",
                tools=ChatTools(...) or None, **editor_kwargs(cfg))
    code = ed.propose(render_context(history, thread), ctx={"cid": cid, "gen_dir": gd, "visible_cids": [...]})
    ed.last_stats   # {"rounds", "tool_calls", "tool_s", "prompt_tokens", "completion_tokens", "wall_s", "stop", ...}

One Editor per thread (its own history). One proposal = one call of propose():
  * tools None or tool_rounds 0: exactly one /chat/completions request (payload without tools / tool_choice),
    the code is the last ```python block of the reply ("" if none) -- the behaviour before 2026-11-21.
  * otherwise the tool loop (2026-11-21): the model may call the chat tools (rsicore.chat_tools: run_python,
    read_record) for up to tool_rounds turns ("round" = one model turn with tool calls, however many parallel
    calls), then one forced final turn with tool_choice "none" (a notice is appended to the last tool message,
    never a new user message). The proposal ends with a reply WITHOUT tool calls that holds a ```python block.
    Empty / cut-at-length / unparsed "<tool_call>" replies get up to 2 nudges during the tool rounds and one
    more on the forced final turn (still tool_choice "none"); without any final block the last block seen in
    the loop is used (stop "fallback"). Guards that force the final turn: rounds >= tool_rounds, tool calls >=
    max_tool_calls, completion tokens >= proposal_tokens, wall clock >= proposal_wall_s, too little room left
    in context_tokens, a tool-round reply that timed out. Per-turn max_tokens = min(turn_max_tokens, or
    max_tokens on the final turn, context_tokens - estimated prompt - 512). The live message list is compacted
    deterministically (reasoning of older turns dropped, older tool outputs and long arguments elided) when the
    estimated prompt passes compact_at x context_tokens; an HTTP 400 about the context length -> aggressive
    compaction and one retry, then the forced final turn. HTTP read timeouts are not retried inside the loop:
    a tool-round turn gets min(timeout_s, time left in proposal_wall_s + 600 s) (at least 60 s) and a timeout
    forces the final turn (forced_by "timeout"); the final turn gets the full timeout_s. If the final turn
    fails too (context or timeout) the fallback block is used when there is one, else EditorReplyError
    (search records an empty candidate and continues). Connection errors / 5xx after 5 attempts stay fatal
    (RuntimeError).
    In-loop reasoning is passed back per reasoning_passback: think_tag (<think>..</think> + content), field
    (reasoning_content) or drop.
Cross-step history is collapsed (one user + one assistant message per earlier proposal: no tool turns, no
reasoning). history_keep=K keeps proposal 0 and the last K earlier proposals in full; the ones in between
become a stub user message + their code block. None keeps everything. editor_kwargs() sets it to None when
tool_rounds is 0, so the single call sends the same payload as before the tool loop.

Transcript (jsonl, format 2): a system line {"role": "system", ..., "format": 2}, then per proposal p:
  {p, role: user, content, cid}                                    the feedback message (cid from ctx)
  {p, r, role: assistant, content, reasoning, tool_calls: [{id, name, arguments}], usage, finish_reason}
  {p, r, role: tool, tool_call_id, name, args, content (as sent), full_chars, ok, seconds}
  {p, r, role: assistant, content, reasoning, usage, finish_reason, nudge}  +  {p, role: user, content, nudge: true}
  {p, role: event, kind: compact | nudge | forced_final | http400 | timeout, detail}
  {p, role: assistant, final: true, content, reasoning, code, usage, finish_reason, stats}
  {p, role: assistant, final: true, failed: true, code: "", error}  (EditorReplyError: not in the history)
On construction the transcript is reloaded. propose() with the same user message as the next COMPLETE recorded
proposal replays it (collapsed history, recorded code and stats; a failed one raises EditorReplyError again)
without calling the model or any tool; a different message drops the rest of the record; a trailing
incomplete proposal (crash inside the loop) is discarded and run again live; the file is rewritten at the
first live proposal. Resume (search skips the cids it already recorded): when ctx carries a cid, the leading
recorded proposals of OTHER cids are consumed first (into the history and the rewritten file, as if
replayed), so a resumed thread keeps its earlier steps. Legacy files (format 1: strict user / assistant pairs)
load as complete proposals with zero rounds (and no cid).
mock_policy: callable(messages: list[dict]) -> program text (str; wrapped in a ```python block when it has
none) or a dict {"content", "tool_calls": [{"name", "arguments"}], "reasoning"}; used instead of the HTTP call.
It sees the full per-proposal messages; ids are synthesised as call_{p}_{r}_{i}; guards and the forced final
turn apply as for the real model.
"""
import json
import os
import re
import shutil
import socket
import time
import urllib.error
from pathlib import Path

from . import _net
from .chat_tools import FORCED_MARK, cut, error_json

_BLOCK = re.compile(r"```(?:python|py|python3)[^\n]*\n(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)

EDITOR_DEFAULTS = {
    "temperature": 0.6, "top_p": 0.95, "max_tokens": 16384,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20, "min_p": 0.0},
    "tool_rounds": 50, "turn_max_tokens": 8192, "max_tool_calls": 100, "proposal_tokens": 160000,
    "proposal_wall_s": 2400, "context_tokens": 131072, "tool_result_chars": 6000, "compact_at": 0.6,
    "keep_tool_rounds": 8, "keep_reasoning_rounds": 2, "reasoning_passback": "think_tag", "history_keep": 2,
    "timeout_s": 3600, "mock_tool_rounds": 1, "history_tool": True,
    # persistent (2026-11-21): one sandbox kernel per proposal for run_python (rsicore.chat_tools; simple: true)
    "explore": {"timeout_s": 300, "mem_gb": 8, "gpu_tools": False, "gpu_seconds": 300, "persistent": False},
}
_EDITOR_KW = ("temperature", "top_p", "max_tokens", "extra_body", "timeout_s", "tool_rounds", "turn_max_tokens",
              "max_tool_calls", "proposal_tokens", "proposal_wall_s", "context_tokens", "tool_result_chars",
              "compact_at", "keep_tool_rounds", "keep_reasoning_rounds", "reasoning_passback", "history_keep")
HISTORY_STUB = "[step context omitted; the latest message has the full candidate table]"
NUDGE = {
    "length": "Your reply was cut off at the token limit before the program. Think less now: reply with a short "
              "plan and exactly one ```python block.",
    "tool_text": "Your function call was not recognised (it arrived as plain text). Either call the function "
                 "properly, or reply with a short plan and exactly one ```python block.",
    "empty": "Your reply was empty. Reply with a short plan and exactly one ```python block.",
    "tool_text_final": "Function calls are not available any more (your call arrived as plain text). Reply "
                       "with a short plan and exactly one ```python block.",
}
FORCED_WHY = {"rounds": "tool-round limit reached", "tool_calls": "tool-call limit reached",
              "tokens": "token budget of this proposal used up", "wall": "time limit of this proposal reached",
              "context": "context nearly full", "http400": "context full", "timeout": "model reply timed out"}
STOP_TEXT = {"length": "final reply cut at max_tokens while thinking", "no_code": "final reply had no code block",
             "fallback": "used the last code block written during the tool rounds"}


class EditorReplyError(RuntimeError):
    """The editor could not produce a reply for this proposal (context errors even after compaction and the
    forced final turn). search records an empty candidate with this error and continues."""


class _ContextError(Exception):
    pass


class _ReplyTimeout(Exception):
    pass


def _is_timeout(e) -> bool:
    if isinstance(e, (TimeoutError, socket.timeout)):
        return True
    return isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None), (TimeoutError, socket.timeout))


def extract_python(text: str) -> str:
    text = _THINK.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    blocks = _BLOCK.findall(text)
    return blocks[-1].strip("\n") + "\n" if blocks else ""


def describe_stop(stats: dict | None) -> str:
    """Short reason for an empty program, e.g. 'final reply cut at max_tokens while thinking'."""
    stop = (stats or {}).get("stop") or ""
    if stop.startswith("forced:"):
        return f"final turn forced ({FORCED_WHY.get(stop[7:], stop[7:])})"
    return STOP_TEXT.get(stop, stop)


# ------------------------------------------------------------------------------------------- config helpers
def editor_settings(cfg: dict | None) -> dict:
    """Resolved editor settings of a paper config (its `editor:` block over EDITOR_DEFAULTS). extra_body is
    taken as given when the key is present (so {} turns the thinking extras off); explore is merged per key."""
    e = dict(((cfg or {}).get("editor") or {}))
    out = {k: (json.loads(json.dumps(v)) if isinstance(v, dict) else v) for k, v in EDITOR_DEFAULTS.items()}
    for k, v in e.items():
        if k == "explore":
            out["explore"].update(dict(v or {}))
        elif k == "extra_body":
            out["extra_body"] = dict(v or {})
        else:
            out[k] = v
    return out


def context_tokens(cfg: dict | None, server_args=None) -> int | None:
    """editor.context_tokens (default 131072). Explicit null -> --max-model-len of server_args (the editor
    server's vLLM args; 40960 when they name none), or None without server_args."""
    s = editor_settings(cfg)
    n = s.get("context_tokens")
    if n is None and server_args is not None:
        from .servers import max_model_len
        n = max_model_len(server_args)
    return None if n is None else int(n)


def editor_kwargs(cfg: dict | None, server_args=None) -> dict:
    """Editor(...) keyword arguments from a paper config's editor block: sampling (temperature, top_p,
    extra_body), max_tokens (final turn), timeout_s, the loop keys and context_tokens (see context_tokens()).
    history_keep is None when tool_rounds is 0 (the single call of before sends the full history)."""
    s = editor_settings(cfg)
    kw = {k: s[k] for k in _EDITOR_KW}
    kw["context_tokens"] = context_tokens(cfg, server_args)
    for k in ("max_tokens", "tool_rounds", "turn_max_tokens", "max_tool_calls", "tool_result_chars",
              "keep_tool_rounds", "keep_reasoning_rounds"):
        kw[k] = int(kw[k])
    for k in ("temperature", "top_p", "compact_at"):
        kw[k] = float(kw[k])
    if kw["history_keep"] is not None:
        kw["history_keep"] = int(kw["history_keep"])
    if kw["tool_rounds"] <= 0:          # the single call of before: the full collapsed history
        kw["history_keep"] = None
    return kw


def apply_context_tokens(args, cfg: dict | None) -> list[str]:
    """vLLM args of a server the editor talks to, with --max-model-len = editor.context_tokens (replacing any
    --max-model-len in args). context_tokens null -> args unchanged."""
    from .servers import set_max_model_len
    n = editor_settings(cfg).get("context_tokens")
    return set_max_model_len(args, None if n is None else int(n))


# ------------------------------------------------------------------------------------------- helpers
def _valid_args(a) -> str:
    """Arguments as a JSON-object string (vLLM json.loads them when it renders the chat template)."""
    if isinstance(a, dict):
        return json.dumps(a, ensure_ascii=False)
    try:
        if isinstance(json.loads(a or "{}"), dict):
            return a or "{}"
    except (TypeError, ValueError):
        pass
    return json.dumps({"_raw": str(a)[:2000]}, ensure_ascii=False)


def _code_block(code: str) -> str:
    return f"```python\n{code}```" if code else "(no program)"


def _collapsed(final: dict) -> str:
    """Assistant content of a finished proposal in the cross-step history."""
    content, code = final.get("content") or "", final.get("code") or ""
    if not code or extract_python(content) == code:
        return content
    return (content.rstrip() + "\n\n" if content.strip() else "") + _code_block(code)


class _Reply:
    __slots__ = ("content", "reasoning", "tool_calls", "finish", "usage")

    def __init__(self, content, reasoning, tool_calls, finish, usage):
        self.content, self.reasoning, self.tool_calls = content or "", reasoning, tool_calls or []
        self.finish, self.usage = finish, usage or {}


class Editor:
    def __init__(self, endpoint, served_name, *, system_prompt: str, seed: int, transcript: Path,
                 temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 8192,
                 mock_policy=None, extra_body: dict | None = None, timeout_s: float = 3600,
                 tools=None, tool_rounds: int = 50, turn_max_tokens: int = 8192, max_tool_calls: int = 100,
                 proposal_tokens: int | None = 160000, proposal_wall_s: float | None = 2400,
                 context_tokens: int | None = None, tool_result_chars: int = 6000, compact_at: float = 0.6,
                 keep_tool_rounds: int = 8, keep_reasoning_rounds: int = 2, reasoning_passback: str = "think_tag",
                 history_keep: int | None = None, clock=None):
        self.endpoint, self.served_name = endpoint, served_name
        self.system_prompt, self.seed = system_prompt, seed
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, int(max_tokens)
        self.mock_policy, self.extra_body, self.timeout_s = mock_policy, dict(extra_body or {}), timeout_s
        self.tools, self.tool_rounds = tools, int(tool_rounds or 0)
        self.loop = tools is not None and self.tool_rounds > 0
        self.turn_max_tokens, self.max_tool_calls = int(turn_max_tokens), int(max_tool_calls)
        self.proposal_tokens, self.proposal_wall_s = proposal_tokens, proposal_wall_s
        self.context_tokens = None if context_tokens is None else int(context_tokens)
        self.tool_result_chars, self.compact_at = int(tool_result_chars), float(compact_at)
        self.keep_tool_rounds, self.keep_reasoning_rounds = int(keep_tool_rounds), int(keep_reasoning_rounds)
        if reasoning_passback not in ("think_tag", "field", "drop"):
            raise ValueError(f"reasoning_passback must be think_tag | field | drop, not {reasoning_passback!r}")
        self.reasoning_passback, self.history_keep = reasoning_passback, history_keep
        self.clock = clock or time.monotonic
        self.transcript = Path(transcript)
        self.transcript.parent.mkdir(parents=True, exist_ok=True)
        self.history = [{"role": "system", "content": system_prompt}]   # full collapsed history (legacy view)
        self._props = []      # finished proposals: {"user", "content" (collapsed), "code"}
        self._n = 0           # proposals done (replayed / skipped / live, failed ones included) = next p
        self._lines = []      # transcript lines (dicts) of the finished proposals (after the system line)
        self._replay = []     # recorded complete proposals not yet consumed: {"user", "lines", "final"}
        self._ratio = 1 / 3   # prompt tokens per character of the request (from the last usage)
        self._ctx_checked = False
        self.last_stats = None
        self._load()
        self.n_calls = 0      # model calls made by this instance (replays excluded)

    # ---------------------------------------------------------------- transcript
    def _load(self):
        self._stale = self.transcript.exists()
        if not self._stale:
            self._write_all()
            return
        lines = []
        for l in self.transcript.read_text().splitlines():
            try:
                d = json.loads(l)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict):
                lines.append(d)
        body = [l for l in lines if l.get("role") != "system"]
        if any("p" in l for l in body):                       # format 2: group by proposal
            groups, order = {}, []
            for l in body:
                if "p" not in l:
                    continue
                if l["p"] not in groups:
                    groups[l["p"]] = []
                    order.append(l["p"])
                groups[l["p"]].append(l)
            for p in order:
                g = groups[p]
                users = [l for l in g if l.get("role") == "user" and not l.get("nudge")]
                finals = [l for l in g if l.get("final")]
                if not users or not finals:
                    break                                     # incomplete: discarded (and everything after it)
                self._replay.append({"user": users[0].get("content"), "lines": g, "final": finals[-1],
                                     "cid": users[0].get("cid"), "failed": bool(finals[-1].get("failed"))})
        else:                                                 # format 1: strict user / assistant pairs
            msgs = [l for l in body if l.get("role") in ("user", "assistant")]
            for i in range(0, len(msgs) - 1, 2):
                u, a = msgs[i], msgs[i + 1]
                if u["role"] != "user" or a["role"] != "assistant":
                    break
                code = a.get("code") if a.get("code") is not None else extract_python(a.get("content"))
                usage = a.get("usage") or {}
                final = {"role": "assistant", "final": True, "content": a.get("content") or "",
                         "reasoning": a.get("reasoning"), "code": code, "usage": usage,
                         "stats": self._stats0(stop="answer" if code else "no_code", usage=usage), "ts": a.get("ts")}
                self._replay.append({"user": u.get("content"), "final": final, "v1": True,
                                     "lines": [{"role": "user", "content": u.get("content"), "ts": u.get("ts")}, final]})

    @staticmethod
    def _stats0(stop="answer", usage=None):
        usage = usage or {}
        return {"rounds": 0, "tool_calls": 0, "tool_s": 0.0, "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0), "wall_s": 0.0, "stop": stop,
                "model_calls": 1, "nudges": 0, "compactions": 0, "finish_reason": None, "forced_by": None}

    def _write_all(self):
        tmp = self.transcript.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps({"role": "system", "content": self.system_prompt, "served_name": self.served_name,
                                "seed": self.seed, "format": 2, "ts": time.strftime("%F %T")},
                               ensure_ascii=False) + "\n")
            for l in self._lines:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        os.replace(tmp, self.transcript)

    def _append(self, line: dict):
        with open(self.transcript, "a") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    # ---------------------------------------------------------------- history
    def _prompt_history(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        n, keep = len(self._props), self.history_keep
        for k, pr in enumerate(self._props):
            if keep is None or k == 0 or k >= n - int(keep):
                msgs += [{"role": "user", "content": pr["user"]}, {"role": "assistant", "content": pr["content"]}]
            else:
                msgs += [{"role": "user", "content": HISTORY_STUB}, {"role": "assistant", "content": _code_block(pr["code"])}]
        return msgs

    def _finish_prop(self, user_msg: str, final: dict, lines: list[dict]):
        content = _collapsed(final)
        self._props.append({"user": user_msg, "content": content, "code": final.get("code") or ""})
        self.history += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": content}]
        self._lines += lines
        self._n += 1

    def _consume(self, rec: dict):
        """Take a recorded complete proposal into the history (replay / resume skip-ahead)."""
        lines = [dict(l, p=self._n) for l in rec["lines"]] if rec.get("v1") else rec["lines"]
        if rec.get("failed"):           # EditorReplyError: recorded, never part of the history
            self._lines += lines
            self._n += 1
        else:
            self._finish_prop(rec["user"], rec["final"], lines)

    def _skip_recorded(self, cid):
        """Resume: consume the leading recorded proposals of other cids (search does not propose the cids it
        has already recorded). Stops at a record of this cid or one without a cid (legacy)."""
        if cid is None:
            return
        while self._replay and self._replay[0].get("cid") not in (None, cid):
            self._consume(self._replay.pop(0))

    # ---------------------------------------------------------------- one model call
    def _mock_reply(self, msgs: list[dict], p: int, r: int, force: bool) -> _Reply:
        out = self.mock_policy([dict(m) for m in msgs])
        if isinstance(out, dict):
            calls = []
            if self.loop and not force:
                for i, tc in enumerate(out.get("tool_calls") or []):
                    calls.append({"id": tc.get("id") or f"call_{p}_{r}_{i}", "name": tc.get("name"),
                                  "arguments": _valid_args(tc.get("arguments") or {})})
            text = out.get("content") or ""
            if not calls and text and "```" not in text:
                text = f"```python\n{text.rstrip()}\n```"
            return _Reply(text, out.get("reasoning"), calls, "tool_calls" if calls else "stop", {"mock": True})
        text = out or ""
        if "```" not in text:
            text = f"```python\n{text.rstrip()}\n```"
        return _Reply(text, None, [], "stop", {"mock": True})

    def _http_reply(self, msgs: list[dict], *, force: bool, max_tokens: int, timeout: float,
                    retry_timeout: bool = True) -> _Reply:
        """retry_timeout False (the tool loop): a read timeout raises _ReplyTimeout at once (the server has
        already spent the whole timeout on this payload; a retry would decode it again)."""
        payload = {"model": self.served_name, "messages": msgs, "temperature": self.temperature,
                   "top_p": self.top_p, "max_tokens": int(max_tokens), "seed": self.seed}
        if self.loop:
            payload["tools"] = self.tools.schemas()
            payload["tool_choice"] = "none" if force else "auto"
        payload.update(self.extra_body)
        last = None
        for attempt in range(5):
            try:
                resp = _net.post_json(self.endpoint.rstrip("/") + "/chat/completions", payload, timeout=timeout)
                ch = resp["choices"][0]
                msg = ch["message"]
                calls = []
                for i, tc in enumerate(msg.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    calls.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name"),
                                  "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str)
                                  else json.dumps(fn.get("arguments") or {}, ensure_ascii=False)})
                return _Reply(msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning"),
                              calls, ch.get("finish_reason"), resp.get("usage") or {})
            except _net.HTTPError as e:
                last = e
                if e.status in (400, 413, 422) and re.search(r"context|token|too long|length|prompt", e.body or "", re.I):
                    raise _ContextError(str(e)) from None
                if e.status < 500 and e.status not in (408, 429):
                    break
            except (OSError, KeyError, IndexError, TypeError, ValueError) as e:
                last = e
                if not retry_timeout and _is_timeout(e):
                    raise _ReplyTimeout(f"{type(e).__name__}: {e}") from None
            time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"editor chat failed: {last}")

    def _server_context(self):
        """min(context_tokens, the server's max_model_len from /v1/models) once (real servers only). A LoRA
        adapter's card has no max_model_len: then its parent (base model) card's, else the smallest one listed."""
        if self._ctx_checked or self.mock_policy is not None or self.context_tokens is None:
            return
        self._ctx_checked = True
        try:
            cards = [m for m in (_net.get_json(self.endpoint.rstrip("/") + "/models", timeout=10).get("data") or [])
                     if isinstance(m, dict)]

            def mml(m):
                n = (m or {}).get("max_model_len")
                return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None
            by_id = {m.get("id"): m for m in cards}
            me = by_id.get(self.served_name)
            if me is None:
                return
            n = mml(me)
            if n is None:
                n = mml(by_id.get(me.get("parent"))) or mml(by_id.get(me.get("root")))
            if n is None:
                n = min((x for x in map(mml, cards) if x), default=None)
            if n is not None and n < self.context_tokens:
                self.context_tokens = n
        except Exception:  # noqa: BLE001 - optional
            pass

    # ---------------------------------------------------------------- loop helpers
    def _render(self, loop: list[dict]) -> list[dict]:
        out = []
        for e in loop:
            t = e["t"]
            if t == "asst":
                content = e["content"]
                m = {"role": "assistant"}
                if e["reasoning"] and not e["drop_reasoning"]:
                    if self.reasoning_passback == "think_tag":
                        content = f"<think>\n{e['reasoning'].strip()}\n</think>\n\n{content}"
                    elif self.reasoning_passback == "field":
                        m["reasoning_content"] = e["reasoning"]
                m["content"] = content
                m["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {
                    "name": tc["name"] or "", "arguments": '{"code": "[elided]"}' if k in e["elided_args"]
                    else _valid_args(tc["arguments"])}} for k, tc in enumerate(e["tool_calls"])]
                out.append(m)
            elif t == "tool":
                out.append({"role": "tool", "tool_call_id": e["id"],
                            "content": (e["elided"] or e["content"]) + e["notice"]})
            elif t == "nasst":
                out.append({"role": "assistant", "content": e["content"]})
            else:   # nudge user
                out.append({"role": "user", "content": e["content"] + e["notice"]})
        return out

    def _estimate(self, msgs: list[dict]) -> int:
        n = len(json.dumps(msgs, ensure_ascii=False))
        if self.loop:
            n += len(json.dumps(self.tools.schemas(), ensure_ascii=False))
        return int(n * self._ratio) + 64

    def _compact(self, loop: list[dict], base: list[dict], *, aggressive: bool) -> dict:
        rounds = [e["r"] for e in loop if e["t"] == "asst"]
        cur = max(rounds, default=0)
        keep_reason = 0 if aggressive else self.keep_reasoning_rounds
        keep_tools = min(2, self.keep_tool_rounds) if aggressive else self.keep_tool_rounds
        keep_r = set(sorted(rounds)[-keep_reason:]) if keep_reason > 0 else set()
        est0 = self._estimate(base + self._render(loop))
        nr = nt = na = 0
        for e in loop:
            if e["t"] == "asst" and e["reasoning"] and not e["drop_reasoning"] and e["r"] not in keep_r:
                e["drop_reasoning"] = True
                nr += 1
            if e["t"] == "tool" and not e["elided"] and e["r"] <= cur - keep_tools:
                e["elided"] = f"[elided: {e['name']} output r{e['r']}, {len(e['content']):,} chars]"
                nt += 1
        if aggressive or (self.context_tokens and self._estimate(base + self._render(loop)) > 0.85 * self.context_tokens):
            for e in loop:
                if e["t"] == "asst" and e["r"] <= cur - keep_tools:
                    for k, tc in enumerate(e["tool_calls"]):
                        if k not in e["elided_args"] and len(tc["arguments"] or "") > 1000:
                            e["elided_args"].add(k)
                            na += 1
        return {"mode": "aggressive" if aggressive else "normal", "est_before": est0,
                "est_after": self._estimate(base + self._render(loop)), "reasoning_dropped": nr,
                "tools_elided": nt, "args_elided": na, "changed": bool(nr or nt or na)}

    # ---------------------------------------------------------------- propose
    def propose(self, user_msg: str, ctx: dict | None = None) -> str:
        cid = (ctx or {}).get("cid")
        if self._replay:
            self._skip_recorded(cid)
        if self._replay:
            rec = self._replay[0]
            if rec["user"] == user_msg:
                self._replay.pop(0)
                self._consume(rec)
                f = rec["final"]
                if rec.get("failed"):
                    self.last_stats = None
                    raise EditorReplyError(f.get("error") or "editor failed (recorded)")
                self.last_stats = dict(f.get("stats") or self._stats0(), replayed=True)
                return f.get("code") if f.get("code") is not None else extract_python(f.get("content"))
            self._replay = []           # diverged: drop the rest of the recorded conversation
        if self._stale:                 # file may hold unconsumed / torn lines: rewrite to the live history
            self._write_all()
            self._stale = False
        p = self._n
        lines = []

        def emit(line):
            line = dict(line, ts=time.strftime("%F %T"))
            lines.append(line)
            self._append(line)

        emit({"p": p, "role": "user", "content": user_msg, **({"cid": str(cid)} if cid is not None else {})})
        base = self._prompt_history() + [{"role": "user", "content": user_msg}]
        try:
            if not self.loop:
                final, stats = self._single(base, p)
            else:
                final, stats = self._run_loop(base, p, ctx or {}, emit)
        except EditorReplyError as e:   # recorded as a failed proposal (complete, outside the history)
            emit({"p": p, "role": "assistant", "final": True, "failed": True, "content": "", "code": "",
                  "error": str(e)[:2000], "stats": {"stop": "error"}})
            self._lines += lines
            self._n += 1
            self.last_stats = None
            raise
        finally:
            if self.loop:               # e.g. ChatTools: shut the proposal's persistent kernel down (2026-11-21)
                end = getattr(self.tools, "end", None)
                if callable(end):
                    try:
                        end(ctx or {})
                    except Exception:  # noqa: BLE001 - housekeeping only
                        pass
        final = dict(final, p=p, role="assistant", final=True, stats=stats)
        emit(final)
        self._finish_prop(user_msg, final, lines)
        self.last_stats = stats
        return final["code"]

    def _single(self, base: list[dict], p: int):
        t0 = self.clock()
        try:
            r = (self._mock_reply(base, p, 1, True) if self.mock_policy is not None else
                 self._http_reply(base, force=True, max_tokens=self.max_tokens, timeout=self.timeout_s))
        except _ContextError as e:
            raise EditorReplyError(f"editor request rejected: {e}") from None
        self.n_calls += 1
        code = extract_python(r.content)
        stats = self._stats0(stop="answer" if code else ("length" if r.finish == "length" else "no_code"), usage=r.usage)
        stats.update(finish_reason=r.finish, wall_s=round(self.clock() - t0, 2))
        return {"content": r.content, "reasoning": r.reasoning, "code": code, "usage": r.usage,
                "finish_reason": r.finish}, stats

    def _run_loop(self, base: list[dict], p: int, ctx: dict, emit):
        t0 = self.clock()
        self._server_context()
        begin = getattr(self.tools, "begin", None)
        if callable(begin):             # e.g. ChatTools: clear explore/<cid>/ of a discarded earlier attempt
            try:
                begin(ctx)
            except Exception:  # noqa: BLE001 - housekeeping only
                pass
        st = {"rounds": 0, "tool_calls": 0, "tool_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
              "wall_s": 0.0, "stop": None, "model_calls": 0, "nudges": 0, "compactions": 0,
              "finish_reason": None, "forced_by": None}
        loop, fallback, forced_by, aggressive, noticed, final_nudged = [], "", None, False, False, False

        def elapsed():
            return self.clock() - t0

        def wall_left():
            return None if not self.proposal_wall_s else float(self.proposal_wall_s) - elapsed()

        def compact(aggr):
            info = self._compact(loop, base, aggressive=aggr)
            if info.pop("changed"):
                st["compactions"] += 1
                emit({"p": p, "role": "event", "kind": "compact", "detail": info})
            return info

        def give_up(why: str):
            """The final turn failed (context / timeout): the fallback block if any, else EditorReplyError."""
            if not fallback:
                raise EditorReplyError(why)
            st.update(stop="fallback", finish_reason=None, forced_by=forced_by, wall_s=round(elapsed(), 2),
                      error=why[:1000])
            return {"content": "", "reasoning": None, "code": fallback, "usage": {}, "finish_reason": None}, st

        while True:
            if forced_by is None:
                if st["rounds"] >= self.tool_rounds:
                    forced_by = "rounds"
                elif st["tool_calls"] >= self.max_tool_calls:
                    forced_by = "tool_calls"
                elif self.proposal_tokens and st["completion_tokens"] >= int(self.proposal_tokens):
                    forced_by = "tokens"
                elif wall_left() is not None and wall_left() <= 0:
                    forced_by = "wall"
            final = forced_by is not None
            if final and not noticed:
                noticed = True
                notice = (f"\n\n{FORCED_MARK} ({FORCED_WHY.get(forced_by, forced_by)}). Do not call any function. "
                          "Reply now with a short plan and exactly one ```python block.]")
                tgt = next((e for e in reversed(loop) if e["t"] in ("tool", "nuser")), None)
                if tgt is not None:
                    tgt["notice"] = notice
                emit({"p": p, "role": "event", "kind": "forced_final",
                      "detail": {"reason": forced_by, "rounds": st["rounds"], "notice": notice if tgt else None}})
            cap = self.max_tokens if final else self.turn_max_tokens
            msgs = base + self._render(loop)
            mt = cap
            if self.context_tokens:
                est = self._estimate(msgs)
                if est > self.compact_at * self.context_tokens:
                    compact(False)
                    msgs = base + self._render(loop)
                    est = self._estimate(msgs)
                if self.context_tokens - est - 512 < 2048:
                    compact(True)
                    msgs = base + self._render(loop)
                    est = self._estimate(msgs)
                room = self.context_tokens - est - 512
                if room < 2048 and not final:
                    forced_by = "context"
                    continue
                mt = max(256, min(cap, room))
            wl = wall_left()
            if final or wl is None:     # the final turn always gets the full timeout_s
                timeout = float(self.timeout_s)
            else:
                timeout = max(60.0, min(float(self.timeout_s), max(0.0, wl) + 600))
            try:
                r = (self._mock_reply(msgs, p, st["rounds"] + 1, final) if self.mock_policy is not None else
                     self._http_reply(msgs, force=final, max_tokens=mt, timeout=timeout, retry_timeout=False))
            except _ContextError as e:
                emit({"p": p, "role": "event", "kind": "http400", "detail": {"error": str(e)[:1000], "final": final,
                                                                             "aggressive_done": aggressive}})
                self._ratio *= 1.25
                if not aggressive:
                    aggressive = True
                    compact(True)
                    continue
                if not final:
                    forced_by = "http400"
                    continue
                return give_up(f"editor request rejected even after compaction: {e}")
            except _ReplyTimeout as e:
                emit({"p": p, "role": "event", "kind": "timeout", "detail": {"error": str(e)[:1000], "final": final,
                                                                             "timeout_s": round(timeout, 1)}})
                if not final:
                    forced_by = "timeout"
                    continue
                return give_up(f"editor reply timed out after {timeout:.0f}s on the final turn: {e}")
            self.n_calls += 1
            st["model_calls"] += 1
            u = r.usage or {}
            st["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            st["completion_tokens"] += int(u.get("completion_tokens") or 0)
            if u.get("prompt_tokens"):
                self._ratio = max(0.05, float(u["prompt_tokens"]) / max(1, len(json.dumps(msgs, ensure_ascii=False))
                                                                          + len(json.dumps(self.tools.schemas()))))
            if r.tool_calls and not final:
                st["rounds"] += 1
                rr = st["rounds"]
                blk = extract_python(r.content)
                if blk:
                    fallback = blk
                loop.append({"t": "asst", "r": rr, "content": r.content, "reasoning": r.reasoning,
                             "tool_calls": r.tool_calls, "drop_reasoning": False, "elided_args": set()})
                emit({"p": p, "r": rr, "role": "assistant", "content": r.content, "reasoning": r.reasoning,
                      "tool_calls": r.tool_calls, "usage": r.usage, "finish_reason": r.finish})
                left = self.tool_rounds - rr
                for i, tc in enumerate(r.tool_calls):
                    wl = wall_left()
                    if st["tool_calls"] >= self.max_tool_calls:
                        res = {"content": error_json("skipped: tool-call cap of this proposal reached"), "ok": False,
                               "seconds": 0.0, "args": None}
                    elif wl is not None and wl <= 0:
                        res = {"content": error_json("skipped: time limit of this proposal reached"), "ok": False,
                               "seconds": 0.0, "args": None}
                    else:
                        st["tool_calls"] += 1
                        res = self._run_tool(tc, ctx, rr, i, wl)
                    st["tool_s"] = round(st["tool_s"] + float(res.get("seconds") or 0.0), 3)
                    full = str(res.get("content") or "")
                    sent = cut(full, self.tool_result_chars)
                    if 0 < left <= 5:
                        sent += f"\n[{left} tool round{'s' if left != 1 else ''} left]"
                    loop.append({"t": "tool", "r": rr, "id": tc["id"], "name": tc["name"] or "?", "content": sent,
                                 "elided": None, "notice": ""})
                    emit({"p": p, "r": rr, "role": "tool", "tool_call_id": tc["id"], "name": tc["name"],
                          "args": res.get("args"), "content": sent, "full_chars": len(full), "ok": bool(res.get("ok")),
                          "seconds": res.get("seconds")})
                continue
            code = extract_python(r.content)
            if not code and ((not final and st["nudges"] < 2) or (final and not final_nudged)):
                reason = ("length" if r.finish == "length" else
                          "tool_text" if ("<tool_call>" in r.content or "<function=" in r.content) else
                          "empty" if not r.content.strip() else None)
                if reason:
                    st["nudges"] += 1
                    final_nudged = final_nudged or final      # one nudge on the forced final turn
                    text = NUDGE["tool_text_final" if final and reason == "tool_text" else reason]
                    loop += [{"t": "nasst", "content": r.content},
                             {"t": "nuser", "content": text, "notice": ""}]
                    emit({"p": p, "r": st["rounds"], "role": "assistant", "content": r.content, "reasoning": r.reasoning,
                          "usage": r.usage, "finish_reason": r.finish, "nudge": reason})
                    emit({"p": p, "role": "event", "kind": "nudge", "detail": {"reason": reason, "final": final}})
                    emit({"p": p, "role": "user", "content": text, "nudge": True})
                    continue
            break
        if code:
            stop = f"forced:{forced_by}" if final else "answer"
        elif fallback:
            code, stop = fallback, "fallback"
        else:
            stop = "length" if r.finish == "length" else "no_code"
        st.update(stop=stop, finish_reason=r.finish, forced_by=forced_by, wall_s=round(elapsed(), 2))
        return {"content": r.content, "reasoning": r.reasoning, "code": code, "usage": r.usage,
                "finish_reason": r.finish}, st

    def _run_tool(self, tc: dict, ctx: dict, r: int, i: int, deadline_s) -> dict:
        try:
            res = self.tools.call(tc.get("name"), tc.get("arguments"), ctx, r, i, deadline_s=deadline_s)
            if not isinstance(res, dict) or "content" not in res:
                res = {"content": str(res), "ok": True, "seconds": 0.0, "args": None}
            return res
        except Exception as e:  # noqa: BLE001 - tools never stop the proposal
            return {"content": error_json(e), "ok": False, "seconds": 0.0, "args": None}
