"""The editor (改造者): a multi-turn chat with the current model M_t that returns programs.

    ed = Editor(endpoint, served_name, system_prompt=SYS, seed=rc.seed + thread,
                transcript=gen_dir / "editor" / f"thread_{thread}.jsonl")
    code = ed.propose(render_context(history, thread))     # last ```python block of the reply ("" if none)

One Editor per thread (its own history). The transcript (jsonl: one line per message with role, content,
reasoning, extracted code, usage) is reloaded on construction; propose() with the same user message as
the next recorded turn replays the recorded reply without calling the model, so a resumed search
reproduces the same programs. A different message truncates the transcript at that point.
mock_policy: callable(history: list[{"role","content"}]) -> program text (or a reply containing a
```python block); used instead of the HTTP call (mock mode).
"""
import json
import os
import re
import time
from pathlib import Path

from . import _net

_BLOCK = re.compile(r"```(?:python|py|python3)[^\n]*\n(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def extract_python(text: str) -> str:
    text = _THINK.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    blocks = _BLOCK.findall(text)
    return blocks[-1].strip("\n") + "\n" if blocks else ""


class Editor:
    def __init__(self, endpoint, served_name, *, system_prompt: str, seed: int, transcript: Path,
                 temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 8192,
                 mock_policy=None, extra_body: dict | None = None, timeout_s: float = 3600):
        self.endpoint, self.served_name = endpoint, served_name
        self.system_prompt, self.seed = system_prompt, seed
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.mock_policy, self.extra_body, self.timeout_s = mock_policy, dict(extra_body or {}), timeout_s
        self.transcript = Path(transcript)
        self.transcript.parent.mkdir(parents=True, exist_ok=True)
        self.history = [{"role": "system", "content": system_prompt}]
        self._lines = []      # transcript lines (dicts) of the current history (after the system line)
        self._replay = []     # recorded (user_line, assistant_line) pairs not yet consumed
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
                lines.append(json.loads(l))
            except json.JSONDecodeError:
                pass
        msgs = [l for l in lines if l.get("role") in ("user", "assistant")]
        pairs = [(msgs[i], msgs[i + 1]) for i in range(0, len(msgs) - 1, 2)
                 if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"]
        self._replay = pairs

    def _write_all(self):
        tmp = self.transcript.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps({"role": "system", "content": self.system_prompt, "served_name": self.served_name,
                                "seed": self.seed, "ts": time.strftime("%F %T")}, ensure_ascii=False) + "\n")
            for l in self._lines:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        os.replace(tmp, self.transcript)

    def _append(self, line: dict):
        self._lines.append(line)
        with open(self.transcript, "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- chat
    def _chat(self) -> tuple[str, str | None, dict]:
        if self.mock_policy is not None:
            text = self.mock_policy([dict(m) for m in self.history])
            text = text or ""
            if "```" not in text:
                text = f"```python\n{text.rstrip()}\n```"
            return text, None, {"mock": True}
        payload = {"model": self.served_name, "messages": self.history, "temperature": self.temperature,
                   "top_p": self.top_p, "max_tokens": self.max_tokens, "seed": self.seed}
        payload.update(self.extra_body)
        last = None
        for attempt in range(5):
            try:
                resp = _net.post_json(self.endpoint.rstrip("/") + "/chat/completions", payload, timeout=self.timeout_s)
                msg = resp["choices"][0]["message"]
                return msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning"), resp.get("usage") or {}
            except (_net.HTTPError, OSError, KeyError, ValueError) as e:
                last = e
                if isinstance(e, _net.HTTPError) and e.status < 500 and e.status not in (408, 429):
                    break
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"editor chat failed: {last}")

    def propose(self, user_msg: str) -> str:
        if self._replay:
            u, a = self._replay[0]
            if u.get("content") == user_msg:
                self._replay.pop(0)
                self.history += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": a["content"]}]
                self._lines += [u, a]
                return a.get("code") if a.get("code") is not None else extract_python(a["content"])
            self._replay = []           # diverged: drop the rest of the recorded conversation
        if self._stale:                 # file may hold unconsumed / torn lines: rewrite to the live history
            self._write_all()
            self._stale = False
        self.history.append({"role": "user", "content": user_msg})
        self._append({"role": "user", "content": user_msg, "ts": time.strftime("%F %T")})
        content, reasoning, usage = self._chat()
        self.n_calls += 1
        code = extract_python(content)
        self.history.append({"role": "assistant", "content": content})
        self._append({"role": "assistant", "content": content, "reasoning": reasoning, "code": code,
                      "usage": usage, "ts": time.strftime("%F %T")})
        return code
