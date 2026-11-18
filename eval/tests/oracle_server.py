"""Tiny OpenAI-compatible "oracle" server for GPU-free end-to-end tests of run_eval.py.

It rebuilds the exact requests run_eval.py would send for the given tasks, indexes them by their
user/system message contents, and answers with the gold response (--mode gold) or a deliberately
wrong one (--mode wrong). It also plays the optional LLM judge (MT-Bench ratings / AlpacaEval m|M).

  eval/.venv/bin/python -I eval/tests/oracle_server.py --tasks gsm8k,mmlu_pro --mode gold --port 8011

Endpoints: POST /v1/chat/completions, GET /v1/models, GET /stats. Binds 127.0.0.1 only.
"""
import argparse
import hashlib
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from evallib.common import EVAL_DIR, load_items  # noqa: E402
from evallib.tasks import load_tasks  # noqa: E402

GOLD_FREEFORM = "A helpful, well-structured answer."


def msg_key(messages):
    keep = [(m["role"], m["content"]) for m in messages if m.get("role") in ("system", "user")]
    return hashlib.sha256(json.dumps(keep, ensure_ascii=False).encode()).hexdigest()


class Oracle:
    def __init__(self, task_names, mode):
        self.mode = mode
        tasks = load_tasks(yaml.safe_load(open(EVAL_DIR / "configs" / "tasks.yaml")))
        self.index = {}
        self.stats = {"requests": 0, "hits": 0, "misses": 0, "judge": 0, "collisions": 0}
        self.lock = threading.Lock()
        for tn in task_names:
            t = tasks[tn]
            for it in load_items(t.dataset):
                for stage in range(t.stages):
                    prev = {"turn1": [{"content": GOLD_FREEFORM}]} if stage == 1 else {}
                    for uid, req in t.units(it, stage, prev):
                        k = msg_key(req["messages"])
                        if k in self.index and self.index[k][1]["id"] != it["id"]:
                            self.stats["collisions"] += 1
                            continue
                        self.index[k] = (t, it, uid)
        print(f"oracle: indexed {len(self.index)} requests for {task_names} mode={mode} "
              f"(collisions skipped: {self.stats['collisions']})", flush=True)

    def judge_answer(self, messages):
        text = "\n".join(m["content"] for m in messages)
        if "## Best Model Identifier" in text:  # AlpacaEval
            m_out = re.search(r'"model_identifier": "m",\s*"output": """(.*?)"""', text, re.S)
            # in gold mode the model output is GOLD_FREEFORM and should win; in wrong mode it is "no." and loses
            if self.mode == "gold":
                return "m" if (m_out and GOLD_FREEFORM in m_out.group(1)) else "M"
            return "M" if (m_out and m_out.group(1).strip() == "no.") else "m"
        if "[[rating]]" in text or "Rating: [[" in text:
            return "The response is evaluated. Rating: [[10]]" if self.mode == "gold" else "Rating: [[1]]"
        return None

    def respond(self, req):
        with self.lock:
            self.stats["requests"] += 1
        msgs = req["messages"]
        n = int(req.get("n", 1) or 1)
        ja = self.judge_answer(msgs)
        logprobs = None
        if ja is not None:
            with self.lock:
                self.stats["judge"] += 1
            out = {"content": ja}
            if req.get("logprobs") and ja in ("m", "M"):
                other = "M" if ja == "m" else "m"
                logprobs = {"content": [{"token": ja, "logprob": -0.01,
                                         "top_logprobs": [{"token": ja, "logprob": -0.01}, {"token": other, "logprob": -4.6}]}]}
        else:
            hit = self.index.get(msg_key(msgs))
            if hit is None:
                with self.lock:
                    self.stats["misses"] += 1
                out = {"content": "UNKNOWN REQUEST"}
            else:
                with self.lock:
                    self.stats["hits"] += 1
                t, it, uid = hit
                out = t.oracle(it, uid, self.mode)
        choices = []
        for i in range(n):
            m = {"role": "assistant", "content": out.get("content", "")}
            if out.get("tool_calls"):
                m["tool_calls"] = out["tool_calls"]
            ch = {"index": i, "message": m, "finish_reason": "tool_calls" if out.get("tool_calls") else "stop"}
            if logprobs:
                ch["logprobs"] = logprobs
            choices.append(ch)
        return {"id": "oracle-" + hashlib.md5(json.dumps(req, sort_keys=True).encode()).hexdigest()[:12],
                "object": "chat.completion", "created": int(time.time()), "model": req.get("model"),
                "choices": choices, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def make_handler(oracle):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": "oracle", "object": "model"}]})
            if self.path.rstrip("/") == "/stats":
                return self._send(200, oracle.stats)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._send(200, oracle.respond(req))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--mode", choices=["gold", "wrong"], default="gold")
    ap.add_argument("--port", type=int, default=8011)
    a = ap.parse_args()
    oracle = Oracle([t for t in a.tasks.split(",") if t], a.mode)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(oracle))
    print(f"oracle listening on http://127.0.0.1:{a.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
