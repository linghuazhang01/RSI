"""Mock OpenAI-compatible server for GPU-free runs of the whole RSI loop (INTERFACES.md §5).

  eval/.venv/bin/python -I eval/rsicore/mock_server.py --port 8123 --served-name base \
      --registry run/mock_registry.json [--tasks dapo_math,math500] [--enable-lora] [--seed 42]

* /v1/chat/completions: requests built by eval/run_eval.py are recognised through an index of
  (system+user messages) -> (task, item, unit), built like eval/tests/oracle_server.py and cached under
  eval/.cache/mock_index/. Each choice is the gold reply with probability `quality` and a wrong reply
  otherwise, decided by u = sha256(seed, request, choice index) < quality (deterministic). The served name is
  NOT hashed (common random numbers): every model sees the same u per item, so on the same items a model
  with higher quality is right on a superset of the items a lower-quality model gets right, and scores
  move monotonically with quality (2026-11-19). Greedy requests
  (temperature 0 / absent) ignore the request seed; sampled requests (temperature > 0) also hash the
  request's `seed` and `temperature`, so repeated evaluations with different seeds differ (noise band)
  while each (seed, temperature, prompt) stays deterministic. ifeval_train gets a
  constructed reply satisfying its single constraint (rsicore/_mock_answers.py); held-out IFEval/IFBench
  only have the hand-made fixtures, so most of their items score 0 in mock mode.
  Editor chats (metadata.mock_editor, or a model name ending in "editor") get a trivial program.
* quality: registry JSON {name: value} for served names and LoRA names, re-read whenever the file
  changes (builders register candidates while the server runs); default 0.7. value = float, or a dict
  {"default": q, <domain>: q, <task>: q} (task > domain > default; domain = the task's `category` in
  configs/tasks.yaml), optionally wrapped as {"quality": value, ...} (rsicore/mock_quality.py).
* /v1/completions with `prompt_logprobs: k`: deterministic fake vLLM-format prompt logprobs over a
  word/punctuation "tokenisation"; logits = base + 3 * (1 - q) * noise with q = mock_quality.scalar
  (mean over the five domains; = the float / "default" when nothing else is given), so KL between two names
  is 0 for equal quality and grows with the quality gap (KL to the 0.7 default grows as quality falls).
* /v1/load_lora_adapter, /v1/unload_lora_adapter (only with --enable-lora, like vLLM), /v1/models,
  /health, /stats. Unknown model names get vLLM's 404. Binds 127.0.0.1 only.
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL))

import yaml  # noqa: E402

from evallib.common import EVAL_DIR, PROC_DIR, load_items  # noqa: E402
from evallib.tasks import load_tasks  # noqa: E402
from rsicore import mock_quality as MQ  # noqa: E402

TRAIN_TASKS = ["dapo_math", "taco_verified", "triviaqa_train", "ifeval_train", "sealtool_train"]
HELDOUT_TASKS = ["math500", "gsm8k", "aime24", "aime25", "humanevalplus", "mbppplus", "livecodebench",
                 "gpqa_diamond", "mmlu_pro", "triviaqa", "ifeval", "ifbench", "sealtool", "apibank", "bfcl"]
INDEX_VERSION = 1
EDITOR_REPLY = "```python\ndef main(api):\n    return {}\n```"


def msg_key(messages):
    keep = [(m["role"], m["content"]) for m in messages if m.get("role") in ("system", "user")]
    return hashlib.sha256(json.dumps(keep, ensure_ascii=False).encode()).hexdigest()


def _u(*parts) -> float:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


class Mock:
    def __init__(self, a):
        self.a = a
        self.served = a.served_name
        self.loras: dict[str, str] = {}
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "hits": 0, "misses": 0, "gold": 0, "wrong": 0, "editor": 0,
                      "completions": 0, "lora_loads": 0}
        self.tasks = load_tasks(yaml.safe_load(open(EVAL_DIR / "configs" / "tasks.yaml")))
        names = [t for t in (a.tasks.split(",") if a.tasks else TRAIN_TASKS + HELDOUT_TASKS) if t]
        if not a.tasks:                                # eval-set targets (objective eval_overfit), when built and
            cfg = yaml.safe_load(open(EVAL_DIR / "configs" / "tasks.yaml"))   # their sources are not indexed already
            for k, src in (cfg.get("eval_targets") or {}).items():            # (same prompts -> same requests)
                if (PROC_DIR / f"{k}_eval.jsonl").exists() and not all(t in names for t, _ in src):
                    names.append(f"{k}_eval")
        self.index = self._index(names)
        self.items: dict[str, dict] = {}
        self.if_ok: dict[str, bool] = {}
        self._reg = (None, {})

    # ------------------------------------------------------------------ index
    def _index(self, names):
        h = hashlib.sha256()
        h.update(f"v{INDEX_VERSION}|{sorted(names)}".encode())
        h.update((EVAL_DIR / "configs" / "tasks.yaml").read_bytes())
        for n in sorted(names):
            st = (PROC_DIR / f"{self.tasks[n].dataset}.jsonl").stat()
            h.update(f"{n}:{st.st_size}:{st.st_mtime_ns}".encode())
        cp = EVAL_DIR / ".cache" / "mock_index" / f"{h.hexdigest()[:24]}.json"
        if cp.exists() and not self.a.no_index_cache:
            idx = json.loads(cp.read_text())
            print(f"mock: loaded index {cp.name} ({len(idx)} requests)", flush=True)
            return idx
        idx = {}
        for n in names:
            t = self.tasks[n]
            for it in load_items(t.dataset):
                for stage in range(t.stages):
                    for uid, req in t.units(it, stage, {}):
                        idx.setdefault(msg_key(req["messages"]), [n, it["id"], uid])
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(f".tmp{id(self)}")
        tmp.write_text(json.dumps(idx))
        tmp.replace(cp)
        print(f"mock: built index for {names}: {len(idx)} requests", flush=True)
        return idx

    def item(self, task, iid):
        ds = self.tasks[task].dataset
        with self.lock:
            if ds not in self.items:
                self.items[ds] = {r["id"]: r for r in load_items(ds)}
            return self.items[ds][iid]

    # ------------------------------------------------------------------ quality registry
    def _reg_value(self, name):
        p = Path(self.a.registry) if self.a.registry else None
        if p is None or not p.exists():
            return None
        mt = p.stat().st_mtime_ns
        with self.lock:
            if self._reg[0] != mt:
                try:
                    self._reg = (mt, json.loads(p.read_text() or "{}"))
                except json.JSONDecodeError:   # being rewritten; keep the previous view
                    pass
            return self._reg[1].get(name)

    def domain_of(self, task) -> str | None:
        t = self.tasks.get(task)
        return MQ.CATEGORY_DOMAIN.get(t.cfg.get("category")) if t is not None else None

    def quality(self, name, task: str | None = None) -> float:
        """Quality of `name` for a request of `task` (None: the scalar quality, used for logprobs)."""
        v = self._reg_value(name)
        if task is None:
            return MQ.scalar(v, self.a.default_quality)
        return MQ.resolve(v, task=task, domain=self.domain_of(task), default=self.a.default_quality)

    def known(self, model) -> bool:
        return model == self.served or model in self.loras

    # ------------------------------------------------------------------ answers
    def gold_reply(self, t, it, uid):
        if t.cfg["grader"] == "rlvr_if":
            from rsicore._mock_answers import rlvr_if_gold
            txt = rlvr_if_gold(it["reference"])
            if it["id"] not in self.if_ok:
                from evallib.graders.rlvr_if import grade_rlvr_if
                try:
                    self.if_ok[it["id"]] = grade_rlvr_if(txt, it["reference"])["score"] == 1.0
                except Exception:  # noqa: BLE001
                    self.if_ok[it["id"]] = False
            return {"content": txt}
        return t.oracle(it, uid, "gold")

    def chat(self, req):
        model = req.get("model")
        n = int(req.get("n", 1) or 1)
        md = req.get("metadata") or {}
        with self.lock:
            self.stats["requests"] += 1
        hit_info = None
        if md.get("mock_editor") or str(model).endswith("editor"):
            with self.lock:
                self.stats["editor"] += 1
            outs = [{"content": EDITOR_REPLY}] * n
        else:
            hit = self.index.get(msg_key(req["messages"]))
            outs = []
            if hit is None:
                with self.lock:
                    self.stats["misses"] += 1
                outs = [{"content": "UNKNOWN REQUEST"}] * n
            else:
                task, iid, uid = hit
                hit_info = (task, iid)
                q = self.quality(model, task)
                t, it = self.tasks[task], self.item(task, iid)
                temp = float(req.get("temperature") or 0.0)
                # common random numbers: u does not depend on the model name, so scores are monotone in quality.
                # greedy: one u per prompt; sampled: u also depends on (seed, temperature)
                samp = ("sampled", req.get("seed"), round(temp, 4)) if temp > 0 else ()
                for i in range(n):
                    good = _u(self.a.seed, task, iid, uid, i, *samp) < q
                    with self.lock:
                        self.stats["hits"] += 1
                        self.stats["gold" if good else "wrong"] += 1
                    if good:
                        outs.append(self.gold_reply(t, it, uid))
                    elif t.cfg["grader"] == "rlvr_if":
                        outs.append({"content": ""})
                    else:
                        outs.append(t.oracle(it, uid, "wrong"))
        # thinking requested (chat_template_kwargs.enable_thinking, 2026-11-21): a short deterministic reasoning_content
        # per sample, like vLLM's qwen3 reasoning parser returns it (run_eval keeps it as the record's "reasoning")
        think = bool((req.get("chat_template_kwargs") or {}).get("enable_thinking")) and hit_info is not None
        choices = []
        for i, o in enumerate(outs):
            rc = f"[mock thinking] {hit_info[0]} item {hit_info[1]} sample {i}: working it out." if think else None
            m = {"role": "assistant", "content": o.get("content", ""), "reasoning_content": rc}
            if o.get("tool_calls"):
                m["tool_calls"] = o["tool_calls"]
            choices.append({"index": i, "message": m, "finish_reason": "tool_calls" if o.get("tool_calls") else "stop"})
        return {"id": "mock-" + hashlib.md5(json.dumps(req, sort_keys=True).encode()).hexdigest()[:12],
                "object": "chat.completion", "created": int(time.time()), "model": model, "choices": choices,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    # ------------------------------------------------------------------ fake prompt logprobs
    def prompt_logprobs(self, text, q, k):
        toks = re.findall(r"\w+|[^\w\s]", text) or [text or " "]
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        amp = 3.0 * (1.0 - q)
        out = [None]
        for j in range(1, len(toks)):
            r = random.Random(f"{sha}:{j}")
            ids = r.sample(range(1000, 150000), k)
            actual = 1000 + int(hashlib.md5(toks[j].encode()).hexdigest(), 16) % 149000
            if actual not in ids:
                ids.append(actual)
            base = [r.gauss(0, 2) for _ in ids]
            nr = random.Random(f"{sha}:{j}:noise")
            logits = [b + amp * nr.gauss(0, 1) for b in base]
            m = max(logits + [1.0])
            z = m + math.log(sum(math.exp(x - m) for x in logits) + math.exp(1.0 - m))
            lps = [x - z for x in logits]
            order = sorted(range(len(ids)), key=lambda i: -lps[i])
            pos = {}
            for rank, i in enumerate(order, 1):
                if rank <= k or ids[i] == actual:
                    pos[str(ids[i])] = {"logprob": lps[i], "rank": rank,
                                        "decoded_token": toks[j] if ids[i] == actual else f"t{ids[i]}"}
            out.append(pos)
        return out

    def completions(self, req):
        model = req.get("model")
        prompts = req.get("prompt")
        prompts = prompts if isinstance(prompts, list) else [prompts]
        k = req.get("prompt_logprobs")
        q = self.quality(model)
        with self.lock:
            self.stats["completions"] += 1
        choices = []
        for i, p in enumerate(prompts):
            p = p if isinstance(p, str) else " ".join(map(str, p))
            ch = {"index": i, "text": " ok", "logprobs": None, "finish_reason": "length", "stop_reason": None}
            if k is not None:
                ch["prompt_logprobs"] = self.prompt_logprobs(p, q, int(k) if int(k) > 0 else 20)
            choices.append(ch)
        return {"id": "mockcmpl-" + hashlib.md5(json.dumps(req, sort_keys=True).encode()).hexdigest()[:12],
                "object": "text_completion", "created": int(time.time()), "model": model, "choices": choices,
                "usage": {"prompt_tokens": 0, "completion_tokens": len(prompts), "total_tokens": 0}}


def make_handler(mock: Mock):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = (obj if isinstance(obj, str) else json.dumps(obj)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain" if isinstance(obj, str) else "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _err(self, code, msg, typ="BadRequestError"):
            self._send(code, {"error": {"message": msg, "type": typ, "code": code}})

        def do_GET(self):
            p = self.path.rstrip("/")
            if p == "/v1/models":
                data = [{"id": mock.served, "object": "model", "root": mock.a.model_path, "parent": None}]
                data += [{"id": n, "object": "model", "root": path, "parent": mock.served} for n, path in mock.loras.items()]
                return self._send(200, {"object": "list", "data": data})
            if p == "/health":
                return self._send(200, "")
            if p == "/stats":
                return self._send(200, dict(mock.stats, loras=dict(mock.loras), served=mock.served))
            self._err(404, "not found", "NotFoundError")

        def do_POST(self):
            p = self.path.rstrip("/")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self._err(400, "invalid JSON")
            if p in ("/v1/load_lora_adapter", "/v1/unload_lora_adapter"):
                if not mock.a.enable_lora:
                    return self._err(400, "LoRA is not enabled (start with --enable-lora)")
                name = req.get("lora_name")
                if p.endswith("/load_lora_adapter"):
                    path = req.get("lora_path") or ""
                    if name in mock.loras:
                        return self._err(400, f"The lora adapter '{name}' has already been loaded.")
                    if not name or not Path(path).exists():
                        return self._err(400, f"lora_path does not exist: {path}")
                    with mock.lock:
                        mock.loras[name] = path
                        mock.stats["lora_loads"] += 1
                    return self._send(200, f"Success: LoRA adapter '{name}' added successfully.")
                if name not in mock.loras:
                    return self._err(404, f"The lora adapter '{name}' cannot be found.", "NotFoundError")
                with mock.lock:
                    mock.loras.pop(name)
                return self._send(200, f"Success: LoRA adapter '{name}' removed successfully.")
            if p not in ("/v1/chat/completions", "/v1/completions"):
                return self._err(404, "not found", "NotFoundError")
            model = req.get("model")
            is_editor = (req.get("metadata") or {}).get("mock_editor") or str(model).endswith("editor")
            if not is_editor and not mock.known(model):
                return self._err(404, f"The model `{model}` does not exist.", "NotFoundError")
            try:
                return self._send(200, mock.chat(req) if p.endswith("chat/completions") else mock.completions(req))
            except Exception as e:  # noqa: BLE001
                return self._err(500, f"{type(e).__name__}: {e}", "InternalServerError")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--registry", default=None, help="JSON {name: quality}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--tasks", default=None, help="comma-separated tasks to index (default: all train + held-out)")
    ap.add_argument("--enable-lora", action="store_true")
    ap.add_argument("--max-loras", type=int, default=8)
    ap.add_argument("--default-quality", type=float, default=0.7)
    ap.add_argument("--no-index-cache", action="store_true")
    a = ap.parse_args()
    mock = Mock(a)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 512   # must be set before listen(); avoids resets under run_eval's concurrency

    srv = Server(("127.0.0.1", a.port), make_handler(mock))
    print(f"mock listening on http://127.0.0.1:{a.port}/v1 served={a.served_name}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
