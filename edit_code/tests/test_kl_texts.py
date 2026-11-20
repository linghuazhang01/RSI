"""Paper A KL / calibration texts: prompt + M_0's own response (kl.self_generated) and the per-candidate subset.

  python -I -m pytest -q edit_code/tests/test_kl_texts.py
"""
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

from edit_code.problem import QuantProblem, add_self_generated  # noqa: E402


class _Fake(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.requests.append(body)
        choices = [{"index": i, "text": f" answer to {p[-12:]}", "finish_reason": "stop" if i % 2 == 0 else "length"}
                   for i, p in enumerate(body["prompt"])]
        b = json.dumps({"choices": list(reversed(choices))}).encode()   # out of order on purpose
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class SelfGeneratedTest(unittest.TestCase):
    def setUp(self):
        _Fake.requests = []
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.srv.shutdown()

    def test_prompt_plus_response_batched_and_cached(self):
        texts = [("math", f"<|im_start|>user\nq{i}<|im_end|>\n<|im_start|>assistant\n") for i in range(5)]
        cfg = {"kl": {"self_generated_max_tokens": 2048, "self_generated_batch": 2}}
        calls = []

        def endpoint():
            calls.append(1)
            return self.url

        out = add_self_generated(texts, endpoint, "m0", cfg, self.tmp / "c.jsonl")
        self.assertEqual(len(_Fake.requests), 3)                         # 5 prompts in batches of 2
        self.assertEqual(_Fake.requests[0]["max_tokens"], 2048)
        self.assertEqual(_Fake.requests[0]["temperature"], 0.0)
        for (d, t), (d2, o) in zip(texts, out):
            self.assertEqual(d, d2)
            self.assertTrue(o.startswith(t))                              # prompt first, then the response
            self.assertIn(f" answer to {t[-12:]}", o)                      # matched by index, not by order
        self.assertTrue(out[0][1].endswith("<|im_end|>"))                  # finished (index 0 of its batch)
        self.assertFalse(out[1][1].endswith("<|im_end|>"))                 # cut by max_tokens
        again = add_self_generated(texts, endpoint, "m0", cfg, self.tmp / "c.jsonl")
        self.assertEqual(again, out)
        self.assertEqual((len(calls), len(_Fake.requests)), (1, 3))       # resume: no server, no request


class EvalSubsetTest(unittest.TestCase):
    def test_evenly_spaced_subset_keeps_every_domain(self):
        texts = [(d, f"{d}{i}") for d in ("math", "code", "knowledge", "if", "tool") for i in range(50)]
        prob = SimpleNamespace(kl_texts=texts, cfg={"kl": {"eval_texts": 25}})
        sub = QuantProblem.kl_eval_texts(prob)
        self.assertEqual(len(sub), 25)
        self.assertEqual({d for d, _ in sub}, {"math", "code", "knowledge", "if", "tool"})
        self.assertEqual(sum(d == "math" for d, _ in sub), 5)
        prob.cfg = {"kl": {"eval_texts": None}}
        self.assertEqual(QuantProblem.kl_eval_texts(prob), texts)


if __name__ == "__main__":
    unittest.main()
