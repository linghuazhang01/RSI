"""Using every GPU (2026-11-20): split scoring over replicas, the two-phase search
(editor phase / eval phase), and vLLM sleep mode (mock: state; real: command / env).

  python -I -m pytest -q eval/tests/rsicore/test_gpu_use.py      (mock servers, no GPU)
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from rsicore import evaluate, search, servers  # noqa: E402


def _pool_ids(n):
    return json.loads((ROOT / "data" / "splits" / "dapo_math" / "pool.json").read_text())[:n]


class ReplicaScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gpu_use_"))
        reg = cls.tmp / "reg.json"
        reg.write_text(json.dumps({"m": 0.7}))
        cls.srvs = [servers.VLLMServer("mock-m", "m", [i], mock=True, mock_registry=reg, seed=42,
                                       log_dir=cls.tmp / "logs", extra_args=["--tasks", "dapo_math"])
                    for i in range(3)]
        servers.start_all(cls.srvs)

    @classmethod
    def tearDownClass(cls):
        for s in cls.srvs:
            s.stop()

    def test_score_split_matches_single(self):
        ids = _pool_ids(60)
        eps = [s.endpoint for s in self.srvs]
        one = evaluate.score(eps[0], "m", "dapo_math", pool="pool", ids=ids, out_dir=self.tmp / "one", mock=True)
        split = evaluate.score_split(eps, "m", "dapo_math", pool="pool", ids=ids, out_dir=self.tmp / "split", mock=True)
        self.assertEqual(split.summary["split_over_replicas"], 3)
        self.assertEqual(sorted(r["id"] for r in split.records), sorted(ids))           # every item once
        self.assertEqual({r["id"]: r["score"] for r in split.records},                   # greedy: same answers
                         {r["id"]: r["score"] for r in one.records})
        self.assertAlmostEqual(split.summary["accuracy"], one.summary["accuracy"], places=6)
        merged = [json.loads(x) for x in (self.tmp / "split" / "dapo_math.jsonl").read_text().splitlines()]
        self.assertEqual(len(merged), 60)
        self.assertIn("dapo_math", json.loads((self.tmp / "split" / "summary.json").read_text()))
        single = evaluate.score_split(eps[:1], "m", "dapo_math", pool="pool", ids=ids[:5], out_dir=self.tmp / "s1",
                                      mock=True)
        self.assertNotIn("split_over_replicas", single.summary)                        # one replica: plain score


class SleepModeTest(unittest.TestCase):
    def test_command_env_and_mock_state(self):
        s = servers.VLLMServer("/models/x", "x", [0, 1], sleep_mode=True)
        self.assertIn("--enable-sleep-mode", s.command())
        self.assertNotIn("--enable-sleep-mode", servers.VLLMServer("/models/x", "x", [0]).command())
        with tempfile.TemporaryDirectory() as d:
            m = servers.VLLMServer("mock-x", "x", [0], mock=True, sleep_mode=True, log_dir=Path(d))
            m.start()
            try:
                m.sleep()
                self.assertTrue(m.sleeping)
                m.wake_up()
                self.assertFalse(m.sleeping)
            finally:
                m.stop()
            n = servers.VLLMServer("mock-y", "y", [0], mock=True, log_dir=Path(d))
            n.start()
            try:
                with self.assertRaises(RuntimeError):                                   # needs sleep_mode
                    n.sleep()
            finally:
                n.stop()


class TwoPhaseSearchTest(unittest.TestCase):
    def test_phases(self):
        rc = SimpleNamespace(run_dir=Path(tempfile.mkdtemp()))
        events, lock = [], threading.Lock()

        class Ed:
            def __init__(self, th):
                self.th = th

            def propose(self, msg):
                with lock:
                    events.append(("propose", self.th))
                time.sleep(0.05)
                return f"prog {self.th}"

        class Phases:
            def editor_phase(self):
                events.append(("editor",))

            def eval_phase(self):
                events.append(("eval",))
                return 3

        def build(program, cid):
            with lock:
                events.append(("build", cid))
            return {"cid": cid}, None

        def screen(art, cid):
            return float(cid[-1]) / 10, {}
        ranked = search.inner_search(rc, 1, editor_factory=Ed, render_context=lambda h, th: "", build_fn=build,
                                     screen_fn=screen, threads=3, steps=2,
                                     max_candidates=6, parallel=2, phases=Phases(), out_dir=rc.run_dir)
        self.assertEqual(len(ranked), 6)
        kinds = [e[0] for e in events]
        # per step: editor phase, all proposals, eval phase, all builds; the search ends in the eval phase
        first = kinds[:kinds.index("eval")]
        self.assertEqual(first[0], "editor")
        self.assertEqual(first.count("propose"), 3)
        self.assertEqual(kinds[-1], "eval")
        self.assertEqual(kinds.count("editor"), 2)
        self.assertEqual(kinds.count("eval"), 3)                 # one per step + the final one
        self.assertEqual({c.cid for c in ranked[:3]}, {f"g01_t{t}_s01" for t in range(3)})      # by screen score
        self.assertTrue(all(c.full is None and c.score == c.screen["score"] for c in ranked))
        events.clear()                                           # resumed: nothing to do, still ends in the eval phase
        again = search.inner_search(rc, 1, editor_factory=Ed, render_context=lambda h, th: "", build_fn=build,
                                    screen_fn=screen, threads=3, steps=2, max_candidates=6, parallel=2,
                                    phases=Phases(), out_dir=rc.run_dir)
        self.assertEqual(([c.cid for c in again], events), ([c.cid for c in ranked], [("eval",)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class GDNBackendTest(unittest.TestCase):
    def test_env_override_and_explicit_argument_precedence(self):
        with patch.dict(os.environ, {"RSI_GDN_PREFILL_BACKEND": "triton"}):
            cmd = servers.VLLMServer("/models/x", "x", [0]).command()
            self.assertEqual(cmd[cmd.index("--gdn-prefill-backend") + 1], "triton")
            for args in (["--gdn-prefill-backend", "flashinfer"],
                         ["--gdn-prefill-backend=flashinfer"]):
                cmd = servers.VLLMServer("/models/x", "x", [0], extra_args=args).command()
                self.assertNotIn("triton", cmd)
                self.assertEqual(sum(a.startswith("--gdn-prefill-backend") for a in cmd), 1)
            cmd = servers.VLLMServer("mock-x", "x", [0], mock=True).command()
            self.assertFalse(any(a.startswith("--gdn-prefill-backend") for a in cmd))
        with patch.dict(os.environ, {}, clear=True):
            cmd = servers.VLLMServer("/models/x", "x", [0]).command()
            self.assertNotIn("--gdn-prefill-backend", cmd)
