"""Using every GPU (2026-11-20): split scoring over replicas, parallel held-out tasks, the two-phase search
(editor phase / eval phase), parallel gate submissions, and vLLM sleep mode (mock: state; real: command / env).

  python -I -m pytest -q eval/tests/rsicore/test_gpu_use.py      (mock servers, no GPU)
"""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from rsicore import evaluate, heldout, search, servers  # noqa: E402


def _pool_ids(n):
    return json.loads((ROOT / "data" / "splits" / "dapo_math" / "pool.json").read_text())[:n]


class ReplicaScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gpu_use_"))
        reg = cls.tmp / "reg.json"
        reg.write_text(json.dumps({"m": 0.7}))
        cls.srvs = [servers.VLLMServer("mock-m", "m", [i], mock=True, mock_registry=reg, seed=42,
                                       log_dir=cls.tmp / "logs", extra_args=["--tasks", "dapo_math,math500,gsm8k,math500_eval"])
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

    def test_heldout_tasks_in_parallel(self):
        eps = [s.endpoint for s in self.srvs]
        par = heldout.run_heldout(eps, "m", self.tmp / "hpar", domains="math", mock=True, limit=5)
        seq = heldout.run_heldout(eps[0], "m", self.tmp / "hseq", domains="math", mock=True, limit=5)
        self.assertEqual(list(par), [t for t, _ in heldout.HELDOUT["math"]])          # HELDOUT order
        self.assertEqual({t: v["accuracy"] for t, v in par.items()}, {t: v["accuracy"] for t, v in seq.items()})
        allsum = json.loads((self.tmp / "hpar" / "summary.json").read_text())
        self.assertEqual(set(allsum), set(par))
        again = heldout.run_heldout(eps, "m", self.tmp / "hpar", domains="math", mock=True, limit=5)   # resume
        self.assertEqual(again, par)

    def test_single_task_split_over_replicas(self):
        """eval_target (one task, 2026-11-21): the items are split over every replica, same scores, resumable."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("bet", ROOT / "data" / "scripts" / "build_eval_target.py")
        bet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bet)
        try:
            bet.build("math500")                                                       # -> math500_eval
        except SystemExit:
            self.skipTest("math500 source data not present")
        cfg = {"heldout": {"domains": "eval_target"}, "eval_target": {"dataset": "math500_eval"}}
        dom = heldout.heldout_domains(cfg)
        self.assertEqual(heldout.heldout_tasks(dom), [("math500_eval", "pool")])
        self.assertNotIn(("math500_eval", "pool"), heldout.heldout_tasks("all"))
        eps = [s.endpoint for s in self.srvs]
        sp = heldout.run_heldout(eps, "m", self.tmp / "hsplit", domains=dom, mock=True, limit=9)
        seq = heldout.run_heldout(eps[0], "m", self.tmp / "hseq1", domains=dom, mock=True, limit=9)
        summ = json.loads((self.tmp / "hsplit" / "summary.json").read_text())["math500_eval"]
        self.assertEqual((summ["split_over_replicas"], summ["n"], summ["limit"]), (3, 9, 9))
        self.assertEqual(sp["math500_eval"]["accuracy"], seq["math500_eval"]["accuracy"])
        again = heldout.run_heldout(eps, "m", self.tmp / "hsplit", domains=dom, mock=True, limit=9)   # resume
        self.assertEqual(again, sp)
        self.assertEqual(len(list((self.tmp / "hsplit").glob("_part*"))), 3)

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
    def test_phases_and_parallel_gate(self):
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
                                     screen_fn=screen, full_fn=lambda a, c: (0.5, {}), threads=3, steps=2,
                                     max_candidates=6, top_full=2, parallel=2, phases=Phases(), out_dir=rc.run_dir)
        self.assertEqual(len(ranked), 6)
        kinds = [e[0] for e in events]
        # per step: editor phase, all proposals, eval phase, all builds; the search ends in the eval phase
        first = kinds[:kinds.index("eval")]
        self.assertEqual(first[0], "editor")
        self.assertEqual(first.count("propose"), 3)
        self.assertEqual(kinds[-1], "eval")
        self.assertEqual(kinds.count("editor"), 2)
        seen = []

        def gate_fn(c):
            with lock:
                seen.append(c.cid)
            time.sleep(0.05)
            return c.cid == ranked[1].cid, {}
        t0 = time.time()
        acc = search.submit_to_gate(rc, 1, ranked, gate_fn, max_submissions=3, out_dir=rc.run_dir, parallel=3)
        self.assertLess(time.time() - t0, 0.14)                                         # evaluated at once
        self.assertEqual(acc.cid, ranked[1].cid)                                        # first pass in rank order
        log = json.loads((rc.run_dir / "gate.json").read_text())
        self.assertEqual([s["cid"] for s in log["submissions"]], [ranked[0].cid, ranked[1].cid])


if __name__ == "__main__":
    unittest.main(verbosity=2)
