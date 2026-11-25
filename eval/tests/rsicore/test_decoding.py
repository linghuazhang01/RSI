"""Decoding plumbing (2026-11-21): rsicore.evaluate.score / score_split `decoding=`, config.stage_decoding /
check_decoding; n > 1 per-record means against the mock server; thinking truncation.

  python -I -m pytest -q eval/tests/rsicore/test_decoding.py
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from rsicore import config, draws, evaluate, records, servers  # noqa: E402

SAMPLED = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "thinking": True, "max_tokens": 16384, "n": 4}


class DecodingUnitTest(unittest.TestCase):
    def test_flags(self):
        evaluate.set_max_tokens({"dapo_math": 4096})
        self.addCleanup(evaluate.set_max_tokens, None)
        f = evaluate._decoding_flags
        self.assertEqual(f("dapo_math", {}, ()), ["--max-tokens", "4096"])            # the table, as before
        self.assertEqual(f("dapo_math", dict(SAMPLED, seed=7), ()),
                         ["--seed", "7", "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20", "--n", "4",
                          "--thinking", "on", "--max-tokens", "16384"])               # explicit max_tokens wins
        self.assertEqual(f("dapo_math", {"thinking": False, "n": 2}, ["--n", "3", "--max-tokens", "9"]),
                         ["--thinking", "off"])                                        # extra wins

    def test_check_and_stage(self):
        self.assertIsNone(config.check_decoding(None))
        self.assertIsNone(config.check_decoding({"n": None}))
        self.assertEqual(config.check_decoding({"min_p": 0.0, "n": 2}), {"min_p": 0.0, "n": 2})
        for bad in ({"min_p": 0.05}, {"nope": 1}, {"n": 0}, {"thinking": "yes"}, {"max_tokens": 1.5}, {"top_p": -1}):
            with self.assertRaises(SystemExit, msg=bad):
                config.check_decoding(bad)
        cfg = {"decoding": {"screen": dict(SAMPLED, n=1, seed=None), "gate": dict(SAMPLED)}}
        self.assertEqual(config.stage_decoding(cfg, "screen", 42), dict(SAMPLED, n=1, seed=42))
        self.assertEqual(config.stage_decoding(cfg, "full", 42), dict(SAMPLED, n=1, seed=42))   # full -> screen
        self.assertEqual(config.stage_decoding(cfg, "gate", 5)["seed"], 5)
        self.assertIsNone(config.stage_decoding({}, "gate", 42))
        self.assertIsNone(config.stage_decoding({"decoding": {"screen": None}}, "screen", 42))
        with self.assertRaises(ValueError):
            config.stage_decoding(cfg, "heldout", 42)

    def test_run_config(self):
        """decoding blocks are validated at load time and a null seed becomes the run's --seed (locked paper_cfg)."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))

        def load(*sets):
            argv = ["--domain", "math", "--mock", "--runs-dir", tmp, "--seed", "9", "--run-id", f"r{abs(hash(sets))}"]
            for x in sets:
                argv += ["--set", x]
            return config.load_run_config("B", config.build_arg_parser("B").parse_args(argv))

        rc = load("decoding.gate={temperature: 0.6, n: 4, thinking: true, max_tokens: 16384}")
        self.assertEqual(rc.paper_cfg["decoding"]["gate"]["seed"], 9)
        self.assertEqual(config.stage_decoding(rc.paper_cfg, "gate", rc.seed)["n"], 4)
        # (2026-11-22) the simple decoding is base.yaml's own now: screen keeps it, its null seed becomes --seed
        self.assertEqual(config.stage_decoding(rc.paper_cfg, "screen", rc.seed)["seed"], 9)
        self.assertIsNone(config.stage_decoding(load("decoding.screen=null").paper_cfg, "screen", 9))
        for bad in ("decoding.final={n: 2}", "decoding.gate={min_p: 0.1}"):
            with self.assertRaises(SystemExit, msg=bad):
                load(bad)

    def test_thinking_truncation(self):
        """thinking on: a sample cut at max_tokens while still thinking (no reasoning, no </think>) scores 0."""
        out = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(out, ignore_errors=True))
        recs = [{"id": "a", "sample_scores": [1.0, 1.0], "score": 1.0, "finish_reason": ["length", "length"],
                 "raw_output": ["thinking ... 42", "</think> 42"], "reasoning": [None, None]},
                {"id": "b", "sample_scores": [1.0], "score": 1.0, "finish_reason": ["length"],
                 "raw_output": "answer 42", "reasoning": "worked it out"},
                {"id": "c", "sample_scores": [0.0], "score": 0.0, "finish_reason": ["stop"], "raw_output": "",
                 "reasoning": None}]
        s = evaluate._normalise("t", {"accuracy": 2 / 3}, recs, True, out)
        self.assertEqual([r["score"] for r in recs], [0.5, 1.0, 0.0])
        self.assertEqual(recs[0]["scores"], [0.0, 1.0])
        self.assertEqual((s["accuracy"], s["truncated_thinking"]), (0.5, 1))
        self.assertEqual(json.loads((out / "summary.json").read_text())["t"]["accuracy"], 0.5)
        recs2 = [dict(recs[1], score=1.0, sample_scores=[1.0])]
        self.assertEqual(evaluate._normalise("t", {"accuracy": 1.0}, recs2, False, out), {"accuracy": 1.0})


class DecodingMockTest(unittest.TestCase):
    """score / score_split with a decoding dict against the mock server (n = 4: per-record means)."""

    def test_mock_passthrough(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        reg = tmp / "mock_registry.json"
        records.write_json(reg, {"base": 0.5})
        ids = draws.build_eval_set("simple_math_test")[:40]
        with servers.VLLMServer(tmp / "model", "base", [0], mock=True, mock_registry=reg, seed=42, log_dir=tmp / "logs",
                                extra_args=["--tasks", "dapo_math"]) as srv:
            dec = dict(SAMPLED, seed=42)
            a = evaluate.score(srv.endpoint, "base", "simple_math_test", pool="pool", ids=ids, out_dir=tmp / "a",
                               mock=True, decoding=dec)
            b = evaluate.score_split([srv.endpoint, srv.endpoint], "base", "simple_math_test", pool="pool", ids=ids,
                                     out_dir=tmp / "b", mock=True, decoding=dec)
            g = evaluate.score(srv.endpoint, "base", "simple_math_test", pool="pool", ids=ids, out_dir=tmp / "g",
                               mock=True)
            s2 = evaluate.score(srv.endpoint, "base", "simple_math_test", pool="pool", ids=ids, out_dir=tmp / "s2",
                                mock=True, decoding=dict(dec, seed=43))
        d = a.summary["decoding"]
        self.assertEqual((d["temperature"], d["top_p"], d["top_k"], d["n"], d["thinking"], d["seed"]),
                         (0.6, 0.95, 20, 4, True, 42))
        self.assertEqual(d["effective_max_tokens"], 16384)
        self.assertEqual(len(a.records), 40)
        for r in a.records:
            self.assertEqual(len(r["scores"]), 4)
            self.assertAlmostEqual(r["score"], sum(r["scores"]) / 4)
        self.assertTrue(any(0 < r["score"] < 1 for r in a.records))           # samples differ per choice index
        self.assertAlmostEqual(a.accuracy, sum(r["score"] for r in a.records) / 40)
        self.assertEqual(a.summary["avg_at_n"], a.accuracy)
        self.assertEqual([(r["id"], r["scores"]) for r in b.records], [(r["id"], r["scores"]) for r in a.records])
        self.assertAlmostEqual(b.accuracy, a.accuracy)                         # same seed -> same samples
        self.assertEqual((b.summary["n"], b.summary["split_over_replicas"]), (40, 2))
        self.assertAlmostEqual(b.summary["avg_at_n"], a.accuracy)
        self.assertEqual(evaluate.paired_delta(a.records, b.records)["delta"], 0.0)
        self.assertEqual(g.summary["decoding"]["temperature"], 0.0)            # None = tasks.yaml (greedy, n 1)
        self.assertTrue(all("scores" not in r for r in g.records))
        self.assertNotEqual([r["scores"] for r in s2.records], [r["scores"] for r in a.records])


if __name__ == "__main__":
    unittest.main(verbosity=2)
