"""End-to-end Paper A orchestration in mock mode (no weights, no GPU), against the real rsicore.
Fixed-size design (2026-11-19): frozen editor (M_0), every candidate quantizes M_0 to one budget,
a generation is accepted only if it beats the incumbent on the gate.

Run with the eval interpreter (needs pyyaml; torch / transformers are NOT needed in mock mode):
    eval/.venv/bin/python -I -m pytest edit_code/tests/test_mock_flow.py -q      (or run the file)
Takes ~3-4 minutes (two mock runs + one resume).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

try:
    import rsicore.search  # noqa: F401
    import yaml  # noqa: F401
    HAVE_CORE = True
except Exception:  # pragma: no cover
    HAVE_CORE = False


def run(args, runs_dir):
    cmd = [sys.executable, "-I", str(REPO / "edit_code" / "run.py"), "--mock", "--runs-dir", str(runs_dir)] + args
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return r, time.time() - t0


def jl(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


@unittest.skipUnless(HAVE_CORE, "eval/rsicore or pyyaml not importable")
class TestMockFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixed_budget_loop_and_resume(self):
        r, dt = run(["--domain", "math", "--gens", "2", "--run-id", "A_mock_test"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_test"
        st = json.loads((rd / "state.json").read_text())
        for k in ("baseline/gate", "baseline/noise", "baseline/heldout", "baseline/prep", "baseline/budget",
                  "baseline/q0", "baseline/q0_heldout", "gen01/refs", "gen01/search", "gen01/gate", "gen01/export",
                  "gen01/heldout", "gen01/done", "gen02/done"):
            self.assertIn(k, st)
        # one budget for every generation = packed size of the reference config (Q_0)
        budget = json.loads((rd / "baseline" / "budget.json").read_text())
        self.assertEqual(budget["source"], "budget.reference")
        self.assertAlmostEqual(budget["gib"], budget["reference_gib"], places=3)
        self.assertLess(budget["fraction_of_bf16"], 0.3)
        q0 = json.loads((rd / "baseline" / "incumbent.json").read_text())
        self.assertEqual(q0["gen"], 0)
        self.assertTrue((rd / "baseline" / "q0" / "heldout.json").exists())
        # the layer cache / diagnostics are built once, from M_0
        self.assertTrue((rd / "prep" / "spec.json").exists())
        self.assertFalse((rd / "gen_01" / "prep").exists())
        for g in (1, 2):
            gd = rd / f"gen_{g:02d}"
            cands = jl(gd / "candidates.jsonl")
            self.assertGreaterEqual(len(cands), 3)
            for c in cands:
                if not c["error"]:
                    self.assertLessEqual(c["artifact"]["bytes_gib"], budget["gib"] + 1e-9)
                    self.assertEqual(c["artifact"]["budget_gib"], budget["gib"])
            inc_scores = json.loads((gd / "incumbent_scores.json").read_text())
            self.assertIsNotNone(inc_scores["screen"])
            self.assertTrue((gd / "editor" / "thread_0.jsonl").exists())
            evals = jl(gd / "evals.jsonl")
            self.assertLessEqual(sum(1 for e in evals if e["stage"] == "screen" and e["charged"]), 4)
            self.assertTrue(any(e["origin"] == "incumbent" for e in evals))
        # generation 1: the mock editor's bit exchange beats the uniform W3 incumbent on the gate
        acc = json.loads((rd / "gen_01" / "accepted.json").read_text())
        self.assertTrue(acc["accepted"], acc)
        self.assertIn("beats the incumbent", acc["gate_reason"])
        inc1 = json.loads((rd / "gen_01" / "incumbent.json").read_text())
        self.assertEqual(inc1["gen"], 1)
        self.assertGreater(inc1["gate_scores"]["dapo_math"], q0["gate_scores"]["dapo_math"])
        self.assertTrue((rd / "gen_01" / "model" / "mock_model.json").exists())
        # the editor is frozen: always M_0
        self.assertIn("[gen 2] budget", r.stdout)
        self.assertIn("editor = M_0 (frozen)", r.stdout)
        summ = json.loads((rd / "summary.json").read_text())
        f = summ["final"]
        self.assertEqual(f["editor"], "frozen")
        self.assertIn(1, f["accepted_generations"])
        self.assertGreater(f["gain_vs_q0"], 0)
        self.assertGreater(f["gap_closed"], 0)
        for g in ("0", "1", "2"):
            self.assertIn("heldout_by_domain", summ["gens"][g])
            self.assertIn("gate_mean", summ["gens"][g])
        led = json.loads((rd / "draws" / "ledger.json").read_text())
        shown = sum(1 for v in led["items"].get("dapo_math", {}).values() for e in v if e[1] == "shown")
        self.assertGreater(shown, 0)
        # resume: everything is skipped
        before = (rd / "state.json").read_text()
        r2, dt2 = run(["--domain", "math", "--gens", "2", "--run-id", "A_mock_test"], self.runs)
        self.assertEqual(r2.returncode, 0, r2.stderr[-3000:])
        self.assertLess(dt2, 30)
        self.assertEqual(json.loads(before).keys(), json.loads((rd / "state.json").read_text()).keys())

    def test_patience_stop_and_unfrozen_editor(self):
        # loop.editor=current (ablation): the incumbent writes; patience 1: the first generation without a
        # new incumbent stops the loop
        r, _ = run(["--domain", "math", "--gens", "4", "--run-id", "A_mock_stop", "--set", "loop.editor=current",
                    "--set", "loop.patience=1", "--set", "heldout.limit=2"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_stop"
        st = json.loads((rd / "state.json").read_text())
        self.assertIn("stopped", st)
        g = st["stopped"]["gen"]
        self.assertLess(g, 4)
        self.assertNotIn(f"gen{g + 1:02d}/draw", st)
        acc = json.loads((rd / f"gen_{g:02d}" / "accepted.json").read_text())
        self.assertFalse(acc["accepted"])
        inc = json.loads((rd / f"gen_{g:02d}" / "incumbent.json").read_text())
        prev = json.loads(((rd / "baseline") if g == 1 else (rd / f"gen_{g - 1:02d}")).joinpath("incumbent.json").read_text())
        self.assertEqual(inc["config_hash"], prev["config_hash"])          # the incumbent stays
        self.assertEqual(json.loads((rd / "summary.json").read_text())["final"]["editor"], "current")

    def test_budget_below_reference_is_refused(self):
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_small", "--budget-gib", "5",
                    "--set", "heldout.limit=2"], self.runs)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("smaller than budget.reference", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
