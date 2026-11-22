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


def evals_of(gd: Path):
    return jl(gd / "evals.jsonl")


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
        # W3 reference (the default is W4 since 2026-11-20): at W4 the mock damage is too small for the mock
        # editor's bit exchange to clear the gate, and this test checks the accept / export / resume path
        r, dt = run(["--domain", "math", "--gens", "2", "--run-id", "A_mock_test",
                     "--set", "budget.reference.default.bits=3"], self.runs)
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
        # greedy noise (default since 2026-11-20): nothing re-run, band 0
        self.assertEqual(json.loads((rd / "baseline" / "noise.json").read_text()), {"dapo_math": 0.0})
        self.assertFalse((rd / "baseline" / "noise").exists())
        q0 = json.loads((rd / "baseline" / "incumbent.json").read_text())
        self.assertEqual(q0["gen"], 0)
        self.assertTrue((rd / "baseline" / "q0" / "heldout.json").exists())
        # the layer cache / diagnostics are built once, from M_0
        self.assertTrue((rd / "prep" / "spec.json").exists())
        self.assertFalse((rd / "gen_01" / "prep").exists())
        diag = rd / "prep" / "diagnostics"
        prof = json.loads((diag / "profile_table.json").read_text())
        self.assertEqual({r["bits"] for r in prof}, {2, 3, 4})                       # 8 bit dropped
        ht = json.loads((diag / "hessian_table.json").read_text())                   # Hessian-weighted error table
        self.assertEqual({(r["quantizer"], r["bits"]) for r in ht},
                         {(q, b) for q in ("rtn", "gptq", "awq") for b in (2, 3, 4)})
        self.assertEqual(len(ht), len({r["key"] for r in prof}) * 9)
        self.assertTrue(all(r["herr_per_token"] >= 0 for r in ht))
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
            # M_0 is only the KL reference: not scored on the items; candidates are compared with the incumbent
            self.assertFalse((gd / "refs" / "screen").exists())
            ok = [c for c in cands if not c["error"] and c.get("screen")]
            self.assertTrue(ok)
            for c in ok:
                pr = c["screen"]["feedback"]["probes"]["math"]
                self.assertIn("vs_incumbent", pr)
                self.assertNotIn("parent", pr)
                self.assertTrue(c["screen"]["feedback"]["error_top_groups"])
                self.assertNotIn("per_text", c["screen"]["feedback"]["kl"])          # kept on disk instead
                edir = gd / "evals" / c["cid"] / "screen"
                if not edir.exists():                       # identical config: shares an earlier cid's records
                    continue
                self.assertEqual(len(json.loads((edir / "kl_per_text.json").read_text())), 400)
                ge = json.loads((edir / "group_error.json").read_text())
                self.assertAlmostEqual(sum(v["share"] for v in ge["groups"].values()), 1.0, places=2)
            self.assertTrue(any("group_error top" in (c["artifact"] or {}).get("stdout_tail", "")
                                and "compare vs incumbent" in (c["artifact"] or {}).get("stdout_tail", "")
                                for c in ok), "the mock editor's analysis tool calls")
            ie = [e for e in evals_of(gd) if e["origin"] == "incumbent" and e["stage"] == "screen"]
            self.assertEqual(ie[0]["feedback"]["retention"], 1.0)
            self.assertTrue((gd / "editor" / "thread_0.jsonl").exists())
            evals = jl(gd / "evals.jsonl")
            self.assertLessEqual(sum(1 for e in evals if e["stage"] == "screen" and e["charged"]), 4)
            self.assertTrue(any(e["origin"] == "incumbent" for e in evals))
        # generation 1: the mock editor's bit exchange beats the uniform W4 incumbent on the gate
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
        r2, dt2 = run(["--domain", "math", "--gens", "2", "--run-id", "A_mock_test",
                       "--set", "budget.reference.default.bits=3"], self.runs)
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

    def test_noise_subset_fixed_band_and_data_sizes(self):
        # data.* overrides the draw sizes for the run; noise.n scores a subset of the screen items;
        # noise.fixed replaces the re-runs
        # (noise.temperature=0.6: the sampled re-runs; the default is greedy, which re-runs nothing)
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_sizes", "--set", "noise.n=12",
                    "--set", "noise.temperature=0.6",
                    "--set", "data.n=120", "--set", "data.screen_n=40", "--set", "heldout.limit=2"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_sizes"
        ids = rd / "draws" / "gen_00" / "ids"
        self.assertEqual(len(json.loads((ids / "dapo_math.signal.json").read_text())), 120)
        self.assertEqual(len(json.loads((ids / "dapo_math.screen.json").read_text())), 40)
        det = json.loads((rd / "baseline" / "noise_detail.json").read_text())["dapo_math"]
        self.assertEqual((det["n"], det["source"]), (12, "measured"))
        # single-domain: every one of the kl_contexts (default 400) KL / calibration texts is a math text
        self.assertEqual(len(json.loads((ids / "dapo_math.kl.json").read_text())), 400)
        self.assertEqual(sorted(p.name for p in ids.glob("*.kl.json")), ["dapo_math.kl.json"])
        # kl.self_generated (default): one cached M_0 response per KL text and generation
        sg = (rd / "kl_selfgen" / "gen_00.jsonl").read_text().splitlines()
        self.assertEqual(len(sg), 400)
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_fixed", "--set", "noise.fixed={math: 0.04}",
                    "--set", "heldout.limit=2"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_fixed"
        self.assertEqual(json.loads((rd / "baseline" / "noise.json").read_text()), {"dapo_math": 0.04})
        self.assertFalse((rd / "baseline" / "noise").exists())              # no sampled re-runs

    def test_four_gpus_all_used(self):
        """4 GPUs (2026-11-20): M_0's generation-0 gate scoring split over 4 replicas, held-out tasks spread over
        the GPU groups, the editor's GPU lent as a 4th candidate slot while it sleeps, gate submissions at once."""
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_4gpu", "--gpus", "0,1,2,3",
                    "--set", "hardware=h20", "--set", "heldout.limit=2"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_4gpu"
        self.assertEqual(len(list((rd / "baseline" / "gate").glob("_part*"))), 4)
        self.assertEqual({p.name for p in (rd / "baseline" / "heldout" / "_par").iterdir()},
                         {"math500", "gsm8k", "aime24", "aime25"})
        self.assertTrue((rd / "baseline" / "q0" / "heldout" / "_par").exists())
        self.assertIn("serve as candidate slot 3 (4 slots)", r.stdout)
        self.assertEqual(len(jl(rd / "kl_selfgen" / "gen_00.jsonl")), 400)

    def test_eval_overfit(self):
        """objective eval_overfit: optimise directly on math_eval (MATH-500 + GSM8K test); no draws, no KL, gate /
        full reuse the screen records (same items)."""
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_evalfit", "--set", "objective=eval_overfit",
                    "--set", "heldout.limit=2"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_evalfit"
        self.assertIn("datasets=['math_eval']", r.stdout)
        self.assertFalse((rd / "draws").exists())
        self.assertEqual(json.loads((rd / "baseline" / "gate.json").read_text()).keys(), {"math_eval"})
        # held-out = the same 1819 items (greedy), for M_0 and Q_0
        self.assertEqual(set(json.loads((rd / "baseline" / "heldout.json").read_text())), {"math_eval"})
        self.assertEqual(set(json.loads((rd / "baseline" / "q0" / "heldout.json").read_text())), {"math_eval"})
        ev = jl(rd / "gen_01" / "evals.jsonl")
        scr = [e for e in ev if e["stage"] == "screen" and e["origin"] == "candidate"]
        self.assertTrue(scr)
        pr = scr[0]["feedback"]["probes"]["math_eval"]
        self.assertEqual(pr["n"], 1819)
        self.assertIsNone(scr[0]["feedback"]["kl"])                                   # kl_weight 0: not computed
        for e in ev:
            if e["stage"] in ("full", "gate"):                                        # reused, not re-scored
                self.assertFalse((rd / "gen_01" / "evals" / e["cid"] / e["stage"] / "summary.json").exists())

    def test_data_roles_eval_set_gate(self):
        """data_roles: train stays the dapo_math draws, the gate is every item of math500_eval (really scored, not
        reused from screen), the held-out report is math500 alone."""
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_roles_gate",
                    "--set", "data_roles.gate=math500", "--set", "data_roles.eval=math500"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_roles_gate"
        self.assertIn("datasets=['dapo_math'] gate_datasets=['math500_eval']", r.stdout)
        self.assertTrue((rd / "draws" / "gen_01" / "ids" / "dapo_math.screen.json").exists())
        self.assertEqual(json.loads((rd / "baseline" / "gate.json").read_text()).keys(), {"math500_eval"})
        self.assertEqual(set(json.loads((rd / "baseline" / "heldout.json").read_text())), {"math500"})
        q0 = json.loads((rd / "baseline" / "incumbent.json").read_text())
        self.assertEqual(set(q0["gate_scores"]), {"math500_eval"})
        q0e = [e for e in jl(rd / "baseline" / "q0" / "evals.jsonl") if e["stage"] == "gate"]
        self.assertEqual(q0e[0]["feedback"]["probes"]["math500_eval"]["n"], 500)
        ev = jl(rd / "gen_01" / "evals.jsonl")
        scr = [e for e in ev if e["stage"] == "screen" and e["origin"] == "candidate"]
        self.assertTrue(scr)
        self.assertEqual(set(scr[0]["feedback"]["probes"]), {"math"})            # dapo_math -> domain math
        self.assertEqual(scr[0]["feedback"]["probes"]["math"]["dataset"], "dapo_math")
        for e in ev:
            if e["stage"] == "gate":                                              # scored on all of math500_eval
                pr = e["feedback"]["probes"]["math500_eval"]
                self.assertEqual(pr["n"], 500)
                self.assertTrue((rd / "gen_01" / "evals" / e["cid"] / "gate" / "math500_eval.jsonl").exists())
        inc = json.loads((rd / "gen_01" / "incumbent.json").read_text())
        self.assertEqual(set(inc["gate_scores"]), {"math500_eval"})
        summ = json.loads((rd / "summary.json").read_text())
        self.assertIsNotNone(summ["gens"]["0"]["m0_heldout_by_domain"]["math"])

    def test_data_roles_eval_set_train(self):
        """data_roles: the editor works from all of math500_eval (screen n 500), the gate is the fixed dapo_math
        split; feedback examples show the reference answer (eval-set train)."""
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_roles_train",
                    "--set", "data_roles.train=math500", "--set", "data_roles.gate=dapo_math",
                    "--set", "heldout.limit=2"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_roles_train"
        self.assertIn("datasets=['math500_eval'] gate_datasets=['dapo_math'] kl_datasets=['math500_eval']", r.stdout)
        self.assertEqual(json.loads((rd / "baseline" / "gate.json").read_text()).keys(), {"dapo_math"})
        self.assertEqual(json.loads((rd / "baseline" / "noise.json").read_text()), {"math500_eval": 0.0})
        ev = jl(rd / "gen_01" / "evals.jsonl")
        scr = [e for e in ev if e["stage"] == "screen" and e["origin"] == "candidate"]
        self.assertTrue(scr)
        pr = scr[0]["feedback"]["probes"]["math500_eval"]
        self.assertEqual(pr["n"], 500)
        gates = [e for e in ev if e["stage"] == "gate"]
        self.assertTrue(gates)
        for e in gates:
            self.assertEqual(set(e["feedback"]["probes"]), {"math"})
            self.assertEqual(e["feedback"]["probes"]["math"]["dataset"], "dapo_math")
            self.assertTrue((rd / "gen_01" / "evals" / e["cid"] / "gate" / "dapo_math.jsonl").exists())
        regs = [x for e in ev for x in (e["feedback"].get("regressions") or [])]
        self.assertTrue(all("reference" in x for x in regs))
        led = json.loads((rd / "draws" / "ledger.json").read_text())
        self.assertNotIn("math500_eval", led.get("items") or {})                  # eval sets: nothing marked shown

    def test_simple_objective(self):
        """objective simple, FINAL flow (2026-11-21): train role simple_math_pool with a fresh per-generation screen
        (simple.screen_n), gate on simple_math_eval (500 items, thinking n 4), held-out = simple_math_test (avg@4).
        The incumbent answers the screen items once (n 4; shared, shown via the context summary and screen_results /
        screen_output); candidates are not screened, have no api.evaluate and get ONE gate evaluation each
        (gen_XX/gate.json, one entry per candidate); the largest passing gain is accepted. The mock keeps the simple
        layout (mock_objective_overrides: 3 threads x 1 step)."""
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_simple", "--set", "objective=simple",
                    "--set", "simple.screen_n=20"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_simple"
        self.assertIn("datasets=['simple_math_pool'] gate_datasets=['simple_math_eval'] "
                      "kl_datasets=['simple_math_pool']", r.stdout)
        self.assertEqual(json.loads((rd / "baseline" / "gate.json").read_text()).keys(), {"simple_math_eval"})
        for h in (rd / "baseline" / "heldout.json", rd / "baseline" / "q0" / "heldout.json"):
            ho = json.loads(h.read_text())
            self.assertEqual(set(ho), {"simple_math_test"})                          # the test set, avg@4
            self.assertIn("avg_at_n", ho["simple_math_test"])
        ids = rd / "draws" / "gen_01" / "ids"                                         # per-generation draw
        scr_ids = json.loads((ids / "simple_math_pool.screen.json").read_text())
        self.assertEqual(len(scr_ids), 20)
        ev = jl(rd / "gen_01" / "evals.jsonl")
        n = {(e["stage"], v["dataset"]): v.get("n") for e in ev for v in (e["feedback"].get("probes") or {}).values()}
        self.assertEqual(n.get(("screen", "simple_math_pool")), 20)
        self.assertEqual(n.get(("gate", "simple_math_eval")), 500)
        for e in ev:                                  # thinking decoding, n 4 everywhere (screen, full, gate)
            dec = json.loads(e["decoding"])
            self.assertEqual((dec["thinking"], dec["temperature"], dec["top_k"], dec["max_tokens"], dec["seed"],
                              dec["n"]), (True, 0.6, 20, 16384, 42, 4))
        # the incumbent's screen: sampled once (n 4), the only screen record; candidates are not screened
        self.assertEqual(sorted(e["stage"] for e in ev if e["origin"] == "incumbent"), ["full", "screen"])
        self.assertFalse([e for e in ev if e["stage"] in ("screen", "full") and e["origin"] != "incumbent"
                          and not str(e["origin"]).startswith("baseline")])
        inc_recs = jl(rd / "gen_01" / "evals" / "incumbent" / "screen" / "simple_math_pool.jsonl")
        self.assertTrue(len(inc_recs) == 20 and all(len(x["sample_scores"]) == 4 for x in inc_recs))
        self.assertTrue((rd / "gen_01" / "reference_screen.json").exists())
        # 3 threads x 1 step, no api.evaluate, ONE gate evaluation per candidate (n 4), recorded in gate.json
        cands = jl(rd / "gen_01" / "candidates.jsonl")
        self.assertEqual([(c["thread"], c["step"]) for c in cands], [(0, 0), (1, 0), (2, 0)])
        self.assertFalse([c for c in cands if c["error"]])
        self.assertFalse([e for e in ev if str(e["origin"]).startswith("tool:")])
        gates = [e for e in ev if e["stage"] == "gate"]
        self.assertEqual(sorted(e["cid"] for e in gates), sorted({c["cid"] for c in cands}))
        for e in gates:
            recs = jl(rd / "gen_01" / "evals" / e["cid"] / "gate" / "simple_math_eval.jsonl")
            self.assertTrue(all(len(x["sample_scores"]) == 4 for x in recs))
        subs = json.loads((rd / "gen_01" / "gate.json").read_text())["submissions"]
        self.assertEqual([s["cid"] for s in subs], [c["cid"] for c in cands])
        for s, c in zip(subs, cands):
            self.assertTrue({"gain", "ci", "accuracy", "broken", "fixed", "broken_ids"} <= set(s["info"]))
            self.assertEqual(c["screen"]["feedback"]["gate"]["gain"], s["info"]["gain"])
            self.assertEqual(c["score"], s["info"]["gain"])
        acc = json.loads((rd / "gen_01" / "accepted.json").read_text())
        passing = [s for s in subs if s["passed"]]
        if passing:
            best = max(passing, key=lambda s: s["info"]["gain"])
            self.assertEqual((acc["accepted"], acc["cid"]), (True, best["cid"]))
        else:
            self.assertFalse(acc["accepted"])
        st = json.loads((rd / "state.json").read_text())
        self.assertEqual(st["gen01/search"]["evaluations"], 0)
        out = cands[2]["artifact"]["stdout_tail"]                   # the greedy mock program reads the screen
        self.assertIn("screen_results", out)
        self.assertIn("screen_output", out)
        # prompts: session k of N, no api.evaluate, the configured tool rounds, the screen summary
        th1 = jl(rd / "gen_01" / "editor" / "thread_1.jsonl")
        sys_p = th1[0]["content"]
        self.assertIn("There is no api.evaluate.", sys_p)
        self.assertNotIn("api.evaluate(config", sys_p)
        self.assertIn("for up to 20 rounds", sys_p)
        self.assertIn("persist between your run_python calls", sys_p)
        user = next(x for x in th1 if x.get("role") == "user")["content"]
        self.assertIn("You are session 2 of 3 in this round", user)
        self.assertIn("accuracy avg@4", user)
        self.assertNotIn("improve on the best candidate", user)

    def test_search_history_tool(self):
        """search_history (2026-11-21): generation 2's editor gets the tool (system prompt) and it lists generation
        1's candidates with a config description plus generation 2's visible ones; read_record opens generation 1's
        records; the E_t session sees no earlier generation; editor.history_tool=false removes it."""
        from types import SimpleNamespace
        from rsicore.chat_tools import ChatTools
        from rsicore.sandbox import ToolAPI
        from edit_code import run as RUN
        r, _ = run(["--domain", "math", "--gens", "2", "--run-id", "A_mock_hist", "--set", "heldout.limit=2"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_hist"
        sysmsg = jl(rd / "gen_02" / "editor" / "thread_0.jsonl")[0]["content"]
        self.assertIn("search_history(gen, sort, offset)", sysmsg)
        h = RUN.Runner.search_history(SimpleNamespace(rc=SimpleNamespace(run_dir=rd)), 2, "main")
        self.assertEqual(sorted(h.past), [1])
        t = ChatTools(lambda tag, ctx: ToolAPI({}), history=h)
        vis = [c["cid"] for c in jl(rd / "gen_02" / "candidates.jsonl")][:4]
        ctx = {"cid": "g02_t0_s01", "gen_dir": str(rd / "gen_02"), "visible_cids": vis}
        out = json.loads(t.call("search_history", {}, ctx, 1, 0)["content"])
        self.assertTrue(out["ok"])
        self.assertEqual(out["total"], len(vis) + len(jl(rd / "gen_01" / "candidates.jsonl")))
        self.assertEqual(out["generations"][0]["gen"], 1)
        ok = [x for x in out["rows"] if "error" not in x]
        self.assertTrue(ok and all(x["config"].startswith("default ") and x["GiB"] for x in ok), ok[:2])
        self.assertTrue(any(x["gen"] == 1 for x in out["rows"]) or out.get("next_offset"))
        submitted = {s_["cid"] for s_ in json.loads((rd / "gen_01" / "gate.json").read_text())["submissions"]}
        g1 = {x["cid"]: x for x in out["rows"] if x["gen"] == 1}
        for cid in submitted & set(g1):
            self.assertIn("gate", g1[cid])
        c1 = jl(rd / "gen_01" / "candidates.jsonl")[0]["cid"]
        rr = json.loads(t.call("read_record", {"cid": c1, "part": "feedback"}, ctx, 1, 0)["content"])
        self.assertTrue(rr["ok"] and '"config"' in rr["text"])
        he = RUN.Runner.search_history(SimpleNamespace(rc=SimpleNamespace(run_dir=rd)), 2, "e")
        self.assertEqual(he.past, {})
        cfg = load_cfg(["editor.history_tool=false"], self.runs)
        from rsicore.editor import editor_settings
        self.assertFalse(editor_settings(cfg)["history_tool"])

    def test_acceptance_text_follows_config(self):
        """System prompt (2026-11-21): the acceptance rule follows gate.min_delta / require_significant; the
        api-scope rule is stated."""
        from edit_code.prompts import acceptance_text
        self.assertIn("statistically significant", acceptance_text({"gate": {"min_delta": 0.005}}))
        t = acceptance_text({"gate": {"min_delta": 0.005, "require_significant": False}})
        self.assertIn("no significance test", t)
        self.assertNotIn("significant gain", t)
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_nosig_prompt", "--set", "heldout.limit=2",
                    "--set", "gate.require_significant=false"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        sysmsg = jl(self.runs / "A_mock_nosig_prompt" / "gen_01" / "editor" / "thread_0.jsonl")[0]["content"]
        self.assertIn("there is no significance test", sysmsg)
        self.assertIn("it is not a global there", sysmsg)

    def test_editor_tool_loop(self):
        """editor tool loop (2026-11-21): by default every proposal makes editor.mock_tool_rounds (1) run_python
        call on the exploration API before the program (transcript format 2); exploration spends no evaluate and
        does not change the programs; editor.tool_rounds=0 is the single call of before; the rounds are capped by
        editor.tool_rounds (forced final turn)."""
        runs = {}
        for rid, sets in (("A_mock_tools", []), ("A_mock_notools", ["--set", "editor.tool_rounds=0"]),
                          ("A_mock_cap", ["--set", "editor.tool_rounds=1", "--set", "editor.mock_tool_rounds=3"])):
            r, _ = run(["--domain", "math", "--gens", "1", "--run-id", rid, "--set", "heldout.limit=2"] + sets,
                       self.runs)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            runs[rid] = self.runs / rid / "gen_01"
        gd, g0, gc = runs["A_mock_tools"], runs["A_mock_notools"], runs["A_mock_cap"]
        cands = jl(gd / "candidates.jsonl")
        self.assertGreaterEqual(len(cands), 3)
        for c in cands:
            self.assertGreaterEqual(c["cost"]["editor_rounds"], 1)
            self.assertEqual(c["cost"]["editor_tool_calls"], 1)
            self.assertEqual(c["cost"]["editor_stop"], "answer")
            self.assertTrue((gd / "explore" / c["cid"] / "r01_0" / "stdout.txt").exists())
            self.assertTrue((gd / "explore" / c["cid"] / "r01_0" / "result.json").exists())
        for tr in sorted((gd / "editor").glob("thread_*.jsonl")):
            lines = jl(tr)
            self.assertEqual(lines[0].get("format"), 2)
            tools = [x for x in lines if x.get("role") == "tool"]
            finals = [x for x in lines if x.get("final")]
            self.assertTrue(finals)
            self.assertEqual(len(tools), len(finals))                  # one run_python round per proposal
            for t in tools:
                self.assertEqual((t["name"], t["r"], t["ok"]), ("run_python", 1, True))
                out = json.loads(t["content"])
                self.assertTrue(out["ok"], out)
                self.assertEqual(out["tool_calls"], {"profile_summary": 1})
                self.assertIn("b2", out["result"])
            for f in finals:
                self.assertEqual(f["stats"]["rounds"], 1)
                self.assertTrue(f["code"].startswith("def main(api)"))
        # exploration changes neither the programs nor the evaluate budget, and never evaluates
        progs = {p.name: p.read_text() for p in (gd / "programs").glob("*.py")}
        self.assertTrue(progs)
        self.assertEqual(progs, {p.name: p.read_text() for p in (g0 / "programs").glob("*.py")})
        ev, ev0 = jl(gd / "evals.jsonl"), jl(g0 / "evals.jsonl")
        self.assertEqual(sum(1 for e in ev if e.get("charged")), sum(1 for e in ev0 if e.get("charged")))
        self.assertFalse([e for e in ev if "~x" in str(e.get("cid")) or "~x" in str(e.get("origin"))])
        # tool_rounds 0: one call per proposal, no tools, no explore dir
        for c in jl(g0 / "candidates.jsonl"):
            self.assertEqual(c["cost"]["editor_rounds"], 0)
        self.assertFalse((g0 / "explore").exists())
        for tr in (g0 / "editor").glob("thread_*.jsonl"):
            self.assertFalse([x for x in jl(tr) if x.get("role") == "tool" or x.get("tool_calls")])
        sys_tools = jl(gd / "editor" / "thread_0.jsonl")[0]["content"]
        sys_plain = jl(g0 / "editor" / "thread_0.jsonl")[0]["content"]
        self.assertIn("for up to 50 rounds", sys_tools)
        self.assertNotIn("run_python", sys_plain)
        self.assertTrue(sys_plain.endswith("exactly one ```python code block containing the full program."))
        # tool_rounds 1 with a mock that wants 3 rounds: 1 round, then the forced final turn
        for c in jl(gc / "candidates.jsonl"):
            self.assertEqual((c["cost"]["editor_rounds"], c["cost"]["editor_stop"]), (1, "forced:rounds"))
        self.assertEqual({p.name: p.read_text() for p in (gc / "programs").glob("*.py")}, progs)

    def test_budget_below_reference_is_refused(self):
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_small", "--budget-gib", "5",
                    "--set", "heldout.limit=2"], self.runs)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("smaller than budget.reference", r.stdout + r.stderr)



def load_cfg(sets, runs_dir):
    """Paper A's effective config of a mock invocation (rsicore.config + run.effective_config)."""
    from rsicore import config as C
    from edit_code import run as RUN
    ap = C.build_arg_parser("A")
    ap.add_argument("--budget-gib", type=float, default=None)
    args = ap.parse_args(["--domain", "math", "--mock", "--runs-dir", str(runs_dir),
                          "--run-id", "cfg_" + str(abs(hash(tuple(sets))))] + [x for s in sets for x in ("--set", s)])
    return RUN.effective_config(C.load_run_config("A", args), args)


def mml(args):
    return int(args[args.index("--max-model-len") + 1])


@unittest.skipUnless(HAVE_CORE, "eval/rsicore or pyyaml not importable")
class TestEditorConfig(unittest.TestCase):
    """Config control of the editor (2026-11-21): rounds and context length; no subprocess."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_context_tokens_sets_editor_max_model_len(self):
        from edit_code.run import server_args
        from rsicore.editor import editor_kwargs
        for sets, want in (([], 131072), (["hardware=h20"], 131072), (["hardware=h20_80"], 49152),
                           (["editor.context_tokens=65536"], 65536), (["hardware=h20_80", "editor.context_tokens=65536"], 65536)):
            cfg = load_cfg(sets, self.runs)
            ea = server_args(cfg, "editor_args")
            self.assertEqual(mml(ea), want, sets)
            self.assertEqual(ea.count("--max-model-len"), 1)
            self.assertEqual(mml(server_args(cfg, "candidate_args")), 20480, sets)    # candidates keep theirs
            kw = editor_kwargs(cfg, server_args=ea)
            self.assertEqual(kw["context_tokens"], want)
        cfg = load_cfg(["editor.tool_rounds=7", "editor.context_tokens=null"], self.runs)
        ea = server_args(cfg, "editor_args")
        self.assertEqual(ea, list(cfg["vllm"]["editor_args"]))                          # null: args as written
        kw = editor_kwargs(cfg, server_args=ea)
        self.assertEqual((kw["tool_rounds"], kw["context_tokens"]), (7, mml(ea)))
        self.assertEqual(kw["extra_body"]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual((kw["temperature"], kw["top_p"], kw["max_tokens"]), (0.6, 0.95, 16384))

    def test_prompt_protocol_only_with_tools(self):
        import types
        from edit_code import prompts as P
        prob = types.SimpleNamespace(space={"bits": [2, 3, 4, 8, 16], "group_sizes": [32, 64, 128],
                                            "quantizers": ["rtn", "gptq", "awq"]})
        base = P.SYSTEM_PROMPT.format(tools="DOC", bits=prob.space["bits"], group_sizes=prob.space["group_sizes"],
                                      quantizers=prob.space["quantizers"], timeout_s=1800,
                                      acceptance=P.acceptance_text({"gate": {"min_delta": 0.005}}))
        cfg = {"gate": {"min_delta": 0.005}}
        self.assertEqual(P.system_prompt(prob, "DOC", dict(cfg, editor={"tool_rounds": 0})), base)
        t = P.system_prompt(prob, "DOC", dict(cfg, editor={"tool_rounds": 12}))
        self.assertIn("for up to 12 rounds", t)
        self.assertIn("at most 100 calls in all", t)                                  # editor.max_tool_calls default
        self.assertIn("killed after 1800 s", base)                                     # sandbox.timeout_s default
        self.assertIn("api.evaluate is NOT available there", t)
        self.assertIn("group_error there only accepts settings already cached", t)
        self.assertNotIn(P.FINAL_RULE, t)
        self.assertTrue(t.startswith(base[:-len(P.FINAL_RULE)]))
        t = P.system_prompt(prob, "DOC", dict(cfg, editor={"explore": {"gpu_tools": True}}))
        self.assertIn("for up to 50 rounds", t)                                         # default tool_rounds
        self.assertNotIn("only accepts settings already cached", t)

    def test_explore_api(self):
        """make_tool_api(explore=True): no evaluate (and describe() does not list it); group_error refuses
        uncached settings unless editor.explore.gpu_tools."""
        import types
        from edit_code.tools import ToolError, make_tool_api
        prob = types.SimpleNamespace(spec=None, evals_left=5, herr_missing=lambda c: ["0.down.down_proj|gptq|8|128"],
                                     group_error=lambda c, top=10: {"total_per_token": 0.0})
        full = make_tool_api(prob, {}, cid="c", per_program_evaluate=2)
        self.assertIn("evaluate", full.names())
        ex = make_tool_api(prob, {}, cid="c~x01_0", per_program_evaluate=0, explore=True)
        self.assertNotIn("evaluate", ex.names())
        self.assertNotIn("evaluate", ex.budget)
        self.assertNotIn("evaluate(", ex.describe())
        self.assertEqual(set(full.names()) - set(ex.names()), {"evaluate"})
        with self.assertRaisesRegex(ToolError, "uncached setting"):
            ex.call("group_error", {"config": {"default": {"bits": 8}}})
        self.assertEqual(full.call("group_error", {"config": {"default": {"bits": 8}}}), {"total_per_token": 0.0})
        gx = make_tool_api(prob, {"editor": {"explore": {"gpu_tools": True}}}, cid="c~x", per_program_evaluate=0,
                           explore=True)
        self.assertEqual(gx.call("group_error", {"config": {"default": {"bits": 8}}}), {"total_per_token": 0.0})
        prob.herr_missing = lambda c: []                                                # cached: allowed
        self.assertEqual(ex.call("group_error", {"config": {"default": {"bits": 4}}}), {"total_per_token": 0.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
