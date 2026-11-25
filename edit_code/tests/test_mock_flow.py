"""End-to-end Paper A orchestration in mock mode (no weights, no GPU), against the real rsicore.
Fixed-size design: frozen editor (M_0), every candidate quantizes M_0 to one budget. KL only (2026-11-25): generation
0 writes M_0's reference traces on kl_eval.n items of the training pool (mock: 24); every candidate gets ONE KL
evaluation on them; the lowest KL of a generation is accepted when it is below the incumbent's. No screen, no
accuracy gate, no held-out test, no classic baselines (the mock runs 3 threads x 1 step).

Run with the eval interpreter (needs pyyaml; torch / transformers are NOT needed in mock mode):
    eval/.venv/bin/python -I -m pytest edit_code/tests/test_mock_flow.py -q      (or run the file)
Takes ~2 minutes.
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

    def check_kl_set(self, rd):
        kd = rd / "baseline" / "kl_gate"
        tr = json.loads((kd / "traces.json").read_text())
        items = tr["items"]
        self.assertEqual(len(items), 24)                                   # mock_overrides.kl_eval.n
        self.assertEqual({it["ds"] for it in items}, {"simple_math_pool"})
        screen = set(json.loads((rd / "draws" / "gen_00" / "ids" / "simple_math_pool.screen.json").read_text()))
        calib = set(json.loads((rd / "draws" / "gen_00" / "ids" / "simple_math_pool.kl.json").read_text()))
        ids = [it["id"] for it in items]
        self.assertTrue(set(ids) <= screen and not set(ids) & calib)        # generation 0's draw, not calibration
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(it["text"].startswith(it["prompt"]) and len(it["text"]) > len(it["prompt"])
                            for it in items))
        self.assertEqual(len(jl(kd / "ref.jsonl")), 24)
        self.assertEqual(len(jl(kd / "answers.jsonl")), 24)
        return ids

    def test_kl_loop_and_resume(self):
        args = ["--domain", "math", "--gens", "2", "--run-id", "A_mock_test", "--set", "loop.start=q0",
                "--set", "budget.reference.default.bits=3"]
        r, dt = run(args, self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_test"
        st = json.loads((rd / "state.json").read_text())
        for k in ("baseline/draw", "baseline/kl_gate", "baseline/prep", "baseline/budget", "baseline/q0",
                  "gen01/search", "gen01/gate", "gen01/done", "gen02/done"):
            self.assertIn(k, st)
        for gone in ("baseline/gate", "baseline/heldout", "baseline/q0_heldout", "gen01/heldout", "gen01/draw"):
            self.assertNotIn(gone, st)
        self.assertFalse((rd / "baseline" / "heldout.json").exists() or (rd / "baselines").exists())
        self.assertFalse((rd / "draws" / "gen_01").exists())                 # only generation 0 draws
        ids = self.check_kl_set(rd)
        budget = json.loads((rd / "baseline" / "budget.json").read_text())
        self.assertEqual(budget["source"], "budget.reference")
        self.assertAlmostEqual(budget["gib"], budget["reference_gib"], places=3)
        q0 = json.loads((rd / "baseline" / "incumbent.json").read_text())
        self.assertEqual((q0["gen"], q0["cid"]), (0, "q0"))
        self.assertGreater(q0["kl"], 0)
        self.assertEqual(json.loads((rd / "baseline" / "q0" / "record.json").read_text())["kl"], q0["kl"])
        q0recs = jl(Path(q0["kl_records"]))
        self.assertEqual([x["id"] for x in q0recs], ids)
        self.assertAlmostEqual(sum(x["kl"] for x in q0recs) / len(q0recs), q0["kl"], places=5)
        # the layer cache / diagnostics are built once, from M_0
        self.assertTrue((rd / "prep" / "spec.json").exists())
        self.assertFalse((rd / "gen_01" / "prep").exists())
        prof = json.loads((rd / "prep" / "diagnostics" / "profile_table.json").read_text())
        self.assertEqual({x["bits"] for x in prof}, {2, 3, 4})
        prev = q0
        for g in (1, 2):
            gd = rd / f"gen_{g:02d}"
            cands = jl(gd / "candidates.jsonl")
            self.assertEqual([(c["thread"], c["step"]) for c in cands], [(0, 0), (1, 0), (2, 0)])
            subs = json.loads((gd / "gate.json").read_text())["submissions"]
            self.assertEqual([s_["cid"] for s_ in subs], [c["cid"] for c in cands])
            evals = jl(gd / "evals.jsonl")
            self.assertTrue(all(e["stage"] == "kl" for e in evals))
            self.assertFalse([e for e in evals if e["origin"] == "incumbent"])   # the incumbent is never re-scored
            for c, s_ in zip(cands, subs):
                if c["error"]:
                    continue
                self.assertLessEqual(c["artifact"]["bytes_gib"], budget["gib"] + 1e-9)
                info = s_["info"]
                if info.get("kl") is None:
                    continue
                self.assertEqual(info["reference_kl"], prev["kl"])
                self.assertAlmostEqual(info["gain"], prev["kl"] - info["kl"], places=5)
                self.assertAlmostEqual(c["score"], -info["kl"], places=6)
                fb = c["screen"]["feedback"]
                self.assertEqual(fb["kl_eval"]["kl"], info["kl"])
                self.assertTrue(fb["error_top_groups"])
            acc = json.loads((gd / "accepted.json").read_text())
            scored = [s_ for s_ in subs if s_["passed"] and s_["info"].get("kl") is not None]
            best = min(scored, key=lambda s_: (s_["info"]["kl"], s_["cid"])) if scored else None
            if best is not None and best["info"]["kl"] < prev["kl"]:
                self.assertEqual((acc["accepted"], acc["cid"], acc["kl"]), (True, best["cid"], best["info"]["kl"]))
                self.assertEqual(acc["rule"], "the lowest KL of the round, below the incumbent's")
                self.assertTrue((gd / "model" / "mock_model.json").exists())
            else:
                self.assertFalse(acc["accepted"])
            self.assertEqual(acc["incumbent_kl"], prev["kl"])
            prev = json.loads((gd / "incumbent.json").read_text())
        self.assertIn("editor = M_0 (frozen)", r.stdout)
        summ = json.loads((rd / "summary.json").read_text())
        f = summ["final"]
        self.assertEqual((f["editor"], f["kl"]["q0"]), ("frozen", q0["kl"]))
        self.assertEqual(f["kl"]["final"], prev["kl"])
        self.assertEqual(summ["gens"]["0"]["kl_set"], 24)
        for g in ("1", "2"):
            self.assertIn("candidate_kl", summ["gens"][g])
            self.assertFalse({"heldout", "heldout_by_domain", "gate_scores", "gate_mean"} & set(summ["gens"][g]))
        # resume: everything is skipped
        before = (rd / "state.json").read_text()
        r2, dt2 = run(args, self.runs)
        self.assertEqual(r2.returncode, 0, r2.stderr[-3000:])
        self.assertLess(dt2, 30)
        self.assertEqual(json.loads(before).keys(), json.loads((rd / "state.json").read_text()).keys())

    def test_start_bf16(self):
        """loop.start bf16 (default): the incumbent of generation 1 is M_0 itself (KL 0, over the budget); the
        lowest-KL in-budget candidate is accepted whatever its KL; the editor is told so."""
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_bf16"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_bf16"
        inc0 = json.loads((rd / "baseline" / "incumbent.json").read_text())
        self.assertTrue(inc0["original"] and inc0["kl"] == 0.0 and inc0["config"] is None)
        self.assertTrue((rd / "baseline" / "q0" / "record.json").exists())          # Q_0 still recorded
        subs = json.loads((rd / "gen_01" / "gate.json").read_text())["submissions"]
        scored = [s_ for s_ in subs if s_["info"].get("kl") is not None]
        self.assertTrue(scored and all(s_["info"]["reference_kl"] == 0.0 for s_ in scored))
        acc = json.loads((rd / "gen_01" / "accepted.json").read_text())
        best = min(scored, key=lambda s_: (s_["info"]["kl"], s_["cid"]))
        self.assertEqual((acc["accepted"], acc["cid"], acc["first_quantization"]), (True, best["cid"], True))
        self.assertEqual(acc["rule"], "first quantization: the lowest KL among the in-budget candidates")
        sysmsg = jl(rd / "gen_01" / "editor" / "thread_0.jsonl")[0]["content"]
        self.assertIn("the incumbent is the original BF16 model", sysmsg)
        self.assertIn("first incumbent", sysmsg)
        user = next(x for x in jl(rd / "gen_01" / "editor" / "thread_0.jsonl") if x.get("role") == "user")["content"]
        self.assertIn("Find the lowest-KL config within the budget", user)

    def test_patience_stop_and_unfrozen_editor(self):
        # loop.editor=current (ablation): the incumbent writes; patience 1: the first generation without a
        # new incumbent stops the loop
        r, _ = run(["--domain", "math", "--gens", "4", "--run-id", "A_mock_stop", "--set", "loop.editor=current",
                    "--set", "loop.start=q0", "--set", "loop.patience=1"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_stop"
        st = json.loads((rd / "state.json").read_text())
        self.assertIn("stopped", st)
        g = st["stopped"]["gen"]
        self.assertLess(g, 4)
        acc = json.loads((rd / f"gen_{g:02d}" / "accepted.json").read_text())
        self.assertFalse(acc["accepted"])
        inc = json.loads((rd / f"gen_{g:02d}" / "incumbent.json").read_text())
        prev = json.loads(((rd / "baseline") if g == 1 else (rd / f"gen_{g - 1:02d}")).joinpath("incumbent.json").read_text())
        self.assertEqual(inc["config_hash"], prev["config_hash"])          # the incumbent stays
        self.assertEqual(json.loads((rd / "summary.json").read_text())["final"]["editor"], "current")

    def test_calibration_texts_and_removed_knobs(self):
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_sizes", "--set", "noise.temperature=0"],
                   self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_sizes"
        ids = rd / "draws" / "gen_00" / "ids"
        self.assertEqual(len(json.loads((ids / "simple_math_pool.kl.json").read_text())), 400)
        # kl.self_generated (default): one cached M_0 response per calibration text, generation 0 only
        self.assertEqual(len((rd / "kl_selfgen" / "gen_00.jsonl").read_text().splitlines()), 400)
        self.assertEqual(sorted(p.name for p in (rd / "kl_selfgen").iterdir()), ["gen_00.jsonl"])
        for sets in (["noise.temperature=0.6"], ["search.screen_candidates=true"], ["search.kl_weight=0.1"]):
            r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_removed", "--set", sets[0]], self.runs)
            self.assertNotEqual(r.returncode, 0, sets)
            self.assertIn("this knob was removed", r.stdout + r.stderr, sets)
            self.assertFalse((self.runs / "A_mock_removed").exists())

    def test_four_gpus_all_used(self):
        """4 GPUs: M_0's generation-0 jobs on 4 replicas, the editor's GPU lent as a 4th candidate slot while it
        sleeps."""
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_4gpu", "--gpus", "0,1,2,3",
                    "--set", "hardware=h20"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_4gpu"
        self.assertIn("serve as candidate slot 3 (4 slots)", r.stdout)
        self.assertEqual(len(jl(rd / "kl_selfgen" / "gen_00.jsonl")), 400)
        self.check_kl_set(rd)

    def test_search_history_tool(self):
        """search_history: generation 2's editor gets the tool and it lists generation 1's candidates with a config
        description and their KL plus generation 2's visible ones; read_record opens generation 1's records;
        editor.history_tool=false removes it."""
        from types import SimpleNamespace
        from rsicore.chat_tools import ChatTools
        from rsicore.sandbox import ToolAPI
        from edit_code import run as RUN
        r, _ = run(["--domain", "math", "--gens", "2", "--run-id", "A_mock_hist"], self.runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        rd = self.runs / "A_mock_hist"
        sysmsg = jl(rd / "gen_02" / "editor" / "thread_0.jsonl")[0]["content"]
        self.assertIn("search_history(gen, sort, offset)", sysmsg)
        user = next(x for x in jl(rd / "gen_02" / "editor" / "thread_0.jsonl") if x.get("role") == "user")["content"]
        self.assertIn("Archive of earlier rounds", user)
        self.assertIn('"kl": ', user)
        h = RUN.Runner.search_history(SimpleNamespace(rc=SimpleNamespace(run_dir=rd)), 2)
        self.assertEqual(sorted(h.past), [1])
        t = ChatTools(lambda tag, ctx: ToolAPI({}), history=h)
        vis = [c["cid"] for c in jl(rd / "gen_02" / "candidates.jsonl")][:4]
        ctx = {"cid": "g02_t0_s01", "gen_dir": str(rd / "gen_02"), "visible_cids": vis}
        out = json.loads(t.call("search_history", {}, ctx, 1, 0)["content"])
        self.assertTrue(out["ok"])
        self.assertEqual(out["total"], len(vis) + len(jl(rd / "gen_01" / "candidates.jsonl")))
        self.assertEqual(out["generations"][0]["gen"], 1)
        self.assertTrue(out["generations"][0].get("first_quantization"))
        ok = [x for x in out["rows"] if "error" not in x]
        self.assertTrue(ok and all(x["config"].startswith("default ") and x["GiB"] for x in ok), ok[:2])
        g1 = [x for x in out["rows"] if x["gen"] == 1]
        self.assertTrue(any(x.get("kl") is not None for x in g1), g1[:2])
        c1 = jl(rd / "gen_01" / "candidates.jsonl")[0]["cid"]
        rr = json.loads(t.call("read_record", {"cid": c1, "part": "feedback"}, ctx, 1, 0)["content"])
        self.assertTrue(rr["ok"] and '"config"' in rr["text"])
        cfg = load_cfg(["editor.history_tool=false"], self.runs)
        from rsicore.editor import editor_settings
        self.assertFalse(editor_settings(cfg)["history_tool"])

    def test_editor_tool_loop(self):
        """editor tool loop (2026-11-21): by default every proposal makes editor.mock_tool_rounds (1) run_python
        call on the exploration API before the program (transcript format 2); exploration never evaluates and
        does not change the programs; editor.tool_rounds=0 is the single call of before; the rounds are capped by
        editor.tool_rounds (forced final turn)."""
        runs = {}
        for rid, sets in (("A_mock_tools", []), ("A_mock_notools", ["--set", "editor.tool_rounds=0"]),
                          ("A_mock_cap", ["--set", "editor.tool_rounds=1", "--set", "editor.mock_tool_rounds=3"])):
            r, _ = run(["--domain", "math", "--gens", "1", "--run-id", rid] + sets, self.runs)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            runs[rid] = self.runs / rid / "gen_01"
        gd, g0, gc = runs["A_mock_tools"], runs["A_mock_notools"], runs["A_mock_cap"]
        cands = jl(gd / "candidates.jsonl")
        self.assertGreaterEqual(len(cands), 3)
        for c in cands:
            self.assertGreaterEqual(c["cost"]["editor_rounds"], 1)
            self.assertEqual(c["cost"]["editor_tool_calls"], 1)
            self.assertEqual(c["cost"]["editor_stop"], "answer")
            # editor.explore.persistent (base.yaml): one session per proposal, call outputs under session/
            self.assertTrue((gd / "explore" / c["cid"] / "session" / "r01_0" / "result.json").exists())
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
        # exploration changes neither the programs nor the evaluations, and never evaluates
        progs = {p.name: p.read_text() for p in (gd / "programs").glob("*.py")}
        self.assertTrue(progs)
        self.assertEqual(progs, {p.name: p.read_text() for p in (g0 / "programs").glob("*.py")})
        ev, ev0 = jl(gd / "evals.jsonl"), jl(g0 / "evals.jsonl")
        self.assertEqual(sorted((e["cid"], e["kl"]) for e in ev), sorted((e["cid"], e["kl"]) for e in ev0))
        self.assertFalse([e for e in ev if "~x" in str(e.get("cid")) or "~x" in str(e.get("origin"))])
        # tool_rounds 0: one call per proposal, no tools, no explore dir
        for c in jl(g0 / "candidates.jsonl"):
            self.assertEqual(c["cost"]["editor_rounds"], 0)
        self.assertFalse((g0 / "explore").exists())
        for tr in (g0 / "editor").glob("thread_*.jsonl"):
            self.assertFalse([x for x in jl(tr) if x.get("role") == "tool" or x.get("tool_calls")])
        sys_tools = jl(gd / "editor" / "thread_0.jsonl")[0]["content"]
        sys_plain = jl(g0 / "editor" / "thread_0.jsonl")[0]["content"]
        self.assertIn("for up to 20 rounds", sys_tools)
        self.assertNotIn("run_python", sys_plain)
        self.assertTrue(sys_plain.endswith("exactly one ```python code block containing the full program."))
        # tool_rounds 1 with a mock that wants 3 rounds: 1 round, then the forced final turn
        for c in jl(gc / "candidates.jsonl"):
            self.assertEqual((c["cost"]["editor_rounds"], c["cost"]["editor_stop"]), (1, "forced:rounds"))
        self.assertEqual({p.name: p.read_text() for p in (gc / "programs").glob("*.py")}, progs)

    def test_budget_below_reference_is_refused(self):
        r, _ = run(["--domain", "math", "--gens", "1", "--run-id", "A_mock_small", "--budget-gib", "5"], self.runs)
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
            self.assertEqual(mml(server_args(cfg, "candidate_args")), 32768, sets)    # candidate_max_model_len
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
                                            "quantizers": ["rtn", "gptq", "awq"]}, kl_items=lambda: [{}] * 256)
        base = P.system_prompt(prob, "DOC", {"editor": {"tool_rounds": 0}})
        self.assertTrue(base.endswith(P.FINAL_RULE))
        t = P.system_prompt(prob, "DOC", {"editor": {"tool_rounds": 12}})
        self.assertIn("for up to 12 rounds", t)
        self.assertIn("at most 100 calls in all", t)                                  # editor.max_tool_calls default
        self.assertIn("killed after 1800 s", base)                                     # sandbox.timeout_s default
        self.assertIn("group_error there only accepts settings already cached", t)
        self.assertNotIn(P.FINAL_RULE, t)
        self.assertTrue(t.startswith(base[:-len(P.FINAL_RULE)]))
        t = P.system_prompt(prob, "DOC", {"editor": {"explore": {"gpu_tools": True}}})
        self.assertIn("for up to 50 rounds", t)                                         # default tool_rounds
        self.assertNotIn("only accepts settings already cached", t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
