"""Paper B data, GPU layout and config of the one (simple) flow: only the run's domain is drawn, gated and evaluated
(simple_<d>_pool / simple_<d>_eval / simple_<d>_test), no theta0 role, the folded base.yaml values."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from lora_code.run import Run


class DatasetsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rc = SimpleNamespace(domain="math", run_dir=Path(self.tmp.name))

    def test_only_the_run_domain_is_read(self):
        run = Run(self.rc, {})
        self.assertEqual((run.tds, run.gds), ("simple_math_pool", "simple_math_eval"))
        with patch("rsicore.draws.gate_ids", side_effect=lambda rc, ds: ["math-id"]
                   if ds == "simple_math_eval" else self.fail(f"other data accessed: {ds}")):
            self.assertEqual(run.gate_level(), {"simple_math_eval": ["math-id"]})
        seen = []

        def fake_ids(rc, g, ds, role):
            seen.append((ds, role))
            return ["b", "a", "c"]
        with patch("rsicore.draws.ids", side_effect=fake_ids), \
                patch("rsicore.draws.gate_ids", return_value=["g2", "g1"]):
            ids = Run(self.rc, {"eval": {"screen_limit": 2}}).level_ids(0)
        self.assertEqual(ids, {"screen": {"simple_math_pool": ["a", "b"]},
                               "full": {"simple_math_pool": ["a", "b", "c"]},
                               "gate": {"simple_math_eval": ["g1", "g2"]}})
        self.assertEqual(sorted(seen), [("simple_math_pool", "screen"), ("simple_math_pool", "signal")])
        for gone in ("retention_domains", "datasets", "gate_datasets", "kl_texts", "e_ref", "e_branch", "theta0"):
            self.assertFalse(hasattr(run, gone), gone)


class GpuLayoutTest(unittest.TestCase):
    """Every given GPU is used (rsicore.gpu_layout); gpu_roles pins roles, the rest are laid out around them."""

    def test_paper_b_replicas(self):
        """Default since 2026-11-20: train = 1 GPU, main = the rest as replicas (no theta0 role, 2026-11-22)."""
        from lora_code.common import gpu_layout, main_groups

        def cfg(**lay):
            return {"gpu_layout": dict(lay)}

        def rc(n):
            return SimpleNamespace(gpus=list(range(n)))
        self.assertEqual(gpu_layout(rc(4), cfg())[0], {"main": [0, 1, 2], "train": [3]})
        self.assertEqual(main_groups(rc(4), cfg()), [[0], [1], [2]])          # 3 replicas
        self.assertEqual(main_groups(rc(5), cfg(main_tp=2)), [[0, 1], [2, 3]])
        self.assertEqual(gpu_layout(rc(5), cfg(train_gpus=2))[0]["train"], [3, 4])
        self.assertEqual(gpu_layout(rc(2), cfg())[0], {"main": [0], "train": [1]})
        self.assertEqual(gpu_layout(rc(2), cfg(train_gpus=3))[0], {"main": [0], "train": [1]})   # main keeps one
        with self.assertRaises(ValueError):
            gpu_layout(rc(1), cfg())
        self.assertEqual(gpu_layout(rc(1), cfg(share_single_gpu=True))[0], {"main": [0], "train": [0]})
        for n in range(2, 9):
            self.assertEqual(gpu_layout(rc(n), cfg())[1], [], n)                # nothing idle
        roles, _ = gpu_layout(rc(4), dict(cfg(), gpu_roles={"train": [2, 3]}))  # pinned: one main server
        self.assertEqual(roles, {"train": [2, 3], "main": [0, 1]})
        self.assertEqual(main_groups(rc(4), dict(cfg(), gpu_roles={"train": [2, 3]})), [roles["main"]])

    def test_paper_b(self):
        """The earlier layout (gpu_layout.main_replicas: false): main grows by tensor parallelism."""
        from lora_code.common import gpu_layout

        def lay(n, **extra):
            return gpu_layout(SimpleNamespace(gpus=list(range(n))), dict({"gpu_layout": {"main_replicas": False}},
                                                                         **extra))
        self.assertEqual(lay(2)[0], {"main": [0], "train": [1]})
        self.assertEqual(lay(3)[0], {"main": [0, 1], "train": [2]})
        self.assertEqual(lay(5), ({"main": [0, 1], "train": [2, 3, 4]}, []))
        self.assertEqual(lay(6)[0], {"main": [0, 1, 2, 3], "train": [4, 5]})
        for n in range(2, 9):
            self.assertEqual(lay(n)[1], [], n)                         # nothing idle
        roles, _ = lay(4, gpu_roles={"train": [2, 3]})
        self.assertEqual(roles, {"train": [2, 3], "main": [0, 1]})
        with self.assertRaises(ValueError):
            lay(2, gpu_roles={"train": [5]})

    def test_paper_a(self):
        from edit_code.common import gpu_layout
        h20 = {"gpu_layout": {"share_single_gpu": False}}

        def lay(n, **extra):
            return gpu_layout(SimpleNamespace(gpus=[10 + i for i in range(n)]), dict(h20, **extra))
        self.assertEqual(lay(2)[0], {"editor": [10], "candidate": [11]})
        self.assertEqual(lay(3)[0], {"editor": [10], "candidate": [11, 12]})        # editor 1, the rest candidates
        self.assertEqual(lay(8)[0]["candidate"], list(range(11, 18)))
        self.assertEqual(lay(5)[1], [])                                             # nothing idle
        with self.assertRaises(ValueError):
            lay(1)
        self.assertEqual(gpu_layout(SimpleNamespace(gpus=[0]), {})[0], {"editor": [0], "candidate": [0]})  # B300/mock


class ConfigTest(unittest.TestCase):
    """The folded base.yaml (2026-11-22): no objective / scope / data / retention / E_t keys; the former selectors
    objective=simple and scope=single are accepted no-ops (rsicore.config), the run id keeps _single_simple."""

    def rc(self, *sets, domain="math", seed=None, tmp=None):
        from rsicore.config import build_arg_parser, load_run_config
        if tmp is None:
            tmp = tempfile.mkdtemp()
            self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        argv = ["--domain", domain, "--mock", "--runs-dir", tmp]
        if seed is not None:
            argv += ["--seed", str(seed)]
        for x in sets:
            argv += ["--set", x]
        return load_run_config("B", build_arg_parser("B").parse_args(argv))

    def test_default_run_id_and_removed_keys(self):
        rc = self.rc()
        self.assertEqual(rc.run_id, "B_math_s42_single_simple_mock")
        cfg = rc.paper_cfg
        for gone in ("objective", "objective_profiles", "scope", "scope_profiles", "mock_objective_overrides",
                     "data", "data_roles", "draw_config", "retention", "e_t", "eval_target"):
            self.assertNotIn(gone, cfg, gone)
        self.assertNotIn("domains", cfg["heldout"])
        self.assertNotIn("math_eval", cfg["max_tokens"])
        self.assertNotIn("theta0_args", cfg["server"])
        self.assertNotIn("theta0_own_gpu", cfg["gpu_layout"])
        self.assertNotIn("kl_limit", cfg["eval"])
        self.assertFalse({"lambda_retention", "lambda_kl"} & set(cfg["feedback"]))
        self.assertEqual(cfg["simple"]["screen_n"], 500)
        self.assertEqual(self.rc("objective=simple", "scope=single").paper_cfg, cfg)   # no-ops
        for bad in ("objective=train_gate", "objective=eval_overfit", "scope=all", "data.n=40"):   # data.screen_n -> simple.screen_n (accepted)
            with self.assertRaises(SystemExit, msg=bad):
                self.rc(bad)

    def test_heldout_and_gpu_roles(self):
        from rsicore.heldout import heldout_domains

        from lora_code.common import gpu_layout
        cfg = self.rc("hardware=h20", domain="code").paper_cfg
        self.assertEqual(heldout_domains(cfg, "code"), "eval:simple_code_test")
        roles, unused = gpu_layout(SimpleNamespace(gpus=[3, 5]), cfg)       # h20, 2 GPUs: main | train
        self.assertEqual((roles, unused), ({"main": [3], "train": [5]}, []))
        run = Run(SimpleNamespace(domain="code", run_dir=Path(tempfile.mkdtemp())), cfg)
        self.assertEqual((run.tds, run.gds), ("simple_code_pool", "simple_code_eval"))
        b2 = self.rc("gpu_roles.train=[2,3]").paper_cfg                     # explicit --set still wins
        self.assertEqual(b2["gpu_roles"]["train"], [2, 3])

    def test_mock_overrides_fold(self):
        """mock_overrides holds the former mock_objective_overrides.simple (k sessions x 1 step at mock size)."""
        from lora_code.common import deep_merge
        from lora_code.tools import search_settings
        cfg = self.rc().paper_cfg
        m = deep_merge(cfg, cfg["mock_overrides"])
        self.assertEqual(m["search"]["threads"], 3)
        self.assertEqual(search_settings(m), {"threads": 3, "steps": 1, "max_candidates": 3, "parallel_builds": True})
        self.assertEqual((m["eval"]["gate_limit"], m["nonllm"]["evaluate"]), (30, 2))
        self.assertNotIn("baseline", m)

    def test_simple_flow(self):
        """The simple flow v2 (2026-11-21, base.yaml since 2026-11-22): thinking-mode sampled reference screen, gate
        and held-out (avg@4) with the run seed, 8 x 1 search, candidates not screened, one gate evaluation each, no
        api.evaluate, tensor / paper / screen tools on, persistent run_python, nonllm 8."""
        from rsicore.config import stage_decoding

        from lora_code.run import context_shortfall
        from lora_code.tools import (paper_tools_enabled, screen_tools_enabled, search_settings,
                                     tensor_compute_enabled)
        rc = self.rc(seed=7)
        cfg = rc.paper_cfg
        self.assertEqual(cfg["heldout"]["decoding"]["seed"], 7)                     # one seed (null -> --seed)
        samp = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "thinking": True, "max_tokens": 16384, "seed": 7}
        self.assertEqual(stage_decoding(cfg, "screen", 7), dict(samp, n=4))
        self.assertEqual(stage_decoding(cfg, "full", 7), dict(samp, n=4))          # full falls back to screen
        self.assertEqual(stage_decoding(cfg, "gate", 7), dict(samp, n=4))
        hd = cfg["heldout"]["decoding"]
        self.assertEqual({k: hd[k] for k in samp}, samp)
        self.assertEqual((hd["n"], hd.get("n_by_task"), cfg["heldout"]["enabled"]), (4, None, True))
        self.assertEqual(search_settings(cfg), {"threads": 8, "steps": 1, "max_candidates": 8, "parallel_builds": True})
        self.assertEqual(search_settings({"search": {"threads": 2, "steps": 3}})["max_candidates"], 6)   # null: t x s
        self.assertEqual(cfg["gate"]["min_gain"], 0.0)
        self.assertTrue(tensor_compute_enabled(cfg) and paper_tools_enabled(cfg) and screen_tools_enabled(cfg))
        self.assertTrue(tensor_compute_enabled({}) and paper_tools_enabled({}) and screen_tools_enabled({}))   # on
        self.assertTrue(cfg["editor"]["explore"]["persistent"] and cfg["editor"]["explore"]["gpu_tools"])
        self.assertEqual((cfg["nonllm"]["evaluate"], cfg["editor"]["tool_rounds"]), (8, 20))
        self.assertEqual(cfg["tools"]["train_loss_backend"], "vllm")
        self.assertEqual(cfg["method"]["mode"], "scratch")
        self.assertIsNone(context_shortfall(cfg))                   # main replicas 131072 >= prompt + 16384
        small = dict(cfg, editor=dict(cfg["editor"], context_tokens=18432))
        self.assertIn("--max-model-len 18432", context_shortfall(small))
        for hw in ("h20", "h20_80"):
            self.assertIsNone(context_shortfall(self.rc(f"hardware={hw}").paper_cfg))


class RemovedKnobsTest(unittest.TestCase):
    """2026-11-23: the search / gate variants and the noise band are hard-wired away (rsicore.config.REMOVED_KNOBS):
    base.yaml has none of their keys, --set of one is refused unless it is the hard-wired value (a no-op)."""

    KEYS = ("search.screen_candidates", "search.gate_every_candidate", "search.charge_screens",
            "search.program_evaluate_calls", "search.program_evaluate_total", "search.max_evaluate", "search.top_full",
            "search.rerun_seeds", "gate.max_submissions", "gate.reference", "gate.epsilon", "gate.use_noise_band",
            "baseline.noise_repeats", "baseline.noise_temperature", "baseline.noise_top_p", "baseline.noise_n",
            "baseline.noise_fixed")

    def rc(self, *sets):
        return ConfigTest.rc(self, *sets)

    def test_base_has_no_removed_keys(self):
        import yaml
        b = yaml.safe_load((ROOT / "lora_code" / "configs" / "base.yaml").read_text())
        for k in self.KEYS:
            top, leaf = k.split(".")
            self.assertNotIn(leaf, b.get(top) or {}, k)
            self.assertNotIn(leaf, (b["mock_overrides"].get(top) or {}), k)
        self.assertNotIn("baseline", b)
        self.assertEqual(b["eval"]["full_limit"], None)                  # still read (raw / non-LLM full level)
        cfg = self.rc().paper_cfg
        self.assertFalse({"max_evaluate", "top_full", "screen_candidates"} & set(cfg["search"]))

    def test_set_removed_knob(self):
        base = self.rc().paper_cfg
        for ok in ("search.screen_candidates=false", "search.gate_every_candidate=true", "search.top_full=0",
                   "baseline.noise_temperature=0", "gate.reference=parent"):
            self.assertEqual(self.rc(ok).paper_cfg, base, ok)            # the hard-wired value: a no-op
        for bad in ("search.screen_candidates=true", "search.max_evaluate=12", "search.top_full=3",
                    "baseline.noise_fixed=0.05", "gate.reference=raw", "gate.epsilon=0.01",
                    "gate.max_submissions=3", "search.program_evaluate_calls=2"):
            with self.assertRaises(SystemExit, msg=bad) as e:
                self.rc(bad)
            self.assertIn("removed", str(e.exception), bad)

    def test_gate_fallback(self):
        from lora_code.common import gate_fallback
        self.assertEqual(gate_fallback({"method": {"mode": "edit"}}), "raw")
        self.assertEqual(gate_fallback({"method": {"mode": "scratch"}}), "keep")
        self.assertEqual(gate_fallback({"method": {"mode": "weights"}}), "keep")
        self.assertEqual(gate_fallback({"method": {"mode": "edit"}, "gate": {"fallback": "keep"}}), "keep")
        for bad in ({"method": {"mode": "scratch"}, "gate": {"fallback": "raw"}}, {"gate": {"fallback": "sft"}}):
            with self.assertRaises(SystemExit):
                gate_fallback(bad)


class DecodingTest(unittest.TestCase):
    """Decoding helpers."""

    def test_tasks_greedy_scoring_sampled(self):
        """tasks.yaml decoding is greedy; Paper B scores and evaluates the held-out set with the thinking-mode
        sampling of its config (decoding.*, heldout.decoding)."""
        import yaml
        from rsicore.heldout import decoding_args
        tasks = yaml.safe_load((ROOT / "eval" / "configs" / "tasks.yaml").read_text())
        self.assertEqual(tasks["defaults"]["temperature"], 0.0)
        for name, t in tasks["tasks"].items():
            self.assertEqual(t.get("temperature", 0.0), 0.0, name)
            self.assertEqual(t.get("n", 1), 1, name)
        b = yaml.safe_load((ROOT / "lora_code" / "configs" / "base.yaml").read_text())
        dec = dict(b["heldout"]["decoding"], seed=42)                # resolved to the run's --seed at load time
        self.assertEqual((dec["temperature"], dec["top_p"], dec["top_k"], dec["n"]), (0.6, 0.95, 20, 4))
        d = decoding_args(dec, "simple_math_test")
        self.assertEqual(d["extra"][:2], ["--n", "4"])
        self.assertEqual({k: d["kw"][k] for k in ("temperature", "top_p", "top_k", "seed")},
                         {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 42})
        self.assertEqual(decoding_args(None, "gsm8k"), {"kw": {}, "extra": [], "key": None})

    def test_run_eval_sends_top_k_only_when_sampling(self):
        sys.path.insert(0, str(ROOT / "eval"))
        import run_eval
        a = run_eval.parse_args(["--task", "gsm8k", "--pool", "test", "--endpoint", "x", "--model", "m",
                                 "--temperature", "0.7", "--top-p", "0.8", "--top-k", "20"])
        dflt = {"temperature": 0.0, "top_p": 1.0, "top_k": None, "n": 1, "max_tokens": 10, "max_tokens_thinking": 10,
                "thinking": False, "seed": 42, "send_chat_template_kwargs": True}
        p = run_eval.build_payload({"messages": []}, run_eval.decoding_for({}, dflt, a), "m")
        self.assertEqual((p["temperature"], p["top_p"], p["top_k"], p["seed"]), (0.7, 0.8, 20, 42))
        g = run_eval.parse_args(["--task", "gsm8k", "--pool", "test", "--endpoint", "x", "--model", "m"])
        p = run_eval.build_payload({"messages": []}, run_eval.decoding_for({}, dflt, g), "m")
        self.assertEqual(p["temperature"], 0.0)
        self.assertNotIn("top_k", p)
        self.assertNotIn("seed", p)

    def test_raw_train_size(self):
        from lora_code import baselines as BL
        r, n, gb = BL.raw_recipe({})
        self.assertIsNone(n)                                             # raw.n_train null -> all train items
        self.assertNotIn("n_train", r)
        r, n, gb = BL.raw_recipe({"raw": {"n_train": 40}})
        self.assertEqual(n, 40)
        self.assertFalse(hasattr(BL, "sft_n_train") or hasattr(BL, "train_sft"))   # no SFT baseline


class SamplingParamsTest(unittest.TestCase):
    """Training answers use train_data (2026-11-20: 4 samples at 0.7 / 0.8 / top_k 20, the run seed)."""

    def test_train_data_sampling_uses_configured_temperature_and_top_p(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rc = SimpleNamespace(domain="math", run_dir=Path(tmp.name), mock=False, seed=42)
        seen = {}

        def fake_score(endpoint, served, task, **kw):
            seen.update(kw)
            return SimpleNamespace(records=[])

        import yaml
        base = yaml.safe_load((ROOT / "lora_code" / "configs" / "base.yaml").read_text())
        td = base["train_data"]
        self.assertEqual((td["temperature"], td["top_p"], td["top_k"], td["samples"]), (0.7, 0.8, 20, 4))
        run = Run(rc, {"train_data": base["train_data"]})
        with patch("rsicore.evaluate.score", side_effect=fake_score), \
                patch("lora_code.run.load_records", return_value={}):
            run.build_train_data("dapo_math", ["a"], Path(tmp.name) / "td.json", endpoint="http://x", served="m",
                                 mode="self_correct_then_gold")
        self.assertEqual((seen["temperature"], seen["top_p"], seen["top_k"], seen["seed"]), (0.7, 0.8, 20, 42))
        self.assertEqual(seen["extra"], ["--n", "4"])


    def test_items_without_a_correct_sample_are_skipped(self):
        """train_data.mode self_correct (default): first correct sample kept; no correct sample -> no target
        (skipped by training), not the gold answer; self_correct_then_gold falls back to gold."""
        import yaml
        base = yaml.safe_load((ROOT / "lora_code" / "configs" / "base.yaml").read_text())
        self.assertEqual(base["train_data"]["mode"], "self_correct")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        rc = SimpleNamespace(domain="math", run_dir=Path(tmp.name), mock=False, seed=42)

        def fake_score(endpoint, served, task, **kw):
            return SimpleNamespace(records=[{"id": "a", "raw_output": ["w1", "w2"], "sample_scores": [0, 0]},
                                            {"id": "b", "raw_output": ["w", "right"], "sample_scores": [0, 1]}])
        run = Run(rc, {"train_data": base["train_data"]})
        for mode, want_a in (("self_correct", (None, None)), ("self_correct_then_gold", ("GOLD", "gold"))):
            with patch("rsicore.evaluate.score", side_effect=fake_score), \
                    patch("lora_code.run.load_records", return_value={"a": {}, "b": {}}), \
                    patch("lora_code.run.render_messages", return_value=[{"role": "user", "content": "q"}]), \
                    patch("lora_code.run.gold_target", return_value="GOLD"):
                ex = run.build_train_data("dapo_math", ["a", "b"], Path(tmp.name) / f"{mode}.json",
                                          endpoint="http://x", served="m", mode=mode)
            self.assertEqual((ex["b"]["target"], ex["b"]["source"]), ("right", "self"))
            self.assertEqual((ex["a"]["target"], ex["a"]["source"]), want_a)

    def test_answers_are_sampled_only_when_the_raw_lora_is_trained(self):
        """(mock runs force train_data.mode=gold, so this is checked here, not in the mock flow)"""
        rc = SimpleNamespace(domain="math", run_dir=Path(tempfile.mkdtemp()), mock=False)
        td = {"train_data": {"mode": "self_correct"}}
        edit = Run(rc, dict(td, method={"mode": "edit"}))
        self.assertEqual(edit.train_data_mode(0), "self_correct")             # the only training of a run
        self.assertEqual([edit.train_data_mode(g) for g in (1, 2, 5)], ["gold"] * 3)
        scratch = Run(rc, dict(td, method={"mode": "scratch"}))
        self.assertEqual([scratch.train_data_mode(g) for g in (0, 1, 2)], ["gold"] * 3)


class MaxTokensTest(unittest.TestCase):
    """max_tokens in the paper config reaches every run_eval call as --max-tokens."""

    def tearDown(self):
        from rsicore.evaluate import set_max_tokens
        set_max_tokens(None)

    def test_score_passes_the_configured_length(self):
        import yaml
        from rsicore import evaluate as E
        for paper in ("lora_code", "edit_code"):
            mt = yaml.safe_load((ROOT / paper / "configs" / "base.yaml").read_text())["max_tokens"]
            self.assertEqual(mt, {"dapo_math": 4096, "taco_verified": 4096, "ifeval_train": 2048,
                                  "sealtool_train": 1024, "triviaqa_train": 1024}, paper)   # no math_eval (2026-11-22)
        E.set_max_tokens({"dapo_math": 1234})
        cmds = []

        def fake_run(cmd, **kw):
            cmds.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch("rsicore.evaluate.subprocess.run", side_effect=fake_run):
            E.score("http://x", "m", "dapo_math", pool="pool", ids=None, out_dir=Path(tmp.name) / "a")
            E.score("http://x", "m", "gsm8k", pool="test", ids=None, out_dir=Path(tmp.name) / "b")
            E.score("http://x", "m", "dapo_math", pool="pool", ids=None, out_dir=Path(tmp.name) / "c",
                    extra=["--max-tokens", "77"])
        a, b, c = cmds
        self.assertEqual(a[a.index("--max-tokens") + 1], "1234")
        self.assertNotIn("--max-tokens", b)                         # unlisted task: tasks.yaml value
        self.assertEqual(c.count("--max-tokens"), 1)                 # an explicit extra wins
        with self.assertRaises(SystemExit):
            E.set_max_tokens({"dapo_mth": 10})
        with self.assertRaises(SystemExit):
            E.set_max_tokens({"dapo_math": 0})


if __name__ == "__main__":
    unittest.main()
