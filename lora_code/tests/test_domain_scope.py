"""Domain restrictions must exclude non-target data while preserving the default experiment."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from lora_code.common import DOMAIN_DATASET, DOMAINS
from lora_code.run import Run


class DomainScopeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rc = SimpleNamespace(domain="math", run_dir=Path(self.tmp.name))

    def test_math_only_never_reads_non_math_gate_or_kl_data(self):
        run = Run(self.rc, {"retention": {"domains": []}})
        with patch("rsicore.draws.gate_ids", side_effect=lambda rc, ds: ["math-id"]
                   if ds == "dapo_math" else self.fail(f"non-math data accessed: {ds}")):
            self.assertEqual(run.gate_level(), {"dapo_math": ["math-id"]})
        with patch("lora_code.run.load_records", side_effect=AssertionError("cross-domain KL read")):
            self.assertEqual(run.kl_texts(0), [])
        self.assertEqual(run.retention_domains, [])

    def test_default_keeps_all_original_datasets(self):
        run = Run(self.rc, {})
        self.assertEqual(run.datasets, [DOMAIN_DATASET[d] for d in DOMAINS])
        self.assertEqual(run.retention_domains, [d for d in DOMAINS if d != "math"])

    def test_rejects_target_or_unknown_retention_domain(self):
        for domains in (["math"], ["unknown"]):
            with self.subTest(domains=domains), self.assertRaises(ValueError):
                Run(self.rc, {"retention": {"domains": domains}})


class GpuLayoutTest(unittest.TestCase):
    """Every given GPU is used (rsicore.gpu_layout); gpu_roles pins roles, the rest are laid out around them."""

    def test_paper_b_replicas(self):
        """Default since 2026-11-20: train = 1 GPU, theta0 its own GPU only with E_t, main = the rest as replicas."""
        from lora_code.common import gpu_layout, main_groups
        h20 = {"gpu_layout": {"theta0_own_gpu": True}}

        def cfg(e_t, **lay):
            return {"gpu_layout": dict(h20["gpu_layout"], **lay), "e_t": {"enabled": e_t}}

        def rc(n):
            return SimpleNamespace(gpus=list(range(n)))
        self.assertEqual(gpu_layout(rc(4), cfg(False))[0], {"main": [0, 1, 2], "train": [3], "theta0": [3]})
        self.assertEqual(main_groups(rc(4), cfg(False)), [[0], [1], [2]])          # 3 replicas
        self.assertEqual(gpu_layout(rc(4), cfg(True))[0], {"main": [0, 1], "train": [3], "theta0": [2]})
        self.assertEqual(main_groups(rc(5), cfg(False, main_tp=2)), [[0, 1], [2, 3]])
        self.assertEqual(gpu_layout(rc(5), cfg(False, train_gpus=2))[0]["train"], [3, 4])
        self.assertEqual(gpu_layout(rc(2), cfg(False))[0], {"main": [0], "train": [1], "theta0": [1]})
        with self.assertRaises(ValueError):
            gpu_layout(rc(2), cfg(True))                                            # E_t on h20 needs 3
        for n in range(2, 9):
            self.assertEqual(gpu_layout(rc(n), cfg(False))[1], [], n)                # nothing idle
        roles, _ = gpu_layout(rc(4), dict(cfg(True), gpu_roles={"train": [2, 3]}))  # pinned: one main server
        self.assertEqual(main_groups(rc(4), dict(cfg(True), gpu_roles={"train": [2, 3]})), [roles["main"]])

    def test_paper_b(self):
        """The earlier layout (gpu_layout.main_replicas: false): main grows by tensor parallelism."""
        from lora_code.common import gpu_layout
        h20 = {"gpu_layout": {"theta0_own_gpu": True, "main_replicas": False}}

        def lay(n, e_t, **extra):
            return gpu_layout(SimpleNamespace(gpus=list(range(n))), dict(h20, e_t={"enabled": e_t}, **extra))
        self.assertEqual(lay(2, False)[0]["main"], [0])
        self.assertEqual(lay(3, False)[0], {"main": [0, 1], "train": [2], "theta0": [2]})
        self.assertEqual(lay(5, False), ({"main": [0, 1], "train": [2, 3, 4], "theta0": [2]}, []))
        self.assertEqual(lay(4, True)[0], {"main": [0, 1], "theta0": [2], "train": [3]})
        for n in range(3, 9):
            self.assertEqual(lay(n, True)[1], [], n)                     # nothing idle
        with self.assertRaises(ValueError):
            lay(2, True)                                                  # E_t on h20 needs 3
        roles, _ = lay(4, True, gpu_roles={"train": [2, 3]})
        self.assertEqual(roles, {"train": [2, 3], "main": [0], "theta0": [1]})
        with self.assertRaises(ValueError):
            lay(2, False, gpu_roles={"train": [5]})

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


class ScopeConfigTest(unittest.TestCase):
    """scope=single (rsicore.config scope_profiles) restricts both papers to the run's domain."""

    def load(self, paper, domain, *sets):
        from rsicore.config import build_arg_parser, load_run_config
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        argv = ["--domain", domain, "--mock", "--runs-dir", tmp]
        for x in sets:
            argv += ["--set", x]
        return load_run_config(paper, build_arg_parser(paper).parse_args(argv)).paper_cfg

    def test_default_scope_is_single(self):
        b = self.load("B", "math")
        self.assertEqual(b["scope"], "single")
        self.assertEqual(b["retention"]["domains"], [])
        self.assertEqual(b["heldout"]["domains"], "math")
        self.assertFalse(b["e_t"]["enabled"])
        a = self.load("A", "math")
        self.assertEqual((a["scope"], a["kl"]["domains"], a["heldout"]["domains"]), ("single", "run", "math"))
        # the original design on request; Paper A --domain all always uses scope all
        b = self.load("B", "math", "scope=all")
        self.assertIsNone(b["retention"]["domains"])
        self.assertEqual(b["heldout"]["domains"], "all")
        self.assertTrue(b["e_t"]["enabled"])
        a = self.load("A", "all")
        self.assertEqual((a["scope"], a["kl"]["domains"], a["heldout"]["domains"]), ("all", "all", "all"))

    def test_default_run_id_marks_single(self):
        from rsicore.config import build_arg_parser, load_run_config
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        rc = load_run_config("B", build_arg_parser("B").parse_args(["--domain", "math", "--runs-dir", tmp, "--mock"]))
        self.assertEqual(rc.run_id, "B_math_s42_single_mock")
        rc = load_run_config("B", build_arg_parser("B").parse_args(["--domain", "math", "--runs-dir", tmp, "--mock",
                                                                    "--set", "scope=all"]))
        self.assertEqual(rc.run_id, "B_math_s42_mock")

    def test_single_scope_paper_b(self):
        b = self.load("B", "code", "scope=single", "hardware=h20")
        self.assertEqual(b["retention"]["domains"], [])
        self.assertEqual(b["heldout"]["domains"], "code")
        self.assertFalse(b["e_t"]["enabled"])
        from lora_code.common import gpu_layout
        roles, unused = gpu_layout(SimpleNamespace(gpus=[3, 5]), b)   # no E_t: no theta0 card needed on h20
        self.assertEqual((roles["main"], roles["train"], unused), ([3], [5], []))
        run = Run(SimpleNamespace(domain="code", run_dir=Path(tempfile.mkdtemp())), b)
        self.assertEqual(run.datasets, [DOMAIN_DATASET["code"]])
        # explicit --set still wins over the profile
        b2 = self.load("B", "code", "scope=single", "e_t.enabled=true", "gpu_roles.train=[2,3]")
        self.assertTrue(b2["e_t"]["enabled"])
        self.assertEqual(b2["gpu_roles"]["train"], [2, 3])

    def test_single_scope_paper_a(self):
        a = self.load("A", "knowledge", "scope=single")
        self.assertEqual((a["kl"]["domains"], a["heldout"]["domains"]), ("run", "knowledge"))
        with self.assertRaises(SystemExit):
            self.load("A", "all", "scope=single")


class DataSizeAndNoiseTest(unittest.TestCase):
    """data: sizes override the draw config per run; fixed noise band and the noise subset."""

    def rc(self, paper, *sets, tmp=None, seed=None):
        from rsicore.config import build_arg_parser, load_run_config
        if tmp is None:
            tmp = tempfile.mkdtemp()
            self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        argv = ["--domain", "math", "--mock", "--runs-dir", tmp, "--run-id", "t"]
        if seed is not None:
            argv += ["--seed", str(seed)]
        for x in sets:
            argv += ["--set", x]
        return load_run_config(paper, build_arg_parser(paper).parse_args(argv)), tmp

    def test_one_seed_for_draws_and_heldout(self):
        """--seed drives the per-generation draws (effective draw config) and the held-out decoding seed;
        the gate split keeps its own fixed gate.seed."""
        import yaml
        for paper in ("A", "B"):
            rc, _ = self.rc(paper)                                # default 42 = the draw configs' seed
            self.assertEqual(rc.paper_cfg["heldout"]["decoding"]["seed"], 42)
            rc, _ = self.rc(paper, seed=7)
            self.assertEqual(rc.paper_cfg["heldout"]["decoding"]["seed"], 7)
            eff = yaml.safe_load(rc.draw_config.read_text())
            self.assertEqual((eff["seed"], eff["gate"]["seed"]), (7, 20261118))

    def test_simple_profile_paper_b(self):
        """objective simple v2, FINAL flow (2026-11-21): thinking-mode sampled reference screen, gate and held-out
        (avg@4) with the run seed, held-out = the test set, 8 x 1 search, candidates not screened, one gate
        evaluation each, no api.evaluate, tensor / paper / screen tools on, persistent run_python, nonllm 8; other
        objectives keep greedy scoring and the old search accounting."""
        from rsicore.config import stage_decoding

        from lora_code.run import context_shortfall
        from lora_code.tools import (paper_tools_enabled, screen_tools_enabled, search_settings,
                                     tensor_compute_enabled)
        rc, _ = self.rc("B", "objective=simple", seed=7)
        cfg = rc.paper_cfg
        samp = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "thinking": True, "max_tokens": 16384, "seed": 7}
        self.assertEqual(stage_decoding(cfg, "screen", 7), dict(samp, n=4))
        self.assertEqual(stage_decoding(cfg, "full", 7), dict(samp, n=4))          # full falls back to screen
        self.assertEqual(stage_decoding(cfg, "gate", 7), dict(samp, n=4))
        hd = cfg["heldout"]["decoding"]
        self.assertEqual({k: hd[k] for k in samp}, samp)
        self.assertEqual((hd["n"], hd.get("n_by_task"), cfg["heldout"]["enabled"]), (4, None, True))
        self.assertEqual(cfg["data_roles"]["eval"], "eval:simple_math_test")
        self.assertEqual(search_settings(cfg), {"threads": 8, "steps": 1, "charge_screens": False, "pool": 0,
                                                "program_cap": 0, "max_candidates": 8, "parallel_builds": True,
                                                "screen_candidates": False, "gate_every_candidate": True,
                                                "evaluate_tool": False})
        self.assertEqual(cfg["search"]["top_full"], 0)
        self.assertTrue(tensor_compute_enabled(cfg) and paper_tools_enabled(cfg) and screen_tools_enabled(cfg))
        self.assertTrue(cfg["editor"]["explore"]["persistent"])
        self.assertEqual((cfg["nonllm"]["evaluate"], cfg["e_t"]["enabled"], cfg["editor"]["tool_rounds"]),
                         (8, False, 20))
        self.assertEqual(cfg["method"]["mode"], "scratch")
        self.assertIsNone(context_shortfall(cfg))                   # main replicas 131072 >= prompt + 16384
        small = dict(cfg, editor=dict(cfg["editor"], context_tokens=18432))
        self.assertIn("--max-model-len 18432", context_shortfall(small))
        for hw in ("h20", "h20_80"):
            rc, _ = self.rc("B", "objective=simple", f"hardware={hw}")
            self.assertIsNone(context_shortfall(rc.paper_cfg))
        rc, _ = self.rc("B")                                        # train_gate: unchanged
        cfg = rc.paper_cfg
        self.assertEqual([stage_decoding(cfg, lv, 42) for lv in ("screen", "full", "gate")], [None, None, None])
        self.assertEqual(search_settings(cfg), {"threads": 4, "steps": 8, "charge_screens": True, "pool": 32,
                                                "program_cap": 2, "max_candidates": 32, "parallel_builds": False,
                                                "screen_candidates": True, "gate_every_candidate": False,
                                                "evaluate_tool": True})
        self.assertFalse(tensor_compute_enabled(cfg) or paper_tools_enabled(cfg) or screen_tools_enabled(cfg))
        self.assertIsNone(context_shortfall(cfg))

    def test_eval_overfit_objective(self):
        """objective eval_overfit: the run's data is math_eval (MATH-500 + GSM8K test); every role and the gate are
        all of it, kl a fixed subset; nothing is drawn or marked shown; accept on any higher score; run id _evalfit."""
        import json as _json
        from rsicore import draws
        for paper in ("A", "B"):
            tmp = tempfile.mkdtemp()
            self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
            from rsicore.config import build_arg_parser, load_run_config
            argv = ["--domain", "math", "--mock", "--runs-dir", tmp, "--set", "objective=eval_overfit"]
            rc = load_run_config(paper, build_arg_parser(paper).parse_args(argv))
            self.assertTrue(rc.run_id.endswith("_single_evalfit_mock"), rc.run_id)
            cfg = rc.paper_cfg
            self.assertEqual((cfg["objective"], cfg["eval_target"]["dataset"]), ("eval_overfit", "math_eval"))
            gate = cfg["gate"]
            self.assertFalse(gate["require_significant"])
            self.assertEqual((cfg["heldout"]["domains"], cfg["heldout"]["decoding"]), ("eval_target", None))
            self.assertLess(gate.get("min_delta", gate.get("min_gain")), 1e-3)
            self.assertEqual(draws.eval_dataset(rc), "math_eval")
            allids = draws.gate_ids(rc, "math_eval")
            self.assertEqual(len(allids), 1819)
            for role in ("signal", "screen", "train"):
                self.assertEqual(draws.ids(rc, 3, "math_eval", role), allids)
            kl = draws.ids(rc, 0, "math_eval", "kl")
            self.assertEqual(kl, draws.ids(rc, 5, "math_eval", "kl"))                 # same every generation
            self.assertTrue(set(kl) <= set(allids) and len(kl) == 400)
            self.assertEqual(draws.ensure_gate(rc), {"math": {"math_eval": allids}})
            draws.draw_gen(rc, 0, target="math")
            draws.mark_shown(rc, 0, {"math_eval": allids[:3]})
            self.assertFalse((Path(tmp) / rc.run_id / "draws").exists())             # nothing drawn / marked
        meta = _json.loads((ROOT / "data" / "splits" / "math_eval" / "BUILD.json").read_text())
        self.assertEqual(meta["by_source"], {"math": 500, "gsm8k": 1319})
        rc, _ = self.rc("A")                                                          # default objective unchanged
        self.assertIsNone(draws.eval_target(rc))
        self.assertEqual(rc.paper_cfg["objective"], "train_gate")

    def test_eval_overfit_any_single_eval_set(self):
        """2026-11-21: any single eval set (code / instruction following / knowledge ...), one per run."""
        import json as _json
        from rsicore import draws, evalsets
        from rsicore.config import build_arg_parser, load_run_config
        from rsicore.heldout import heldout_domains, heldout_tasks

        def load(paper, domain, *sets):
            tmp = tempfile.mkdtemp()
            self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
            argv = ["--domain", domain, "--mock", "--runs-dir", tmp, "--set", "objective=eval_overfit"]
            for x in sets:
                argv += ["--set", x]
            return load_run_config(paper, build_arg_parser(paper).parse_args(argv))
        cases = [("code", "eval_target.dataset=humanevalplus", "humanevalplus_eval", "_evalfit_humanevalplus"),
                 ("code", "eval_target.dataset=mbppplus_eval", "mbppplus_eval", "_evalfit_mbppplus"),
                 ("instruction_following", None, "ifeval_eval", "_evalfit_ifeval"),          # domain default
                 ("knowledge", "eval_target.dataset=gpqa_diamond", "gpqa_diamond_eval", "_evalfit_gpqa_diamond"),
                 ("math", "eval_target.dataset=aime24", "aime24_eval", "_evalfit_aime24"),
                 ("math", "eval_target.dataset=math", "math_eval", "_evalfit")]
        for paper in ("A", "B"):
            for domain, s, ds, suffix in cases:
                rc = load(paper, domain, *([s] if s else []))
                self.assertTrue(rc.run_id.endswith(f"_single{suffix}_mock"), rc.run_id)
                self.assertEqual(draws.eval_dataset(rc), ds)
                self.assertEqual(rc.paper_cfg["heldout"]["domains"], "eval_target")    # locked config unchanged
                self.assertEqual(heldout_tasks(heldout_domains(rc.paper_cfg)), [(ds, "pool")])
        rc = load("B", "code", "eval_target.dataset=humanevalplus")
        allids = draws.gate_ids(rc, "humanevalplus_eval")
        src = _json.loads((ROOT / "data" / "splits" / "humanevalplus" / "test.json").read_text())
        self.assertEqual(sorted(allids), sorted(src))                                   # the whole test set
        self.assertEqual(draws.ids(rc, 2, "humanevalplus_eval", "train"), allids)
        self.assertEqual(len(draws.ids(rc, 0, "humanevalplus_eval", "kl")), len(allids))   # kl_n > n: all
        for bad in [("code", "eval_target.dataset=ifeval"), ("math", "eval_target.dataset=nope")]:
            with self.assertRaises(SystemExit):
                load("A", *bad)
        self.assertEqual(evalsets.targets()["livecodebench"]["domain"], "code")
        from evallib.tasks import load_tasks
        import yaml as _yaml
        tasks = load_tasks(_yaml.safe_load((ROOT / "eval" / "configs" / "tasks.yaml").read_text()))
        t, base = tasks["humanevalplus_eval"], tasks["humanevalplus"]
        self.assertEqual((t.dataset, t.cfg["grader"], t.cfg["allowed_pools"]), ("humanevalplus_eval", "humanevalplus", ["pool"]))
        self.assertEqual(type(t), type(base))
        self.assertEqual(tasks["math_eval"].cfg["max_tokens"], 4096)                    # explicit entry kept
        self.assertEqual(tasks["livecodebench_eval"].cfg["max_tokens"], 8192)           # inherited

    def test_no_data_sizes_keeps_the_draw_config(self):
        from rsicore.config import DRAW_CONFIGS
        rc, _ = self.rc("B")
        self.assertEqual(rc.draw_config, DRAW_CONFIGS["B"])
        rc, _ = self.rc("A", "scope=all")                        # KL texts from all five domains
        self.assertEqual(rc.draw_config, DRAW_CONFIGS["A"])

    def test_single_domain_paper_a_puts_every_kl_text_in_its_domain(self):
        import yaml
        rc, _ = self.rc("A")                                     # scope single: kl.domains = run
        eff = yaml.safe_load(rc.draw_config.read_text())
        self.assertEqual((eff["kl_domains"], eff["kl_contexts"]), (["math"], 400))       # 2026-11-20: was 256
        self.assertEqual(eff["screen_n"], 400)
        rc, _ = self.rc("A", "data.kl_contexts=64")
        self.assertEqual(yaml.safe_load(rc.draw_config.read_text())["kl_domains"], ["math"])

    def test_data_sizes_write_an_effective_draw_config(self):
        import yaml
        rc, tmp = self.rc("B", "data.n=300", "data.gate_n={math: 120}", "data.screen_n=40",
                          "data.target_train_frac=0.5")
        self.assertEqual(rc.draw_config.name, "draw_config.effective.yaml")
        eff = yaml.safe_load(rc.draw_config.read_text())
        self.assertEqual(eff["domains"]["math"]["n"], 300)
        self.assertEqual(eff["domains"]["code"]["n"], 300)              # a number applies to every domain
        self.assertEqual(eff["domains"]["math"]["gate_n"], 120)
        self.assertEqual(eff["domains"]["code"]["gate_n"], 1000)        # unlisted domains keep the draw value
        self.assertEqual((eff["screen_n"], eff["target_train_frac"]), (40, 0.5))
        txt = rc.draw_config.read_text()
        self.rc("B", "data.n=300", "data.gate_n={math: 120}", "data.screen_n=40", "data.target_train_frac=0.5",
                tmp=tmp)                                                # resume: identical file, lock passes
        self.assertEqual(rc.draw_config.read_text(), txt)
        with self.assertRaises(SystemExit):                             # other sizes on the same run-id
            self.rc("B", "data.n=500", tmp=tmp)
        self.assertEqual(rc.draw_config.read_text(), txt)               # refused before rewriting the file
        a, _ = self.rc("A", "data.kl_contexts=32", "data.n=100")
        eff = yaml.safe_load(a.draw_config.read_text())
        self.assertEqual((eff["kl_contexts"], eff["domains"]["math"]["n"]), (32, 100))

    def test_data_sizes_are_validated(self):
        for p, bad in (("B", "data.kl_contexts=32"), ("A", "data.retention_n=10"), ("B", "data.n=0"),
                       ("B", "data.n=1.5"), ("B", "data.target_train_frac=1.2"), ("B", "data.gate_n={maths: 1}")):
            with self.assertRaises(SystemExit, msg=bad):
                self.rc(p, bad)

    def test_greedy_noise_band_runs_nothing(self):
        from rsicore.gate import measure_noise_band
        with patch("rsicore.evaluate.score", side_effect=AssertionError("greedy must not re-run")):
            nb = measure_noise_band("http://x", "m", "dapo_math", ["a", "b"], Path(tempfile.mkdtemp()), temperature=0.0)
        self.assertEqual((nb["band"], nb["n"]), (0.0, 0))
        self.assertTrue(nb["source"].startswith("greedy"))

    def test_search_and_gate_greedy_heldout_sampled(self):
        """2026-11-20: search / gate / noise greedy (tasks.yaml); held-out sampled at 0.7 / 0.8 / top_k 20 avg@n."""
        import yaml
        from rsicore.heldout import decoding_args
        tasks = yaml.safe_load((ROOT / "eval" / "configs" / "tasks.yaml").read_text())
        self.assertEqual(tasks["defaults"]["temperature"], 0.0)
        for name, t in tasks["tasks"].items():
            self.assertEqual(t.get("temperature", 0.0), 0.0, name)
            self.assertEqual(t.get("n", 1), 1, name)
        a = yaml.safe_load((ROOT / "edit_code" / "configs" / "base.yaml").read_text())
        b = yaml.safe_load((ROOT / "lora_code" / "configs" / "base.yaml").read_text())
        self.assertEqual((a["noise"]["temperature"], b["baseline"]["noise_temperature"]), (0.0, 0.0))
        for cfg in (a, b):
            dec = cfg["heldout"]["decoding"]
            self.assertEqual((dec["temperature"], dec["top_p"], dec["top_k"], dec["seed"]), (0.7, 0.8, 20, None))
            dec = dict(dec, seed=42)                              # resolved to the run's --seed at load time
            d = decoding_args(dec, "aime24")
            self.assertEqual(d["extra"], ["--n", "8"])
            self.assertEqual(d["kw"], {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 42})
            self.assertEqual(decoding_args(dec, "gsm8k")["extra"], ["--n", "1"])
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

    def test_fixed_noise_band_and_subset(self):
        from rsicore.gate import fixed_noise_band, noise_ids
        self.assertIsNone(fixed_noise_band(None, "dapo_math"))
        self.assertEqual(fixed_noise_band(0.03, "dapo_math"), 0.03)
        spec = {"math": 0.04, "taco_verified": 0.05, "default": 0.02}
        self.assertEqual(fixed_noise_band(spec, "dapo_math"), 0.04)       # by domain
        self.assertEqual(fixed_noise_band(spec, "taco_verified"), 0.05)   # by dataset
        self.assertEqual(fixed_noise_band(spec, "ifeval_train"), 0.02)    # default
        self.assertIsNone(fixed_noise_band({"math": 0.04}, "ifeval_train"))
        with self.assertRaises(ValueError):
            fixed_noise_band(True, "dapo_math")
        ids = [f"x/{i}" for i in range(100)]
        sub = noise_ids(ids, 20, 42)
        self.assertEqual(len(sub), 20)
        self.assertEqual(sub, noise_ids(list(reversed(ids)), 20, 42))
        self.assertEqual(noise_ids(ids, None, 42), ids)
        self.assertEqual(noise_ids(ids, 500, 42), ids)

    def test_raw_train_size(self):
        from lora_code import baselines as BL
        r, n, gb = BL.raw_recipe({})
        self.assertIsNone(n)                                             # raw.n_train null -> all train items
        self.assertNotIn("n_train", r)
        r, n, gb = BL.raw_recipe({"raw": {"n_train": 40}})
        self.assertEqual(n, 40)
        self.assertFalse(hasattr(BL, "sft_n_train") or hasattr(BL, "train_sft"))   # no SFT baseline


class SamplingParamsTest(unittest.TestCase):
    """Training answers use train_data (2026-11-20: 4 samples at 0.7 / 0.8 / top_k 20, the run seed); noise is greedy."""

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
        self.assertEqual((base["baseline"]["noise_temperature"], base["baseline"]["noise_top_p"]), (0.0, 1.0))
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
                                  "sealtool_train": 1024, "triviaqa_train": 1024, "math_eval": 4096})
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
