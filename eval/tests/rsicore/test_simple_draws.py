"""The simple flow (2026-11-21; the only flow since 2026-11-22): per-generation screen / train / kl draws of
simple_<d>_pool (rsicore.draws), the gate and held-out test sets, the data / registration they rest on, the default
run id, the removed selectors and the run lock across the removals. Config level only (no servers).

  python -I -m pytest -q eval/tests/rsicore/test_simple_draws.py
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

import yaml  # noqa: E402

from rsicore import config, draws, evalsets, heldout  # noqa: E402
from rsicore.config import build_arg_parser, load_run_config  # noqa: E402

DOMAINS = ("math", "code", "knowledge", "instruction_following", "tool_calling")


class SimpleDrawsTest(unittest.TestCase):
    def paper_dir(self, base: dict | None = None) -> str:
        """A minimal paper dir (the draws do not depend on the papers' base.yaml)."""
        pdir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(pdir, ignore_errors=True))
        (pdir / "configs" / "domains").mkdir(parents=True)
        (pdir / "configs" / "base.yaml").write_text(yaml.safe_dump(base or {"heldout": {"domains": "run"}}))
        return str(pdir)

    def load(self, paper, domain, *sets, run_id=None, pdir=None, runs=None):
        if runs is None:
            runs = tempfile.mkdtemp()
            self.addCleanup(lambda: shutil.rmtree(runs, ignore_errors=True))
        argv = ["--domain", domain, "--mock", "--runs-dir", str(runs), "--paper-dir", pdir or self.paper_dir()]
        if run_id:
            argv += ["--run-id", run_id]
        for x in sets:
            argv += ["--set", x]
        return load_run_config(paper, build_arg_parser(paper).parse_args(argv))

    def test_data_files(self):
        """test.jsonl / pool_ids.json exist; train / eval / test pairwise disjoint and inside the pool."""
        for d in DOMAINS:
            sd = ROOT / "simple_data" / d
            pool = json.loads((sd / "pool_ids.json").read_text())
            self.assertEqual(pool, sorted(set(pool)))
            parts = {}
            for name in ("train", "eval", "test"):
                rows = [json.loads(l) for l in (sd / f"{name}.jsonl").read_text().split("\n") if l.strip()]
                self.assertTrue(all(r["split_pool"] == name for r in rows))
                parts[name] = {r["id"] for r in rows}
            self.assertEqual([len(parts[n]) for n in ("train", "eval", "test")], [1500, 500, 500])
            names = list(parts)
            for i, a in enumerate(names):
                self.assertTrue(parts[a] <= set(pool))
                for b in names[i + 1:]:
                    self.assertTrue(parts[a].isdisjoint(parts[b]), (d, a, b))
            self.assertEqual(json.loads((sd / "build_report.json").read_text())["test"], 500)

    def test_registration_and_roles(self):
        t = evalsets.targets()
        self.assertTrue(all(k.startswith("simple_") for k in t))                # only the simple_data sets
        self.assertNotIn("math_eval", evalsets.task_names())
        for d in DOMAINS:
            self.assertEqual(t[f"simple_{d}_pool"]["file"], f"simple_data/{d}/pool_ids.json")
            self.assertEqual(t[f"simple_{d}_test"]["file"], f"simple_data/{d}/test.jsonl")
            self.assertTrue(evalsets.is_eval_set(f"simple_{d}_pool") and evalsets.is_eval_set(f"simple_{d}_test"))
            self.assertEqual(evalsets.simple_sets(d),
                             {"train": f"simple_{d}_pool", "gate": f"simple_{d}_eval", "test": f"simple_{d}_test"})
            self.assertTrue({f"simple_{d}_{x}" for x in ("pool", "eval", "test")} <= evalsets.task_names())
        self.assertFalse(evalsets.is_eval_set("dapo_math"))

    def test_draw_disjoint_and_sizes(self):
        """screen = signal (500) and train partition pool - gate - test; kl (400) inside train; files + manifest."""
        for d in ("math", "tool_calling"):
            rc = self.load("B", d)
            pool_ds = draws.train_dataset(rc)
            self.assertEqual((pool_ds, draws.gate_dataset(rc), draws.test_dataset(rc)),
                             tuple(evalsets.simple_sets(d)[k] for k in ("train", "gate", "test")))
            gate = set(draws.ensure_gate(rc)[d][f"simple_{d}_eval"])
            test = set(draws.build_eval_set(f"simple_{d}_test"))
            pool = set(draws.build_eval_set(pool_ds))
            m = draws.draw_gen(rc, 0, target=d)
            self.assertTrue(m["simple"] and m["target"] == d and m["dataset"] == pool_ds)
            scr, sig, tr, kl = (draws.ids(rc, 0, pool_ds, r) for r in ("screen", "signal", "train", "kl"))
            self.assertEqual((len(scr), len(kl)), (500, 400))
            self.assertEqual(scr, sig)
            self.assertEqual(set(scr) | set(tr), pool - gate - test)
            self.assertEqual(len(scr) + len(tr), len(pool - gate - test))
            self.assertTrue(set(kl) <= set(tr))
            for a, b in ((scr, gate), (scr, test), (tr, gate), (tr, test), (scr, tr)):
                self.assertTrue(set(a).isdisjoint(b))
            self.assertTrue(gate.isdisjoint(test))
            self.assertEqual(m["n"], {"signal": 500, "screen": 500, "train": len(tr), "kl": 400})
            for r in draws.ROLES:
                self.assertEqual(json.loads(draws.ids_file(rc, 0, pool_ds, r).read_text()),
                                 draws.ids(rc, 0, pool_ds, r))
            self.assertEqual(sorted(p.name for p in (rc.run_dir / "draws").iterdir()), ["gen_00"])
            self.assertEqual(sorted(p.name for p in (rc.run_dir / "draws" / "gen_00").iterdir()),
                             ["ids", "manifest.json"])
            self.assertEqual(draws.gate_ids(rc, f"simple_{d}_eval"), sorted(gate))

    def test_determinism_and_generations(self):
        """Same seed -> the same screens for Paper A and Paper B (any run id); each generation draws anew; a
        different --seed draws differently; files are reused on resume."""
        ra = self.load("A", "math", run_id="simple_det_a")
        rb = self.load("B", "math", run_id="simple_det_b")
        ds = "simple_math_pool"
        # Paper A searches in generations 1..N, Paper B in 0..N-1: A's g and B's g-1 are the same search round
        sa = [draws.ids(ra, g, ds, "screen") for g in range(1, 4)]
        sb = [draws.ids(rb, g, ds, "screen") for g in range(3)]
        self.assertEqual(sa, sb)
        self.assertEqual(draws.search_round(ra, 1), draws.search_round(rb, 0))
        for r in ("train", "kl"):
            self.assertEqual(draws.ids(ra, 2, ds, r), draws.ids(rb, 1, ds, r))
        self.assertTrue(sa[0] != sa[1] != sa[2] and sa[0] != sa[2])
        self.assertLess(len(set(sa[0]) & set(sa[1])), 100)                     # independent (overlap ~ 500^2 / 13.7k)
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        argv = ["--domain", "math", "--mock", "--runs-dir", tmp, "--seed", "7", "--paper-dir", self.paper_dir()]
        rc7 = load_run_config("A", build_arg_parser("A").parse_args(argv))
        self.assertNotEqual(draws.ids(rc7, 1, ds, "screen"), sa[0])
        # resume: the files win over a recomputation (here: a hand-edited file after clearing the process cache)
        f = draws.ids_file(ra, 2, ds, "screen")
        f.write_text(json.dumps(sa[1][:3]))
        draws._SIMPLE.clear()
        self.assertEqual(draws.ids(ra, 2, ds, "screen"), sa[1][:3])

    def test_screen_size_key(self):
        rc = self.load("A", "code", "simple.screen_n=20")
        self.assertEqual(draws.simple_screen_n(rc), 20)
        self.assertEqual(len(draws.ids(rc, 0, "simple_code_pool", "screen")), 20)
        self.assertEqual(draws.simple_screen_n(self.load("A", "code")), 500)
        with self.assertRaises(ValueError):                                     # only the train role is drawn
            draws.ids(rc, 0, "simple_code_eval", "screen")

    def test_heldout_builds_test_set(self):
        """heldout_domains gives eval:simple_<d>_test; run_heldout's ensure_built builds it before it is read."""
        self.assertEqual(heldout.heldout_domains({}, "code"), "eval:simple_code_test")
        self.assertEqual(heldout.heldout_domains({"domain": "math"}), "eval:simple_math_test")          # Paper A
        self.assertEqual(self.load("B", "knowledge").paper_cfg["heldout"]["domains"], "knowledge")      # run -> domain
        self.assertEqual(heldout.heldout_domains(self.load("B", "knowledge").paper_cfg), "eval:simple_knowledge_test")
        with self.assertRaises(ValueError):
            heldout.heldout_domains({"heldout": {"domains": "all"}})
        with self.assertRaises(ValueError):
            heldout.heldout_tasks("math")                                       # public benchmarks: removed
        heldout.ensure_built("eval:simple_tool_calling_test")
        self.assertEqual(heldout.heldout_tasks("eval:simple_tool_calling_test"), [("simple_tool_calling_test", "pool")])
        self.assertEqual(len(heldout._task_ids("simple_tool_calling_test", "pool")), 500)
        self.assertTrue((ROOT / "data" / "processed" / "simple_tool_calling_test.jsonl").exists())

    def test_run_id_and_removed_selectors(self):
        """Default run id keeps the _single_simple suffix; objective=simple / scope=single are accepted no-ops (--set
        and --config); other values, the removed data keys and --domain all are refused."""
        for paper in ("A", "B"):
            rc = self.load(paper, "math", "objective=simple", "scope=single")
            self.assertEqual(rc.run_id, f"{paper}_math_s42_single_simple_mock")
            self.assertNotIn("objective", rc.paper_cfg)
        for bad in ("objective=train_gate", "objective=eval_overfit", "scope=all", "data_roles.eval=test",
                    "data.n=30", "eval_target.dataset=math"):
            with self.assertRaises(SystemExit, msg=bad):
                self.load("B", "math", bad)
        self.assertEqual(self.load("B", "math", "data.screen_n=30").paper_cfg["simple"]["screen_n"], 30)  # renamed
        with self.assertRaises(SystemExit):
            build_arg_parser("A").parse_args(["--domain", "all"])
        f = Path(self.paper_dir()) / "exp.yaml"
        f.write_text("objective: simple\nscope: single\nsearch: {threads: 2}\n")
        argv = ["--domain", "math", "--mock", "--runs-dir", tempfile.mkdtemp(), "--paper-dir", self.paper_dir(),
                "--config", str(f)]
        rc = load_run_config("B", build_arg_parser("B").parse_args(argv))
        self.assertEqual(rc.paper_cfg["search"], {"threads": 2})
        self.assertFalse({"objective", "scope"} & set(rc.paper_cfg))
        f.write_text("objective: train_gate\n")
        with self.assertRaises(SystemExit):
            load_run_config("B", build_arg_parser("B").parse_args(argv))

    def test_lock_across_the_removals(self):
        """A run locked before the removals (objective / scope profiles merged, data / data_roles / retention / E_t /
        draw_config keys, mock_objective_overrides, max_tokens.math_eval, theta0 hardware keys) resumes with the same
        command once base.yaml holds the folded values; a real difference is still refused; a run of a removed
        objective cannot be resumed."""
        new = {"search": {"threads": 8, "steps": 1}, "heldout": {"domains": "run", "enabled": True},
               "max_tokens": {"dapo_math": 4096}, "simple": {"screen_n": 500},
               "mock_overrides": {"search": {"threads": 3, "steps": 1}, "gate": {"min_delta": 0.0}},
               "hardware": "h20", "hardware_profiles": {"h20": {"gpu_layout": {"m0_replicas": True}}},
               "gpu_layout": {"m0_replicas": True}}
        old = {"search": {"threads": 8, "steps": 1}, "heldout": {"domains": "math", "enabled": True},
               "max_tokens": {"dapo_math": 4096, "math_eval": 4096}, "draw_config": "data/draw/paper_b.yaml",
               "data": {"n": None, "screen_n": None, "retention_n": None}, "scope": "single",
               "scope_profiles": {"single": {"retention": {"domains": []}}}, "objective": "simple",
               "objective_profiles": {"simple": {"search": {"steps": 1}}}, "data_roles": {"eval": "eval:simple_math_test"},
               "retention": {"domains": []}, "e_t": {"enabled": False}, "feedback": {"lambda_retention": 1.0},
               "server": {"theta0_args": ["--x"]},
               "mock_overrides": {"search": {"threads": 2, "steps": 2}, "gate": {"min_delta": 0.002}, "e_t": {"steps": 2}},
               "mock_objective_overrides": {"simple": {"search": {"threads": 3, "steps": 1}, "gate": {"min_delta": 0.0}}},
               "hardware": "h20", "gpu_layout": {"m0_replicas": True, "theta0_own_gpu": True},
               "hardware_profiles": {"h20": {"gpu_layout": {"m0_replicas": True, "theta0_own_gpu": True}}}}
        self.assertEqual(config.lock_view(old), config.lock_view(new))
        runs = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(runs, ignore_errors=True))
        rc = self.load("B", "math", run_id="r", pdir=self.paper_dir(new), runs=runs)
        lock = rc.run_dir / "run_config.yaml"
        d = yaml.safe_load(lock.read_text())
        lock.write_text(yaml.safe_dump(dict(d, paper_cfg=dict(old, heldout={"domains": "math", "enabled": True}),
                                            draw_config="data/draw/paper_b.yaml")))
        self.load("B", "math", "objective=simple", run_id="r", pdir=self.paper_dir(new), runs=runs)      # resumes
        with self.assertRaises(SystemExit) as e:
            self.load("B", "math", "search.threads=5", run_id="r", pdir=self.paper_dir(new), runs=runs)
        self.assertIn("paper_cfg.search.threads", str(e.exception))
        with self.assertRaises(SystemExit):                                     # data.screen_n 300 != 500
            lock.write_text(yaml.safe_dump(dict(d, paper_cfg=dict(old, data={"screen_n": 300}))))
            self.load("B", "math", run_id="r", pdir=self.paper_dir(new), runs=runs)
        lock.write_text(yaml.safe_dump(dict(d, paper_cfg=dict(old, objective="train_gate"))))
        with self.assertRaises(SystemExit) as e:
            self.load("B", "math", run_id="r", pdir=self.paper_dir(new), runs=runs)
        self.assertIn("removed", str(e.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
