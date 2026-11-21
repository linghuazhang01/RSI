"""objective simple v2 (2026-11-21): per-generation screen / train / kl draws of simple_<d>_pool (rsicore.draws, no
draw.py), the held-out test set, and the data / registration they rest on. Config level only (no servers).

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

from rsicore import draws, evalsets, heldout  # noqa: E402
from rsicore.config import build_arg_parser, load_run_config  # noqa: E402

DOMAINS = ("math", "code", "knowledge", "instruction_following", "tool_calling")


class SimpleDrawsTest(unittest.TestCase):
    def load(self, paper, domain, *sets, run_id=None):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        argv = ["--domain", domain, "--mock", "--runs-dir", tmp]
        if run_id:
            argv += ["--run-id", run_id]
        for x in ("objective=simple",) + sets:
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
        for d in DOMAINS:
            self.assertEqual(t[f"simple_{d}_pool"]["file"], f"simple_data/{d}/pool_ids.json")
            self.assertEqual(t[f"simple_{d}_test"]["file"], f"simple_data/{d}/test.jsonl")
            self.assertTrue(evalsets.is_eval_set(f"simple_{d}_pool") and evalsets.is_eval_set(f"simple_{d}_test"))
            self.assertEqual(evalsets.default_roles({"objective": "simple"}, d),
                             {"train": f"simple_{d}_pool", "gate": f"simple_{d}_eval", "test": f"simple_{d}_test"})
            self.assertEqual(evalsets.resolve_set("simple_pool", d, "train"), f"simple_{d}_pool")
            self.assertEqual(evalsets.resolve_set("simple_test", d, "gate"), f"simple_{d}_test")
        roles = evalsets.default_roles({"objective": "simple"}, "math")
        self.assertEqual(evalsets.eval_tokens("test", roles), "eval:simple_math_test")
        self.assertEqual(evalsets.eval_tokens("simple_test,math500", roles, "math"), "eval:simple_math_test,math500")
        with self.assertRaises(SystemExit):
            evalsets.eval_tokens("test", evalsets.default_roles({}, "math"))

    def test_draw_disjoint_and_sizes(self):
        """screen = signal (500) and train partition pool - gate - test; kl (400) inside train; files + manifest."""
        for d in ("math", "tool_calling"):
            rc = self.load("B", d, "data_roles.eval=test")
            pool_ds = draws.train_dataset(rc)
            self.assertEqual(draws.simple_pool(rc), pool_ds)
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
            draws.mark_shown(rc, 0, {pool_ds: scr[:5], f"simple_{d}_eval": sorted(gate)[:2]})    # no-op
            self.assertEqual(sorted(p.name for p in (rc.run_dir / "draws").iterdir()), ["gen_00"])
            self.assertEqual(sorted(p.name for p in (rc.run_dir / "draws" / "gen_00").iterdir()),
                             ["ids", "manifest.json"])
            self.assertEqual(draws.gate_ids(rc, f"simple_{d}_eval"), sorted(gate))

    def test_determinism_and_generations(self):
        """Same seed -> the same screens for Paper A and Paper B (any run id); each generation draws anew; a
        different --seed draws differently; files are reused on resume."""
        ra = self.load("A", "math", run_id="simple_det_a")
        rb = self.load("B", "math", "data_roles.eval=test", run_id="simple_det_b")
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
        argv = ["--domain", "math", "--mock", "--runs-dir", tmp, "--seed", "7", "--set", "objective=simple"]
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
        rc = self.load("A", "code", "data.screen_n=30")
        self.assertEqual(len(draws.ids(rc, 0, "simple_code_pool", "screen")), 30)

    def test_heldout_builds_test_set(self):
        """run_heldout's ensure_built builds simple_<d>_test (data/processed + data/splits) before it is read."""
        heldout.ensure_built("eval:simple_tool_calling_test")
        self.assertEqual(len(heldout._task_ids("simple_tool_calling_test", "pool")), 500)
        self.assertTrue((ROOT / "data" / "processed" / "simple_tool_calling_test.jsonl").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
