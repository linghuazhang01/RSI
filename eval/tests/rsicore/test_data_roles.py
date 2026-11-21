"""Data roles (2026-11-21): which dataset a run trains on (train), accepts on (gate) and reports on (eval), resolved by
rsicore.config / rsicore.evalsets and read by rsicore.draws / rsicore.heldout. Config level only (no servers).

  python -I -m pytest -q eval/tests/rsicore/test_data_roles.py
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from rsicore import draws, evalsets  # noqa: E402
from rsicore.config import build_arg_parser, load_run_config  # noqa: E402
from rsicore.heldout import heldout_domains, heldout_tasks  # noqa: E402


class DataRolesTest(unittest.TestCase):
    def load(self, paper, domain, *sets):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        argv = ["--domain", domain, "--mock", "--runs-dir", tmp]
        for x in sets:
            argv += ["--set", x]
        return load_run_config(paper, build_arg_parser(paper).parse_args(argv))

    def test_defaults_unchanged(self):
        for paper in ("A", "B"):
            rc = self.load(paper, "math")
            self.assertNotIn("data_roles", rc.paper_cfg)                     # defaults stay implicit (run lock)
            self.assertEqual(draws.roles(rc), {"train": "dapo_math", "gate": "dapo_math"})
            self.assertEqual(heldout_tasks(heldout_domains(rc.paper_cfg)), [("math500", "test"), ("gsm8k", "test"),
                                                                            ("aime24", "test"), ("aime25", "test")])
            rc = self.load(paper, "code", "objective=eval_overfit", "eval_target.dataset=mbppplus")
            self.assertEqual(draws.roles(rc), {"train": "mbppplus_eval", "gate": "mbppplus_eval"})
            self.assertTrue(rc.run_id.endswith("_single_evalfit_mbppplus_mock"))

    def test_custom_roles(self):
        cases = [  # paper, domain, sets, roles, held-out tasks, run-id ending
            ("B", "math", ["data_roles.gate=math500", "data_roles.eval=math500,gsm8k"],
             ("dapo_math", "math500_eval"), [("math500", "test"), ("gsm8k", "test")], "_single_gate-math500_mock"),
            ("A", "code", ["data_roles.train=humanevalplus", "data_roles.gate=taco_verified", "data_roles.eval=code"],
             ("humanevalplus_eval", "taco_verified"), [("humanevalplus", "test"), ("mbppplus", "test"),
                                                       ("livecodebench", "test")], "_single_train-humanevalplus_mock"),
            ("B", "instruction_following", ["data_roles.gate=ifeval", "data_roles.eval=gate,ifbench"],
             ("ifeval_train", "ifeval_eval"), [("ifeval_eval", "pool"), ("ifbench", "test")], "_single_gate-ifeval_mock"),
            ("A", "math", ["objective=eval_overfit", "data_roles.gate=aime24", "data_roles.eval=math_eval,aime24"],
             ("math_eval", "aime24_eval"), [("math_eval", "pool"), ("aime24", "test")], "_single_evalfit_gate-aime24_mock"),
            ("B", "math", ["data_roles.train=dapo_math"],                       # = the default: no run-id suffix
             ("dapo_math", "dapo_math"), None, "_math_s42_single_mock"),
        ]
        for paper, domain, sets, (tr, gt), tasks, end in cases:
            rc = self.load(paper, domain, *sets)
            self.assertEqual(draws.roles(rc), {"train": tr, "gate": gt}, sets)
            self.assertTrue(rc.run_id.endswith(end), rc.run_id)
            if tasks:
                self.assertEqual(heldout_tasks(heldout_domains(rc.paper_cfg)), tasks)

    def test_eval_set_ids_and_shown(self):
        rc = self.load("B", "math", "data_roles.train=math500", "data_roles.gate=gsm8k")
        src = json.loads((ROOT / "data" / "splits" / "math" / "test.json").read_text())
        self.assertEqual(sorted(draws.ids(rc, 3, "math500_eval", "screen")), sorted(src))
        self.assertEqual(len(draws.gate_ids(rc, "gsm8k_eval")), 1319)
        self.assertEqual(draws.ensure_gate(rc), {"math": {"gsm8k_eval": draws.gate_ids(rc, "gsm8k_eval")}})
        self.assertEqual(draws.draw_gen(rc, 1)["eval_target"], "math500_eval")
        draws.mark_shown(rc, 1, {"math500_eval": src[:3]})                   # ignored: an eval set
        self.assertFalse((rc.run_dir / "draws").exists())                   # nothing drawn at all
        self.assertTrue(draws.is_eval_set("math_eval") and not draws.is_eval_set("dapo_math"))

    def test_simple_objective(self):
        """objective simple (v2, 2026-11-21): train role = the whole clean pool (simple_<d>_pool; a fresh 500-item
        screen every generation, drawn from pool - gate - test), gate = simple_<d>_eval (500), held-out =
        simple_<d>_test (500, data_roles.eval: test, set here explicitly: the base.yaml simple profiles belong to the
        papers); editor at most 20 tool rounds."""
        for paper in ("A", "B"):
            for d in ("math", "code", "knowledge", "instruction_following", "tool_calling"):
                rc = self.load(paper, d, "objective=simple", "data_roles.eval=test")
                r = draws.roles(rc)
                self.assertEqual(r, {"train": f"simple_{d}_pool", "gate": f"simple_{d}_eval"})
                self.assertTrue(rc.run_id.endswith("_single_simple_mock"), rc.run_id)
                self.assertTrue(rc.paper_cfg["heldout"]["enabled"])
                self.assertEqual(heldout_tasks(heldout_domains(rc.paper_cfg)), [(f"simple_{d}_test", "pool")])
                self.assertEqual(rc.paper_cfg["editor"]["tool_rounds"], 20)
                gate, test = draws.gate_ids(rc, r["gate"]), draws.build_eval_set(f"simple_{d}_test")
                pool = draws.build_eval_set(r["train"])
                self.assertEqual((len(gate), len(test)), (500, 500))
                self.assertEqual(sorted(pool), json.loads((ROOT / "simple_data" / d / "pool_ids.json").read_text()))
                scr2, scr3 = (draws.ids(rc, g, r["train"], "screen") for g in (2, 3))
                tr2 = draws.ids(rc, 2, r["train"], "train")
                self.assertEqual(len(scr2), 500)
                self.assertNotEqual(scr2, scr3)                                  # a fresh screen every generation
                self.assertEqual(draws.ids(rc, 2, r["train"], "signal"), scr2)
                self.assertEqual(set(scr2) | set(tr2), set(pool) - set(gate) - set(test))
                for a, b in ((scr2, tr2), (scr2, gate), (scr2, test), (tr2, gate), (tr2, test), (gate, test)):
                    self.assertTrue(set(a).isdisjoint(b))
                self.assertFalse((rc.run_dir / "draws" / "gate").exists())       # draw.py never ran
        rc = self.load("B", "math", "objective=simple", "data_roles.gate=simple_train")      # shorthand
        self.assertEqual(draws.roles(rc)["gate"], "simple_math_train")
        self.assertTrue(rc.run_id.endswith("_simple_gate-simple_math_train_mock"), rc.run_id)
        rc = self.load("A", "code", "objective=simple", "data_roles.eval=simple_test,gate")
        self.assertEqual(heldout_tasks(heldout_domains(rc.paper_cfg)), [("simple_code_test", "pool"),
                                                                        ("simple_code_eval", "pool")])
        rc = self.load("A", "math", "objective=simple", "data_roles.train=simple_train")      # the older 1,500 items
        self.assertEqual(len(draws.ids(rc, 1, "simple_math_train", "screen")), 1500)
        self.assertTrue(rc.run_id.endswith("_simple_train-simple_math_train_mock"), rc.run_id)

    def test_refusals(self):
        bad = [("A", "math", "data_roles.train=humanevalplus"),           # other domain
               ("B", "math", "data_roles.gate=nope"),                      # unknown
               ("A", "math", "data_roles.gate=math500", "scope=all"),      # eval set needs scope single
               ("B", "math", "data_roles.eval=train"),                     # train is a training set
               ("B", "math", "data_roles.eval=test"),                      # test needs objective simple
               ("B", "math", "data_roles.other=x")]
        for paper, domain, *sets in bad:
            with self.assertRaises(SystemExit, msg=sets):
                self.load(paper, domain, *sets)
        self.assertEqual(evalsets.short("humanevalplus_eval"), "humanevalplus")
        self.assertEqual(evalsets.short("dapo_math"), "dapo_math")


if __name__ == "__main__":
    unittest.main(verbosity=2)
