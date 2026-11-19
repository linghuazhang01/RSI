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

    def test_paper_b(self):
        from lora_code.common import gpu_layout
        h20 = {"gpu_layout": {"theta0_own_gpu": True}}

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


if __name__ == "__main__":
    unittest.main()
