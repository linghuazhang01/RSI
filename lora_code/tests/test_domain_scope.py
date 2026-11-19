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


if __name__ == "__main__":
    unittest.main()
