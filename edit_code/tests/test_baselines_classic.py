"""Classic Paper A baselines (2026-11-23, edit_code/baselines.py + edit_code/evopress.py):
  - uniform_config swaps the reference's quantizer everywhere; hawq / kl_greedy fit the budget;
  - the EvoPress level-switch mutation keeps the budget and the GPTQ g128 levels; the search is deterministic and a
    resumed search (from its per-generation checkpoint) ends where an uninterrupted one does;
  - the KL fitness pool (spawned worker processes; CPU here) equals the KL of the exported fake-quant checkpoint.
The end-to-end mock run (gate + held-out, lock-ignored block) is test_mock_flow.test_baselines_still_run.
    python -I -m pytest -q edit_code/tests/test_baselines_classic.py
"""
from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

from edit_code import baselines as BL  # noqa: E402
from edit_code.quant import schema, sizes  # noqa: E402
from edit_code.quant.spec import synthetic_spec  # noqa: E402

try:
    import torch
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
    HAVE_TF = True
except Exception:  # pragma: no cover
    HAVE_TF = False

REF = {"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"}, "embed": 8, "lm_head": 8}


def _prob(spec):
    from edit_code import mock as M
    p = SimpleNamespace(spec=spec, space=dict(schema.DEFAULT_ACTION_SPACE), cfg={"mock": {"lam": 1.2}},
                        source_state=M.fresh_state(0.8))
    p.normalize = lambda c: schema.normalize(c, spec, p.space)
    p.tables = {"profile": M.profile_table(spec, p.source_state, [2, 3, 4, 8], 128)}
    p.tables["hessian"] = [dict(r, quantizer="gptq", herr_per_token=r["dkl_mean"] * 3.0) for r in p.tables["profile"]
                           if r["bits"] < 8]
    return p


class TestConfigs(unittest.TestCase):
    def setUp(self):
        self.spec = synthetic_spec(None)
        self.p = _prob(self.spec)
        self.budget = sizes.config_bytes(self.p.normalize(REF), self.spec)["gib"]

    def test_uniform_and_knapsacks(self):
        c = BL.uniform_config(dict(REF, rules=[{"layers": [0], "bits": 8, "group_size": 128, "quantizer": "gptq"}]),
                              "awq")
        self.assertEqual(c["default"]["quantizer"], "awq")
        self.assertEqual(c["rules"][0]["quantizer"], "awq")
        self.assertEqual(REF["default"]["quantizer"], "gptq")             # the reference is not modified
        for f in (BL.hawq_config, BL.kl_greedy_config):
            cfg = f(self.p, REF, self.budget)
            n = self.p.normalize(cfg)
            self.assertLessEqual(sizes.config_bytes(n, self.spec)["gib"], self.budget + 1e-9, f.__name__)
            self.assertEqual((n["embed"], n["lm_head"]), (8, 8))
            self.assertTrue({s["quantizer"] for s in n["groups"].values() if s["bits"] < 16} <= {"gptq"})
        with self.assertRaises(ValueError):
            BL.check_methods(["evo"])

    def test_mutation_and_resumable_search(self):
        ladder = BL.levels(self.p)
        rng = random.Random(0)
        n = {"groups": {k: BL._level(4) for k in self.spec.group_keys()}, "embed": 8, "lm_head": 8}
        for _ in range(50):
            n = BL.mutate(self.spec, self.budget, n, rng, ladder)
            self.assertLessEqual(sizes.config_bytes(n, self.spec)["gib"], self.budget + 1e-9)
            self.assertTrue(all(s["bits"] in ladder and (s["bits"] == 16 or (s["group_size"], s["quantizer"]) ==
                                                         (128, "gptq")) for s in n["groups"].values()))
        fit = (lambda norms, stage: BL.mock_fitness(self.p, norms, stage))
        kw = dict(offspring=6, survivors=2, max_mutations=3, seed=7, log=lambda *a: None)
        full = BL.evopress_search(self.p, REF, self.budget, fit, generations=6, **kw)
        self.assertLessEqual(full["fitness"], full["trace"][0]["fitness"])
        self.assertEqual(full["evaluations"], 1 + 6 * (6 + 2))
        saved = {}
        BL.evopress_search(self.p, REF, self.budget, fit, generations=3, checkpoint=lambda s: saved.update(s), **kw)
        import json
        state = json.loads(json.dumps(saved))                              # as written to search.json
        resumed = BL.evopress_search(self.p, REF, self.budget, fit, generations=6, state=state, **kw)
        self.assertEqual(resumed["config"], full["config"])
        self.assertEqual(resumed["trace"], full["trace"])
        self.assertLessEqual(sizes.config_bytes(self.p.normalize(full["config"]), self.spec)["gib"], self.budget + 1e-9)


@unittest.skipUnless(HAVE_TF, "transformers not importable")
class TestKLPool(unittest.TestCase):
    def test_pool_kl_equals_exported_checkpoint(self):
        from edit_code.common import write_json
        from edit_code.evopress import KLPool
        from edit_code.quant.cache import QuantCache
        from edit_code.quant.export import export_fake_quant
        from edit_code.quant.hessian import collect_stats, load_hf_model
        from edit_code.quant.kl import kl_to_reference_packed, pack, reference_packed
        from edit_code.quant.spec import spec_from_checkpoint
        from edit_code.tests.test_quant import make_tiny
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = make_tiny(td / "src")
            spec = spec_from_checkpoint(src)
            prep = td / "prep"
            write_json(prep / "spec.json", spec.to_json())
            torch.manual_seed(0)
            ids = [torch.randint(1, 256, (40,)).tolist() for _ in range(6)]
            m0 = load_hf_model(src, "cpu")
            collect_stats(m0, spec, [torch.tensor([x]) for x in ids], prep / "hessians", layers_per_pass=2,
                          log=lambda *a: None)
            dcfg = {"profile_max_len": 32, "profile_batch_tokens": 64}
            norms = [schema.normalize(c, spec) for c in (
                {"default": {"bits": 8, "group_size": 128, "quantizer": "gptq"}, "embed": 8, "lm_head": 8},
                {"default": {"bits": 2, "group_size": 128, "quantizer": "gptq"}, "embed": 8, "lm_head": 8},
                {"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"},
                 "rules": [{"layers": [0], "bits": 16}], "embed": 8, "lm_head": 8})]
            pool = KLPool(["cpu", "cpu"], model_dir=src, spec_path=prep / "spec.json", prep_dir=prep, texts=ids,
                          dcfg=dcfg, embed=8, lm_head=8, log=lambda *a: None)
            try:
                full = pool(spec, norms, "full")
                again = pool(spec, list(reversed(norms)), "full")[::-1]   # other workers, other previous states
                small = pool(spec, norms[:1], "small")
            finally:
                pool.close()
            self.assertLess(full[0], full[2])
            self.assertLess(full[2], full[1])
            for a, b in zip(full, again):
                self.assertAlmostEqual(a, b, places=5)
            self.assertGreater(small[0], 0)
            # the same KL as the exported fake-quant checkpoint of that config
            cache = QuantCache(prep / "cache", spec, src, hessian_dir=prep / "hessians", log=lambda *a: None)
            packed = pack([torch.tensor([x[:32]]) for x in ids], 64)
            refs = reference_packed(m0, packed)
            for n, kl in zip(norms, full):
                out = td / f"fq{abs(hash(str(n))) % 10 ** 6}"
                export_fake_quant(src, out, n, spec, cache, log=lambda *a: None)
                ref_kl = kl_to_reference_packed(load_hf_model(out, "cpu"), packed, refs)["mean"]
                self.assertAlmostEqual(kl, ref_kl, delta=1e-4 + 1e-3 * ref_kl)


if __name__ == "__main__":
    unittest.main(verbosity=2)
