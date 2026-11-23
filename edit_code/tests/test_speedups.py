"""Paper A speed-ups of 2026-11-23 that must not change results:
  - parallel exports: one layer cache per candidate slot (QuantProblem._slot_cache), no global export lock; a
    missing cache entry is computed once even when several caches ask for it at the same time;
  - the accepted candidate's gate checkpoint is moved to gen_XX/model (QuantProblem.adopt_checkpoint) instead of
    being exported again;
  - calibration parts: collect_stats(return_stats=False) does not read its Hessians back;
  - scoring.concurrency is passed to run_eval as --concurrency (rsicore.evaluate.score / score_split).
Tests needing transformers skip cleanly when it is not importable (run them with a torch + transformers Python).
    python -I -m pytest -q edit_code/tests/test_speedups.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
try:
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
    HAVE_TF = torch is not None
except Exception:  # pragma: no cover
    HAVE_TF = False


def _problem(root: Path, src: Path, spec, n_slots: int):
    """A real-mode QuantProblem with only what materialize / adopt_checkpoint use (layer caches on the CPU)."""
    from edit_code.problem import QuantProblem
    p = QuantProblem.__new__(QuantProblem)
    p.mock, p.spec, p.source_dir = False, spec, src
    p.out_dir, p.prep_dir = root / "gen_01", root / "prep"
    p.out_dir.mkdir(parents=True, exist_ok=True)
    p.cfg = {"cache": {"device": "cpu", "lazy": True, "gptq_damp": 0.01}, "candidates": {"keep_checkpoints": 1}}
    p.rc = SimpleNamespace(gpus=[0], run_dir=root)
    p.slots = [[i] for i in range(n_slots)]
    p._lock, p._export_lock, p._slot_caches, p.exported = threading.RLock(), threading.Lock(), {}, {}
    p.tag, p.gen, p.budget_gib, p.log = "g01", 1, 99.0, (lambda *a, **k: None)
    p._device = lambda: "cpu"
    return p


@unittest.skipUnless(HAVE_TF, "transformers not importable")
class TestParallelExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from edit_code.quant.spec import spec_from_checkpoint
        from edit_code.tests.test_quant import make_tiny
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.src = make_tiny(cls.root / "src")
        cls.spec = spec_from_checkpoint(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _norms(self):
        from edit_code.quant import schema
        return [schema.normalize(c, self.spec) for c in (
            {"default": {"bits": 4, "group_size": 32, "quantizer": "rtn"}, "embed": "bf16", "lm_head": 8},
            {"default": {"bits": 3, "group_size": 64, "quantizer": "rtn"}, "embed": 8, "lm_head": "bf16"},
            {"default": {"bits": 2, "group_size": 32, "quantizer": "rtn"},
             "rules": [{"layers": [0, 1], "bits": 8, "group_size": 32, "quantizer": "rtn"}]})]

    def test_entry_computed_once_across_caches(self):
        from edit_code.quant.cache import QuantCache
        root = self.root / "once"
        caches = [QuantCache(root, self.spec, self.src, log=lambda *a: None) for _ in range(6)]
        m = self.spec.modules[0].name
        with ThreadPoolExecutor(6) as ex:
            outs = list(ex.map(lambda c: c.get(m, "rtn", 3, 32).codes, caches))
        self.assertEqual(sum(c.stats["computed"] for c in caches), 1)
        self.assertTrue(all(torch.equal(o, outs[0]) for o in outs))
        self.assertEqual(list(root.rglob("*.tmp*")), [])

    def test_parallel_materialize_equals_serial_export(self):
        from safetensors.torch import load_file
        from edit_code.common import config_hash, write_json
        from edit_code.quant.cache import QuantCache
        from edit_code.quant.export import export_fake_quant
        root = self.root / "par"
        norms = self._norms()
        # serial reference: one shared cache, one export at a time (the old path)
        ref_cache = QuantCache(root / "ref_cache", self.spec, self.src, log=lambda *a: None)
        refs = []
        for i, n in enumerate(norms):
            out = root / f"ref{i}"
            export_fake_quant(self.src, out, n, self.spec, ref_cache, log=lambda *a: None)
            refs.append(out)
        # parallel: one QuantProblem, three slots exporting at the same time into a fresh cache
        p = _problem(root, self.src, self.spec, n_slots=3)
        arts = []
        for n in norms:
            h = config_hash(n)
            write_json(p.out_dir / "configs" / f"{h}.json", n)
            arts.append({"config_hash": h})
        with ThreadPoolExecutor(3) as ex:
            outs = list(ex.map(lambda i: p.materialize(arts[i], slot=i), range(3)))
        self.assertEqual(len(p._slot_caches), 3)
        for out, ref in zip(outs, refs):
            self.assertEqual(out.parent.name, f"slot{outs.index(out)}")
            for f in sorted(ref.glob("*.safetensors")):
                a, b = load_file(str(f)), load_file(str(out / f.name))
                self.assertEqual(set(a), set(b))
                self.assertTrue(all(torch.equal(a[k], b[k]) for k in a), f.name)
            self.assertEqual(json.loads((out / "fake_quant.json").read_text())["config"],
                             json.loads((ref / "fake_quant.json").read_text())["config"])
        # the accepted candidate's checkpoint is moved, not exported again; another config is not adopted
        dst = root / "gen_01" / "model"
        self.assertTrue(p.adopt_checkpoint(arts[1], dst))
        self.assertTrue((dst / "fake_quant.json").exists() and not outs[1].exists())
        for f in sorted(refs[1].glob("*.safetensors")):
            a, b = load_file(str(f)), load_file(str(dst / f.name))
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertFalse(p.adopt_checkpoint(arts[1], root / "again"))      # already moved
        other = {"config_hash": arts[0]["config_hash"]}
        write_json(p.out_dir / "configs" / f"{other['config_hash']}.json", norms[2])   # hash file now disagrees
        self.assertFalse(p.adopt_checkpoint(other, root / "nope"))

    def test_collect_stats_parts_skip_read_back(self):
        from edit_code.quant.hessian import collect_stats, hessian_path, load_hf_model
        model = load_hf_model(self.src, "cpu")
        batches = [torch.randint(0, 256, (1, 24)) for _ in range(2)]
        out = self.root / "parts"
        self.assertEqual(collect_stats(model, self.spec, batches, out, layers_per_pass=2, log=lambda *a: None,
                                       return_stats=False), {})
        self.assertTrue(hessian_path(out, 3, "down_in").exists())
        self.assertIn("3.down_in", collect_stats(model, self.spec, batches, out, layers_per_pass=2,
                                                 log=lambda *a: None))


class TestScoringConcurrency(unittest.TestCase):
    def test_concurrency_flag(self):
        from rsicore import evaluate as E
        cmds = []

        def fake_run(cmd, **kw):
            cmds.append(cmd)
            out = Path(cmd[cmd.index("--out") + 1])
            ids = json.loads(Path(cmd[cmd.index("--ids-file") + 1]).read_text())
            (out / "summary.json").write_text(json.dumps({"t": {"accuracy": 1.0, "n": len(ids)}}))
            (out / "t.jsonl").write_text("".join(json.dumps({"id": i, "score": 1.0}) + "\n" for i in ids))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        orig = E.subprocess.run
        E.subprocess.run = fake_run
        try:
            with tempfile.TemporaryDirectory() as td:
                E.score("http://x", "m", "t", pool="p", ids=["a", "b"], out_dir=Path(td) / "1", concurrency=96)
                E.score("http://x", "m", "t", pool="p", ids=["a", "b"], out_dir=Path(td) / "2")
                E.score_split(["http://x", "http://y"], "m", "t", pool="p", ids=["a", "b", "c"],
                              out_dir=Path(td) / "3", concurrency=96)
        finally:
            E.subprocess.run = orig
        self.assertEqual(cmds[0][cmds[0].index("--concurrency") + 1], "96")
        self.assertNotIn("--concurrency", cmds[1])
        self.assertTrue(all(c[c.index("--concurrency") + 1] == "96" for c in cmds[2:]) and len(cmds) == 4)

    def test_config_default_and_lock(self):
        from rsicore.config import lock_view
        from edit_code.common import scoring_concurrency
        import yaml
        cfg = yaml.safe_load((REPO / "edit_code" / "configs" / "base.yaml").read_text())
        self.assertEqual(scoring_concurrency(cfg), 96)
        self.assertIsNone(scoring_concurrency(dict(cfg, scoring={"concurrency": None})))
        self.assertEqual(lock_view(cfg, "A"), lock_view(dict(cfg, scoring={"concurrency": 32}), "A"))
        old = {k: v for k, v in cfg.items() if k != "scoring"}           # a lock made before the key existed
        self.assertEqual(lock_view(old, "A"), lock_view(cfg, "A"))

    def test_old_locks_resume_under_every_profile(self):
        """A lock written before scoring.concurrency existed (no top-level key, no hardware_profiles.h20_80.scoring)
        compares equal to today's config under every hardware profile (the stored hardware_profiles block is part of
        the lock view, so the key is ignored at any depth)."""
        import copy
        import yaml
        from rsicore import config as C
        new = yaml.safe_load((REPO / "edit_code" / "configs" / "base.yaml").read_text())
        old = copy.deepcopy(new)
        old.pop("scoring")
        old["hardware_profiles"]["h20_80"].pop("scoring")
        for hw in new["hardware_profiles"]:
            a = C._apply_hardware(dict(copy.deepcopy(old), hardware=hw), [])
            b = C._apply_hardware(dict(copy.deepcopy(new), hardware=hw), [])
            self.assertIn("hardware_profiles", b)                   # the stored block is compared by the lock
            self.assertEqual(C._diff_paths(C.lock_view(a, "A"), C.lock_view(b, "A")), [], hw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
