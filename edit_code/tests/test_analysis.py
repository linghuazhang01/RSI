"""Paper A analysis tools (2026-11-20): Hessian-weighted error (real path, with a fake layer cache and real
Hessian files), group_error and the hessian table (compare was removed 2026-11-23: candidates have no screen records).

  python -I -m pytest -q edit_code/tests/test_analysis.py      (needs torch + safetensors; no transformers)
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

try:
    import torch
    from safetensors.torch import save_file
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

TINY = {"num_layers": 4, "interval": 4, "hidden": 32, "intermediate": 48, "heads": 2, "head_dim": 8,
        "kv_heads": 1, "lin_k_heads": 2, "lin_v_heads": 2, "lin_head_dim": 8, "vocab": 64}


def _problem(tmp: Path, mock: bool):
    """A QuantProblem with just the attributes the analysis methods use (no servers, no rsicore run)."""
    from edit_code.problem import QuantProblem
    from edit_code.quant import schema
    from edit_code.quant.spec import synthetic_spec
    p = QuantProblem.__new__(QuantProblem)
    p.mock = mock
    p.spec = synthetic_spec(TINY)
    p.prep_dir = tmp / "prep"
    p.out_dir = tmp / "gen"
    p.out_dir.mkdir(parents=True, exist_ok=True)
    p.cfg = {"diagnostics": {"profile_group_size": 16, "profile_bits": [2, 3, 4],
                             "hessian_quantizers": ["rtn", "gptq", "awq"]}}
    p.space = dict(schema.DEFAULT_ACTION_SPACE, group_sizes=[16, 32])
    p.datasets = ["dapo_math"]
    p._herr, p._herr_lock, p._export_lock = None, threading.Lock(), threading.Lock()
    p._evals = {}
    p.prep_gpus = None                       # one worker (the hessian table on several GPUs is the GPU path)
    p.log = lambda *a, **k: None
    p.tag = "t"
    return p


class _FakeCache:
    """weight() = a fixed random matrix per module; dequant() = RTN of it (like the real cache's entries)."""

    def __init__(self, spec):
        self.device = "cpu"
        g = torch.Generator().manual_seed(0)
        self.W = {m.name: torch.randn(*m.shape, generator=g) * 0.1 for m in spec.modules}

    def weight(self, module):
        return self.W[module]

    def dequant(self, module, setting, dtype=torch.float32):
        from edit_code.quant import quantizers as Q
        q = Q.rtn(self.W[module], setting["bits"], setting["group_size"])
        return Q.dequant(q, dtype=dtype)


@unittest.skipUnless(HAVE_TORCH, "torch / safetensors not available")
class TestHessianError(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = _problem(Path(self.tmp.name), mock=False)
        self.p.cache = _FakeCache(self.p.spec)
        from edit_code.quant.hessian import hessian_path
        g = torch.Generator().manual_seed(1)
        self.H = {}
        for m in self.p.spec.modules:                    # one Hessian per (layer, input key), like the real pass
            k = (m.layer, m.input_key)
            if k in self.H:
                continue
            X = torch.randn(200, m.shape[1], generator=g)
            H = X.T @ X
            self.H[k] = H
            path = hessian_path(self.p.prep_dir / "hessians", m.layer, m.input_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"H": H, "n": torch.tensor(200), "absmax": X.abs().max(0).values}, str(path))

    def test_herr_matches_definition_and_is_memoised(self):
        from edit_code.quant import quantizers as Q
        m = self.p.spec.modules[0]
        s = {"bits": 3, "group_size": 16, "quantizer": "rtn"}
        err, ref, n = self.p.herr_module(m.name, s)
        W = self.p.cache.weight(m.name)
        Wq = self.p.cache.dequant(m.name, s)
        H = self.H[(m.layer, m.input_key)]
        self.assertAlmostEqual(err, Q.h_error(W, Wq, H), places=2)
        self.assertAlmostEqual(ref, float(((W @ H) * W).sum()), places=1)
        self.assertEqual(n, 200)
        self.assertGreater(self.p.herr_module(m.name, {**s, "bits": 2})[0], err)   # fewer bits, larger error
        self.assertEqual(self.p.herr_module(m.name, {**s, "bits": 16})[:1], (0.0,))
        self.p.cache.W[m.name] = W * 0                 # memoised: the cache is not read again
        self.assertEqual(self.p.herr_module(m.name, s)[0], err)

    def test_group_error_and_table(self):
        cfg = {"default": {"bits": 4, "group_size": 16, "quantizer": "rtn"},
               "groups": {"0.down": {"bits": 2}, "3.qk": {"bits": 16}}}
        ge = self.p.group_error(cfg, top=3)
        self.assertAlmostEqual(sum(v["share"] for v in ge["groups"].values()), 1.0, places=3)
        self.assertEqual(ge["groups"]["3.qk"]["herr_per_token"], 0.0)               # kept in BF16
        self.assertEqual(ge["top"][0][0], "0.down")                                 # the 2-bit group dominates
        self.assertEqual(len(ge["top"]), 3)
        self.assertTrue((self.p.prep_dir / "diagnostics" / "herr_cache.json").exists())   # persisted
        self.p.cfg["diagnostics"]["hessian_quantizers"] = ["rtn"]
        rows = self.p.hessian_table()
        self.assertEqual(len(rows), len(self.p.spec.group_keys()) * 3)
        r2 = {(r["key"], r["bits"]): r["herr_rel"] for r in rows}
        for k in self.p.spec.group_keys():
            self.assertGreater(r2[(k, 2)], r2[(k, 4)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
