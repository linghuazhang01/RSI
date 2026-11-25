"""Profile-table KL without full-vocab logits on the GPU (2026-11-25), on the tiny Qwen3.5 model of test_quant.py.

    eval/.venv/bin/python -m pytest edit_code/tests/test_kl_hidden.py -q

- quant/kl.py packed path: backbone + LM head split, reference kept as final hidden states per chunk; the same mean
  KL / p99 / top-1 agreement as the unpacked oracle kl_to_reference with a few quantized groups, and as the
  full-logits path; a changed LM head is refused;
- fallback: a model without the split (no .model / .lm_head), with a logit transform in the config, or whose forward
  differs from lm_head(backbone) keeps full-vocab log-prob references and still matches the oracle;
- diagnostics.profile_meta: kl_path and profile_batch_tokens are part of it;
- quant/hessian.py collect_stats: no accumulator of the previous window is alive during the next window's forward.
"""
from __future__ import annotations

import gc
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]

from test_quant import HAVE_TF, make_tiny  # noqa: E402

if HAVE_TF:
    import torch


@unittest.skipUnless(HAVE_TF, "transformers not importable")
class TestHiddenStateKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = make_tiny(Path(cls.tmp.name) / "src")
        from edit_code.quant.hessian import load_hf_model
        from edit_code.quant.kl import pack
        from edit_code.quant.spec import spec_from_checkpoint
        cls.spec = spec_from_checkpoint(cls.src)
        cls.model = load_hf_model(cls.src, "cpu")
        g = torch.Generator().manual_seed(5)
        cls.bs = [torch.randint(1, 256, (1, n), generator=g) for n in (48, 17, 40, 30, 9, 33, 70)]
        cls.packed = pack(cls.bs, max_tokens=100, pad_id=0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _quantized(self, model, keys, bits=2):
        """Context: the modules of these group keys RTN-quantized (as profile_table_real does), restored after."""
        from contextlib import contextmanager

        from edit_code.quant import quantizers as Q

        @contextmanager
        def ctx():
            mods = [m for k in keys for m in self.spec.groups()[k]]
            params = {m.name: model.get_parameter(m.name) for m in mods}
            orig = {n: p.data.clone() for n, p in params.items()}
            try:
                with torch.no_grad():
                    for n, p in params.items():
                        p.data.copy_(Q.rtn(orig[n], bits, 32).dequant(p.dtype, device=p.device))
                yield
            finally:
                with torch.no_grad():
                    for n, p in params.items():
                        p.data.copy_(orig[n])
        return ctx()

    def test_hidden_path_matches_oracle(self):
        from edit_code.quant.kl import kl_to_reference, kl_to_reference_packed, reference, reference_packed
        model = self.model
        refs_h = reference_packed(model, self.packed, chunk=16)
        self.assertEqual(refs_h.kind, "hidden")
        self.assertTrue(all(r.dtype == torch.bfloat16 and r.shape[1] == 128 for r in refs_h))   # [n, hidden]
        self.assertEqual(sum(r.shape[0] for r in refs_h), sum(b.shape[-1] - 1 for b in self.bs))
        refs_l = reference_packed(model, self.packed, chunk=16, split=False)
        self.assertEqual(refs_l.kind, "logprobs")
        self.assertEqual([r.shape[0] for r in refs_l], [r.shape[0] for r in refs_h])
        refs = reference(model, self.bs)
        self.assertLess(kl_to_reference_packed(model, self.packed, refs_h, chunk=16)["mean"], 1e-3)   # bf16 refs only
        keys = ["0.gdn_qkv", "1.down", "3.qk", "2.gate_up"]
        with self._quantized(model, keys):
            a = kl_to_reference(model, self.bs, refs)
            h = kl_to_reference_packed(model, self.packed, refs_h, chunk=16)
            f = kl_to_reference_packed(model, self.packed, refs_l, chunk=16)
        self.assertEqual(a["tokens"], h["tokens"])
        self.assertGreater(a["mean"], 1e-3)
        # same tolerances as test_quant.test_packed_kl_matches_unpacked (bf16 rounding of batched matmuls)
        self.assertAlmostEqual(a["mean"], h["mean"], delta=0.05 * a["mean"] + 1e-6)
        self.assertAlmostEqual(a["p99"], h["p99"], delta=0.1 * a["p99"] + 1e-6)
        self.assertAlmostEqual(a["top1_agree"], h["top1_agree"], delta=0.02)
        # hidden-state path vs full-logits path on the same packing: only lm_head(h[r, s:e]) vs lm_head(h)[r, s:e]
        self.assertAlmostEqual(f["mean"], h["mean"], delta=1e-3 * f["mean"] + 1e-7)
        self.assertAlmostEqual(f["p99"], h["p99"], delta=1e-3 * f["p99"] + 1e-7)
        self.assertAlmostEqual(f["top1_agree"], h["top1_agree"], delta=1e-3)

    def test_changed_head_refused(self):
        from edit_code.quant.kl import kl_to_reference_packed, reference_packed
        model = self.model
        refs = reference_packed(model, self.packed, chunk=16)
        w = model.lm_head.weight
        orig = w.data.clone()
        try:
            with torch.no_grad():
                w.data.mul_(1.5)
            with self.assertRaises(RuntimeError):
                kl_to_reference_packed(model, self.packed, refs, chunk=16)
        finally:
            with torch.no_grad():
                w.data.copy_(orig)

    def test_fallback_without_split(self):
        from edit_code.quant.kl import head_split, kl_to_reference, kl_to_reference_packed, reference, reference_packed
        inner = self.model

        class NoSplit(torch.nn.Module):          # no .model / .lm_head attributes
            def __init__(self):
                super().__init__()
                self.inner = inner

            def forward(self, **kw):
                return self.inner(**kw)

        class Scaled(torch.nn.Module):           # has the attributes, but its forward scales the logits
            def __init__(self):
                super().__init__()
                self.model, self.lm_head = inner.model, inner.lm_head

            def forward(self, **kw):
                return SimpleNamespace(logits=inner(**kw).logits * 2)

        for wrap in (NoSplit(), Scaled()):
            refs_p = reference_packed(wrap, self.packed, chunk=16)
            self.assertEqual(refs_p.kind, "logprobs", type(wrap).__name__)
            refs = reference(wrap, self.bs)
            with self._quantized(inner, ["1.down", "3.qk"]):
                a = kl_to_reference(wrap, self.bs, refs)
                b = kl_to_reference_packed(wrap, self.packed, refs_p, chunk=16)
            self.assertGreater(a["mean"], 1e-3)
            self.assertAlmostEqual(a["mean"], b["mean"], delta=0.05 * a["mean"] + 1e-6)
            self.assertAlmostEqual(a["top1_agree"], b["top1_agree"], delta=0.02)
        tc = inner.config.text_config
        self.assertIsNotNone(head_split(inner))
        try:
            tc.final_logit_softcapping = 30.0
            self.assertIsNone(head_split(inner))
        finally:
            del tc.final_logit_softcapping
        self.assertIsNotNone(head_split(inner))

    def test_profile_rows_hidden_path(self):
        """profile_table_real with the packed (hidden-state) KL: rows like the oracle path, 2-bit worse than 4-bit."""
        from edit_code.diagnostics import profile_table_real
        from edit_code.quant.kl import kl_to_reference, kl_to_reference_packed, reference, reference_packed
        keys = self.spec.group_keys()[:4]
        kw = dict(bits_list=[2, 4], group_size=32, log=lambda *a: None, groups=keys)
        hp = profile_table_real(self.model, self.spec, self.packed, reference_packed(self.model, self.packed),
                                kl=kl_to_reference_packed, **kw)
        un = profile_table_real(self.model, self.spec, self.bs, reference(self.model, self.bs),
                                kl=kl_to_reference, **kw)
        self.assertEqual([(r["key"], r["bits"]) for r in hp], [(r["key"], r["bits"]) for r in un])
        for x, y in zip(hp, un):
            self.assertAlmostEqual(x["dkl_mean"], y["dkl_mean"], delta=0.05 * y["dkl_mean"] + 1e-5)


    def test_profile_worker_in_process(self):
        """profile_worker end to end on cpu, in this process (test_quant's spawned version skips when the fixture
        tokenizer does not load): rows with the new meta, the allocator setting only in the worker's environment."""
        import os
        from unittest import mock

        from edit_code.common import read_json, write_json
        from edit_code.diagnostics import profile_meta, profile_worker
        texts = [f"t{i}" for i in range(len(self.bs))]
        dcfg = {"profile_max_len": 80, "profile_bits": [2, 4], "profile_group_size": 32, "profile_batch_tokens": 100}
        keys = self.spec.group_keys()[:3]
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, clear=False):
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            os.environ.pop("PYTORCH_ALLOC_CONF", None)
            td = Path(td)
            write_json(td / "spec.json", self.spec.to_json())
            logs = []
            with mock.patch("edit_code.quant.hessian.tokenize_texts", lambda d, t, n: list(self.bs)), \
                    mock.patch("builtins.print", lambda *a, **k: logs.append(" ".join(map(str, a)))):
                profile_worker(str(self.src), "cpu", str(td / "spec.json"), texts, dcfg, str(td / "rows"), keys, "t")
            self.assertEqual(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "expandable_segments:True")
            self.assertTrue(any("reference kept as final hidden states" in x for x in logs), logs)
            meta = profile_meta(texts, dcfg)
            for k in keys:
                rec = read_json(td / "rows" / f"{k}.json")
                self.assertEqual(rec["meta"], meta)
                self.assertEqual([r["bits"] for r in rec["rows"]], [2, 4])

    def test_free_memory_check(self):
        """check_free_memory: the BF16 model + headroom must be free, else a RuntimeError that names nvidia-smi."""
        from unittest import mock

        from edit_code.common import GIB
        from edit_code.diagnostics import check_free_memory
        from edit_code.quant import sizes
        need = sizes.bf16_bytes(self.spec) + 8 * GIB
        with mock.patch("torch.cuda.mem_get_info", lambda d: (int(need + GIB), 140 * GIB)):
            check_free_memory("cuda:0", self.spec, log=lambda *a: None)
        with mock.patch("torch.cuda.mem_get_info", lambda d: (int(need - GIB), 140 * GIB)):
            with self.assertRaises(RuntimeError) as cm:
                check_free_memory("cuda:0", self.spec, log=lambda *a: None)
        self.assertIn("nvidia-smi --query-compute-apps=pid,used_memory --format=csv", str(cm.exception))
        self.assertIn("GiB free", str(cm.exception))


class TestProfileMeta(unittest.TestCase):
    def test_meta_has_kl_path_and_batch_tokens(self):
        from edit_code.diagnostics import PROFILE_KL_PATH, profile_meta
        dcfg = {"profile_max_len": 512, "profile_bits": [2, 3, 4], "profile_group_size": 128}
        m0 = profile_meta(["a", "b"], dcfg)
        self.assertEqual(m0["kl_path"], PROFILE_KL_PATH)
        self.assertEqual(m0["profile_batch_tokens"], 8192)
        self.assertEqual(profile_meta(["a", "b"], dict(dcfg, profile_batch_tokens=8192)), m0)
        self.assertNotEqual(profile_meta(["a", "b"], dict(dcfg, profile_batch_tokens=4096)), m0)
        try:
            from edit_code.quant.kl import KL_PATH
        except ImportError:  # pragma: no cover - no torch
            return
        self.assertEqual(KL_PATH, PROFILE_KL_PATH)


@unittest.skipUnless(HAVE_TF, "transformers not importable")
class TestCollectStatsWindows(unittest.TestCase):
    def test_previous_window_freed(self):
        """2026-11-25: when the next window's forward starts, no Hessian accumulator of the previous window is alive
        (the loop variable used to keep the last one, often down_in)."""
        from edit_code.quant.hessian import collect_stats, load_hf_model
        from edit_code.quant.spec import spec_from_checkpoint
        with tempfile.TemporaryDirectory() as td:
            src = make_tiny(Path(td) / "src")
            spec = spec_from_checkpoint(src)
            model = load_hf_model(src, "cpu")
            dims = {m.shape[1] for m in spec.modules}          # Hessians are [in, in] fp32

            def live_hessians():
                gc.collect()
                with warnings.catch_warnings():      # isinstance() on deprecated torch.distributed objects warns
                    warnings.simplefilter("ignore")
                    return sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor)
                               and o.dtype == torch.float32 and o.dim() == 2 and o.shape[0] == o.shape[1]
                               and o.shape[0] in dims)
            seen = []
            first = model.get_submodule(f"{spec.layers_prefix}.0")
            h = first.register_forward_pre_hook(lambda mod, args: seen.append(live_hessians()))
            try:
                collect_stats(model, spec, [torch.randint(0, 256, (1, 24))], Path(td) / "h", layers_per_pass=2,
                              log=lambda *a: None, return_stats=False)
            finally:
                h.remove()
            self.assertEqual(len(seen), 2)                     # one forward per window (layers 0-1, 2-3)
            self.assertEqual(seen, [seen[0], seen[0]], seen)   # nothing left over from window 0-1


if __name__ == "__main__":
    unittest.main()
