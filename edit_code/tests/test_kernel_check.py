"""CPU tests of edit_code/packed_kernel_check.py (2026-11-28): everything except the GPU GEMMs.

Checked here: the vectorized bit reader against the reference reader, the GPTQ tensors of unpacked codes against the
export's own packed_tensors, the synthetic case list (Qwen3.8-27B shapes, the shapes the format cannot express), the
case tensors (the reference weight W' is what the packed tensors decode to), the cases read from a written packed
checkpoint (and the export check against its dequant twin), the worker restart logic after crashes, and the report.

Run:  eval/.venv/bin/python -I edit_code/tests/test_kernel_check.py   (the checkpoint tests need transformers with
      Qwen3.5; the others need torch only)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
try:
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
    HAVE_TF = torch is not None
except Exception:  # pragma: no cover
    HAVE_TF = False

PACKED_SPACE = {"packed": True, "quantizers": ["rtn", "gptq"], "embed": ["bf16"], "lm_head": ["bf16"]}


def args(**kw):
    d = dict(model=None, modules=None, bits=[2, 3, 4, 8], group_sizes=[32, 64, 128], ckpt=None, dequant=None,
             per_kind=1, m=[1], impls=["vllm"], gpu=0, blocking=False, stall_timeout=1800, poll=0.05)
    d.update(kw)
    return argparse.Namespace(**d)


@unittest.skipIf(torch is None, "torch not importable")
class TestReaders(unittest.TestCase):
    def test_decode_rows_matches_reference(self):
        from edit_code.quant.export import decode_rows, pack_rows_int32, unpack_rows_int32
        g = torch.Generator().manual_seed(1)
        for b in (2, 3, 4, 8):
            R, C = 9, 128 if b != 3 else 96
            codes = torch.randint(0, 2 ** b, (R, C), generator=g, dtype=torch.int64).to(torch.uint8)
            codes[0, :4] = torch.tensor([0, 2 ** b - 1, 0, 2 ** b - 1], dtype=torch.uint8)
            w = pack_rows_int32(codes, b)
            ref = unpack_rows_int32(w, b, C)
            self.assertTrue(torch.equal(decode_rows(w, b, C), ref), b)
            self.assertTrue(torch.equal(decode_rows(w, b, C, chunk=4), ref), f"chunked b={b}")
        with self.assertRaises(ValueError):
            decode_rows(torch.zeros(1, 2, dtype=torch.int32), 3, 32)

    def test_gptq_tensors_decode_to_dequant(self):
        from edit_code.quant import quantizers as Q
        from edit_code.quant.export import decode_rows, gptq_tensors
        g = torch.Generator().manual_seed(2)
        for b, gs in ((2, 32), (3, 64), (4, 128), (8, 32)):
            W = torch.randn(64, 256, generator=g) * 0.02
            q = Q.rtn(W, b, gs, scale_dtype=torch.bfloat16)
            t = gptq_tensors(q.codes, q.zero, q.scale, b)
            self.assertEqual(tuple(t["qweight"].shape), (256 * b // 32, 64))
            self.assertEqual(tuple(t["qzeros"].shape), (256 // gs, 64 * b // 32))
            self.assertEqual(tuple(t["scales"].shape), (256 // gs, 64))
            codes = decode_rows(t["qweight"].t().contiguous(), b, 256)
            zero = decode_rows(t["qzeros"], b, 64).t()
            col = torch.arange(256) // gs
            w = ((codes.float() - zero.float()[:, col]) * t["scales"].t().float()[:, col]).to(torch.bfloat16)
            self.assertTrue(torch.equal(w, Q.dequant(q, dtype=torch.bfloat16)), (b, gs))


@unittest.skipIf(torch is None, "torch not importable")
class TestCases(unittest.TestCase):
    def test_synthetic_cases(self):
        from edit_code import packed_kernel_check as K
        cases, skipped = K.synthetic_cases(args())
        mods = {c["module"] for c in cases}
        self.assertEqual(mods, set(K.FUSED_ORDER))
        self.assertEqual(len(cases), 7 * 4 * 3 - 3)                      # in_proj_ba (48 + 48) has no 3-bit form
        self.assertEqual({(s["module"], s["bits"]) for s in skipped}, {("in_proj_ba", 3)})
        by = {c["module"]: c for c in cases}
        self.assertEqual(by["qkv_proj"]["shards"], [12288, 1024, 1024])
        self.assertEqual(by["in_proj_qkvz"]["shards"], [10240, 6144])
        self.assertEqual(by["in_proj_ba"]["shards"], [48, 48])
        self.assertEqual(by["gate_up_proj"]["shards"], [17408, 17408])
        self.assertEqual((by["down_proj"]["k"], by["out_proj"]["k"], by["o_proj"]["k"]), (17408, 6144, 6144))
        self.assertEqual(by["qkv_proj"]["head_dim"], 256)
        self.assertTrue(by["qkv_proj"]["prefix"].startswith("language_model.model.layers.3.self_attn."))
        self.assertTrue(by["in_proj_ba"]["prefix"].startswith("language_model.model.layers.0.linear_attn."))
        for c in cases:                                                   # vLLM resolves each case's own setting
            from edit_code.quant.export import resolve_humming
            r = resolve_humming(c["qcfg"], c["prefix"])
            self.assertEqual((r["bits"], r["group_size"], r["sym"]), (c["bits"], c["group_size"], False))
        quick, _ = K.synthetic_cases(args(group_sizes=[128], modules="down_proj,in_proj_ba"))
        self.assertEqual(sorted((c["module"], c["bits"]) for c in quick),
                         sorted([("down_proj", b) for b in (2, 3, 4, 8)] + [("in_proj_ba", b) for b in (2, 4, 8)]))

    def test_case_tensors_synthetic(self):
        from edit_code import packed_kernel_check as K
        from edit_code.quant.export import decode_rows
        case = {"source": "synthetic", "module": "gate_up_proj", "prefix": "x", "shards": [64, 96], "k": 256,
                "bits": 3, "group_size": 32, "seed": 5}
        gptq, wref, extra, parts = K.case_tensors(case, torch.device("cpu"))
        self.assertEqual(extra, {})
        self.assertEqual(len(parts), 2)
        self.assertEqual(tuple(gptq["qweight"].shape), (256 * 3 // 32, 160))
        self.assertEqual(tuple(gptq["qzeros"].shape), (8, 160 * 3 // 32))
        self.assertEqual(tuple(wref.shape), (160, 256))
        codes = decode_rows(gptq["qweight"].t().contiguous(), 3, 256)
        zero = decode_rows(gptq["qzeros"], 3, 160).t()
        self.assertEqual(int(zero.min()), 0)                              # both zero-point extremes occur
        self.assertEqual(int(zero.max()), 7)
        col = torch.arange(256) // 32
        w = ((codes.float() - zero.float()[:, col]) * gptq["scales"].t().float()[:, col]).to(torch.bfloat16)
        self.assertTrue(torch.equal(w, wref))

    def test_expressible(self):
        from edit_code.packed_kernel_check import expressible
        self.assertIsNone(expressible([48, 48], 5120, 4, 128))
        self.assertIn("int32", expressible([48, 48], 5120, 3, 128))
        self.assertIn("group size", expressible([64], 5120, 4, 96))
        self.assertIsNone(expressible([17408, 17408], 5120, 3, 32))


@unittest.skipIf(not HAVE_TF, "transformers with Qwen3.5 not importable")
class TestCheckpointCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from edit_code.quant import schema
        from edit_code.quant.cache import QuantCache
        from edit_code.quant.export import export_checkpoint
        from edit_code.quant.spec import spec_from_checkpoint
        from edit_code.tests.test_quant import make_tiny
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.src = make_tiny(root / "src")
        cls.spec = spec_from_checkpoint(cls.src)
        cls.cache = QuantCache(root / "cache", cls.spec, cls.src, scale_dtype=torch.bfloat16, log=lambda *a: None)
        cls.norm = schema.normalize({"default": {"bits": 4, "group_size": 32, "quantizer": "rtn"},
                                     "rules": [{"layers": [0], "groups": ["gate_up"], "bits": 3, "group_size": 64},
                                               {"layers": [1], "groups": ["down"], "bits": 2, "group_size": 128},
                                               {"layers": [3], "groups": ["qk", "v"], "bits": 8},
                                               {"layers": [2], "groups": ["gdn_out"], "bits": 16},
                                               {"groups": ["gdn_ba"], "bits": 16},
                                               {"layers": [0], "groups": ["gdn_ba"], "bits": 8}]},
                                    cls.spec, PACKED_SPACE)
        cls.pk, cls.dq = root / "packed", root / "dequant"
        export_checkpoint("packed", cls.src, cls.pk, cls.norm, cls.spec, cls.cache, log=lambda *a: None)
        export_checkpoint("dequant", cls.src, cls.dq, cls.norm, cls.spec, cls.cache, log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_gptq_tensors_equal_packed_tensors(self):
        from edit_code.quant import quantizers as Q
        from edit_code.quant.export import gptq_tensors, packed_tensors
        from edit_code.quant.schema import module_assignment
        assign = module_assignment(self.norm, self.spec)
        n = 0
        for m in self.spec.modules:
            s = assign[m.name]
            if s["bits"] >= 16:
                continue
            d = self.cache.raw(m.name, s["quantizer"], s["bits"], s["group_size"])
            o, i, zo, zg = (int(x) for x in d["shape"].tolist())
            codes = Q.unpack(d["codes"], s["bits"], o * i).reshape(o, i)
            zero = Q.unpack(d["zero"], s["bits"], zo * zg).reshape(zo, zg)
            a, b = packed_tensors(self.cache, m.name, s), gptq_tensors(codes, zero, d["scale"], s["bits"])
            for k in ("qweight", "qzeros", "scales"):
                self.assertTrue(torch.equal(a[k], b[k]), (m.name, k))
            n += 1
        self.assertGreater(n, 10)

    def test_ckpt_cases_decode_to_dequant(self):
        from edit_code import packed_kernel_check as K
        from edit_code.quant.export import fused_settings
        cases, skipped = K.ckpt_cases(args(ckpt=str(self.pk), dequant=str(self.dq), per_kind=1))
        fused = fused_settings(self.norm, self.spec)
        kinds = {(p.rsplit(".", 1)[1], s["bits"], s["group_size"]) for p, s in fused.items() if s["bits"] < 16}
        self.assertEqual({(c["module"], c["bits"], c["group_size"]) for c in cases}, kinds)
        self.assertEqual(skipped, [])
        for c in cases:
            self.assertEqual(fused[c["prefix"]]["bits"], c["bits"])
            self.assertEqual(c["k"], self.spec.module(c["bases"][0] + ".weight").shape[1])
            gptq, wref, extra, parts = K.case_tensors(c, torch.device("cpu"))
            self.assertEqual(extra.get("export_mismatch_elems"), 0, c["prefix"])
            self.assertNotIn("export_missing", extra)
            self.assertEqual(wref.shape[0], sum(c["shards"]))
            self.assertEqual(len(parts), len(K.FUSED_ORDER[c["module"]]))
        more, _ = K.ckpt_cases(args(ckpt=str(self.pk), per_kind=99))
        self.assertGreater(len(more), len(cases))


FAKE_WORKER = textwrap.dedent('''
    import argparse, json, os, sys
    ap = argparse.ArgumentParser()
    for k in ("--cases", "--results", "--m", "--impls"):
        ap.add_argument(k)
    ap.add_argument("--start", type=int)
    ap.add_argument("--worker", action="store_true")
    a = ap.parse_args()
    f = open(a.results, "a")
    def emit(e):
        f.write(json.dumps(e) + "\\n"); f.flush()
    def good(ci):
        emit({"event": "begin", "case": ci})
        emit({"event": "fmt", "case": ci, "mismatch_elems": 0, "elems": 4})
        emit({"event": "built", "case": ci, "impl": "vllm", "meta": "m"})
        emit({"event": "result", "case": ci, "impl": "vllm", "m": 1, "ok": True, "finite": True, "rel_err": 0.001,
              "rel_err_cublas": 0.001, "max_abs_rel": 0.0, "rel_err_by_shard": [0.001],
              "worst_block": {"cols": [0, 4], "rel_err": 0.001}})
        emit({"event": "end", "case": ci})
    mode = os.environ.get("FAKE_MODE", "crashes")
    if mode == "startup":
        print("CUDA init failed", flush=True)
        sys.exit(1)
    if mode == "stall":
        if a.start == 0:
            good(0)
            emit({"event": "begin", "case": 1})
            emit({"event": "stage", "case": 1, "stage": "gemm:vllm:M=8192"})
            import time
            time.sleep(600)
        for ci in range(a.start, 4):
            good(ci)
        emit({"event": "done"})
        sys.exit(0)
    if mode == "late_exit":
        for ci in range(4):
            good(ci)
        emit({"event": "done"})
        sys.exit(1)
    if a.start == 0:
        good(0)
        emit({"event": "begin", "case": 1})
        emit({"event": "stage", "case": 1, "stage": "gemm:vllm:M=1"})
        print("segfault-ish", flush=True)
        os._exit(139)
    if a.start == 2:
        emit({"event": "begin", "case": 2})
        emit({"event": "crash", "case": 2, "stage": "build:vllm", "error": "RuntimeError: CUDA error: illegal memory access"})
        sys.exit(3)
    good(3)
    emit({"event": "done"})
''')


@unittest.skipIf(torch is None, "torch not importable")
class TestWorkersAndReport(unittest.TestCase):
    def test_restarts_after_crashes(self):
        from edit_code import packed_kernel_check as K
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            script = d / "fake_worker.py"
            script.write_text(FAKE_WORKER)
            cases = [{"source": "synthetic", "module": "down_proj", "prefix": f"p{i}", "shards": [4], "k": 32,
                      "bits": 4, "group_size": 32} for i in range(4)]
            (d / "cases.json").write_text(json.dumps(cases))
            a = args(m=[1], impls=["vllm"])
            K.run_workers(a, d / "cases.json", d / "results.jsonl", len(cases), script=str(script))
            rep = K.summarize(cases, [], K.read_events(d / "results.jsonl"), a)
            by = {r["case"]: r for r in rep["rows"]}
            self.assertTrue(by[0]["ok"] and by[3]["ok"])
            self.assertFalse(by[1]["ok"] or by[2]["ok"])
            self.assertEqual(by[1]["crash"][0]["stage"], "gemm:vllm:M=1")
            self.assertEqual(by[1]["crash"][0]["returncode"], 139)
            self.assertIn("segfault-ish", by[1]["crash"][0]["error"])
            self.assertEqual(by[2]["crash"][0]["stage"], "build:vllm")
            self.assertEqual(len(by[2]["crash"]), 1)                     # no duplicate crash event
            self.assertEqual(rep["failed"], 2)
            self.assertEqual([r["case"] for r in rep["rows"]][:2], [1, 2])   # failures first
            text = "\n".join(K.diagnosis(rep))
            self.assertIn("CUDA crash at gemm:vllm:M=1", text)

    def _run(self, mode, **kw):
        import os
        from edit_code import packed_kernel_check as K
        d = Path(self.enterContext(tempfile.TemporaryDirectory()))
        script = d / "fake_worker.py"
        script.write_text(FAKE_WORKER)
        cases = [{"source": "synthetic", "module": "down_proj", "prefix": f"p{i}", "shards": [4], "k": 32,
                  "bits": 4, "group_size": 32} for i in range(4)]
        (d / "cases.json").write_text(json.dumps(cases))
        a = args(m=[1], impls=["vllm"], **kw)
        old = os.environ.get("FAKE_MODE")
        os.environ["FAKE_MODE"] = mode
        try:
            K.run_workers(a, d / "cases.json", d / "results.jsonl", len(cases), script=str(script))
        finally:
            if old is None:
                os.environ.pop("FAKE_MODE")
            else:
                os.environ["FAKE_MODE"] = old
        evs = K.read_events(d / "results.jsonl")
        return K.summarize(cases, [], evs, a), evs

    def test_stalled_worker_is_killed(self):
        rep, evs = self._run("stall", stall_timeout=1.5)
        by = {r["case"]: r for r in rep["rows"]}
        self.assertTrue(by[0]["ok"] and by[2]["ok"] and by[3]["ok"])
        self.assertFalse(by[1]["ok"])
        self.assertEqual(by[1]["crash"][0]["stage"], "gemm:vllm:M=8192")
        self.assertTrue(any(e.get("stalled_s") for e in evs if e.get("event") == "crash"))

    def test_startup_failures_end_the_run(self):
        rep, evs = self._run("startup")
        self.assertEqual(rep["failed"], 4)
        crashes = [e for e in evs if e.get("event") == "crash"]
        self.assertEqual([(e["case"], e["stage"]) for e in crashes], [(0, "worker-start"), (1, "worker-start")])
        self.assertIn("CUDA init failed", crashes[0]["error"])

    def test_nonzero_exit_after_done_is_not_a_crash(self):
        rep, evs = self._run("late_exit")
        self.assertEqual(rep["failed"], 0)
        self.assertFalse(any(e.get("event") == "crash" for e in evs))

    def test_loader_attribution(self):
        from edit_code import packed_kernel_check as K
        cases = [{"source": "synthetic", "module": "qkv_proj", "prefix": "q", "shards": [4], "k": 32, "bits": 3,
                  "group_size": 32}]
        evs = [{"event": "fmt", "case": 0, "mismatch_elems": 0, "elems": 128},
               {"event": "built", "case": 0, "impl": "loader", "meta": "m",
                "load_mismatch": {"qweight": 0, "qzeros": 12, "scales": 0}}]
        for impl, ok in (("vllm", True), ("loader", False)):
            evs.append({"event": "result", "case": 0, "impl": impl, "m": 1, "ok": ok, "finite": True,
                        "rel_err": 0.001 if ok else 0.7, "rel_err_cublas": 0.001, "max_abs_rel": 0,
                        "rel_err_by_shard": [0.1], "worst_block": {"cols": [0, 4], "rel_err": 0.1}})
        rep = K.summarize(cases, [], evs, args(m=[1], impls=["vllm", "loader"]))
        text = "\n".join(K.diagnosis(rep))
        self.assertIn("weight_loader_v2 shard offsets", text)
        self.assertIn("'qzeros': 12", text)

    def test_diagnosis_layers(self):
        from edit_code import packed_kernel_check as K
        cases = [{"source": "synthetic", "module": "down_proj", "prefix": f"p{i}", "shards": [4], "k": 32,
                  "bits": 3, "group_size": 32} for i in range(3)]

        def res(ci, impl, ok, err):
            return {"event": "result", "case": ci, "impl": impl, "m": 1, "ok": ok, "finite": True, "rel_err": err,
                    "rel_err_cublas": 0.001, "max_abs_rel": 0, "rel_err_by_shard": [err],
                    "worst_block": {"cols": [0, 4], "rel_err": err}}
        evs = [{"event": "fmt", "case": 0, "mismatch_elems": 3, "elems": 128},
               res(0, "vllm", False, 0.9), res(0, "native", True, 0.001),
               {"event": "fmt", "case": 1, "mismatch_elems": 0, "elems": 128},
               res(1, "vllm", False, 0.5), res(1, "native", False, 0.4),
               {"event": "fmt", "case": 2, "mismatch_elems": 0, "elems": 128},
               res(2, "vllm", True, 0.001), res(2, "native", True, 0.001)]
        a = args(m=[1], impls=["vllm", "native"])
        rep = K.summarize(cases, [], evs, a)
        self.assertEqual(rep["failed"], 2)
        text = "\n".join(K.diagnosis(rep))
        self.assertIn("p0 3b g32 shards [4]: humming READS the tensors differently", text)
        self.assertIn("p1 3b g32 shards [4]: vllm wrong at M=[1]; native wrong at M=[1] -> the humming GEMM itself", text)
        self.assertNotIn("p2", text)
        rep2 = K.summarize(cases[2:], [], [dict(e, case=0) for e in evs if e.get("case") == 2], a)
        self.assertIn("every case passed", K.diagnosis(rep2)[0])

    def test_compare_metrics(self):
        from edit_code.packed_kernel_check import compare
        g = torch.Generator().manual_seed(3)
        yref = torch.randn(8, 512, generator=g)
        ycub = yref.to(torch.bfloat16)
        r = compare(yref.to(torch.bfloat16), yref, ycub, [256, 256])
        self.assertTrue(r["ok"])
        bad = yref.clone()
        bad[:, 256:] = torch.randn(8, 256, generator=g)                   # second shard wrong
        r = compare(bad.to(torch.bfloat16), yref, ycub, [256, 256])
        self.assertFalse(r["ok"])
        self.assertLess(r["rel_err_by_shard"][0], 0.01)
        self.assertGreater(r["rel_err_by_shard"][1], 0.5)
        self.assertEqual(r["worst_block"]["cols"][0], 256)
        nan = yref.clone()
        nan[0, 0] = float("nan")
        self.assertFalse(compare(nan, yref, ycub, [512])["ok"])


if __name__ == "__main__":
    unittest.main()
