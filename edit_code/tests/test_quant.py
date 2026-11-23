"""CPU unit tests for edit_code/quant on a tiny randomly initialised Qwen3.5 model.

Run:  <python with torch+transformers+safetensors> -I -m pytest edit_code/tests/test_quant.py -q
      (or `python -I edit_code/tests/test_quant.py`). Tests needing transformers skip cleanly when it
is not importable (e.g. eval/.venv has torch only).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
try:
    import transformers  # noqa: F401
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    HAVE_TF = torch is not None
except Exception:  # pragma: no cover
    HAVE_TF = False

from edit_code.quant import schema, sizes
from edit_code.quant.spec import synthetic_spec

TINY_TEXT = dict(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_key_head_dim=32,
                 linear_value_head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
                 max_position_embeddings=512)
TINY_VISION = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=128,
                   num_position_embeddings=16)


def make_tiny(path: Path, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    cfg = Qwen3_5Config(text_config=TINY_TEXT, vision_config=TINY_VISION)
    m = Qwen3_5ForConditionalGeneration(cfg)
    # make weights a bit heavier-tailed than the default init so quantization error is visible
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2 and "layers" in n:
                p.normal_(0, 0.05)
                p.add_(torch.randn_like(p).pow(3) * 0.01)
    m = m.to(torch.bfloat16)
    m.save_pretrained(str(path))
    return path


@unittest.skipIf(torch is None, "torch not importable")
class TestQuantizers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.W = torch.randn(96, 256) * 0.05
        X = torch.randn(2048, 256) * torch.linspace(0.2, 3.0, 256)  # anisotropic inputs
        X[:, :4] *= 10                                               # a few outlier channels
        self.H = X.t() @ X
        self.n = X.shape[0]

    def test_pack_roundtrip(self):
        from edit_code.quant.quantizers import pack, unpack
        for b in (2, 3, 4, 8):
            c = torch.randint(0, 2 ** b, (37, 29), dtype=torch.uint8)
            p = pack(c, b)
            self.assertEqual(p.numel(), ((c.numel() + 7) // 8) * b)
            self.assertTrue(torch.equal(unpack(p, b, c.numel()).reshape(c.shape), c))

    def test_error_decreases_with_bits(self):
        from edit_code.quant import quantizers as Q
        for qz in ("rtn", "gptq", "awq"):
            errs = []
            for b in (2, 3, 4, 8):
                q = Q.quantize(self.W, qz, b, 64, H=self.H, n_tokens=self.n)
                errs.append(Q.h_error(self.W, q.dequant(torch.float32), self.H))
            self.assertTrue(all(e1 > e2 for e1, e2 in zip(errs, errs[1:])), f"{qz}: {errs}")

    def test_group_size_helps(self):
        from edit_code.quant import quantizers as Q
        e = [float((self.W - Q.rtn(self.W, 3, g).dequant(torch.float32)).pow(2).sum()) for g in (128, 64, 32)]
        self.assertTrue(e[0] > e[1] > e[2], e)

    def test_gptq_awq_beat_rtn_on_output_error(self):
        from edit_code.quant import quantizers as Q
        for b in (2, 3, 4):
            r = Q.h_error(self.W, Q.rtn(self.W, b, 64).dequant(torch.float32), self.H)
            g = Q.h_error(self.W, Q.gptq(self.W, self.H, b, 64).dequant(torch.float32), self.H)
            a = Q.h_error(self.W, Q.awq_lite(self.W, self.H, self.n, b, 64).dequant(torch.float32), self.H)
            self.assertLess(g, r, f"gptq b{b}")
            self.assertLessEqual(a, r * (1 + 1e-6), f"awq b{b}")

    def test_codes_in_range_and_ragged_groups(self):
        from edit_code.quant import quantizers as Q
        W = torch.randn(8, 100)
        for b in (2, 3):
            q = Q.rtn(W, b, 32)
            self.assertEqual(q.scale.shape, (8, 4))
            self.assertLessEqual(int(q.codes.max()), 2 ** b - 1)
            self.assertTrue(torch.isfinite(q.dequant(torch.float32)).all())
        H = torch.eye(100) * 10
        q = Q.gptq(W, H, 4, 32)
        self.assertTrue(torch.isfinite(q.dequant(torch.float32)).all())


class TestBytesAndSchema(unittest.TestCase):
    def setUp(self):
        self.spec = synthetic_spec()

    def test_weight_bytes_formula(self):
        # 4-bit, g128: 0.5 B/weight + per group (2 B scale + 0.5 B zero) / 128 weights
        b = sizes.weight_bytes(1024, 4096, 4, 128)
        self.assertAlmostEqual(b, 1024 * 4096 * 0.5 + 1024 * 32 * 2.5)
        self.assertEqual(sizes.weight_bytes(10, 20, 16, 128), 400)
        self.assertAlmostEqual(sizes.weight_bytes(4, 256, 3, 64, "awq"), 4 * 256 * 3 / 8 + 4 * 4 * (2 + 3 / 8) + 512)

    def test_config_bytes_monotone_and_bf16(self):
        sp = self.spec
        bf = schema.normalize({"default": {"bits": 16, "group_size": 128, "quantizer": "rtn"}}, sp)
        self.assertAlmostEqual(sizes.config_bytes(bf, sp)["total_bytes"], sizes.bf16_bytes(sp), delta=8)
        prev = None
        for b in (8, 4, 3, 2):
            n = schema.normalize({"default": {"bits": b, "group_size": 128, "quantizer": "gptq"}}, sp)
            g = sizes.config_bytes(n, sp)["gib"]
            if prev is not None:
                self.assertLess(g, prev)
            prev = g
        n4 = schema.normalize({"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"}}, sp)
        n4s = schema.normalize({"default": {"bits": 4, "group_size": 32, "quantizer": "gptq"}}, sp)
        self.assertLess(sizes.config_bytes(n4, sp)["gib"], sizes.config_bytes(n4s, sp)["gib"])
        # 27B-like: BF16 ~ 50 GiB; uniform 4-bit g128 with BF16 embed/head ~ 15-17 GiB
        self.assertTrue(45 < sizes.bf16_bytes(sp) / 2 ** 30 < 56)
        self.assertTrue(13 < sizes.config_bytes(n4, sp)["gib"] < 18)

    def test_schema_resolution(self):
        sp = self.spec
        cfg = {"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"},
               "rules": [{"layers": "60-63", "groups": ["down"], "bits": 8}, {"layers": [-1], "groups": "all", "group_size": 64}],
               "groups": {"gdn_ba": {"bits": 16}, "3.qk": {"bits": 3, "quantizer": "rtn"}},
               "embed": 8, "lm_head": "bf16"}
        n = schema.normalize(cfg, sp)
        self.assertEqual(n["groups"]["60.down"]["bits"], 8)
        self.assertEqual(n["groups"]["63.down"], {"bits": 8, "group_size": 64, "quantizer": "gptq"})
        self.assertEqual(n["groups"]["0.gdn_ba"]["bits"], 16)
        self.assertEqual(n["groups"]["3.qk"], {"bits": 3, "group_size": 128, "quantizer": "rtn"})
        self.assertEqual(n["groups"]["5.down"]["bits"], 4)
        self.assertEqual(n["embed"], 8)
        # compress() round-trips
        self.assertEqual(schema.normalize(schema.compress(n, sp), sp), n)

    def test_schema_errors_are_clear(self):
        sp = self.spec
        with self.assertRaises(schema.ConfigError) as e:
            schema.normalize({"default": {"bits": 5, "group_size": 100, "quantizer": "foo"},
                              "groups": {"999.down": {"bits": 4}, "0.qk": {"bits": 4}}, "bogus": 1}, sp)
        msg = str(e.exception)
        for frag in ("bits=5", "group_size=100", "quantizer='foo'", "bogus", "999", "0.qk"):
            self.assertIn(frag, msg)
        with self.assertRaises(schema.ConfigError):
            schema.normalize([1, 2], sp)
        with self.assertRaises(schema.ConfigError):
            schema.normalize({"default": {"bits": 4, "group_size": 128, "quantizer": "awq"}}, sp,
                             {"quantizers": ["rtn", "gptq"]})


@unittest.skipUnless(HAVE_TF, "transformers not importable")
class TestTinyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.src = make_tiny(cls.root / "src")
        from edit_code.quant.spec import spec_from_checkpoint
        cls.spec = spec_from_checkpoint(cls.src)
        torch.manual_seed(1)
        cls.batches = [torch.randint(0, 256, (1, 48)) for _ in range(6)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_spec_names(self):
        sp = self.spec
        self.assertEqual(sp.layers_prefix, "model.language_model.layers")
        self.assertEqual(sp.layer_types, ["linear_attention"] * 3 + ["full_attention"])
        keys = set(sp.group_keys())
        self.assertIn("0.gdn_qkv", keys)
        self.assertIn("0.gdn_ba", keys)
        self.assertIn("3.qk", keys)
        self.assertIn("3.down", keys)
        self.assertNotIn("0.qk", keys)
        projs = {m.proj for m in sp.modules}
        self.assertEqual(projs, {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b",
                                 "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"})
        self.assertIsNotNone(sp.lm_head)
        self.assertGreater(sp.other_bytes, 0)   # norms, conv1d, A_log, visual tower

    def test_hessian_cache_export_reload(self):
        from edit_code.quant import quantizers as Q
        from edit_code.quant.cache import QuantCache
        from edit_code.quant.export import export_fake_quant
        from edit_code.quant.hessian import collect_stats, load_hf_model, load_hessian
        from edit_code.quant.kl import kl_to_reference, reference
        model = load_hf_model(self.src, "cpu")
        hdir = self.root / "hess"
        stats = collect_stats(model, self.spec, self.batches, hdir, layers_per_pass=2, log=lambda *a: None)
        self.assertIn("3.down_in", stats)
        self.assertIn("0.gdn_in", stats)
        H, n, _ = load_hessian(hdir, 3, "mlp_in")
        self.assertEqual(n, 6 * 48)
        self.assertEqual(tuple(H.shape), (128, 128))
        refs = reference(model, self.batches)
        cache = QuantCache(self.root / "cache", self.spec, self.src, hessian_dir=hdir, log=lambda *a: None)
        # cache assembly == direct quantization
        m = next(x for x in self.spec.modules if x.proj == "down_proj")
        W = cache.weight(m.name)
        Hd, nd, _ = load_hessian(hdir, m.layer, m.input_key)
        for qz in ("rtn", "gptq", "awq"):
            direct = Q.quantize(W, qz, 3, 64, H=Hd, n_tokens=nd).dequant(torch.bfloat16)
            first = cache.dequant(m.name, {"bits": 3, "group_size": 64, "quantizer": qz})
            again = cache.dequant(m.name, {"bits": 3, "group_size": 64, "quantizer": qz})  # from disk
            self.assertTrue(torch.equal(direct, first), qz)
            self.assertTrue(torch.equal(first, again), qz)
        self.assertGreaterEqual(cache.stats["hits"], 3)
        # export at several bit widths, reload with transformers, logits finite, KL shrinks with bits
        kls = {}
        for b in (2, 4, 8):
            norm = schema.normalize({"default": {"bits": b, "group_size": 32, "quantizer": "gptq"},
                                     "embed": "bf16", "lm_head": 8}, self.spec)
            out = self.root / f"fq{b}"
            info = export_fake_quant(self.src, out, norm, self.spec, cache, log=lambda *a: None)
            self.assertTrue((out / "config.json").exists())
            self.assertTrue((out / "fake_quant.json").exists())
            self.assertGreater(info["quantized_tensors"], len(self.spec.modules))
            m2 = load_hf_model(out, "cpu")
            with torch.no_grad():
                lo = m2(input_ids=self.batches[0]).logits
            self.assertTrue(torch.isfinite(lo).all())
            kls[b] = kl_to_reference(m2, self.batches, refs)["mean"]
            # untouched tensors are bit-identical, quantized ones lie on the grid
            sd_src, sd_q = model.state_dict(), m2.state_dict()
            nm = "model.language_model.norm.weight"
            self.assertTrue(torch.equal(sd_src[nm], sd_q[nm]))
            self.assertTrue(torch.equal(sd_q[m.name], cache.dequant(m.name, norm["groups"][f"{m.layer}.{m.group}"])))
        self.assertGreater(kls[2], kls[4])
        self.assertGreater(kls[4], kls[8])
        self.assertGreater(kls[2], 0)
        # re-quantizing a fake-quant model at the same setting (RTN) is (almost) idempotent
        from edit_code.quant.spec import spec_from_checkpoint
        sp4 = spec_from_checkpoint(self.root / "fq4")
        c4 = QuantCache(self.root / "cache4", sp4, self.root / "fq4", log=lambda *a: None)
        n4 = schema.normalize({"default": {"bits": 4, "group_size": 32, "quantizer": "rtn"}}, sp4)
        w4 = c4.weight(m.name).float()
        rq = c4.dequant(m.name, n4["groups"][f"{m.layer}.{m.group}"]).float()
        self.assertLess(float((w4 - rq).abs().max()), 0.02 * float(w4.abs().max()) + 1e-6)

    def test_profile_table_real_path(self):
        from edit_code.diagnostics import profile_table_real
        from edit_code.quant.hessian import load_hf_model
        from edit_code.quant.kl import reference
        model = load_hf_model(self.src, "cpu")
        refs = reference(model, self.batches[:3])
        rows = profile_table_real(model, self.spec, self.batches[:3], refs, bits_list=[2, 4], group_size=32,
                                  log=lambda *a: None)
        self.assertEqual(len(rows), len(self.spec.group_keys()) * 2)
        by = {(r["key"], r["bits"]): r for r in rows}
        worse = sum(by[(k, 2)]["dkl_mean"] > by[(k, 4)]["dkl_mean"] for k in self.spec.group_keys())
        self.assertGreaterEqual(worse, len(self.spec.group_keys()) - 1)
        # weights restored after profiling
        lo = model(input_ids=self.batches[0]).logits[0, :-1].float()
        self.assertTrue(torch.allclose(torch.log_softmax(lo, -1), refs[0].float(), atol=0.05))


    def test_packed_kl_matches_unpacked(self):
        """2026-11-21: right-padded batches give the same reference / KL as one context per forward pass."""
        from edit_code.quant import quantizers as Q
        from edit_code.quant.hessian import load_hf_model
        from edit_code.quant.kl import kl_to_reference, kl_to_reference_packed, pack, reference, reference_packed
        model = load_hf_model(self.src, "cpu")
        g = torch.Generator().manual_seed(5)
        bs = [torch.randint(1, 256, (1, n), generator=g) for n in (48, 17, 40, 30, 9, 33)]
        packed = pack(bs, max_tokens=100, pad_id=0)
        self.assertGreater(len(packed), 1)
        self.assertTrue(all(ids.shape[0] * ids.shape[1] <= 100 or ids.shape[0] == 1 for ids, _ in packed))
        self.assertEqual(sorted(n for _, lens in packed for n in lens), sorted(b.shape[-1] for b in bs))
        refs_p = reference_packed(model, packed, chunk=16)
        self.assertEqual(sum(r.shape[0] for r in refs_p), sum(b.shape[-1] - 1 for b in bs))
        refs = reference(model, bs)
        self_p = kl_to_reference_packed(model, packed, refs_p, chunk=16)["mean"]   # only the bf16 storage of refs
        self_u = kl_to_reference(model, bs, refs)["mean"]
        self.assertLess(self_p, 1e-3)
        self.assertAlmostEqual(self_p, self_u, delta=0.2 * self_u + 1e-5)
        m = next(x for x in self.spec.modules if x.proj == "down_proj")
        p = model.get_parameter(m.name)
        orig = p.data.clone()
        with torch.no_grad():
            p.data.copy_(Q.rtn(orig, 2, 32).dequant(p.dtype, device=p.device))
        a = kl_to_reference(model, bs, refs)
        b = kl_to_reference_packed(model, packed, refs_p, chunk=16)
        with torch.no_grad():
            p.data.copy_(orig)
        self.assertEqual(a["tokens"], b["tokens"])
        self.assertGreater(a["mean"], 0)
        self.assertAlmostEqual(a["mean"], b["mean"], delta=0.05 * a["mean"] + 1e-6)
        self.assertAlmostEqual(a["top1_agree"], b["top1_agree"], delta=0.02)

    def test_profile_rows_resume(self):
        """2026-11-21: each finished group is saved to rows_dir; a rerun with the same meta reads it back, another
        meta recomputes."""
        from edit_code.diagnostics import profile_table_real
        from edit_code.quant.hessian import load_hf_model
        from edit_code.quant.kl import kl_to_reference_packed, pack, reference_packed
        model = load_hf_model(self.src, "cpu")
        packed = pack(self.batches[:3], max_tokens=96)
        refs = reference_packed(model, packed)
        keys = self.spec.group_keys()[:3]
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            kw = dict(bits_list=[2, 4], group_size=32, log=lambda *a: None, groups=keys, rows_dir=rd, meta={"v": 1})
            rows = profile_table_real(model, self.spec, packed, refs, kl=kl_to_reference_packed, **kw)
            self.assertEqual(len(rows), 6)
            self.assertEqual(sorted(f.stem for f in rd.glob("*.json")), sorted(keys))

            def boom(*a, **k):
                raise AssertionError("recomputed a saved group")
            again = profile_table_real(model, self.spec, packed, refs, kl=boom, **kw)
            self.assertEqual(again, rows)
            (rd / f"{keys[1]}.json").unlink()                        # one group lost -> only it is recomputed
            calls = []

            def count(*a, **k):
                calls.append(1)
                return kl_to_reference_packed(*a, **k)
            third = profile_table_real(model, self.spec, packed, refs, kl=count, **kw)
            self.assertEqual(len(calls), 2)
            self.assertEqual([r["key"] for r in third], [r["key"] for r in rows])
            for x, y in zip(third, rows):
                self.assertAlmostEqual(x["dkl_mean"], y["dkl_mean"], places=6)
            kw["meta"] = {"v": 2}                                    # other settings -> everything recomputed
            calls.clear()
            profile_table_real(model, self.spec, packed, refs, kl=count, **kw)
            self.assertEqual(len(calls), 6)

    def test_profile_worker_process(self):
        """2026-11-21: the profile table runs in a separate spawned process per GPU (here: cpu) and writes
        rows_dir/<key>.json; the parent only reads them."""
        import multiprocessing as mp
        import random
        from edit_code.common import read_json, write_json
        from edit_code.diagnostics import profile_meta, profile_worker
        from edit_code.quant.hessian import tokenize_texts
        save_tiny_tokenizer(self.src)
        rng = random.Random(0)
        texts = [" ".join(f"w{rng.randrange(1, 256)}" for _ in range(40)) for _ in range(4)]
        if not tokenize_texts(self.src, texts[:1], 32):
            self.skipTest(f"fixture tokenizer unusable with transformers {transformers.__version__}")
        dcfg = {"profile_max_len": 32, "profile_bits": [2, 4], "profile_group_size": 32, "profile_batch_tokens": 64}
        keys = self.spec.group_keys()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            write_json(td / "spec.json", self.spec.to_json())
            ctx = mp.get_context("spawn")
            procs = [ctx.Process(target=profile_worker, args=(str(self.src), "cpu", str(td / "spec.json"), texts, dcfg,
                                                              str(td / "rows"), keys[i::2], "t")) for i in range(2)]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join()
            self.assertEqual([pr.exitcode for pr in procs], [0, 0])
            meta = profile_meta(texts, dcfg)
            for k in keys:
                rec = read_json(td / "rows" / f"{k}.json")
                self.assertEqual(rec["meta"], meta)
                self.assertEqual([r["bits"] for r in rec["rows"]], [2, 4])


def save_tiny_tokenizer(path: Path, vocab_size: int = 256):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"[UNK]": 0}
    for i in range(1, vocab_size):
        vocab[f"w{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(str(path))


@unittest.skipUnless(HAVE_TF, "transformers not importable")
class TestRealPrepPathOnCPU(unittest.TestCase):
    """QuantProblem.prepare() real path (Hessians, profile table, cache warm) + build + fake-quant
    materialize + ToolAPI tools, on the tiny model with cache.device=cpu (vLLM serving is GPU-only)."""

    def test_problem_prepare_build_materialize(self):
        import random
        from types import SimpleNamespace

        import yaml
        sys.path.insert(0, str(REPO / "eval"))
        try:
            import rsicore.sandbox  # noqa: F401
        except Exception:
            self.skipTest("rsicore not importable")
        from edit_code.problem import QuantProblem
        from edit_code.quant.hessian import load_hf_model
        from edit_code.tools import make_tool_api
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = make_tiny(td / "src", seed=3)
            save_tiny_tokenizer(src)
            cfg = yaml.safe_load((REPO / "edit_code" / "configs" / "base.yaml").read_text())
            cfg["cache"]["device"] = "cpu"
            cfg["cache"]["warm"] = [{"quantizer": "rtn", "bits": 4, "group_size": 128},
                                    {"quantizer": "gptq", "bits": 3, "group_size": 64}]
            cfg["calibration"].update(max_len=64, layers_per_pass=2)
            cfg["diagnostics"].update(profile_contexts=3, profile_max_len=32, profile_bits=[2, 4, 8])
            rng = random.Random(0)
            texts = [("math", " ".join(f"w{rng.randrange(1, 256)}" for _ in range(60))) for _ in range(8)]
            from edit_code.quant.hessian import tokenize_texts
            if not tokenize_texts(src, [texts[0][1]], 64):
                # newer transformers (e.g. 5.23) force Qwen3_5Tokenizer for model_type qwen3_5, which
                # cannot use the fixture's word-level tokenizer.json (the target env 5.9.0 can)
                self.skipTest(f"fixture tokenizer unusable with transformers {transformers.__version__}")
            rc = SimpleNamespace(run_dir=td / "run", gpus=[0, 1], mock=False, seed=42, run_id="t")
            prob = QuantProblem(rc, cfg, tag="g01", gen=1, out_dir=td / "run" / "gen_01", prep_dir=td / "run" / "gen_01" / "prep",
                                source_dir=src, source_state=None, budget_gib=1.0, datasets=["dapo_math"],
                                data={"dapo_math": {"screen": [], "signal": []}}, kl_texts=texts,
                                log=lambda *a: None)
            prob.prepare()
            n_groups = len(prob.spec.group_keys())
            self.assertEqual(len(prob.tables["profile"]), n_groups * 3)
            self.assertTrue(prob.tables["activation"])
            self.assertTrue((prob.prep_dir / "hessians" / "L3.down_in.safetensors").exists())
            self.assertGreater(prob.cache.disk_bytes(), 0)
            # tools: bytes / solve_allocation / validation errors, as the sandboxed program sees them
            api = make_tool_api(prob, cfg, cid="t0")
            bf = api.call("model_info", {})["bf16_gib"]
            budget = 0.55 * bf
            prob.budget_gib = budget
            sol = api.call("solve_allocation", {"budget_gib": budget, "weights": {"*.down": 2.0},
                                                "bits": [2, 3, 4, 8], "group_sizes": [64, 128], "lm_head": "bf16"})
            b = api.call("bytes", {"config": sol})
            self.assertTrue(b["fits_budget"], (b, sol.get("note")))
            with self.assertRaises(Exception):
                api.call("bytes", {"config": {"default": {"bits": 7}}})
            with self.assertRaises(Exception):
                api.call("evaluate", {"config": sol})    # no evaluate tool (2026-11-23)
            # build -> artifact -> fake-quant checkpoint -> transformers reload
            art = prob.build(sol, "t0")
            self.assertLessEqual(art["bytes_gib"], budget + 1e-9)
            with self.assertRaises(ValueError):
                prob.build({"default": {"bits": 16, "group_size": 128, "quantizer": "rtn"}}, "t1")  # over budget
            ck = prob.materialize(art)
            self.assertTrue((ck / "fake_quant.json").exists())
            self.assertTrue((ck / "tokenizer.json").exists())
            m = load_hf_model(ck, "cpu")
            with torch.no_grad():
                lo = m(input_ids=torch.randint(0, 256, (1, 16))).logits
            self.assertTrue(torch.isfinite(lo).all())
            # a second generation re-quantizes the fake-quant checkpoint (recursion on CPU)
            prob2 = QuantProblem(rc, cfg, tag="g02", gen=2, out_dir=td / "run" / "gen_02", prep_dir=td / "run" / "gen_02" / "prep",
                                 source_dir=ck, source_state=None, budget_gib=budget, datasets=["dapo_math"],
                                 data={"dapo_math": {"screen": [], "signal": []}}, kl_texts=texts,
                                 log=lambda *a: None)
            prob2.prepare()
            self.assertIsNotNone(prob2.parent_config())
            prof2 = {(r["key"], r["bits"]): r["dkl_mean"] for r in prob2.tables["profile"]}
            prof1 = {(r["key"], r["bits"]): r["dkl_mean"] for r in prob.tables["profile"]}
            # groups already at low bits in M_1 show less additional damage at 8 bits than at 2 bits
            self.assertTrue(all(prof2[(k, 8)] <= prof2[(k, 2)] + 1e-6 for k in prob2.spec.group_keys()))
            self.assertEqual(set(prof1), set(prof2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
