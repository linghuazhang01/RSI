"""CPU tests of the real packed export (2026-11-27): GPTQ-format tensors for vLLM's Humming kernels.

What is checked here (no GPU, no vLLM): the int32 bit layout against an independent reader of humming's stream format,
the packed-format rules of schema.normalize, the tied knapsack, the quantization_config and its resolution with vLLM's
rules, and an export of a tiny Qwen3.5 checkpoint whose packed tensors decode to exactly the weights of the dequant
export (bf16 scales). Whether the humming kernels compute the same numbers is checked on a GPU by
edit_code/packed_check.py.

Run:  eval/.venv/bin/python -I edit_code/tests/test_packed.py   (the tiny-model test needs transformers with Qwen3.5;
      the others need torch only)
"""
from __future__ import annotations

import json
import sys
import tempfile
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

from edit_code.quant import schema
from edit_code.quant.spec import TIED_GROUPS, synthetic_spec

PACKED_SPACE = {"packed": True, "quantizers": ["rtn", "gptq"], "embed": ["bf16"], "lm_head": ["bf16"]}


def bitstream_reference(words: list[int], bits: int, n: int) -> list[int]:
    """Value j = bits [j*b, j*b + b) of the little-endian concatenation of 32-bit words (humming pack_weight.cuh)."""
    big = 0
    for i, w in enumerate(words):
        big |= (w & 0xFFFFFFFF) << (32 * i)
    return [(big >> (j * bits)) & ((1 << bits) - 1) for j in range(n)]


@unittest.skipIf(torch is None, "torch not importable")
class TestBitLayout(unittest.TestCase):
    def test_pack_rows_matches_stream_reference(self):
        from edit_code.quant.export import pack_rows_int32, unpack_rows_int32
        g = torch.Generator().manual_seed(0)
        for b in (2, 3, 4, 8):
            R, C = 5, 96 if b != 3 else 64
            codes = torch.randint(0, 2 ** b, (R, C), generator=g, dtype=torch.int64).to(torch.uint8)
            codes[0, 0], codes[0, 1] = 0, 2 ** b - 1                 # extremes
            w = pack_rows_int32(codes, b)
            self.assertEqual(tuple(w.shape), (R, C * b // 32))
            self.assertEqual(w.dtype, torch.int32)
            for r in range(R):
                ref = bitstream_reference(w[r].tolist(), b, C)
                self.assertEqual(ref, codes[r].tolist(), f"b={b} row={r}")
            self.assertTrue(torch.equal(unpack_rows_int32(w, b, C), codes.to(torch.int64)))

    def test_three_bit_values_cross_words(self):
        from edit_code.quant.export import pack_rows_int32
        codes = torch.zeros(1, 32, dtype=torch.uint8)
        codes[0, 10] = 0b111                                         # bits 30..32: words 0 and 1
        w = pack_rows_int32(codes, 3)[0].tolist()
        self.assertEqual(w[0] & 0xFFFFFFFF, 0b11 << 30)
        self.assertEqual(w[1] & 0xFFFFFFFF, 0b1)
        self.assertEqual(w[2], 0)

    def test_scale_dtype_bf16(self):
        from edit_code.quant import quantizers as Q
        W = torch.randn(64, 128) * 0.02
        q16, qb = Q.rtn(W, 8, 32), Q.rtn(W, 8, 32, torch.bfloat16)
        self.assertEqual(q16.scale.dtype, torch.float16)
        self.assertEqual(qb.scale.dtype, torch.bfloat16)
        # the dequantized weight is exactly what a bf16-scale kernel computes: bf16((code - zero) * bf16(scale))
        idx = torch.arange(128) // 32
        kern = ((qb.codes.float() - qb.zero.float()[:, idx]) * qb.scale.float()[:, idx]).to(torch.bfloat16)
        self.assertTrue(torch.equal(Q.dequant(qb, torch.bfloat16), kern))
        # an fp16-scale entry served with bf16 scales would NOT match at 8 bit (why the cache uses bf16 scales)
        kern16 = ((q16.codes.float() - q16.zero.float()[:, idx]) * q16.scale.to(torch.bfloat16).float()[:, idx])
        self.assertFalse(torch.equal(Q.dequant(q16, torch.bfloat16), kern16.to(torch.bfloat16)))


class TestPackedSchema(unittest.TestCase):
    def setUp(self):
        self.spec = synthetic_spec()

    def norm(self, cfg):
        return schema.normalize(cfg, self.spec, PACKED_SPACE)

    def test_uniform_ok_and_dequant_space_unchanged(self):
        n = self.norm({"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"}})
        self.assertEqual(n["embed"], "bf16")
        # the default (dequant) space still accepts what it accepted before
        schema.normalize({"default": {"bits": 4, "group_size": 128, "quantizer": "awq"},
                          "groups": {"3.v": {"bits": 8}, "0.gdn_ba": {"bits": 3}}, "embed": 8, "lm_head": 4},
                         self.spec)

    def test_rejections(self):
        base = {"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"}}
        cases = {"tie_qk_v": dict(base, groups={"3.v": {"bits": 8}}),
                 "tie_gdn": dict(base, groups={"0.gdn_z": {"group_size": 64}}),
                 "tie_bf16": dict(base, groups={"0.gdn_qkv": {"bits": 16}}),
                 "gdn_ba_3bit": dict(base, groups={"0.gdn_ba": {"bits": 3}})}
        for name, cfg in cases.items():
            with self.assertRaises(schema.ConfigError, msg=name) as cm:
                self.norm(cfg)
            text = str(cm.exception)
            self.assertTrue(("one vLLM module" in text) if name.startswith("tie") else ("32-bit words" in text), text)
        with self.assertRaises(schema.ConfigError):              # quantizer / embed outside the packed space
            self.norm(dict(base, default={"bits": 4, "group_size": 128, "quantizer": "awq"}))
        with self.assertRaises(schema.ConfigError):
            self.norm(dict(base, embed=8))
        # packed_problems itself also refuses awq / quantized embed (a space that forgot to drop them)
        n = schema.normalize(dict(base, embed=8), self.spec)
        self.assertTrue(any("embed" in p for p in schema.packed_problems(n, self.spec)))
        n = schema.normalize(dict(base, default={"bits": 4, "group_size": 128, "quantizer": "awq"}), self.spec)
        self.assertTrue(any("no packed form" in p for p in schema.packed_problems(n, self.spec)))

    def test_tied_pair_ok(self):
        self.norm({"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"},
                   "rules": [{"layers": [3, 7], "groups": ["qk", "v"], "bits": 8},
                             {"layers": "0-2", "groups": ["gdn_qkv", "gdn_z"], "bits": 16},
                             {"groups": "gdn_ba", "bits": 2}]})

    def test_solver_ties_and_packable(self):
        from edit_code import mock as M
        from edit_code.allocation import solve_allocation
        prof = M.profile_table(self.spec, None, [2, 3, 4], 128)
        space = dict(schema.DEFAULT_ACTION_SPACE, **PACKED_SPACE)
        for budget in (15.5, 17.4, 19.0):
            cfg = solve_allocation(self.spec, prof, budget, quantizer="gptq", embed="bf16", action_space=space,
                                   bits=[2, 3, 4, 8, 16], group_sizes=[64, 128])
            n = schema.normalize(cfg, self.spec, space)           # raises on any packed problem
            self.assertEqual(n["lm_head"], "bf16")
            for i in range(self.spec.num_layers):
                for a, b in TIED_GROUPS:
                    if f"{i}.{a}" in n["groups"]:
                        sa, sb = n["groups"][f"{i}.{a}"], n["groups"][f"{i}.{b}"]
                        self.assertEqual((sa["bits"], sa["group_size"]), (sb["bits"], sb["group_size"]))
        with self.assertRaises(ValueError):
            solve_allocation(self.spec, prof, 17.4, quantizer="awq", action_space=space)
        # dequant space: the default lm_head is still 8 (the solver output itself was compared with the pre-change
        # solver on 42 budget / weight / fixed / embed cases, 2026-11-27: identical)
        c1 = solve_allocation(self.spec, prof, 15.0, quantizer="gptq")
        self.assertEqual(c1["lm_head"], 8)
        # a fixed group fixes its tied partner too
        cf = solve_allocation(self.spec, prof, 17.4, quantizer="gptq", action_space=space,
                              fixed={"3.qk": {"bits": 8, "group_size": 128, "quantizer": "gptq"}})
        self.assertEqual(schema.normalize(cf, self.spec, space)["groups"]["3.v"]["bits"], 8)

    def test_quant_config_resolution(self):
        from edit_code.quant.export import packed_quant_config, resolve_humming
        from edit_code.quant.schema import module_assignment
        n = self.norm({"default": {"bits": 4, "group_size": 128, "quantizer": "gptq"},
                       "rules": [{"layers": [3], "groups": ["qk", "v"], "bits": 8, "group_size": 64},
                                 {"layers": [0], "groups": ["gdn_qkv", "gdn_z"], "bits": 16},
                                 {"layers": [1], "groups": ["gate_up"], "bits": 3, "group_size": 32},
                                 {"layers": [2], "groups": ["down"], "bits": 2}]})
        qc = packed_quant_config(n, self.spec)
        self.assertEqual((qc["quant_method"], qc["bits"], qc["group_size"], qc["sym"], qc["desc_act"]),
                         ("gptq", 4, 128, False, False))
        assign = module_assignment(n, self.spec)
        for m in self.spec.modules:
            s, got = assign[m.name], resolve_humming(qc, m.vllm_prefix)
            if s["bits"] >= 16:
                self.assertIsNone(got, m.name)
            else:
                self.assertEqual((got["bits"], got["group_size"], got["sym"]), (s["bits"], s["group_size"], False),
                                 m.name)
            # the text-only naming (model.layers...) resolves the same way
            alt = m.vllm_prefix.replace("language_model.model.", "model.", 1)
            self.assertEqual(resolve_humming(qc, alt), got)
        for other in ("visual.blocks.0.attn.qkv", "visual.merger.linear_fc1", "language_model.lm_head",
                      "language_model.model.embed_tokens", "language_model.model.layers.0.linear_attn.conv1d",
                      "mtp.layers.0.mlp.gate_up_proj", "mtp.fc"):
            self.assertIsNone(resolve_humming(qc, other), other)
        self.assertIsNone(packed_quant_config(self.norm({"default": {"bits": 16, "group_size": 128,
                                                                     "quantizer": "rtn"}}), self.spec))


@unittest.skipIf(not HAVE_TF, "transformers with Qwen3.5 not importable")
class TestPackedExport(unittest.TestCase):
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

    def test_export_decodes_to_dequant_weights(self):
        from safetensors import safe_open
        from edit_code.quant.cache import QuantCache
        from edit_code.quant.export import (export_checkpoint, read_marker, server_args, unpack_rows_int32,
                                            validate_packed)
        from edit_code.quant.spec import checkpoint_tensors
        cache = QuantCache(self.root / "cache", self.spec, self.src, scale_dtype=torch.bfloat16, log=lambda *a: None)
        n = schema.normalize({"default": {"bits": 4, "group_size": 32, "quantizer": "rtn"},
                              "rules": [{"layers": [0], "groups": ["gate_up"], "bits": 3, "group_size": 64},
                                        {"layers": [1], "groups": ["down"], "bits": 2, "group_size": 128},
                                        {"layers": [1], "groups": ["gdn_qkv", "gdn_z"], "bits": 3},
                                        {"layers": [3], "groups": ["qk", "v"], "bits": 8},
                                        {"layers": [2], "groups": ["gdn_out"], "bits": 16},
                                        {"groups": ["gdn_ba"], "bits": 16},
                                        {"layers": [0], "groups": ["gdn_ba"], "bits": 8}]},
                             self.spec, PACKED_SPACE)
        pk, dq = self.root / "packed", self.root / "dequant"
        info = export_checkpoint("packed", self.src, pk, n, self.spec, cache, log=lambda *a: None)
        export_checkpoint("dequant", self.src, dq, n, self.spec, cache, log=lambda *a: None)
        self.assertEqual(info["format"], "packed")
        self.assertEqual(server_args(pk), ["--quantization", "humming"])
        self.assertEqual(server_args(dq), [])
        self.assertEqual(read_marker(dq)["scale_dtype"], "bfloat16")
        cfg = json.loads((pk / "config.json").read_text())
        self.assertEqual(cfg["quantization_config"]["quant_method"], "gptq")
        idx = json.loads((pk / "model.safetensors.index.json").read_text())["weight_map"] \
            if (pk / "model.safetensors.index.json").exists() else None
        have = checkpoint_tensors(pk)
        if idx is not None:
            self.assertEqual(set(idx), set(have))

        def tensor(d, name):
            fn = checkpoint_tensors(d)[name]["file"]
            with safe_open(str(d / fn), framework="pt") as f:
                return f.get_tensor(name)

        from edit_code.quant.schema import module_assignment
        assign = module_assignment(n, self.spec)
        checked = 0
        for m in self.spec.modules:
            s, base = assign[m.name], m.module_path
            ref = tensor(dq, m.name)
            if s["bits"] >= 16:
                self.assertTrue(torch.equal(tensor(pk, m.name), ref))
                continue
            b, gs = s["bits"], s["group_size"]
            out_f, in_f = m.shape
            qw, qz, sc = tensor(pk, base + ".qweight"), tensor(pk, base + ".qzeros"), tensor(pk, base + ".scales")
            codes = unpack_rows_int32(qw.t().contiguous(), b, in_f)                 # [out, in]
            zero = unpack_rows_int32(qz, b, out_f).t()                              # [out, in/g]
            col = torch.arange(in_f) // gs
            w = ((codes.float() - zero.float()[:, col]) * sc.t().float()[:, col]).to(torch.bfloat16)
            self.assertTrue(torch.equal(w, ref), m.name)
            checked += 1
        self.assertGreater(checked, 10)
        # untouched tensors (norms, vision tower, embedding, LM head) are copied unchanged
        for name in ("model.language_model.norm.weight", "lm_head.weight", "model.language_model.embed_tokens.weight"):
            if name in have:
                self.assertTrue(torch.equal(tensor(pk, name), tensor(self.src, name)), name)
        # the validator catches a missing zero point (vLLM would load garbage silently)
        bad = self.root / "bad"
        import shutil
        shutil.copytree(pk, bad)
        m = next(x for x in self.spec.modules if assign[x.name]["bits"] < 16)
        fn = checkpoint_tensors(bad)[m.module_path + ".qzeros"]["file"]
        from safetensors.torch import load_file, save_file
        d = load_file(str(bad / fn))
        d.pop(m.module_path + ".qzeros")
        save_file(d, str(bad / fn), metadata={"format": "pt"})
        if (bad / "model.safetensors.index.json").exists():
            ix = json.loads((bad / "model.safetensors.index.json").read_text())
            ix["weight_map"].pop(m.module_path + ".qzeros")
            (bad / "model.safetensors.index.json").write_text(json.dumps(ix))
        with self.assertRaises(ValueError):
            validate_packed(bad, n, self.spec)

    def test_packed_refuses_bad_inputs(self):
        from edit_code.quant.cache import QuantCache
        from edit_code.quant.export import export_packed
        fp16 = QuantCache(self.root / "cache16", self.spec, self.src, log=lambda *a: None)
        n = schema.normalize({"default": {"bits": 4, "group_size": 32, "quantizer": "rtn"},
                              "groups": {"gdn_ba": {"bits": 16}}}, self.spec, PACKED_SPACE)
        with self.assertRaises(ValueError):                    # fp16-scale cache
            export_packed(self.src, self.root / "x", n, self.spec, fp16, log=lambda *a: None)
        bf = QuantCache(self.root / "cachebf", self.spec, self.src, scale_dtype=torch.bfloat16, log=lambda *a: None)
        n2 = schema.normalize({"default": {"bits": 4, "group_size": 32, "quantizer": "rtn"},
                               "groups": {"3.v": {"bits": 8}, "gdn_ba": {"bits": 16}}}, self.spec)
        with self.assertRaises(ValueError):                    # qk != v
            export_packed(self.src, self.root / "y", n2, self.spec, bf, log=lambda *a: None)


LAUNCHER = "#!/bin/sh\n'''exec' \"{py}\" \"$0\" \"$@\"\n' '''\nimport sys\n"   # pip / uv long-path launcher


class TestRunIntegration(unittest.TestCase):
    """Regressions found in review (2026-11-27)."""

    def _problem(self, tmp, cfg):
        from types import SimpleNamespace
        from edit_code import mock as M
        from edit_code.problem import QuantProblem
        rc = SimpleNamespace(mock=True, run_dir=tmp, run_id="t", seed=42, gpus=[0, 1])
        base = {"gpu_layout": {"candidate_group_size": 1}, "candidates": {"keep_checkpoints": 1}}
        p = QuantProblem(rc, dict(base, **cfg), tag="t", gen=1, out_dir=tmp / "p", prep_dir=tmp / "prep",
                         source_dir=None, source_state=None, budget_gib=17.4, datasets=["simple_math_pool"],
                         kl_texts=[])
        p.spec = synthetic_spec()
        p.tables = {"profile": M.profile_table(p.spec, None, [2, 3, 4], 128), "hessian": []}
        return p

    def test_run_py_as_script_serves_checkpoints(self):
        """run.py runs as a script: the checkpoint-args import in Runner._server must work there (absolute)."""
        import runpy
        from types import SimpleNamespace
        ns = runpy.run_path(str(REPO / "edit_code" / "run.py"), run_name="not_main")
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "ckpt"
            d.mkdir()
            (d / "fake_quant.json").write_text(json.dumps({"format": "packed",
                                                           "vllm_args": ["--quantization", "humming"]}))
            fake = SimpleNamespace(rc=SimpleNamespace(mock=False, seed=0), registry=Path(t) / "reg.json",
                                   run_dir=Path(t), cfg={"vllm": {"candidate_args": ["--max-logprobs", "20"]}})
            srv = ns["Runner"]._server(fake, "x", d, None, [0], "candidate_args")
            self.assertEqual(srv.kw["extra_args"][-2:], ["--quantization", "humming"])
            m0 = ns["Runner"]._server(fake, "m0", Path(t), None, [0], "candidate_args")   # no marker: no flag
            self.assertNotIn("--quantization", m0.kw["extra_args"])

    def test_packed_run_checks(self):
        from edit_code.problem import check_packed_run
        packed = {"candidates": {"format": "packed"}, "action_space": PACKED_SPACE,
                  "gpu_layout": {"candidate_group_size": 1}, "loop": {"editor": "frozen"}}
        self.assertEqual(check_packed_run(packed, mock=True, candidate_slots=[[1], [2]]), "packed")
        with self.assertRaises(ValueError):            # TP 2 candidate servers
            check_packed_run(dict(packed, gpu_layout={"candidate_group_size": 2}), mock=True)
        with self.assertRaises(ValueError):            # a lent 2-GPU editor slot
            check_packed_run(packed, mock=True, candidate_slots=[[1], [2, 3]])
        with self.assertRaises(ValueError):
            check_packed_run(dict(packed, loop={"editor": "current"}), mock=True)
        with self.assertRaises(ValueError):            # profile half applied
            check_packed_run({"candidates": {"format": "packed"}}, mock=True)
        self.assertEqual(check_packed_run({"candidates": {"format": "dequant"}, "action_space": PACKED_SPACE,
                                           "loop": {"editor": "current"}}, mock=True), "dequant")

    def test_vllm_python_launchers(self):
        import os
        from edit_code.quant.export import vllm_python
        with tempfile.TemporaryDirectory() as t:
            py = sys.executable
            sh, env, direct = Path(t) / "vllm_sh", Path(t) / "vllm_env", Path(t) / "vllm_direct"
            sh.write_text(LAUNCHER.format(py=py))
            env.write_text("#!/usr/bin/env python3\n")
            direct.write_text(f"#!{py}\n")
            old = os.environ.get("RSI_VLLM")
            try:
                for f in (sh, direct):
                    os.environ["RSI_VLLM"] = str(f)
                    self.assertEqual(vllm_python(), py, f.name)
                os.environ["RSI_VLLM"] = str(env)
                self.assertTrue(Path(vllm_python()).name.startswith("python"))
            finally:
                if old is None:
                    os.environ.pop("RSI_VLLM", None)
                else:
                    os.environ["RSI_VLLM"] = old

    def test_prompts_and_tools(self):
        from edit_code.prompts import system_prompt
        from edit_code.tools import make_tool_api
        with tempfile.TemporaryDirectory() as t:
            dq = self._problem(Path(t) / "a", {})
            pk = self._problem(Path(t) / "b", {"action_space": PACKED_SPACE, "candidates": {"format": "packed"}})
            api_dq, api_pk = make_tool_api(dq, {}), make_tool_api(pk, {})
            text_dq = system_prompt(dq, api_dq.describe(), {})
            text_pk = system_prompt(pk, api_pk.describe(), {})
            self.assertNotIn("Packed format", text_dq)
            self.assertNotIn("fused module", text_dq)
            self.assertIn('"lm_head": 8,         # "bf16" | 8 | 4', text_dq)        # the code's default space
            bf = self._problem(Path(t) / "c", {"action_space": {"embed": ["bf16"], "lm_head": ["bf16"]}})
            text_bf = system_prompt(bf, make_tool_api(bf, {}).describe(), {})       # base.yaml since 2026-11-27
            self.assertIn('"embed": "bf16",       # "bf16" only: the embedding is not quantized', text_bf)
            self.assertIn('"lm_head": "bf16",     # "bf16" only: the LM head is not quantized', text_bf)
            self.assertNotIn('"lm_head": 8', text_bf)
            self.assertEqual(schema.normalize(make_tool_api(bf, {})._tools["solve_allocation"][0](), bf.spec,
                                              bf.space)["lm_head"], "bf16")
            # an action space without 16 (every linear module quantized): model_info, the solver with fixed rules
            # and the prompt all work (model_info used to size a bits-16 config and failed)
            nb = self._problem(Path(t) / "d", {"action_space": {"bits": [2, 3, 4, 8], "embed": ["bf16"],
                                                                "lm_head": ["bf16"]}})
            api_nb = make_tool_api(nb, {})
            self.assertGreater(api_nb._tools["model_info"][0]()["bf16_gib"], 40)
            cfg = api_nb._tools["solve_allocation"][0](fixed_rules=[{"layers": [3], "groups": ["down"], "bits": 8}])
            n = schema.normalize(cfg, nb.spec, nb.space)
            self.assertEqual(n["groups"]["3.down"]["bits"], 8)
            self.assertTrue(all(g["bits"] < 16 for g in n["groups"].values()))
            text_nb = system_prompt(nb, api_nb.describe(), {})
            self.assertIn("no BF16 groups", text_nb)
            self.assertIn('"gdn_ba": {"bits": 8}', text_nb)
            self.assertIn('"gdn_ba": {"bits": 16}', text_dq)
            self.assertIn("ONE fused module", text_pk)
            self.assertIn("Packed format", text_pk)
            self.assertNotIn("awq", text_pk.lower())
            self.assertIn("bf16 scales", text_pk)
            # fixing one group of a tied pair through the tool: the solver ties the partner
            fn = api_pk._tools["solve_allocation"][0]
            cfg = fn(fixed_rules=[{"layers": [3], "groups": ["qk"], "bits": 8}])
            n = schema.normalize(cfg, pk.spec, pk.space)
            self.assertEqual(n["groups"]["3.v"]["bits"], 8)
            self.assertEqual(n["lm_head"], "bf16")
            self.assertTrue(api_pk._tools["model_info"][0]()["packed_format"])
            self.assertNotIn("packed_format", api_dq._tools["model_info"][0]())


class TestProfiles(unittest.TestCase):
    def test_inference_profiles(self):
        import yaml
        from rsicore import config as C
        base = yaml.safe_load((REPO / "edit_code" / "configs" / "base.yaml").read_text())
        d = C._apply_hardware(base, [])
        self.assertEqual(d["inference"], "dequant")
        self.assertEqual(d["candidates"]["format"], "dequant")
        self.assertFalse(d["action_space"]["packed"])
        p = C._apply_hardware(yaml.safe_load((REPO / "edit_code" / "configs" / "base.yaml").read_text()),
                              ["inference=packed"])
        self.assertEqual(p["candidates"]["format"], "packed")
        self.assertTrue(p["action_space"]["packed"])
        self.assertEqual(p["action_space"]["quantizers"], ["rtn", "gptq"])
        self.assertEqual(p["budget"]["reference"]["embed"], "bf16")
        self.assertEqual(d["budget"]["reference"]["embed"], "bf16")          # every mode: only linear weights
        self.assertEqual((d["action_space"]["embed"], d["action_space"]["lm_head"]), (["bf16"], ["bf16"]))
        self.assertEqual(p["budget"]["reference"]["default"]["bits"], 4)      # deep merge keeps the rest
        # the run lock: a config without the new keys (locked before 2026-11-27) equals today's dequant default
        old = {k: v for k, v in d.items() if k not in ("inference", "inference_profiles")}
        old["candidates"] = {k: v for k, v in d["candidates"].items() if k != "format"}
        old["action_space"] = {k: v for k, v in d["action_space"].items() if k != "packed"}
        self.assertEqual(C.lock_view(old, "A"), C.lock_view(d, "A"))
        self.assertNotEqual(C.lock_view(p, "A"), C.lock_view(d, "A"))

    def test_problem_space_check(self):
        from edit_code.problem import check_packed_space
        sp = dict(schema.DEFAULT_ACTION_SPACE)
        check_packed_space(sp, "dequant")
        with self.assertRaises(ValueError):
            check_packed_space(sp, "packed")
        with self.assertRaises(ValueError):
            check_packed_space(dict(sp, packed=True), "packed")          # awq / 8-bit embed still allowed
        check_packed_space(dict(sp, **PACKED_SPACE), "packed")
        check_packed_space(dict(sp, **PACKED_SPACE), "dequant")           # packed_dequant


if __name__ == "__main__":
    unittest.main()
