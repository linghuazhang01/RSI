"""GPU check of the real packed format (2026-11-27): does vLLM + Humming serve the packed checkpoint as intended?

One mixed-precision config (2 / 3 / 4 / 8 bit, group sizes 32 / 64 / 128, BF16 modules, tied fused modules; RTN by
default, so no calibration is needed) is exported twice from the same bf16-scale layer cache:
    dequant: the fake-quant BF16 checkpoint (plain vLLM BF16)
    packed : the GPTQ-format checkpoint, served with --quantization humming
Both decode to the same weights (edit_code/tests/test_packed.py checks that on the CPU), so the served models must
agree up to kernel arithmetic. Each is served on one GPU (TP 1, as the candidate servers) and reads the same texts:
    KL(M_0 || dequant), KL(M_0 || packed)   the quantization damage, seen through each format
    KL(dequant || packed)                    what the humming kernels add; ~0 when the format is right
A wrong bit layout, zero-point convention or module mapping shows up as a large KL(dequant || packed) (vLLM loads
such checkpoints silently), and the server logs are searched for vLLM's "skip loading" warnings.

    python -I edit_code/packed_check.py --model <BF16 Qwen3.8-27B dir> --out <scratch dir> --gpu 0
        [--traces <run_dir>/baseline/kl_gate/traces.json] [--n 32] [--prep <run_dir>/prep --quantizer gptq]
        [--config my_config.json] [--skip-m0]

Writes <out>/packed_check.json; exit code 0 = PASS. Needs the eval environment (vLLM 0.24.0 + humming-kernels 0.1.6).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from edit_code.quant import schema  # noqa: E402
from edit_code.quant.spec import ModelSpec, spec_from_checkpoint  # noqa: E402

PACKED_SPACE = {"packed": True, "quantizers": ["rtn", "gptq"], "embed": ["bf16"], "lm_head": ["bf16"]}
BAD_LOG = re.compile(r"not found in params_dict|skip loading|Traceback|out of memory", re.I)
FALLBACK_TEXTS = [
    "The derivative of x^3 + 2x is 3x^2 + 2. To see this, apply the power rule term by term and add the results.",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint(fib(10))",
    "Photosynthesis converts light energy into chemical energy: carbon dioxide and water become glucose and oxygen.",
    "Let n be a positive integer. The sum 1 + 2 + ... + n equals n(n + 1)/2, which can be shown by induction on n.",
    "The capital of Australia is Canberra, chosen in 1908 as a compromise between Sydney and Melbourne.",
    "If a train travels 180 km in 2.5 hours, its average speed is 180 / 2.5 = 72 km per hour.",
    "A prime number has exactly two positive divisors. The first primes are 2, 3, 5, 7, 11, 13, 17, 19, 23 and 29.",
    "In Python, a list comprehension such as [x * x for x in range(5)] builds [0, 1, 4, 9, 16] in one expression.",
]


def default_config(spec: ModelSpec, quantizer: str = "rtn") -> dict:
    """A config that exercises every packed case: 2/3/4/8 bit, g32/64/128, BF16 modules, tied fused modules."""
    full = [i for i, t in enumerate(spec.layer_types) if t == "full_attention"]
    lin = [i for i, t in enumerate(spec.layer_types) if t != "full_attention"]
    q = quantizer
    rules = [{"layers": lin[0:2], "groups": ["gate_up"], "bits": 3, "group_size": 64},
             {"layers": lin[2:4], "groups": ["down"], "bits": 2, "group_size": 32},
             {"layers": lin[4:6], "groups": ["gdn_qkv", "gdn_z"], "bits": 3, "group_size": 32},
             {"layers": lin[6:8], "groups": ["gdn_out"], "bits": 8, "group_size": 128},
             {"layers": full[0:2], "groups": ["qk", "v"], "bits": 8, "group_size": 64},
             {"layers": full[2:3], "groups": ["o"], "bits": 16},
             {"layers": lin[8:9], "groups": ["gdn_qkv", "gdn_z"], "bits": 16},
             {"layers": lin[0:2], "groups": ["gdn_ba"], "bits": 8},
             {"layers": lin[2:4], "groups": ["gdn_ba"], "bits": 2},
             {"layers": lin[4:], "groups": ["gdn_ba"], "bits": 4}]
    rules = [r for r in rules if r["layers"]]
    return {"default": {"bits": 4, "group_size": 128, "quantizer": q}, "rules": rules, "embed": "bf16",
            "lm_head": "bf16"}


def load_items(traces: str | None, n: int, tokenizer_dir: Path) -> list[dict]:
    """[{"prompt", "text"}]: the run's KL traces (first n), else fixed texts split after the first sentence."""
    if traces:
        items = json.loads(Path(traces).read_text())["items"][:n]
        return [{"prompt": it["prompt"], "text": it["text"]} for it in items]
    out = []
    for t in FALLBACK_TEXTS[:n]:
        cut = max(t.find(". ") + 2, len(t) // 3)
        out.append({"prompt": t[:cut], "text": t})
    return out


def serve_logprobs(name: str, model_dir: Path, gpu: int, args: list[str], logs: Path, fn):
    """Start vLLM on one GPU, run fn(endpoint, served_name), stop it; also return the log's suspicious lines."""
    from rsicore.servers import VLLMServer
    from edit_code.quant.export import server_args
    extra = list(args) + server_args(model_dir)
    t0 = time.time()
    with VLLMServer(model_dir, name, [gpu], log_dir=logs, extra_args=extra) as srv:
        up = time.time() - t0
        res = fn(srv.endpoint, name)
        log = Path(srv.log_path)
    bad = [ln for ln in log.read_text(errors="replace").splitlines() if BAD_LOG.search(ln)]
    return res, {"startup_s": round(up, 1), "log": str(log), "suspicious_log_lines": bad[:20], "vllm_args": extra}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the original BF16 checkpoint (M_0)")
    ap.add_argument("--out", required=True, help="scratch dir (two ~20-55 GB checkpoints + a layer cache)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--config", help="a config JSON (editor schema) instead of the built-in mixed config")
    ap.add_argument("--quantizer", default="rtn", choices=["rtn", "gptq"], help="of the built-in config")
    ap.add_argument("--prep", help="a run's prep dir: its hessians (gptq) and the run's cache.gptq_damp; finished "
                                   "bf16-scale entries of its layer cache are copied (the run's cache is never written)")
    ap.add_argument("--traces", help="<run_dir>/baseline/kl_gate/traces.json (default: a few fixed texts)")
    ap.add_argument("--n", type=int, default=32, help="texts")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-gib", default="82", help="vLLM memory per server in GiB (rsicore "
                                                            "--rsi-gpu-memory-gib; the h20 candidate servers use 82)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048,
                    help="prefill chunk: prompt logprobs over the 248k vocabulary cost ~10 B x chunk x vocab outside "
                         "vLLM's budget (h20 runs use 2048; 1024 on 80 GB cards)")
    ap.add_argument("--skip-m0", action="store_true", help="only KL(dequant || packed)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="PASS when KL(dequant || packed) <= tol x KL(M_0 || dequant) (or <= 1e-3)")
    a = ap.parse_args(argv)
    import torch
    from edit_code.common import free_cuda
    from edit_code.quant.cache import QuantCache
    from edit_code.quant.export import export_checkpoint, humming_check
    from rsicore.kl import reference_logprobs, token_kl

    humming_check()
    out, model = Path(a.out).resolve(), Path(a.model).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = spec_from_checkpoint(model)
    cfg = json.loads(Path(a.config).read_text()) if a.config else default_config(spec, a.quantizer)
    norm = schema.normalize(cfg, spec, PACKED_SPACE)          # raises on anything the packed format cannot express
    prep = Path(a.prep).resolve() if a.prep else None
    damp = 0.01
    if prep is not None:                       # the run's GPTQ damping (the cache file names do not record it)
        import yaml
        lock = prep.parent / "run_config.yaml"
        if lock.exists():
            damp = float(((yaml.safe_load(lock.read_text()) or {}).get("paper_cfg") or {}).get("cache", {})
                         .get("gptq_damp", 0.01))
    cache = QuantCache(out / "cache", spec, model, hessian_dir=(prep / "hessians") if prep else None,
                       device=f"cuda:{a.gpu}", damp=damp, scale_dtype=torch.bfloat16, log=print)
    if prep is not None:                       # reuse the run's finished entries without writing into its cache
        import shutil
        from edit_code.quant.schema import module_assignment
        for mod, s in module_assignment(norm, spec).items():
            if s["bits"] < 16:
                src = QuantCache.path(cache, mod, s["quantizer"], s["bits"], s["group_size"]).relative_to(out / "cache")
                if (prep / "cache" / src).exists() and not (out / "cache" / src).exists():
                    (out / "cache" / src).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(prep / "cache" / src, out / "cache" / src)
    ck = {}
    for fmt in ("dequant", "packed"):
        ck[fmt] = out / fmt
        info = export_checkpoint(fmt, model, ck[fmt], norm, spec, cache, meta={"check": "packed_check"}, log=print)
        print(f"[check] {fmt}: {info['bytes']['gib']:.2f} GiB analytic")
    del cache
    free_cuda()

    for tag in ("m0", "dequant"):             # references of an earlier check in this --out are stale
        (out / f"ref_{tag}.jsonl").unlink(missing_ok=True)
    items = load_items(a.traces, a.n, model)
    texts = [it["text"] for it in items]
    vargs = ["--rsi-gpu-memory-gib", str(a.gpu_memory_gib), "--gpu-memory-utilization", "0.58",
             "--max-model-len", str(a.max_model_len), "--max-num-batched-tokens", str(a.max_num_batched_tokens),
             "--max-logprobs", str(a.top_k)]
    logs = out / "logs"
    report = {"model": str(model), "config": schema.compress(norm, spec), "n_texts": len(texts), "servers": {}}

    def ref(tag):
        return lambda ep, name: reference_logprobs([ep], name, items, out / f"ref_{tag}.jsonl", top_k=a.top_k)

    def kl(tag):
        return lambda ep, name: token_kl(out / f"ref_{tag}.jsonl", [ep], name, texts, top_k=a.top_k)

    if not a.skip_m0:
        _, report["servers"]["m0"] = serve_logprobs("check-m0", model, a.gpu, vargs, logs, ref("m0"))
    def dequant_job(ep, name):                     # reference for the packed server; its own KL to M_0
        ref("dequant")(ep, name)
        return None if a.skip_m0 else kl("m0")(ep, name)

    def packed_job(ep, name):
        return kl("dequant")(ep, name), (None if a.skip_m0 else kl("m0")(ep, name))

    r_dq, report["servers"]["dequant"] = serve_logprobs("check-dequant", ck["dequant"], a.gpu, vargs, logs,
                                                        dequant_job)
    r_pk, report["servers"]["packed"] = serve_logprobs("check-packed", ck["packed"], a.gpu, vargs, logs, packed_job)
    kl_dp = r_pk[0]
    report["kl_dequant_packed"] = {"mean": kl_dp["mean"], "token_mean": kl_dp["token_mean"],
                                   "max_text": max(kl_dp["per_text"]) if kl_dp["per_text"] else 0.0}
    if not a.skip_m0:
        report["kl_m0_dequant"] = {"mean": r_dq["mean"], "token_mean": r_dq["token_mean"]}
        report["kl_m0_packed"] = {"mean": r_pk[1]["mean"], "token_mean": r_pk[1]["token_mean"]}
    base = report.get("kl_m0_dequant", {}).get("mean")
    lim = max(1e-3, a.tol * base) if base else 1e-3
    bad_logs = {k: v["suspicious_log_lines"] for k, v in report["servers"].items() if v["suspicious_log_lines"]}
    ok = kl_dp["mean"] <= lim and not any("params_dict" in x or "skip loading" in x
                                          for v in bad_logs.values() for x in v)
    report.update({"limit": lim, "suspicious_logs": bad_logs, "verdict": "PASS" if ok else "FAIL"})
    (out / "packed_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in report if k not in ("config", "servers")}, indent=1))
    print(f"[check] {report['verdict']}: KL(dequant || packed) {kl_dp['mean']:.3g} (limit {lim:.3g}); "
          f"report {out / 'packed_check.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
