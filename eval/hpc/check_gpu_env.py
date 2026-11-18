"""Verify the GPU stack: pinned versions, CUDA 13.0 build, GPU visibility, and (optionally) that
vLLM / Transformers know the model architecture.

  python -I eval/hpc/check_gpu_env.py [--model /path/or/hf-id]
Exit code 1 on any mismatch.
"""
import argparse
import importlib.metadata as md
import sys

PINS = {"torch": "2.11.0", "vllm": "0.24.0", "transformers": "5.9.0", "verl": "0.9.1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model path or HF id to check architecture support")
    a = ap.parse_args()
    bad = []
    for pkg, want in PINS.items():
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            got = None
        ok = got is not None and got.split("+")[0] == want
        print(f"{pkg:13s} {got or 'MISSING':18s} {'OK' if ok else 'want ' + want}")
        bad += [] if ok else [pkg]
    import torch
    cuda = torch.version.cuda
    print(f"torch CUDA build {cuda}; cuda available {torch.cuda.is_available()}; devices {torch.cuda.device_count()}")
    if not (cuda or "").startswith("13.0"):
        bad.append("torch CUDA build is not 13.0")
    if not torch.cuda.is_available():
        print("WARNING: no GPU visible (fine on a login node; required on compute nodes)")
    for name in torch.cuda.device_count() and [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or []:
        print("  gpu:", name)
    for mod in ("vllm", "verl", "peft", "accelerate", "ray", "safetensors", "compressed_tensors"):
        try:
            __import__(mod)
            print(f"import {mod:18s} OK")
        except Exception as e:  # noqa: BLE001
            print(f"import {mod:18s} FAILED: {type(e).__name__}: {e}")
            bad.append(f"import {mod}")
    if a.model:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.model)
        archs = cfg.architectures or []
        print("model architectures:", archs)
        from vllm.model_executor.models import ModelRegistry
        supported = set(ModelRegistry.get_supported_archs())
        miss = [x for x in archs if x not in supported]
        if miss:
            bad.append(f"vLLM 0.24.0 does not list architecture(s) {miss}")
        print("vLLM supports:", "yes" if not miss else f"NO {miss}")
    print("RESULT:", "OK" if not bad else f"PROBLEMS: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
