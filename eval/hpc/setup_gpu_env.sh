#!/usr/bin/env bash
# GPU environment for lora_code/ (Paper B) and edit_code/ (Paper A), and for vLLM serving.
# Stack (pinned by the user, 2026-11-18): verl 0.9.1 | vLLM 0.24.0 | Transformers 5.9.0 | PyTorch 2.11.0 + cu130
# Target: Linux x86_64, NVIDIA driver supporting CUDA 13.0, Python 3.11 (verl requires <3.13).
#
#   bash eval/hpc/setup_gpu_env.sh [--reuse-venv] [VENV_DIR]        # default ~/venvs/rsi-gpu
#   (--reuse-venv: keep an existing venv at VENV_DIR instead of re-running `uv venv` on it)
#
# Installs exactly eval/hpc/requirements-gpu.lock (resolved with uv for x86_64-manylinux_2_28,
# Python 3.11, PyPI + the PyTorch cu130 index). Separate from eval/.venv (the CPU-only client).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REUSE=0; [ "${1:-}" = "--reuse-venv" ] && { REUSE=1; shift; }
VENV="${1:-$HOME/venvs/rsi-gpu}"
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/ (pip install uv)"; exit 1; }
[ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ] || { echo "needs Linux x86_64"; exit 1; }
if [ "$REUSE" = 1 ] && [ -x "$VENV/bin/python" ]; then echo "reusing venv $VENV"; else uv venv --python 3.11 "$VENV"; fi
# Optional download mirrors; the package versions still come from the same lock.
INDEX_ARGS=(--index-url "${RSI_PYPI_INDEX_URL:-https://pypi.org/simple}")
if [ -n "${RSI_TORCH_WHEEL_LINKS:-}" ]; then
  INDEX_ARGS+=(--find-links "$RSI_TORCH_WHEEL_LINKS")
else
  INDEX_ARGS+=(--extra-index-url "${RSI_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}")
fi
if [ -n "${RSI_EXTRA_INDEX_URL:-}" ]; then
  INDEX_ARGS+=(--extra-index-url "$RSI_EXTRA_INDEX_URL")
fi
uv pip install --python "$VENV/bin/python" \
  "${INDEX_ARGS[@]}" \
  --index-strategy unsafe-best-match -r "$HERE/requirements-gpu.lock"
"$VENV/bin/python" -I "$HERE/check_gpu_env.py"
echo "done: $VENV"
