#!/usr/bin/env bash
# Launch one RSI run: Paper A (self-quantization, edit_code/) or Paper B (self-written LoRA, lora_code/).
#
# Usage (from any directory):
#   ./run_rsi.sh --paper A|B --domain DOMAIN [--gens N] [--model NAME|PATH] [--seed S] [--gpus 0,1]
#                [--run-id ID] [--mock] [paper-specific args ...]
#
#   --paper    A -> python -I edit_code/run.py ; B -> python -I lora_code/run.py   (required)
#   --domain   math | code | knowledge | instruction_following | tool_calling
#   --mock     no GPU / no real model: mock OpenAI server + mock builders (runs on a laptop)
#   Every argument except --paper is passed to run.py unchanged (unknown ones too).
#
# Python: $RSI_PYTHON if set; else eval/.venv/bin/python with --mock; else ~/venvs/rsi-gpu/bin/python.
# Outputs: experiment_records/<run-id>/ ; rerunning the same run-id resumes (finished steps are skipped).
# Examples:
#   ./run_rsi.sh --paper A --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id A_math_s42_single_simple
#   ./run_rsi.sh --paper B --domain math --gens 2 --mock
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$ROOT/run_rsi.sh"; }

PAPER=""
MOCK=0
HAVE_DOMAIN=0
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --paper) [ $# -ge 2 ] || { echo "ERROR: --paper needs a value" >&2; exit 2; }; PAPER="$2"; shift 2; continue ;;
    --paper=*) PAPER="${1#--paper=}"; shift; continue ;;
    --mock) MOCK=1 ;;
    --domain|--domain=*) HAVE_DOMAIN=1 ;;
  esac
  ARGS+=("$1")
  shift
done

case "$PAPER" in
  A|a) CODE_DIR=edit_code ;;
  B|b) CODE_DIR=lora_code ;;
  *) echo "ERROR: --paper A|B is required" >&2; usage >&2; exit 2 ;;
esac
if [ "$HAVE_DOMAIN" -eq 0 ]; then
  echo "ERROR: --domain is required" >&2; usage >&2; exit 2
fi

RUN_PY="$ROOT/$CODE_DIR/run.py"
if [ ! -f "$RUN_PY" ]; then
  echo "ERROR: $RUN_PY does not exist yet: Paper $PAPER's code ($CODE_DIR/) is not implemented." >&2
  echo "       The shared runtime is in eval/rsicore/ (see eval/rsicore/INTERFACES.md)." >&2
  exit 3
fi

if [ -n "${RSI_PYTHON:-}" ]; then
  PY="$RSI_PYTHON"
elif [ "$MOCK" -eq 1 ]; then
  PY="$ROOT/eval/.venv/bin/python"
else
  PY="$HOME/venvs/rsi-gpu/bin/python"
fi
if [ ! -x "$PY" ]; then
  echo "ERROR: Python not found at $PY." >&2
  if [ "$MOCK" -eq 1 ]; then
    echo "       Build it with: bash $ROOT/eval/setup_env.sh   (or set RSI_PYTHON)" >&2
  else
    echo "       Build it with: bash $ROOT/eval/hpc/setup_gpu_env.sh   (or set RSI_PYTHON; add --mock for a laptop run)" >&2
  fi
  exit 1
fi

# bash 3.2 + set -u: an empty "${ARGS[@]}" is an unbound-variable error, hence the ${ARGS[@]+...} form.
exec "$PY" -I "$RUN_PY" ${ARGS[@]+"${ARGS[@]}"}
