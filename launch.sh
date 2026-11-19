#!/usr/bin/env bash
# Formal launcher: start the RSI runs (Paper A self-quantization, Paper B self-written LoRA) for several
# domains at once, each run on its own GPUs in the background with its own log. Wraps ./run_rsi.sh.
#
# Usage:
#   ./launch.sh [--paper A|B|both] [--domains all|math,code,...] [--gens 5] [--hw b300|h20]
#               [--gpus 0,1,2,...] [--per-run N] [--model qwen3.8-27b] [--seed 42] [--with-all] [--smoke]
#               [--all-domains] [--set KEY=VALUE ...] [--dry-run] [--mock] [--skip-checks]
#
#   --paper     A, B or both (default both)
#   --domains   all = math,code,knowledge,instruction_following,tool_calling (default all)
#   --gens      generations per run (default 5)
#   --hw        hardware profile (default b300). Qwen3.8-27B: minimum GPUs per run
#                 Paper A 2 (editor | candidates); Paper B 2 (main | training), 3 on h20 with --all-domains
#                 (+ original-model server for E_t)
#   --gpus      GPU ids to use, any order / gaps (default: every GPU nvidia-smi lists). ALL of them are used:
#               every run gets its minimum, the rest are handed out one by one over the runs, and each run
#               spreads its GPUs over its roles (more tensor parallelism / a split training model; see
#               gpu_layout in edit_code/common.py and lora_code/common.py). Runs get consecutive groups.
#   --per-run   fixed GPUs per run instead (the rest stay idle)
#   --set       passed to every run.py (repeatable), e.g. --set search.max_evaluate=12, or a manual GPU
#               split --set 'gpu_roles.train=[2,3]' (indices into the run's GPUs; overrides the automatic one)
#   --with-all  also start Paper A's five-domain main experiment (--domain all)
#   --smoke     1 generation, run ids end in _smoke (first real-hardware check; separate from formal runs)
#   (default)   each run uses ONLY its own domain (scope single): training, gate, feedback, KL and held-out
#               on that domain; Paper B has no retention probes and no E_t. Run ids end in _single.
#   --all-domains  the original design (scope all): the other four domains are retention probes / KL texts,
#               held-out on all five domains, Paper B E_t on. Run ids without _single.
#   --dry-run   print the preflight result and the commands, start nothing
#   --mock      laptop test: mock model server, no GPU (fake GPU ids), uses eval/.venv
#   --skip-checks  skip the preflight checks
#
# Run ids: <A|B>_<domain>_s<seed>[_single][_smoke] (A_all_s<seed> for --with-all), in experiment_records/.
# Logs and pids: experiment_records/_launch/<timestamp>/. Rerunning the same command resumes every run
# (finished steps are skipped); a run whose process is still alive is not started twice.
#
# Examples:
#   ./launch.sh --paper both --smoke                 # all 10 single-domain runs, 1 generation (20 B300)
#   ./launch.sh --paper both --gens 5 --with-all     # formal: 10 single-domain runs + Paper A all (22 B300)
#   ./launch.sh --hw h20 --paper both --gens 5       # on H20: 5 x 2 + 5 x 3 = 25 GPUs
#   ./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 0,1
#   ./launch.sh --paper A --domains math --gens 5 --gpus 0,1
#   ./launch.sh --paper B --domains code,math --gens 5 --gpus 4,5,6,7
#   ./launch.sh --hw h20 --paper both --domains math --gens 5 --gpus 0,1,2,3   # A on 0,1 and B on 2,3
#   ./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 4,5,6,7      # one run: editor 1 + 3 candidate servers
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$ROOT/launch.sh"; }

PAPER=both; DOMAINS=all; GENS=5; GPUS=""; PER_RUN=""; MODEL=qwen3.8-27b; SEED=42; HW=b300
WITH_ALL=0; SMOKE=0; DRY=0; MOCK=0; SKIP=0; SINGLE=1; SETS=()
need() { [ $# -ge 2 ] || { echo "ERROR: $1 needs a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --paper) need "$@"; PAPER="$2"; shift ;;
    --domains) need "$@"; DOMAINS="$2"; shift ;;
    --gens) need "$@"; GENS="$2"; shift ;;
    --gpus) need "$@"; GPUS="$2"; shift ;;
    --per-run) need "$@"; PER_RUN="$2"; shift ;;
    --hw) need "$@"; HW="$2"; shift ;;
    --set) need "$@"; SETS+=(--set "$2"); shift ;;
    --model) need "$@"; MODEL="$2"; shift ;;
    --seed) need "$@"; SEED="$2"; shift ;;
    --with-all) WITH_ALL=1 ;;
    --smoke) SMOKE=1 ;;
    --single-domain) SINGLE=1 ;;
    --all-domains) SINGLE=0 ;;
    --dry-run) DRY=1 ;;
    --mock) MOCK=1 ;;
    --skip-checks) SKIP=1 ;;
    *) echo "ERROR: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

ALL_DOMAINS="math code knowledge instruction_following tool_calling"
case "$HW" in b300|h20) ;; *) echo "ERROR: --hw must be b300 or h20" >&2; exit 2 ;; esac
min_of() {  # paper domain -> minimum GPUs of one run
  if [ "$HW" = h20 ] && [ "$1" = B ] && { [ "$SINGLE" -eq 0 ] || [ "$2" = all ]; }; then echo 3
  else echo 2; fi
}
case "$PAPER" in A|a) PAPERS="A" ;; B|b) PAPERS="B" ;; both) PAPERS="A B" ;;
  *) echo "ERROR: --paper must be A, B or both" >&2; exit 2 ;; esac
if [ "$DOMAINS" = all ]; then DOMS="$ALL_DOMAINS"; else DOMS="$(echo "$DOMAINS" | tr ',' ' ')"; fi
for d in $DOMS; do
  case " $ALL_DOMAINS " in *" $d "*) ;; *) echo "ERROR: unknown domain $d" >&2; exit 2 ;; esac
done
[ "$SMOKE" -eq 1 ] && GENS=1
SUFFIX=""; [ "$SINGLE" -eq 1 ] && SUFFIX="_single"
[ "$SINGLE" -eq 0 ] && SETS=(--set scope=all ${SETS[@]+"${SETS[@]}"})   # explicit --set values come later and win
SMOKE_SUFFIX=""; [ "$SMOKE" -eq 1 ] && SMOKE_SUFFIX="_smoke"

# ---------------------------------------------------------------- run list
RUNS=()   # "paper domain run_id"
for p in $PAPERS; do
  for d in $DOMS; do RUNS+=("$p $d ${p}_${d}_s${SEED}${SUFFIX}${SMOKE_SUFFIX}"); done
done
if [ "$WITH_ALL" -eq 1 ]; then   # Paper A's five-domain run always uses scope all
  case " $PAPERS " in *" A "*) RUNS+=("A all A_all_s${SEED}${SMOKE_SUFFIX}") ;;
    *) echo "WARNING: --with-all only applies to Paper A; ignored" >&2 ;; esac
fi
N_RUNS=${#RUNS[@]}
NEED=0
for r in "${RUNS[@]}"; do set -- $r; m="$(min_of "$1" "$2")"
  [ -n "$PER_RUN" ] && { [ "$PER_RUN" -lt "$m" ] && { echo "ERROR: --per-run $PER_RUN < minimum $m for $3" >&2; exit 2; }; m=$PER_RUN; }
  NEED=$((NEED + m)); done

# ---------------------------------------------------------------- GPUs
if [ -z "$GPUS" ]; then
  if [ "$MOCK" -eq 1 ]; then
    GPUS="$(i=0; while [ $i -lt $NEED ]; do printf "%s," $i; i=$((i + 1)); done)"; GPUS="${GPUS%,}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -)"
  else
    echo "ERROR: nvidia-smi not found; pass --gpus (or --mock for a laptop test)" >&2; exit 1
  fi
fi
IFS=, read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -lt "$NEED" ]; then
  echo "ERROR: $N_RUNS runs need at least $NEED GPUs (--hw $HW), only ${#GPU_LIST[@]} given ($GPUS)." >&2
  echo "       Start fewer runs (--paper / --domains) or give more GPUs (--gpus)." >&2
  exit 1
fi
# GPUs per run: the minimum (or --per-run), then every remaining GPU one by one over the runs
COUNTS=()
for r in "${RUNS[@]}"; do set -- $r; if [ -n "$PER_RUN" ]; then COUNTS+=("$PER_RUN"); else COUNTS+=("$(min_of "$1" "$2")"); fi; done
if [ -z "$PER_RUN" ]; then
  EXTRA=$(( ${#GPU_LIST[@]} - NEED )); i=0
  while [ "$EXTRA" -gt 0 ]; do COUNTS[$i]=$(( COUNTS[$i] + 1 )); EXTRA=$((EXTRA - 1)); i=$(( (i + 1) % N_RUNS )); done
  NEED=${#GPU_LIST[@]}
fi

# ---------------------------------------------------------------- preflight
if [ -n "${RSI_PYTHON:-}" ]; then PY="$RSI_PYTHON"
elif [ "$MOCK" -eq 1 ]; then PY="$ROOT/eval/.venv/bin/python"
else PY="$HOME/venvs/rsi-gpu/bin/python"; fi

ds_of() {  # domain -> training set (data/draw/paper_*.yaml)
  case "$1" in math) echo dapo_math ;; code) echo taco_verified ;; knowledge) echo triviaqa_train ;;
    instruction_following) echo ifeval_train ;; tool_calling) echo sealtool_train ;; esac
}
FAILS=0
ok()  { echo "  OK    $*"; }
bad() { echo "  FAIL  $*"; FAILS=$((FAILS + 1)); }
if [ "$SKIP" -eq 0 ]; then
  echo "== preflight"
  if [ -x "$PY" ]; then ok "python $PY"; else bad "python not found at $PY (./setup_env.sh gpu, or set RSI_PYTHON)"; fi
  if [ "$MOCK" -eq 0 ] && [ -x "$PY" ]; then
    if "$PY" -I eval/hpc/check_gpu_env.py >/dev/null 2>&1; then ok "GPU stack (verl 0.9.1 / vLLM 0.24.0 / Transformers 5.9.0 / torch 2.11.0+cu130)"
    else bad "GPU stack check failed: run $PY -I eval/hpc/check_gpu_env.py"; fi
    MDIR="$MODEL"; [ -d "$MDIR" ] || MDIR="${MODELS_DIR:-$HOME/models}/$MODEL"
    if [ -f "$MDIR/config.json" ]; then ok "model $MDIR"; else bad "model not found at $MDIR (./download_models.sh $MODEL)"; fi
  fi
  CHECK_DOMS="$DOMS"; [ "$WITH_ALL" -eq 1 ] && CHECK_DOMS="$ALL_DOMAINS"
  for d in $CHECK_DOMS; do
    ds="$(ds_of "$d")"
    if [ -s "data/processed/$ds.jsonl" ] && [ -s "data/splits/$ds/pool.json" ]; then ok "data $ds"
    else bad "data $ds missing (./download_data.sh)"; fi
  done
  case " $CHECK_DOMS " in *" code "*)
    if [ "$MOCK" -eq 1 ]; then ok "code grading: mock (no Docker needed)"
    elif command -v docker >/dev/null 2>&1 && docker image inspect rsi-eval-sandbox:0.1 >/dev/null 2>&1; then
      ok "Docker sandbox image rsi-eval-sandbox:0.1 (code grading)"
    else bad "code domain needs Docker + image rsi-eval-sandbox:0.1 (./setup_env.sh sandbox)"; fi ;;
  esac
  if [ "$FAILS" -gt 0 ]; then
    echo "preflight: $FAILS problem(s). Fix them, or rerun with --skip-checks." >&2
    [ "$DRY" -eq 1 ] || exit 1
  fi
fi

# ---------------------------------------------------------------- launch
STAMP="$(date +%Y%m%d_%H%M%S)"
LDIR="experiment_records/_launch/$STAMP"
RUNS_DIR="${RSI_RUNS_DIR:-$ROOT/experiment_records}"
[ "$DRY" -eq 1 ] || mkdir -p "$LDIR"
echo "== $N_RUNS run(s), $GENS generation(s) each, model $MODEL, seed $SEED, hardware $HW, $NEED GPUs$([ "$MOCK" -eq 1 ] && echo ', MOCK')"
off=0; k=0
for r in "${RUNS[@]}"; do
  set -- $r; p=$1; d=$2; rid=$3
  n="${COUNTS[$k]}"; k=$((k + 1))
  g="$(IFS=,; echo "${GPU_LIST[*]:$off:$n}")"
  off=$((off + n))
  cmd=(./run_rsi.sh --paper "$p" --domain "$d" --gens "$GENS" --model "$MODEL" --seed "$SEED"
       --gpus "$g" --run-id "$rid" --runs-dir "$RUNS_DIR" --set "hardware=$HW" ${SETS[@]+"${SETS[@]}"})
  [ "$MOCK" -eq 1 ] && cmd+=(--mock)
  printf '  %-34s GPUs %-8s %s\n' "$rid" "$g" "${cmd[*]}"
  [ "$DRY" -eq 1 ] && continue
  old="$(ls -t experiment_records/_launch/*/"$rid".pid 2>/dev/null | head -1 || true)"
  if [ -n "$old" ] && kill -0 "$(cat "$old")" 2>/dev/null; then
    echo "    already running (pid $(cat "$old"), $old); not started again"; continue
  fi
  # CUDA_VISIBLE_DEVICES is left alone: run.py places its servers on --gpus itself.
  nohup "${cmd[@]}" > "$LDIR/$rid.log" 2>&1 &
  echo $! > "$LDIR/$rid.pid"
done
[ "$DRY" -eq 1 ] && { echo "(dry run: nothing started)"; exit 0; }

cat <<EOF

Started. Logs and pids: $LDIR/
  follow a run:   tail -f $LDIR/<run-id>.log
  progress:       cat $RUNS_DIR/<run-id>/state.json
  still running:  for f in $LDIR/*.pid; do kill -0 \$(cat \$f) 2>/dev/null && echo "running \$(basename \$f .pid)"; done
  stop one run:   kill \$(cat $LDIR/<run-id>.pid)     (rerun the same command later to resume)
EOF
