#!/usr/bin/env bash
# Formal launcher: start the RSI runs (Paper A self-quantization, Paper B self-written LoRA) for several
# domains at once, each run on its own GPUs in the background with its own log. Wraps ./run_rsi.sh.
#
# Usage:
#   ./launch.sh [--paper A|B|both] [--domains all|math,code,...] [--gens 5] [--hw b300|h20|h20_80]
#               [--gpus 0,1,2,...] [--per-run N] [--model qwen3.8-27b] [--seed 42]
#               [--config FILE] [--set KEY=VALUE ...] [--tag NAME] [--baseline-from DIR] [--dry-run] [--mock]
#               [--skip-checks]
#
#   --paper     A, B or both (default both)
#   --domains   all = math,code,knowledge,instruction_following,tool_calling (default all); one run per paper and domain
#   --gens      generations per run (default 5)
#   --hw        hardware profile (default b300): b300 (B300 planned at 220 GB per card), h20 (H20 planned at 110 GB per card),
#               h20_80 (80 GB H20). Qwen3.8-27B (one model per card): minimum GPUs per run
#                 Paper A 2 (editor | candidates); Paper B 2 (main | training)
#   --gpus      GPU ids to use, any order / gaps (default: every GPU nvidia-smi lists). ALL of them are used:
#               every run gets its minimum, the rest are handed out one by one over the runs, and each run
#               spreads its GPUs over its roles (more tensor parallelism / a split training model; see
#               gpu_layout in edit_code/common.py and lora_code/common.py). Runs get consecutive groups.
#   --per-run   fixed GPUs per run instead (the rest stay idle)
#   --set       passed to every run.py (repeatable), e.g. --set search.threads=4, --set method.mode=weights
#               (Paper B: scratch | weights | edit), --set merge.mode=adapter (Paper B: merge | adapter), or a manual
#               GPU split --set 'gpu_roles.train=[2,3]' (indices into the run's GPUs; overrides the automatic one)
#   What every run does (the only flow; 2026-11-22): each run uses ONLY its own domain's simple_data/<domain>/.
#               Every generation screens a fresh 500 items drawn from the domain's clean pool minus its eval and test
#               items (same items for Papers A and B; the rest of the pool is the train data), accepts on the 500 eval
#               items simple_<domain>_eval (the usual paired gate) and reports held-out on the 500 test items
#               simple_<domain>_test. Hard-wired, no knobs: candidates are never screened; each candidate gets ONE gate
#               evaluation during the search (paired vs the current model); the passing candidate with the largest
#               paired gain is accepted. There is no api.evaluate / evaluate budget, no top_full re-scoring, no noise
#               band and no KL term; --set / --config of those removed keys (search.screen_candidates, max_evaluate,
#               top_full, kl_weight, noise.*, ...) is refused unless it gives the hard-wired value.
#               Decoding per the paper config's decoding / heldout.decoding blocks.
#               simple_data/ (pool_ids.json, eval.jsonl, test.jsonl) comes from data/scripts/make_simple_data.py.
#               Merged configs: plan/configs/{A,B}_<domain>_simple.yaml. A changed config cannot resume an older
#               run (locked config): use --tag / --run-id.
#   --simple    accepted and ignored (the flow above is the only one; kept so older commands still work)
#   --config FILE  experiment config (2026-11-21; repeatable): YAML overrides merged after the hardware profile and
#               before --set. Hardware stays on --hw, so one file runs on B300 and H20.
#               Example: plan/configs/run/B_math_r30_nosig.yaml
#   --baseline-from DIR  generation 0 reads M_0's gate / held-out scores (and Paper A's KL gate reference) from an
#               earlier run instead of scoring M_0 again: a run directory, its baseline/ folder, or a run id in
#               experiment_records/ (= --set reuse.baseline_from=DIR; checked, a mismatch stops the run). Runs made
#               before 2026-11-24 also need --set reuse.trust_unrecorded_code=true
#   --tag NAME  append _NAME to every run id (letters, digits, - and _), e.g. --tag nosig ->
#               B_math_s42_single_simple_nosig: a separate run (own locked config) next to the untagged one,
#               for runs that change --set values of an existing run id.
#   --dry-run   print the preflight result and the commands, start nothing
#   --mock      laptop test: mock model server, no GPU (fake GPU ids), uses eval/.venv
#   --skip-checks  skip the preflight checks
#
# Run ids: <A|B>_<domain>_s<seed>_single_simple[_<tag>], in experiment_records/ (the suffix is kept from the time
# other objectives existed, so older runs resume with the same command).
# --seed is the one experiment seed: per-generation screen draws, vLLM servers, editor sampling, training answers,
# held-out sampling and bootstrap CIs all use it (the simple_data split keeps its fixed seed 20261118).
# Logs and pids: experiment_records/_launch/<timestamp>/. Rerunning the same command resumes every run
# (finished steps are skipped); a run whose process is still alive is not started twice.
#
# Examples:
#   ./launch.sh --paper both --gens 5                # formal: 10 runs (2 papers x 5 domains), 20 B300
#   ./launch.sh --hw h20 --paper both --gens 5       # on H20: 5 x 2 x 2 = 20 GPUs
#   ./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 0,1
#   ./launch.sh --paper A --domains math --gens 5 --gpus 0,1
#   ./launch.sh --paper B --domains code,math --gens 5 --gpus 4,5,6,7
#   ./launch.sh --hw h20 --paper both --domains math --gens 5 --gpus 0,1,2,3   # A on 0,1 and B on 2,3
#   ./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 4,5,6,7      # one run: editor 1 + 3 candidate servers
#   ./launch.sh --hw b300 --paper B --domains math --gens 5 --gpus 0,1,2,3,4,5,6,7 --set method.mode=weights --tag weights
#   ./launch.sh --hw b300 --paper A --domains math --gens 5 --gpus 0,1,2,3,4,5,6,7 \
#       --config plan/configs/run/A_math_r30_nosig.yaml --tag r30_nosig
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$ROOT/launch.sh"; }

PAPER=both; DOMAINS=all; GENS=5; GPUS=""; PER_RUN=""; MODEL=qwen3.8-27b; SEED=42; HW=b300
DRY=0; MOCK=0; SKIP=0; TAG=""; SETS=(); CONFIGS=()
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
    --tag) need "$@"; TAG="$2"; shift ;;
    --baseline-from) need "$@"; SETS+=(--set "reuse.baseline_from=$2"); shift ;;
    --config) need "$@"; [ -f "$2" ] || { echo "ERROR: --config $2: file not found" >&2; exit 2; }
              CONFIGS+=(--config "$2"); shift ;;
    --model) need "$@"; MODEL="$2"; shift ;;
    --seed) need "$@"; SEED="$2"; shift ;;
    --simple|--single-domain) ;;   # accepted no-ops: the single-domain simple flow is the only one
    --dry-run) DRY=1 ;;
    --mock) MOCK=1 ;;
    --skip-checks) SKIP=1 ;;
    *) echo "ERROR: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

ALL_DOMAINS="math code knowledge instruction_following tool_calling"
case "$HW" in b300|h20|h20_80) ;; *) echo "ERROR: --hw must be b300, h20 or h20_80" >&2; exit 2 ;; esac
MIN_GPUS=2   # minimum GPUs of one run, both papers and every hardware profile
case "$PAPER" in A|a) PAPERS="A" ;; B|b) PAPERS="B" ;; both) PAPERS="A B" ;;
  *) echo "ERROR: --paper must be A, B or both" >&2; exit 2 ;; esac
if [ "$DOMAINS" = all ]; then DOMS="$ALL_DOMAINS"; else DOMS="$(echo "$DOMAINS" | tr ',' ' ')"; fi
for d in $DOMS; do
  case " $ALL_DOMAINS " in *" $d "*) ;; *) echo "ERROR: unknown domain $d" >&2; exit 2 ;; esac
done
SUFFIX="_single_simple"   # default run-id suffix (same as rsicore.config.RUN_SUFFIX)
if [ -n "$TAG" ]; then
  case "$TAG" in *[!A-Za-z0-9_-]*) echo "ERROR: --tag may only contain letters, digits, - and _" >&2; exit 2 ;; esac
  SUFFIX="${SUFFIX}_${TAG}"
fi

# ---------------------------------------------------------------- run list
RUNS=()   # "paper domain run_id"
for p in $PAPERS; do
  for d in $DOMS; do RUNS+=("$p $d ${p}_${d}_s${SEED}${SUFFIX}"); done
done
N_RUNS=${#RUNS[@]}
NEED=0
for r in "${RUNS[@]}"; do set -- $r; m=$MIN_GPUS
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
for r in "${RUNS[@]}"; do set -- $r; if [ -n "$PER_RUN" ]; then COUNTS+=("$PER_RUN"); else COUNTS+=("$MIN_GPUS"); fi; done
if [ -z "$PER_RUN" ]; then
  EXTRA=$(( ${#GPU_LIST[@]} - NEED )); i=0
  while [ "$EXTRA" -gt 0 ]; do COUNTS[$i]=$(( COUNTS[$i] + 1 )); EXTRA=$((EXTRA - 1)); i=$(( (i + 1) % N_RUNS )); done
  NEED=${#GPU_LIST[@]}
fi

# ---------------------------------------------------------------- preflight
if [ -n "${RSI_PYTHON:-}" ]; then PY="$RSI_PYTHON"
elif [ "$MOCK" -eq 1 ]; then PY="$ROOT/eval/.venv/bin/python"
else PY="$HOME/venvs/rsi-gpu/bin/python"; fi

ds_of() {  # domain -> training set (the source of simple_<domain>_pool)
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
  for d in $DOMS; do
    if [ -s "simple_data/$d/pool_ids.json" ] && [ -s "simple_data/$d/eval.jsonl" ] && [ -s "simple_data/$d/test.jsonl" ]; then
      ok "simple_data/$d (pool ids / 500 gate / 500 test)"
    else bad "simple_data/$d incomplete: needs pool_ids.json, eval.jsonl, test.jsonl ($PY -I data/scripts/make_simple_data.py --only $d)"; fi
    ds="$(ds_of "$d")"   # simple_<d>_pool is built from the training set's processed records
    if [ -s "data/processed/$ds.jsonl" ]; then ok "data $ds (source of simple_${d}_pool)"
    else bad "data $ds missing (./download_data.sh; source of simple_${d}_pool)"; fi
  done
  case " $DOMS " in *" code "*)
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
       --gpus "$g" --run-id "$rid" --runs-dir "$RUNS_DIR" ${CONFIGS[@]+"${CONFIGS[@]}"} --set "hardware=$HW" ${SETS[@]+"${SETS[@]}"})
  [ "$MOCK" -eq 1 ] && cmd+=(--mock)
  printf '  %-34s GPUs %-8s %s\n' "$rid" "$g" "${cmd[*]}"
  [ "$DRY" -eq 1 ] && continue
  old="$(ls -t experiment_records/_launch/*/"$rid".pid 2>/dev/null | head -1 || true)"
  if [ -n "$old" ] && kill -0 "$(cat "$old")" 2>/dev/null && [[ "$(ps -o stat= -p "$(cat "$old")")" != *Z* ]]; then
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
