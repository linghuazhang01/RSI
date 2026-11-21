#!/usr/bin/env bash
# Formal launcher: start the RSI runs (Paper A self-quantization, Paper B self-written LoRA) for several
# domains at once, each run on its own GPUs in the background with its own log. Wraps ./run_rsi.sh.
#
# Usage:
#   ./launch.sh [--paper A|B|both] [--domains all|math,code,...] [--gens 5] [--hw b300|h20]
#               [--gpus 0,1,2,...] [--per-run N] [--model qwen3.8-27b] [--seed 42] [--with-all] [--eval-overfit]
#               [--eval-set humanevalplus,ifeval,...] [--simple] [--train-set X] [--gate-set X] [--eval-on X,Y]
#               [--all-domains] [--set KEY=VALUE ...] [--tag NAME] [--dry-run] [--mock] [--skip-checks]
#
#   --paper     A, B or both (default both)
#   --domains   all = math,code,knowledge,instruction_following,tool_calling (default all)
#   --gens      generations per run (default 5)
#   --hw        hardware profile (default b300): b300 (B300 planned at 220 GB per card), h20 (H20 planned at 110 GB per card),
#               h20_80 (80 GB H20). Qwen3.8-27B (one model per card): minimum GPUs per run
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
#   (default)   each run uses ONLY its own domain (scope single): training, gate, feedback, KL and held-out
#               on that domain; Paper B has no retention probes and no E_t. Run ids end in _single.
#   --eval-overfit optimise directly on ONE eval set per run (objective eval_overfit): every stage and the held-out
#               evaluation are the whole set (greedy), accepted when the score is higher. With --domains, each
#               domain's default set: math -> math (MATH-500 + GSM8K test, 1819), code -> humanevalplus,
#               knowledge -> gpqa_diamond, instruction_following -> ifeval, tool_calling -> sealtool.
#   --eval-set  the eval set(s) to overfit on, comma-separated, one run per set and paper (implies --eval-overfit;
#               the run's domain follows from the set, --domains is ignored). Any of: math math500 gsm8k aime24
#               aime25 amc23 | humanevalplus mbppplus livecodebench | gpqa_diamond mmlu_pro triviaqa |
#               ifeval ifbench | sealtool apibank bfcl  (eval/configs/tasks.yaml eval_targets).
#               Run ids end in _evalfit (math) or _evalfit_<set>, e.g. A_code_s42_single_evalfit_humanevalplus.
#   --simple    objective simple (2026-11-21): train on simple_data/<domain>'s 1,500 items (all of them every
#               generation), accept on its 500 (the usual paired gate), held-out report on the same 500 (greedy);
#               editor at most 20 tool rounds per candidate. simple_data/ comes from data/scripts/make_simple_data.py.
#               Run ids end in _simple, e.g. B_code_s42_single_simple. Merged configs: plan/configs/{A,B}_<domain>_simple.yaml.
#   Data roles (2026-11-21, paper config data_roles; each run's train / gate must belong to its domain):
#   --train-set what the editor works from (signal / screen items, Paper B's train_example / acts / train_loss,
#               Paper A's KL / calibration prompts): the domain's training set (dapo_math, taco_verified,
#               triviaqa_train, ifeval_train, sealtool_train: per-generation draws) or an eval set (all items).
#   --gate-set  what accepts a candidate: a training set (its fixed gate split) or an eval set (all items).
#   --eval-on   the held-out report, comma-separated: domains, held-out tasks (math500, humanevalplus, ...),
#               eval sets (math_eval = MATH-500 + GSM8K merged), or train / gate (when that role is an eval set).
#               Default: --eval-overfit -> the eval set; otherwise the run's domain (sampled avg@n, heldout.decoding).
#               Defaults of train / gate: the domain's training set; with --eval-overfit / --eval-set the eval set;
#               with --simple simple_<domain>_train / simple_<domain>_eval (also accepted: simple_train / simple_eval).
#               A train / gate other than the default adds _train-<x> / _gate-<x> to the run id.
#   --all-domains  the original design (scope all): the other four domains are retention probes / KL texts,
#               held-out on all five domains, Paper B E_t on. Run ids without _single.
#   --tag NAME  append _NAME to every run id (letters, digits, - and _), e.g. --tag nosig ->
#               B_math_s42_single_simple_nosig: a separate run (own locked config) next to the untagged one,
#               for runs that change --set values of an existing run id. Not applied to A_all (--with-all).
#   --dry-run   print the preflight result and the commands, start nothing
#   --mock      laptop test: mock model server, no GPU (fake GPU ids), uses eval/.venv
#   --skip-checks  skip the preflight checks
#
# Run ids: <A|B>_<domain>_s<seed>[_single][_simple][_<tag>][_evalfit[_<set>]] (A_all_s<seed> for --with-all), in experiment_records/.
# --seed is the one experiment seed: per-generation draws, vLLM servers, editor sampling, training answers,
# held-out sampling and bootstrap CIs all use it (the data / gate split keeps its fixed seed 20261118).
# Logs and pids: experiment_records/_launch/<timestamp>/. Rerunning the same command resumes every run
# (finished steps are skipped); a run whose process is still alive is not started twice.
#
# Examples:
#   ./launch.sh --paper both --gens 5 --with-all     # formal: 10 single-domain runs + Paper A all (22 B300)
#   ./launch.sh --hw h20 --paper both --gens 5       # on H20: 5 x 2 + 5 x 3 = 25 GPUs
#   ./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 0,1
#   ./launch.sh --paper A --domains math --gens 5 --gpus 0,1
#   ./launch.sh --paper B --domains code,math --gens 5 --gpus 4,5,6,7
#   ./launch.sh --hw h20 --paper both --domains math --gens 5 --gpus 0,1,2,3   # A on 0,1 and B on 2,3
#   ./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 4,5,6,7      # one run: editor 1 + 3 candidate servers
#   ./launch.sh --hw h20 --paper both --eval-set humanevalplus --gens 5 --gpus 0,1,2,3   # overfit HumanEval+ (A 0,1 / B 2,3)
#   ./launch.sh --hw h20 --paper A --eval-set ifeval,gpqa_diamond --gens 5 --gpus 0,1,2,3   # two runs, one set each
#   ./launch.sh --hw h20 --paper B --domains math --gate-set math500 --eval-on math500,gsm8k --gens 5 --gpus 0,1,2,3
#       # train on DAPO-Math draws, accept on all of MATH-500, report MATH-500 + GSM8K: B_math_s42_single_gate-math500
#   ./launch.sh --hw h20 --paper A --domains code --train-set humanevalplus --gate-set mbppplus --eval-on livecodebench
#   ./launch.sh --hw h20 --paper both --simple --gens 5          # 1,500 / 500 per domain, five domains, both papers
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$ROOT/launch.sh"; }

PAPER=both; DOMAINS=all; GENS=5; GPUS=""; PER_RUN=""; MODEL=qwen3.8-27b; SEED=42; HW=b300
WITH_ALL=0; DRY=0; MOCK=0; SKIP=0; SINGLE=1; EVALFIT=0; EVAL_SETS=""; SIMPLE=0; TRAIN_SET=""; GATE_SET=""; EVAL_ON=""; TAG=""; SETS=()
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
    --model) need "$@"; MODEL="$2"; shift ;;
    --seed) need "$@"; SEED="$2"; shift ;;
    --with-all) WITH_ALL=1 ;;
    --single-domain) SINGLE=1 ;;
    --all-domains) SINGLE=0 ;;
    --eval-overfit) EVALFIT=1 ;;
    --eval-set|--eval-sets) need "$@"; EVAL_SETS="$2"; EVALFIT=1; shift ;;
    --simple) SIMPLE=1 ;;
    --train-set) need "$@"; TRAIN_SET="$2"; shift ;;
    --gate-set) need "$@"; GATE_SET="$2"; shift ;;
    --eval-on) need "$@"; EVAL_ON="$2"; shift ;;
    --dry-run) DRY=1 ;;
    --mock) MOCK=1 ;;
    --skip-checks) SKIP=1 ;;
    *) echo "ERROR: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

ALL_DOMAINS="math code knowledge instruction_following tool_calling"
case "$HW" in b300|h20|h20_80) ;; *) echo "ERROR: --hw must be b300, h20 or h20_80" >&2; exit 2 ;; esac
min_of() {  # paper domain -> minimum GPUs of one run
  if [ "$HW" != b300 ] && [ "$1" = B ] && { [ "$SINGLE" -eq 0 ] || [ "$2" = all ]; }; then echo 3
  else echo 2; fi
}
case "$PAPER" in A|a) PAPERS="A" ;; B|b) PAPERS="B" ;; both) PAPERS="A B" ;;
  *) echo "ERROR: --paper must be A, B or both" >&2; exit 2 ;; esac
if [ "$DOMAINS" = all ]; then DOMS="$ALL_DOMAINS"; else DOMS="$(echo "$DOMAINS" | tr ',' ' ')"; fi
for d in $DOMS; do
  case " $ALL_DOMAINS " in *" $d "*) ;; *) echo "ERROR: unknown domain $d" >&2; exit 2 ;; esac
done
SUFFIX=""; [ "$SINGLE" -eq 1 ] && SUFFIX="_single"
[ "$SINGLE" -eq 0 ] && SETS=(--set scope=all ${SETS[@]+"${SETS[@]}"})   # explicit --set values come later and win
[ "$EVALFIT" -eq 1 ] && SETS=(--set objective=eval_overfit ${SETS[@]+"${SETS[@]}"})
[ "$EVALFIT" -eq 1 ] && [ "$WITH_ALL" -eq 1 ] && { echo "ERROR: --with-all (domain all) cannot overfit an eval set" >&2; exit 2; }
if [ "$SIMPLE" -eq 1 ]; then
  [ "$EVALFIT" -eq 1 ] && { echo "ERROR: --simple and --eval-overfit / --eval-set are different objectives" >&2; exit 2; }
  [ "$WITH_ALL" -eq 1 ] && { echo "ERROR: --with-all (domain all) cannot use --simple" >&2; exit 2; }
  [ "$SINGLE" -eq 0 ] && { echo "ERROR: --simple needs scope single (drop --all-domains)" >&2; exit 2; }
  SETS=(--set objective=simple ${SETS[@]+"${SETS[@]}"}); SUFFIX="${SUFFIX}_simple"
fi
if [ -n "$TAG" ]; then
  case "$TAG" in *[!A-Za-z0-9_-]*) echo "ERROR: --tag may only contain letters, digits, - and _" >&2; exit 2 ;; esac
  SUFFIX="${SUFFIX}_${TAG}"
fi

# eval sets (objective eval_overfit; mirrors eval/configs/tasks.yaml eval_targets / eval_target_defaults)
set_domain() {
  case "$1" in math|math500|gsm8k|aime24|aime25|amc23) echo math ;; humanevalplus|mbppplus|livecodebench) echo code ;;
    gpqa_diamond|mmlu_pro|triviaqa) echo knowledge ;; ifeval|ifbench) echo instruction_following ;;
    sealtool|apibank|bfcl) echo tool_calling ;; *) echo "" ;; esac
}
default_set() {
  case "$1" in math) echo math ;; code) echo humanevalplus ;; knowledge) echo gpqa_diamond ;;
    instruction_following) echo ifeval ;; tool_calling) echo sealtool ;; esac
}
set_sources() {  # eval set -> its source datasets (data/splits/<ds>/test.json)
  case "$1" in math) echo "math gsm8k" ;; math500) echo math ;; *) echo "$1" ;; esac
}
evalfit_suffix() { if [ "$1" = math ]; then echo _evalfit; else echo "_evalfit_$1"; fi; }
norm_set() {  # math_eval -> math, humanevalplus_eval -> humanevalplus, training / simple sets unchanged
  case "$1" in simple_*) echo "$1" ;; *_eval) echo "${1%_eval}" ;; *) echo "$1" ;; esac
}
train_of() {  # domain -> training set (same table as ds_of below)
  case "$1" in math) echo dapo_math ;; code) echo taco_verified ;; knowledge) echo triviaqa_train ;;
    instruction_following) echo ifeval_train ;; tool_calling) echo sealtool_train ;; esac
}
any_domain() {  # training set, simple set or eval set -> its domain
  case "$1" in simple_*_train|simple_*_eval) s="${1#simple_}"; s="${s%_train}"; echo "${s%_eval}" ;; dapo_math) echo math ;; taco_verified) echo code ;; triviaqa_train) echo knowledge ;;
    ifeval_train) echo instruction_following ;; sealtool_train) echo tool_calling ;; *) set_domain "$1" ;; esac
}

# ---------------------------------------------------------------- run list
RUNS=()   # "paper domain run_id eval_set" (eval_set "-" outside eval_overfit)
if [ -n "$EVAL_SETS" ]; then
  for e in $(echo "$EVAL_SETS" | tr ',' ' '); do
    d="$(set_domain "$e")"; [ -n "$d" ] || { echo "ERROR: unknown eval set $e (see --help)" >&2; exit 2; }
    for p in $PAPERS; do RUNS+=("$p $d ${p}_${d}_s${SEED}${SUFFIX}$(evalfit_suffix "$e") $e"); done
  done
  DOMS="$(for r in "${RUNS[@]}"; do set -- $r; echo "$2"; done | sort -u | tr '\n' ' ')"
else
  for p in $PAPERS; do
    for d in $DOMS; do
      if [ "$EVALFIT" -eq 1 ]; then e="$(default_set "$d")"; RUNS+=("$p $d ${p}_${d}_s${SEED}${SUFFIX}$(evalfit_suffix "$e") $e")
      else RUNS+=("$p $d ${p}_${d}_s${SEED}${SUFFIX} -"); fi
    done
  done
fi
# data roles: validate against each run's domain, add _train-<x> / _gate-<x> when not the default
if [ -n "$TRAIN_SET$GATE_SET$EVAL_ON" ]; then
  [ "$WITH_ALL" -eq 1 ] && { echo "ERROR: --train-set / --gate-set / --eval-on need single-domain runs (no --with-all)" >&2; exit 2; }
  [ "$SINGLE" -eq 0 ] && [ -n "$TRAIN_SET$GATE_SET" ] && { echo "ERROR: --train-set / --gate-set need scope single (drop --all-domains)" >&2; exit 2; }
  NEW=()
  for r in "${RUNS[@]}"; do
    set -- $r; p=$1; d=$2; rid=$3; e=$4
    for pair in "train:$TRAIN_SET" "gate:$GATE_SET"; do
      role="${pair%%:*}"; v="$(norm_set "${pair#*:}")"; [ -n "$v" ] || continue
      case "$v" in simple_train|simple_eval) v="simple_${d}_${v#simple_}" ;; esac
      dflt="$e"; [ "$e" = - ] && dflt="$(train_of "$d")"
      [ "$SIMPLE" -eq 1 ] && { if [ "$role" = train ]; then dflt="simple_${d}_train"; else dflt="simple_${d}_eval"; fi; }
      vd="$(any_domain "$v")"
      [ -n "$vd" ] || { echo "ERROR: --$role-set: unknown dataset $v (see --help)" >&2; exit 2; }
      [ "$vd" = "$d" ] || { echo "ERROR: --$role-set $v is a $vd dataset but run $rid is $d" >&2; exit 2; }
      [ "$v" != "$dflt" ] && rid="${rid}_${role}-$v"
    done
    NEW+=("$p $d $rid $e")
  done
  RUNS=("${NEW[@]}")
fi
if [ "$WITH_ALL" -eq 1 ]; then   # Paper A's five-domain run always uses scope all
  case " $PAPERS " in *" A "*) RUNS+=("A all A_all_s${SEED} -") ;;
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
  ESETS="$(for r in "${RUNS[@]}"; do set -- $r; [ "$4" != - ] && echo "$4"; done
            for v in "$TRAIN_SET" "$GATE_SET"; do v="$(norm_set "$v")"; [ -n "$v" ] && [ -n "$(set_domain "$v")" ] && echo "$v"; done
            true)"
  if [ -n "$ESETS" ]; then   # the eval sets' source data (the K_eval copy is built on demand)
    for e in $(echo "$ESETS" | sort -u); do
      for ds in $(set_sources "$e"); do
        if [ -s "data/processed/$ds.jsonl" ] && [ -s "data/splits/$ds/test.json" ]; then ok "eval set $e: data $ds"
        else bad "eval set $e: data $ds missing (./download_data.sh)"; fi
      done
    done
  fi
  if [ "$SIMPLE" -eq 1 ]; then
    for d in $CHECK_DOMS; do
      if [ -s "simple_data/$d/train.jsonl" ] && [ -s "simple_data/$d/eval.jsonl" ]; then ok "simple_data/$d (1,500 train / 500 gate)"
      else bad "simple_data/$d missing ($PY -I data/scripts/make_simple_data.py --only $d)"; fi
    done
  elif [ "$EVALFIT" -eq 0 ]; then
    for d in $CHECK_DOMS; do
      ds="$(ds_of "$d")"
      if [ -s "data/processed/$ds.jsonl" ] && [ -s "data/splits/$ds/pool.json" ]; then ok "data $ds"
      else bad "data $ds missing (./download_data.sh)"; fi
    done
  fi
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
  set -- $r; p=$1; d=$2; rid=$3; e=$4
  n="${COUNTS[$k]}"; k=$((k + 1))
  g="$(IFS=,; echo "${GPU_LIST[*]:$off:$n}")"
  off=$((off + n))
  cmd=(./run_rsi.sh --paper "$p" --domain "$d" --gens "$GENS" --model "$MODEL" --seed "$SEED"
       --gpus "$g" --run-id "$rid" --runs-dir "$RUNS_DIR" --set "hardware=$HW" ${SETS[@]+"${SETS[@]}"})
  [ "$e" != - ] && cmd+=(--set "eval_target.dataset=$e")
  [ -n "$TRAIN_SET" ] && cmd+=(--set "data_roles.train=$TRAIN_SET")
  [ -n "$GATE_SET" ] && cmd+=(--set "data_roles.gate=$GATE_SET")
  [ -n "$EVAL_ON" ] && cmd+=(--set "data_roles.eval=$EVAL_ON")
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
