#!/usr/bin/env bash
# setup_env.sh -- one entry point for this project's environments. It wraps the existing scripts
# (eval/setup_env.sh, eval/hpc/setup_gpu_env.sh, eval/sandbox/Dockerfile) and does not repeat them.
#
# Usage:
#   ./setup_env.sh <mode> [options]
#   ./setup_env.sh --check [mode]          # report what is installed; installs nothing
#
# Modes:
#   cpu       CPU evaluation client env (default eval/.venv): uv + Python 3.11 + eval/requirements.lock,
#             pinned third_party harness repos, IFBench, NLTK data  (runs eval/setup_env.sh). macOS/Linux.
#             An existing venv is kept and only checked against the lock; it is never deleted or recreated.
#   gpu       GPU stack (verl 0.9.1, vLLM 0.24.0, Transformers 5.9.0, PyTorch 2.11.0+cu130, Python 3.11)
#             into --gpu-venv (default ~/venvs/rsi-gpu) via eval/hpc/setup_gpu_env.sh, which then
#             runs eval/hpc/check_gpu_env.py. Linux x86_64 only.
#   sandbox   Build the Docker code-execution image rsi-eval-sandbox:0.1 (eval/sandbox/Dockerfile),
#             only if the Docker daemon is already running. Never starts Docker itself.
#   all       cpu + sandbox on macOS; cpu + gpu + sandbox on Linux x86_64.
#
# Options:
#   --check            Report Python/package versions in each venv, third_party pins, Docker image.
#   --dry-run          Print the commands a mode would run, without running them.
#   --cpu-venv DIR     CPU venv path (default: eval/.venv; env RSI_CPU_VENV).
#   --gpu-venv DIR     GPU venv path (default: ~/venvs/rsi-gpu; env RSI_GPU_VENV).
#   --force-sync       Allow `uv pip install -r eval/requirements.lock` into an existing CPU venv that
#                      differs from the lock (default: refuse and show the differences).
#   -h, --help         Show this help.
#
# Examples:
#   ./setup_env.sh cpu
#   ./setup_env.sh gpu --gpu-venv /scratch/$USER/venvs/rsi-gpu
#   ./setup_env.sh all --dry-run
#   ./setup_env.sh --check
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="$ROOT/eval"
LOCK="$EVAL/requirements.lock"
IMAGE="rsi-eval-sandbox:0.1"
BUILD_CMD="docker build -t $IMAGE -f eval/sandbox/Dockerfile eval"
GPU_PINS="torch=2.11.0 vllm=0.24.0 transformers=5.9.0 verl=0.9.1"
CPU_KEYS="bfcl-eval math-verify ifbench httpx huggingface-hub pyarrow numpy torch antlr4-python3-runtime"
OS="$(uname -s)"
ARCH="$(uname -m)"

usage() { sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n== %s ==\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; "~"|"~/"*) printf '%s\n' "$HOME${1#\~}" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
run() {  # print and run a command (only print with --dry-run)
  printf '+ %s\n' "$*"
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  "$@"
}

# ---------------------------------------------------------------- arguments
MODE="" CHECK=0 DRY_RUN=0 FORCE_SYNC=0
CPU_VENV="${RSI_CPU_VENV:-$EVAL/.venv}"
GPU_VENV="${RSI_GPU_VENV:-$HOME/venvs/rsi-gpu}"
[ $# -gt 0 ] || { usage; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    cpu|gpu|sandbox|all)
      [ -z "$MODE" ] || die "only one mode allowed (got '$MODE' and '$1')"; MODE=$1; shift ;;
    --check) CHECK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force-sync) FORCE_SYNC=1; shift ;;
    --cpu-venv|--gpu-venv)
      [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a directory argument"
      if [ "$1" = --cpu-venv ]; then CPU_VENV=$2; else GPU_VENV=$2; fi; shift 2 ;;
    --cpu-venv=*) CPU_VENV=${1#*=}; shift ;;
    --gpu-venv=*) GPU_VENV=${1#*=}; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument '$1' (see ./setup_env.sh --help)" ;;
  esac
done
CPU_VENV="$(abspath "$CPU_VENV")"; CPU_VENV="${CPU_VENV%/}"
GPU_VENV="$(abspath "$GPU_VENV")"; GPU_VENV="${GPU_VENV%/}"
[ "$CHECK" = 1 ] || [ -n "$MODE" ] || die "no mode given: use cpu, gpu, sandbox, all, or --check"

is_linux_x86() { [ "$OS" = Linux ] && [ "$ARCH" = x86_64 ]; }

# ---------------------------------------------------------------- shared helpers
need_uv() {
  command -v uv >/dev/null 2>&1 && return 0
  cat >&2 <<'EOF'
ERROR: uv is not installed (https://docs.astral.sh/uv/). Install it yourself, then rerun, e.g.:
    pip install --user uv                    # any Python >= 3.8
    brew install uv                          # macOS with Homebrew
    # or the official installer: read https://docs.astral.sh/uv/getting-started/installation/
EOF
  exit 1
}

need_tools() {  # command names
  local t missing=""
  for t in "$@"; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
  [ -z "$missing" ] || die "missing required command(s):$missing"
}

venv_pyver() {  # venv dir -> "3.11.10" or "" if no usable python
  [ -x "$1/bin/python" ] || return 0
  "$1/bin/python" -I -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || true
}

pkg_versions() {  # venv dir, package names... -> "name version" lines (MISSING if absent)
  local venv=$1; shift
  "$venv/bin/python" -I - "$@" <<'EOF'
import importlib.metadata as md, sys
for name in sys.argv[1:]:
    try:
        v = md.version(name)
    except md.PackageNotFoundError:
        v = "MISSING"
    print("%-24s %s" % (name, v))
EOF
}

ver_ge() {  # ver_ge A B: true if dotted version A >= B (numeric fields)
  local a=$1 b=$2 x y
  while [ -n "$a" ] || [ -n "$b" ]; do
    x=${a%%.*}; y=${b%%.*}; x=${x:-0}; y=${y:-0}
    if [ "$x" -gt "$y" ] 2>/dev/null; then return 0; fi
    if [ "$x" -lt "$y" ] 2>/dev/null; then return 1; fi
    case "$a" in *.*) a=${a#*.} ;; *) a="" ;; esac
    case "$b" in *.*) b=${b#*.} ;; *) b="" ;; esac
  done
  return 0
}

nvidia_report() {  # prints driver / CUDA info; warns if CUDA 13.0 cannot run. Never fails.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found: cannot check the NVIDIA driver (fine on a login node; compute nodes need a driver supporting CUDA 13.0, i.e. >= 580)"
    return 0
  fi
  local drv cuda
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1 || true)"
  say "nvidia-smi: driver ${drv:-unknown}, max CUDA ${cuda:-unknown}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /' || true
  if [ -n "$cuda" ] && ! ver_ge "$cuda" 13.0; then
    warn "driver reports CUDA $cuda < 13.0: the cu130 PyTorch/vLLM build will not run on this driver (needs driver >= 580)"
  elif [ -z "$cuda" ] && [ -n "$drv" ] && ! ver_ge "$drv" 580; then
    warn "driver $drv < 580: cannot run CUDA 13.0 builds"
  elif [ -z "$drv" ] && [ -z "$cuda" ]; then
    warn "nvidia-smi gave no driver/CUDA info (no GPU visible or driver not loaded)"
  fi
}

docker_state() {  # echo: nocli | nodaemon | running
  if ! command -v docker >/dev/null 2>&1; then echo nocli
  elif docker info >/dev/null 2>&1; then echo running
  else echo nodaemon; fi
}

docker_hint() {
  say "Start Docker yourself, then rerun './setup_env.sh sandbox' (or run the build command directly):"
  say "    cd $ROOT && $BUILD_CMD"
  if [ "$OS" = Darwin ]; then
    say "  macOS: open Docker Desktop (e.g. 'open -a Docker'), wait until 'docker info' succeeds."
  else
    say "  Linux: 'sudo systemctl start docker' (or ask your admin); on HPC without Docker see eval/hpc/README.md (Apptainer)."
  fi
}

# ---------------------------------------------------------------- cpu
do_cpu() {
  hdr "cpu: evaluation client env -> $CPU_VENV"
  case "$OS" in Darwin|Linux) ;; *) die "cpu mode supports macOS and Linux only (this is $OS)" ;; esac
  need_uv
  need_tools git curl unzip
  [ -f "$LOCK" ] || die "missing $LOCK"
  local ver args
  if [ -e "$CPU_VENV" ]; then
    ver="$(venv_pyver "$CPU_VENV")"
    [ -n "$ver" ] || die "$CPU_VENV exists but has no working bin/python. Not touching it; remove it yourself or choose --cpu-venv DIR."
    case "$ver" in 3.11.*) ;; *) die "$CPU_VENV has Python $ver, need 3.11. Not touching it; remove it yourself or choose --cpu-venv DIR." ;; esac
    say "existing venv kept (Python $ver); checking it against eval/requirements.lock ..."
    local out
    out="$(uv pip install --dry-run --python "$CPU_VENV/bin/python" -r "$LOCK" 2>&1)" || { say "$out" >&2; die "uv could not resolve eval/requirements.lock against $CPU_VENV"; }
    if printf '%s\n' "$out" | grep -q 'Would make no changes'; then
      say "venv already matches the lock (no package changes)."
    elif [ "$FORCE_SYNC" = 1 ]; then
      say "venv differs from the lock; --force-sync given, it will be synced to the pinned versions:"
      printf '%s\n' "$out" | grep -E '^ [-+~] ' || true
    else
      printf '%s\n' "$out" | grep -E '^ [-+~] ' >&2 || printf '%s\n' "$out" >&2
      die "$CPU_VENV differs from eval/requirements.lock (changes above). Not modifying it. Rerun with --force-sync to install the pinned versions, or use --cpu-venv DIR for a fresh venv."
    fi
    args="--reuse-venv"
  else
    say "creating a new venv"
    args=""
  fi
  # shellcheck disable=SC2086
  run bash "$EVAL/setup_env.sh" --venv "$CPU_VENV" $args
  if [ "$DRY_RUN" = 0 ]; then
    say "installed:"; say "  python $(venv_pyver "$CPU_VENV")"
    # shellcheck disable=SC2086
    pkg_versions "$CPU_VENV" $CPU_KEYS | sed 's/^/  /'
  fi
  say "cpu: OK"
}

# ---------------------------------------------------------------- gpu
do_gpu() {
  hdr "gpu: verl/vLLM/Transformers/PyTorch cu130 stack -> $GPU_VENV"
  is_linux_x86 || die "gpu mode needs Linux x86_64 (this machine is $OS $ARCH). The GPU lock (eval/hpc/requirements-gpu.lock) is resolved for x86_64-manylinux_2_28 + CUDA 13.0; run it on the HPC/Linux GPU host."
  need_uv
  [ -f "$EVAL/hpc/requirements-gpu.lock" ] || die "missing eval/hpc/requirements-gpu.lock"
  nvidia_report
  local ver args=""
  if [ -e "$GPU_VENV" ]; then
    ver="$(venv_pyver "$GPU_VENV")"
    [ -n "$ver" ] || die "$GPU_VENV exists but has no working bin/python. Remove it yourself or choose --gpu-venv DIR."
    case "$ver" in 3.11.*) ;; *) die "$GPU_VENV has Python $ver, need 3.11. Remove it yourself or choose --gpu-venv DIR." ;; esac
    say "existing venv kept (Python $ver); syncing it to eval/hpc/requirements-gpu.lock"
    args="--reuse-venv"
  fi
  # setup_gpu_env.sh installs the lock and then runs eval/hpc/check_gpu_env.py (exit 1 on mismatch).
  # shellcheck disable=SC2086
  run bash "$EVAL/hpc/setup_gpu_env.sh" $args "$GPU_VENV"
  say "gpu: OK (check a model with: $GPU_VENV/bin/python -I eval/hpc/check_gpu_env.py --model /path/to/model)"
}

# ---------------------------------------------------------------- sandbox
SANDBOX_SKIPPED=0
do_sandbox() {
  hdr "sandbox: Docker image $IMAGE"
  if [ ! -d "$EVAL/third_party/LiveCodeBench/lcb_runner" ]; then
    if [ "$DRY_RUN" = 1 ] && [ "$MODE" = all ]; then
      say "(eval/third_party/LiveCodeBench would be cloned by the cpu step first)"
    else
      die "eval/third_party/LiveCodeBench/lcb_runner is missing (the image copies it). Run './setup_env.sh cpu' first."
    fi
  fi
  case "$(docker_state)" in
    nocli)
      say "docker CLI not found. Install Docker (Desktop on macOS, Engine on Linux), start it, then run:"
      say "    cd $ROOT && $BUILD_CMD"
      SANDBOX_SKIPPED=1; return 0 ;;
    nodaemon)
      say "The Docker daemon is not running; not building (this script never starts Docker)."
      docker_hint
      SANDBOX_SKIPPED=1; return 0 ;;
  esac
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    say "image $IMAGE already present; rebuilding with the layer cache (no-op if nothing changed)"
  fi
  (cd "$ROOT" && run docker build -t "$IMAGE" -f eval/sandbox/Dockerfile eval)
  say "sandbox: OK"
}

# ---------------------------------------------------------------- check
pin_status() {  # third_party pins from eval/third_party/pins.yaml vs checked-out commits
  local pins="$EVAL/third_party/pins.yaml" name commit head
  [ -f "$pins" ] || { say "  (no eval/third_party/pins.yaml)"; return 0; }
  awk '/^[A-Za-z0-9_.-]+:[[:space:]]*$/ {n=$1; sub(":","",n)} /^[[:space:]]+commit:/ {print n, $2}' "$pins" |
  while read -r name commit; do
    if [ "$name" = nltk_data ]; then
      head="$(cat "$EVAL/.nltk_data/NLTK_DATA_COMMIT" 2>/dev/null || echo MISSING)"
      if [ "$head" = "$commit" ]; then say "  $(printf '%-24s' ".nltk_data") ${commit:0:12} OK"
      else say "  $(printf '%-24s' ".nltk_data") ${head:0:12} (want ${commit:0:12})"; fi
    elif [ -d "$EVAL/third_party/$name/.git" ]; then
      head="$(git -C "$EVAL/third_party/$name" rev-parse HEAD 2>/dev/null || echo '?')"
      if [ "$head" = "$commit" ]; then say "  $(printf '%-24s' "$name") ${commit:0:12} OK"
      else say "  $(printf '%-24s' "$name") ${head:0:12} (want ${commit:0:12})"; fi
    else
      say "  $(printf '%-24s' "$name") MISSING (want ${commit:0:12})"
    fi
  done
}

check_cpu() {
  hdr "cpu venv: $CPU_VENV"
  local ver out
  ver="$(venv_pyver "$CPU_VENV")"
  if [ -z "$ver" ]; then say "  not installed (run ./setup_env.sh cpu)"; else
    say "  python $ver"
    # shellcheck disable=SC2086
    pkg_versions "$CPU_VENV" $CPU_KEYS | sed 's/^/  /'
    if command -v uv >/dev/null 2>&1; then
      out="$(uv pip install --dry-run --offline --python "$CPU_VENV/bin/python" -r "$LOCK" 2>&1 || true)"
      if printf '%s\n' "$out" | grep -q 'Would make no changes'; then say "  lock: matches eval/requirements.lock"
      else say "  lock: DIFFERS from eval/requirements.lock (or could not be checked offline):"
           printf '%s\n' "$out" | grep -E '^ [-+~] |error' | head -n 20 | sed 's/^/    /'; fi
    fi
  fi
  say "  pinned repos (eval/third_party/pins.yaml):"; pin_status
}

check_gpu() {
  hdr "gpu venv: $GPU_VENV"
  is_linux_x86 || say "  (gpu mode is Linux x86_64 only; this machine is $OS $ARCH)"
  local ver p
  ver="$(venv_pyver "$GPU_VENV")"
  if [ -z "$ver" ]; then say "  not installed"; else
    say "  python $ver"
    for p in $GPU_PINS; do
      pkg_versions "$GPU_VENV" "${p%%=*}" | awk -v want="${p#*=}" '{v=$2; sub(/\+.*/,"",v); printf "  %-24s %-18s %s\n", $1, $2, (v==want ? "OK" : "want " want)}'
    done
    say "  full check: $GPU_VENV/bin/python -I eval/hpc/check_gpu_env.py [--model PATH]"
  fi
  if is_linux_x86; then nvidia_report 2>&1 | sed 's/^/  /'; fi
}

check_sandbox() {
  hdr "sandbox: Docker image $IMAGE"
  case "$(docker_state)" in
    nocli) say "  docker CLI: not installed" ;;
    nodaemon) say "  docker CLI: $(docker --version 2>/dev/null)"; say "  daemon: not running (image status unknown)" ;;
    running)
      say "  docker CLI: $(docker --version 2>/dev/null)"; say "  daemon: running"
      if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        say "  image: present ($(docker image inspect -f '{{.Id}} created {{.Created}}' "$IMAGE" | cut -c1-80))"
      else say "  image: not built (run ./setup_env.sh sandbox)"; fi ;;
  esac
}

do_check() {
  hdr "system"
  say "  $OS $ARCH, bash $BASH_VERSION"
  if command -v uv >/dev/null 2>&1; then say "  uv: $(uv --version)"; else say "  uv: NOT INSTALLED (see https://docs.astral.sh/uv/)"; fi
  local t; for t in git curl unzip; do
    command -v "$t" >/dev/null 2>&1 && say "  $t: $(command -v "$t")" || say "  $t: NOT FOUND"
  done
  case "${MODE:-all}" in
    cpu) check_cpu ;; gpu) check_gpu ;; sandbox) check_sandbox ;;
    all) check_cpu; check_gpu; check_sandbox ;;
  esac
}

# ---------------------------------------------------------------- main
if [ "$CHECK" = 1 ]; then do_check; exit 0; fi
[ "$DRY_RUN" = 1 ] && say "(dry run: commands are printed, not executed; read-only checks still run)"
case "$MODE" in
  cpu) do_cpu ;;
  gpu) do_gpu ;;
  sandbox) do_sandbox; [ "$SANDBOX_SKIPPED" = 0 ] || exit 1 ;;
  all)
    do_cpu
    if is_linux_x86; then do_gpu; else say ""; say "(gpu skipped: $OS $ARCH; the GPU stack is Linux x86_64 only)"; fi
    do_sandbox
    DONE=done; [ "$DRY_RUN" = 0 ] || DONE="dry run only"
    hdr "summary"
    say "cpu: $DONE"
    if is_linux_x86; then say "gpu: $DONE"; else say "gpu: skipped (not Linux x86_64)"; fi
    if [ "$SANDBOX_SKIPPED" = 1 ]; then say "sandbox: SKIPPED (Docker not available; see the command above)"; else say "sandbox: $DONE"; fi ;;
esac
