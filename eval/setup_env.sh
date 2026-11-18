#!/usr/bin/env bash
# Recreate the client/grader environment (Linux or macOS; no GPU needed).
#   bash eval/setup_env.sh [--venv DIR] [--reuse-venv]
# Creates eval/.venv (Python 3.11 via uv), installs eval/requirements.lock, clones pinned
# third_party harness repos, installs IFBench (--no-deps), fetches pickle-free NLTK data.
# Optional (defaults unchanged): --venv DIR puts the venv at DIR instead of eval/.venv;
# --reuse-venv keeps an existing venv at that path instead of re-running `uv venv` on it.
set -euo pipefail
EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=.venv REUSE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --venv) [ $# -ge 2 ] || { echo "--venv needs a path" >&2; exit 2; }
            case "$2" in /*) VENV=$2 ;; *) VENV="$PWD/$2" ;; esac; shift 2 ;;
    --reuse-venv) REUSE=1; shift ;;
    *) echo "unknown argument: $1 (usage: bash eval/setup_env.sh [--venv DIR] [--reuse-venv])" >&2; exit 2 ;;
  esac
done
cd "$EVAL"

if [ "$REUSE" = 1 ] && [ -x "$VENV/bin/python" ]; then echo "reusing venv $VENV"; else uv venv --python 3.11 "$VENV"; fi
uv pip install --python "$VENV/bin/python" -r requirements.lock

clone() {  # name repo commit [sparse paths...]
  local name=$1 repo=$2 commit=$3; shift 3
  [ -d "third_party/$name/.git" ] && { echo "have $name"; return; }
  mkdir -p "third_party/$name" && pushd "third_party/$name" >/dev/null
  git init -q && git remote add origin "https://github.com/$repo.git"
  if [ $# -gt 0 ]; then git sparse-checkout init --cone && git sparse-checkout set "$@"; \
     git fetch -q --depth 1 --filter=blob:none origin "$commit"; else git fetch -q --depth 1 origin "$commit"; fi
  git checkout -q FETCH_HEAD && echo "$name $(git rev-parse HEAD)"
  popd >/dev/null
}
clone IFBench allenai/IFBench 488918665d53f867fa908a293769ec6e71e74aec
clone T-Eval open-compass/T-Eval 58f22406404d7e2a4f36856a19c7f4dc28a0a5f0
clone LiveCodeBench LiveCodeBench/LiveCodeBench 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
clone google-research-ifeval google-research/google-research 7117df5f368e1301e34bd014be9c93e24a4e73c6 instruction_following_eval
clone DAMO-ConvAI-apibank AlibabaResearch/DAMO-ConvAI faf754671f5b850e1b4d1ff870db37a6a2480df6 api-bank
clone MetaTool HowieHwong/MetaTool 35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb src
clone evalplus evalplus/evalplus 26d6d00bb1fd0fa37f39c99d5290da67891d1c5e
rm -rf third_party/IFBench/__pycache__
if [ "$REUSE" = 1 ] && "$VENV/bin/python" -I -c 'import importlib.metadata as m, sys; sys.exit(m.version("ifbench") != "0.2.0")' 2>/dev/null; then
  echo "have ifbench 0.2.0"
else uv pip install --python "$VENV/bin/python" --no-deps ./third_party/IFBench; fi

NLTK_SHA=550b6625bcef1f2abff2ff770a5a0d272c9c6b2a
for p in tokenizers/punkt_tab corpora/stopwords taggers/averaged_perceptron_tagger_eng; do
  d=".nltk_data/$(dirname $p)"; mkdir -p "$d"
  [ -d ".nltk_data/$p" ] && continue
  curl -sSL -o "$d/$(basename $p).zip" "https://raw.githubusercontent.com/nltk/nltk_data/$NLTK_SHA/packages/$p.zip"
  (cd "$d" && unzip -q -o "$(basename $p).zip" && rm "$(basename $p).zip")
done
mkdir -p .nltk_data/tokenizers/punkt   # empty on purpose: stops IFBench from auto-downloading pickled punkt
chmod -R go-w .nltk_data
echo "$NLTK_SHA" > .nltk_data/NLTK_DATA_COMMIT
echo "eval environment ready"
