#!/usr/bin/env bash
# Rebuild all training and evaluation data from the pinned sources in data/scripts/sources.yaml.
#
# Usage (from any directory):
#   ./download_data.sh [--only a,b] [--force] [--skip-checks] [--skip-gated] [--download-only]
#
#   --only a,b       download only these sources (names = top-level keys of sources.yaml).
#                    Normalization and splitting still run over everything present in data/raw,
#                    because make_splits.py decontaminates every training set against every eval set.
#   --force          re-download files that already exist (default: existing files are skipped)
#   --skip-checks    skip check_data.py --ngram
#   --skip-gated     do not touch the gated (kind: manual) sources at all
#   --download-only  stop after downloading (no normalize / make_splits / checks)
#
# Steps (the existing pipeline; nothing is reimplemented here):
#   1. data/scripts/download.py     every non-manual source, HF token disabled
#   2. gated sources                every `kind: manual` entry of sources.yaml that has repo/revision/files,
#                                   fetched with YOUR OWN existing Hugging Face login (see below)
#   3. data/scripts/normalize.py    raw -> processed (resets split_pool)
#   4. data/scripts/make_splits.py  splits + MANIFEST.json, back-fills split_pool
#   5. data/scripts/check_data.py --ngram
#
# Python: eval/.venv/bin/python, always run with -I. Override with RSI_PYTHON=/path/to/python.
# Create the environment first with ./setup_env.sh (or bash eval/setup_env.sh).
#
# Gated sources (currently gpqa_diamond and gpqa_main from Idavidrein/gpqa): this script never reads,
# prints, writes or passes a token. It relies on the credentials that `hf auth login` already cached.
# If you are not logged in, or have not accepted the dataset terms, it prints what to do and skips
# them; everything else still runs (check_data.py may then report the missing GPQA sets).
# GPQA must NOT be redistributed publicly: keep data/raw/gpqa_*/ and data/processed/gpqa_*.jsonl
# out of any public repository or release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYBIN="${RSI_PYTHON:-$ROOT/eval/.venv/bin/python}"

ONLY="" FORCE=0 SKIP_CHECKS=0 SKIP_GATED=0 DOWNLOAD_ONLY=0
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$ROOT/download_data.sh"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --only) [ $# -ge 2 ] || { echo "--only needs a comma-separated list" >&2; exit 2; }
            ONLY=$2; shift 2 ;;
    --only=*) ONLY=${1#--only=}; shift ;;
    --force) FORCE=1; shift ;;
    --skip-checks) SKIP_CHECKS=1; shift ;;
    --skip-gated) SKIP_GATED=1; shift ;;
    --download-only) DOWNLOAD_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

if [ ! -x "$PYBIN" ]; then
  echo "ERROR: Python not found at $PYBIN" >&2
  echo "  Create the environment first:  bash $ROOT/setup_env.sh   (or: bash $ROOT/eval/setup_env.sh)" >&2
  echo "  or point RSI_PYTHON at a Python that has pyyaml, pandas, pyarrow and huggingface_hub." >&2
  exit 1
fi
if ! "$PYBIN" -I -c "import yaml, pandas, huggingface_hub" 2>/dev/null; then
  echo "ERROR: $PYBIN lacks pyyaml / pandas / huggingface_hub; run bash $ROOT/setup_env.sh" >&2
  exit 1
fi
py() { "$PYBIN" -I "$@"; }
SCRIPTS="$ROOT/data/scripts"

if [ -n "$ONLY" ]; then  # catch typos: download.py silently ignores unknown names
  py - "$SCRIPTS/sources.yaml" "$ONLY" <<'PYEOF'
import sys, yaml
known = set(yaml.safe_load(open(sys.argv[1])))
bad = sorted(set(filter(None, sys.argv[2].split(","))) - known)
if bad:
    sys.exit(f"ERROR: unknown source(s) for --only: {', '.join(bad)}\n  known: {', '.join(sorted(known))}")
PYEOF
fi

echo "== 1/5 download (pinned, no token)"
DL_ARGS=""
[ -n "$ONLY" ] && DL_ARGS="--only $ONLY"
[ "$FORCE" = 1 ] && DL_ARGS="$DL_ARGS --force"
# shellcheck disable=SC2086  # DL_ARGS is intentionally word-split (names contain no spaces)
py "$SCRIPTS/download.py" $DL_ARGS

GATED_RC=0
if [ "$SKIP_GATED" = 1 ]; then
  echo "== 2/5 gated sources: skipped (--skip-gated)"
else
  echo "== 2/5 gated sources (your own Hugging Face login)"
  set +e
  # The embedded program gets the data dir, --only list and force flag as arguments.
  py - "$ROOT/data" "$ONLY" "$FORCE" <<'PYEOF'
import hashlib, json, os, shutil, sys
from pathlib import Path

import yaml

data, only, force = Path(sys.argv[1]), set(filter(None, sys.argv[2].split(","))), sys.argv[3] == "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Use the cached login implicitly; never read or pass a token ourselves.
os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError  # noqa: E402

sources = yaml.safe_load(open(data / "scripts" / "sources.yaml"))
manual = {n: s for n, s in sources.items()
          if s.get("kind") == "manual" and (not only or n in only)}
if not manual:
    print("  (no gated sources selected)")
    sys.exit(0)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_prov(name, spec, out, note):
    prov = {k: v for k, v in spec.items() if k != "files"}
    prov["note"] = note
    prov["files"] = {f: {"sha256": sha256(out / f), "bytes": (out / f).stat().st_size} for f in spec["files"]}
    json.dump(prov, open(out / "_provenance.json", "w"), indent=1, sort_keys=True)


def howto(repo, url):
    print(f"    To get it: (1) open {url} while logged in and accept the dataset terms;")
    print("               (2) run `hf auth login` once in this shell (paste a read token when it asks);")
    print("               (3) re-run this script (e.g. ./download_data.sh --only "
          + ",".join(sorted(manual)) + ").")


skipped, api, login_ok, login_msg = [], HfApi(), None, ""
for name, spec in manual.items():
    repo, rev, files = spec.get("repo"), spec.get("revision"), spec.get("files")
    url = spec.get("url") or f"https://huggingface.co/datasets/{repo}"
    if not (repo and rev and files):
        print(f"  [skip] {name}: manual source without repo/revision/files; see data/README.md")
        skipped.append(name)
        continue
    out = data / "raw" / name
    have = all((out / f).exists() for f in files)
    if have and not force:
        if not (out / "_provenance.json").exists():
            write_prov(name, spec, out, "provenance computed from files already present locally "
                                        "(fetched earlier with the user's HF login)")
            print(f"  [have] {name}: files present; wrote _provenance.json")
        else:
            print(f"  [have] {name}: files present")
        continue
    if login_ok is None:  # check the login once, only when something has to be fetched
        try:
            api.whoami()  # uses the cached `hf auth login` credentials, if any
            login_ok = True
        except Exception as e:  # noqa: BLE001  (not logged in / expired / offline)
            login_ok, login_msg = False, type(e).__name__
    if not login_ok:
        print(f"  [skip] {name}: no usable Hugging Face login ({login_msg}).")
        howto(repo, url)
        skipped.append(name)
        continue
    try:
        api.auth_check(repo, repo_type="dataset")
    except GatedRepoError:
        print(f"  [skip] {name}: logged in, but this account has not been granted access to {repo}.")
        howto(repo, url)
        skipped.append(name)
        continue
    except RepositoryNotFoundError:
        print(f"  [skip] {name}: {repo} not found (or not visible to this account).")
        howto(repo, url)
        skipped.append(name)
        continue
    print(f"  [get ] {name} ({repo}@{rev[:12]}, gated)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for fn in files:
        if (out / fn).exists() and not force:
            continue
        p = hf_hub_download(repo_id=repo, repo_type="dataset", filename=fn, revision=rev,
                            local_dir=str(out), force_download=force)
        assert Path(p).resolve() == (out / fn).resolve(), (p, out / fn)
    shutil.rmtree(out / ".cache", ignore_errors=True)
    write_prov(name, spec, out, "downloaded with the user's existing HF login after terms were accepted")

if skipped:
    print("  gated sources skipped: " + ", ".join(skipped))
    sys.exit(3)
PYEOF
  GATED_RC=$?
  set -e
  if [ "$GATED_RC" != 0 ] && [ "$GATED_RC" != 3 ]; then
    echo "ERROR: gated-source step failed (exit $GATED_RC)" >&2
    exit "$GATED_RC"
  fi
fi

if [ "$DOWNLOAD_ONLY" = 1 ]; then
  echo "== stopping after download (--download-only)"
  exit 0
fi

echo "== 3/5 normalize (all sources present in data/raw)"
py "$SCRIPTS/normalize.py"
echo "== 4/5 make_splits"
py "$SCRIPTS/make_splits.py"

if [ "$SKIP_CHECKS" = 1 ]; then
  echo "== 5/5 checks skipped (--skip-checks)"
else
  if [ "$GATED_RC" = 3 ]; then
    echo "NOTE: some gated sources were skipped; check_data.py may fail on the missing GPQA sets." >&2
  fi
  echo "== 5/5 check_data --ngram"
  py "$SCRIPTS/check_data.py" --ngram
fi

if [ "$GATED_RC" = 3 ]; then
  echo "done (gated sources skipped; see instructions above)"
else
  echo "done"
fi
