#!/usr/bin/env bash
# Download the model weights used by the two papers, at pinned Hugging Face commits.
#
# Usage (from any directory):
#   ./download_models.sh [--list] [--dry-run] [NAME ...|all]
#
#   NAME       one or more names from --list (default: qwen3.8-27b qwen3-8b, the Paper A and
#              Paper B main models); `all` selects every model
#   --list     show names, repos, pinned revisions, sizes, licenses (offline, no network)
#   --dry-run  query the Hub, verify each pinned revision and file list, print the disk space
#              needed vs available and the exact `hf download` commands, but download nothing
#
# Files go to ${MODELS_DIR:-$HOME/models}/<name>/ (outside the repo). Only safetensors weights plus
# config / tokenizer / chat-template / license files are fetched; other weight formats
# (original/, *.bin, *.pth, *.gguf, ...) and repo code files (*.py) are never downloaded.
# Re-running is safe: files already present with the expected size are not fetched again.
#
# Credentials: public models are downloaded anonymously (implicit token disabled). For a gated
# model the script uses the login that `hf auth login` already cached; it never reads, prints,
# writes or passes a token. None of the models pinned below is gated (checked 2026-11-19).
#
# Tools: Python with huggingface_hub (default eval/.venv/bin/python, override RSI_PYTHON) and the
# `hf` CLI (default: next to that Python, else on PATH; override HF_CLI). --list needs only python3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [ -n "${RSI_PYTHON:-}" ]; then
  PYBIN=$RSI_PYTHON
elif [ -x "$ROOT/eval/.venv/bin/python" ]; then
  PYBIN=$ROOT/eval/.venv/bin/python
else
  PYBIN=$(command -v python3 || true)
fi
if [ -z "$PYBIN" ] || [ ! -x "$PYBIN" ]; then
  echo "ERROR: no Python found. Run bash $ROOT/setup_env.sh, or set RSI_PYTHON." >&2
  exit 1
fi
for a in "$@"; do
  case "$a" in
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$ROOT/download_models.sh"; exit 0 ;;
  esac
done

export RSI_MODELS_DIR="${MODELS_DIR:-$HOME/models}"
exec "$PYBIN" -I - "$@" <<'PYEOF'
import argparse, fnmatch, os, shutil, subprocess, sys
from pathlib import Path

# name | repo | pinned commit | license | gated | bytes to download | architectures | role
# Metadata from HfApi(token=False).model_info(repo, files_metadata=True) and config.json, 2026-11-19.
# "bytes" = sum of the files selected by want() below at that commit.
MODELS = [
    ("qwen3.8-27b", "Qwen/Qwen3.8-27B", "37561816e66b5b9153b1b64c10968c18c1e6f4a0", "apache-2.0", False,
     55586105445, "Qwen3_5ForConditionalGeneration", "Paper A main (dense 27B, BF16; also Paper B extension)"),
    ("qwen3-8b", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "apache-2.0", False,
     16397459696, "Qwen3ForCausalLM", "Paper B main (dense 8B, BF16)"),
    ("qwen3.6-27b", "Qwen/Qwen3.6-27B", "8f33b98937c6031bb49e713dee3e8e185aafd26f", "apache-2.0", False,
     55586099687, "Qwen3_5ForConditionalGeneration", "Paper A backup (dense 27B, BF16)"),
    ("gemma-4-31b", "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", "apache-2.0", False,
     62578684366, "Gemma4ForConditionalGeneration", "cross-family check (instruction-tuned 31B, BF16)"),
    # Base (pre-trained) Gemma 4 31B, if the cross-family check needs it:
    # ("gemma-4-31b-base", "google/gemma-4-31B", "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89", ...),
]
DEFAULT = ["qwen3.8-27b", "qwen3-8b"]
BY_NAME = {m[0]: m for m in MODELS}

KEEP_SUFFIX = (".safetensors", ".json", ".jinja", ".txt", ".model", ".tiktoken")
KEEP_NAMES = {"LICENSE", "LICENSE.txt", "NOTICE", "README.md"}
SKIP_DIRS = ("original/", ".eval_results/")


def want(path):
    if path.startswith(SKIP_DIRS) or "gguf" in path.lower():
        return False
    return path in KEEP_NAMES or path.endswith(KEEP_SUFFIX)


def gib(n):
    return f"{n / 2**30:7.1f} GiB"


def table(rows):
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)).rstrip())


ap = argparse.ArgumentParser(prog="download_models.sh", add_help=False)
ap.add_argument("names", nargs="*")
ap.add_argument("--list", action="store_true")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

if a.list:
    rows = [("name", "repo", "revision", "size", "license", "gated", "architecture", "role")]
    for n, repo, rev, lic, gated, size, arch, role in MODELS:
        rows.append((n + (" *" if n in DEFAULT else ""), repo, rev[:12], gib(size).strip(), lic,
                     "yes" if gated else "no", arch, role))
    table(rows)
    print("  (* = downloaded by default; target dir: " + os.environ["RSI_MODELS_DIR"] + "/<name>)")
    sys.exit(0)

names = a.names or DEFAULT
if names == ["all"]:
    names = [m[0] for m in MODELS]
bad = [n for n in names if n not in BY_NAME]
if bad:
    sys.exit(f"ERROR: unknown model(s): {', '.join(bad)}; known: {', '.join(BY_NAME)} (or `all`)")

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit(f"ERROR: huggingface_hub not importable by {sys.executable}; run setup_env.sh or set RSI_PYTHON")

hf = os.environ.get("HF_CLI") or (str(Path(sys.executable).parent / "hf")
                                  if (Path(sys.executable).parent / "hf").exists() else shutil.which("hf"))
if not hf and not a.dry_run:
    sys.exit("ERROR: `hf` CLI not found (pip install huggingface_hub, or set HF_CLI)")

base = Path(os.environ["RSI_MODELS_DIR"]).expanduser()
anon = HfApi(token=False)
plan, login_checked, login_ok = [], False, False
print(f"== checking pinned revisions on the Hub (target {base})")
for n in names:
    _, repo, rev, lic, gated_rec, size_rec, arch, role = BY_NAME[n]
    info = anon.model_info(repo, revision=rev, files_metadata=True)
    if info.sha != rev:
        sys.exit(f"ERROR: {repo}: Hub returned revision {info.sha}, expected {rev}")
    files = sorted((s.rfilename, s.size or 0) for s in info.siblings if want(s.rfilename))
    if not any(f.endswith(".safetensors") for f, _ in files):
        sys.exit(f"ERROR: {repo}@{rev[:12]}: no safetensors files")
    total = sum(sz for _, sz in files)
    dest = base / n
    missing = sum(sz for f, sz in files if not ((dest / f).is_file() and (dest / f).stat().st_size == sz))
    gated = bool(info.gated)
    hub_lic = (info.card_data or {}).get("license") if info.card_data else None
    notes = []
    if total != size_rec:
        notes.append(f"size differs from recorded {size_rec}")
    if gated != gated_rec:
        notes.append(f"gated status is now {info.gated!r}")
    if hub_lic and hub_lic != lic:
        notes.append(f"license tag is now {hub_lic}")
    print(f"  {n:13s} {repo}@{rev[:12]}  {len(files):2d} files  total {gib(total)}  missing {gib(missing)}"
          + ("  [gated]" if gated else "") + ("  NOTE: " + "; ".join(notes) if notes else ""))
    if gated and not login_checked:
        login_checked = True
        try:
            HfApi().whoami()  # cached `hf auth login` credentials, never passed explicitly
            login_ok = True
        except Exception as e:  # noqa: BLE001
            print(f"  ! {n} is gated and no usable Hugging Face login was found ({type(e).__name__}).")
    if gated:
        try:
            if login_ok:
                HfApi().auth_check(repo)
            else:
                raise PermissionError("not logged in")
        except Exception as e:  # noqa: BLE001
            print(f"  ! {n}: no access ({type(e).__name__}). Accept the terms at https://huggingface.co/{repo}, "
                  "run `hf auth login`, then re-run.")
            if not a.dry_run:
                sys.exit(1)
    plan.append((n, repo, rev, gated, files, dest, missing))

need = sum(p[6] for p in plan)
probe = base
while not probe.exists():
    probe = probe.parent
free = shutil.disk_usage(probe).free
print(f"== disk: need {gib(need).strip()} more, available {gib(free).strip()} on the filesystem of {probe}")
if need > free:
    msg = "not enough free disk space; set MODELS_DIR to a larger filesystem"
    if a.dry_run:
        print("  WARNING: " + msg)
    else:
        sys.exit("ERROR: " + msg)

for n, repo, rev, gated, files, dest, missing in plan:
    cmd = [hf or "hf", "download", repo, *[f for f, _ in files], "--revision", rev, "--local-dir", str(dest)]
    shown = f"{cmd[0]} download {repo} <{len(files)} files> --revision {rev} --local-dir {dest}"
    if a.dry_run:
        print(f"[dry-run] {n}: {shown}")
        print("          files: " + " ".join(f for f, _ in files))
        continue
    if missing == 0:
        print(f"[have] {n}: all files present in {dest}")
        continue
    env = dict(os.environ)
    if not gated:  # public repo: download anonymously
        env.pop("HF_TOKEN", None)
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {n}: {shown}", flush=True)
    subprocess.run(cmd, env=env, check=True)
    bad = [f for f, sz in files if not ((dest / f).is_file() and (dest / f).stat().st_size == sz)]
    if bad:
        sys.exit(f"ERROR: {n}: size mismatch / missing after download: {bad}")
    print(f"[ok  ] {n}: {len(files)} files, {gib(sum(sz for _, sz in files)).strip()} in {dest}")
print("done" + (" (dry run, nothing downloaded)" if a.dry_run else ""))
PYEOF
