"""Download all pinned sources listed in sources.yaml into data/raw/<name>/.

Usage (from anywhere; run with -I so no local modules shadow stdlib):
    eval/.venv/bin/python -I data/scripts/download.py [--only name1,name2] [--force]

* Never uses credentials: HF token disabled explicitly, no git auth.
* Downloaded files are treated as untrusted data: nothing inside data/raw is ever executed.
* Writes data/raw/<name>/_provenance.json (repo, revision, url, sha256 per file).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

import yaml  # noqa: E402

DATA = Path(__file__).resolve().parents[1]
RAW = DATA / "raw"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_url(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dest)


def download_hf(name, spec, out: Path, force: bool):
    from huggingface_hub import hf_hub_download

    files = []
    for fn in spec["files"]:
        dest = out / fn
        if dest.exists() and not force:
            files.append(fn)
            continue
        p = hf_hub_download(
            repo_id=spec["repo"], repo_type="dataset", filename=fn,
            revision=spec["revision"], local_dir=str(out), token=False,
        )
        assert Path(p).resolve() == dest.resolve(), (p, dest)
        files.append(fn)
    # hf local_dir leaves a .cache folder with metadata; remove it to keep raw clean
    shutil.rmtree(out / ".cache", ignore_errors=True)
    return files


def download_github(name, spec, out: Path, force: bool):
    files = []
    for fn in spec["files"]:
        dest = out / fn
        if not dest.exists() or force:
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['revision']}/{fn}"
            fetch_url(url, dest)
        files.append(fn)
    return files


def download_url(name, spec, out: Path, force: bool):
    files = []
    for fn, url in spec["urls"].items():
        dest = out / fn
        if not dest.exists() or force:
            fetch_url(url, dest)
        files.append(fn)
    return files


def download_pip(name, spec, out: Path, force: bool):
    """Copy data shipped inside an installed pinned package (no code executed: we locate the
    package directory via importlib.util.find_spec, which does not import it)."""
    import importlib.metadata as md
    import importlib.util

    ver = md.version(spec["package"])
    assert ver == spec["version"], f"{spec['package']} {ver} != pinned {spec['version']}"
    top, *rest = spec["subdir"].split("/")
    s = importlib.util.find_spec(top)
    src = Path(list(s.submodule_search_locations)[0]).joinpath(*rest)
    if out.exists() and force:
        shutil.rmtree(out)
    files = []
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.suffix in {".json", ".jsonl"}:
            rel = p.relative_to(src)
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(p, dest)
            files.append(str(rel))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sources = yaml.safe_load(open(DATA / "scripts" / "sources.yaml"))
    only = set(filter(None, args.only.split(",")))
    for name, spec in sources.items():
        if only and name not in only:
            continue
        out = RAW / name
        kind = spec["kind"]
        if kind == "manual":
            print(f"[skip] {name}: gated/manual, see data/README.md")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"[get ] {name} ({kind})", flush=True)
        fn = {"hf": download_hf, "github": download_github, "pip": download_pip, "url": download_url}[kind]
        files = fn(name, spec, out, args.force)
        prov = {k: v for k, v in spec.items() if k not in ("files", "urls")}
        if "urls" in spec:
            prov["urls"] = spec["urls"]
        prov["files"] = {f: {"sha256": sha256(out / f), "bytes": (out / f).stat().st_size} for f in files}
        json.dump(prov, open(out / "_provenance.json", "w"), indent=1, sort_keys=True)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
