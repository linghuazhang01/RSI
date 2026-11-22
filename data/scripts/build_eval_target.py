#!/usr/bin/env python
"""Build an eval-set dataset (the simple_data sets) registered in eval/configs/tasks.yaml `eval_targets`.

    python -I data/scripts/build_eval_target.py --name simple_math_test | --all

Forms of an eval_targets entry: {task, file, dataset} = the items of a jsonl file (simple_data/<d>/{train,eval,
test}.jsonl); {task, ids_file, dataset} (2026-11-21) = the records of the task's dataset (data/processed/<dataset>.jsonl)
whose id is in the JSON list `ids_file` (simple_<d>_pool from simple_data/<d>/pool_ids.json; rebuilt when the ids file
or the processed file's size changes). Written to data/processed/<dataset>.jsonl (every item in split pool `pool`, ids
and fields unchanged) and data/splits/<dataset>/pool.json. Deterministic and idempotent: rebuilt only when missing or
a source changed. rsicore.draws.build_eval_set calls build() on demand.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC, SPLITS = ROOT / "data" / "processed", ROOT / "data" / "splits"
TASKS_YAML = ROOT / "eval" / "configs" / "tasks.yaml"


def targets() -> dict:
    """{dataset: [("file", path)] | [("ids_file", path, source dataset)]} from tasks.yaml eval_targets (the
    {task, file, dataset} form -> the items of `file`, e.g. simple_data/math/train.jsonl; the {task, ids_file, dataset}
    form -> the task dataset's records listed in `ids_file`)."""
    import yaml
    cfg = yaml.safe_load(TASKS_YAML.read_text())
    tasks, out = cfg["tasks"], {}
    for v in (cfg.get("eval_targets") or {}).values():
        if v.get("ids_file"):
            out[v["dataset"]] = [("ids_file", str(v["ids_file"]), tasks[v["task"]]["dataset"])]
        else:
            out[v["dataset"]] = [("file", str(v["file"]))]
    return out


def _sources_sha(sources) -> str:
    h = hashlib.sha256()
    for ds, pool, *src in sources:
        h.update(f"{ds}:{pool}:".encode())
        if ds == "ids_file":                       # the ids, plus the source's size (not its 7-70 MB of bytes)
            h.update((ROOT / pool).read_bytes())
            h.update(f"{src[0]}:{(PROC / f'{src[0]}.jsonl').stat().st_size}".encode())
            continue
        h.update((ROOT / pool).read_bytes())    # "file"
    return h.hexdigest()


def build(name: str, force: bool = False) -> dict:
    """Write the dataset if needed; returns {"name", "n", "by_source", "rebuilt"}."""
    tg = targets()
    if name not in tg:
        raise SystemExit(f"unknown eval target {name!r}; known: {sorted(tg)}")
    sources = tg[name]
    for ds, pool, *src in sources:
        if ds == "ids_file":
            if not (ROOT / pool).exists():
                raise SystemExit(f"missing {pool} (eval/.venv/bin/python -I data/scripts/make_simple_data.py)")
            if not (PROC / f"{src[0]}.jsonl").exists():
                raise SystemExit(f"missing source data {src[0]} (./download_data.sh)")
        elif not (ROOT / pool).exists():                  # "file"
            raise SystemExit(f"missing {pool} (eval/.venv/bin/python -I data/scripts/make_simple_data.py)")
    sha = _sources_sha(sources)
    out, split = PROC / f"{name}.jsonl", SPLITS / name / "pool.json"
    meta_p = SPLITS / name / "BUILD.json"
    # (2026-11-21) Paper A and Paper B runs of one domain start together and build the same sets: one process at a
    # time checks and rebuilds (file lock), and every output is replaced atomically, BUILD.json last.
    with _lock(SPLITS / name / ".lock"):
        if not force and out.exists() and split.exists() and meta_p.exists():
            meta = json.loads(meta_p.read_text())
            if meta.get("sources_sha256") == sha:
                return dict(meta, rebuilt=False)
        return _rebuild(name, sources, sha, out, split, meta_p)


@contextlib.contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except ImportError:                       # pragma: no cover - non-POSIX
            pass
        yield


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _rebuild(name, sources, sha, out: Path, split: Path, meta_p: Path) -> dict:
    rows, by = [], {}
    for ds, pool, *srcds in sources:
        is_file, is_ids = ds == "file", ds == "ids_file"
        if is_ids:                       # the source dataset's records listed in the ids file
            ds = srcds[0]
            keep, src = set(json.loads((ROOT / pool).read_text())), PROC / f"{ds}.jsonl"
        else:                            # every item of the file
            keep, src = None, ROOT / pool
        n0 = len(rows)
        for line in src.read_text().split("\n"):   # not splitlines(): U+2028 inside text
            if not line.strip():
                continue
            r = json.loads(line)
            if is_file:
                ds = r.get("dataset") or "file"         # the training set the item came from
            if keep is None or r["id"] in keep:
                r = dict(r, dataset=name, split_pool="pool", metadata=dict(r.get("metadata") or {}, source_dataset=ds))
                rows.append(r)
                by[ds] = by.get(ds, 0) + 1
        if is_ids and len(rows) - n0 != len(keep):
            raise SystemExit(f"{name}: {len(keep) - (len(rows) - n0)} listed ids not found in {src}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids across the eval sources")
    rows.sort(key=lambda r: r["id"])
    _write_atomic(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    _write_atomic(split, json.dumps(sorted(ids)))
    meta = {"name": name, "n": len(rows), "by_source": by, "sources": sources, "sources_sha256": sha}
    _write_atomic(meta_p, json.dumps(meta, indent=1))
    return dict(meta, rebuilt=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="a dataset of eval_targets, e.g. simple_math_test")
    ap.add_argument("--all", action="store_true", help="build every target whose source data is present")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not a.all and not a.name:
        ap.error("give --name <dataset> or --all")
    if a.all:
        for n in targets():
            try:
                r = build(n, a.force)
                print(json.dumps({k: r[k] for k in ("name", "n", "rebuilt")}))
            except SystemExit as e:
                print(f"{n}: skipped ({e})")
    else:
        print(json.dumps(build(a.name, a.force)))
    sys.exit(0)
