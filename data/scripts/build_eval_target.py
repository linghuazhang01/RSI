#!/usr/bin/env python
"""Build an eval-set target dataset (objective eval_overfit): an evaluation set copied into one dataset that the
RSI runs optimise on directly (no train / signal / gate split; the editor sees it all).

    python -I data/scripts/build_eval_target.py [--name math] [--all]

The targets are eval/configs/tasks.yaml `eval_targets` (2026-11-21: any single eval set, e.g. humanevalplus,
ifeval, gpqa_diamond): key K -> source (task, pool) pairs -> dataset K_eval. math -> math_eval = MATH-500
(data/processed/math.jsonl, pool `test`) + GSM8K test = 1819 items. Written to data/processed/K_eval.jsonl
(dataset = K_eval, every item in split pool `pool`, ids and fields unchanged) and data/splits/K_eval/pool.json.
--name takes K or K_eval. Deterministic and idempotent: rebuilt only when missing or a source changed.

Forms of an eval_targets entry: [[task, pool], ...] (above); {task, file, dataset} = the items of a jsonl file
(simple_data/<d>/{train,eval,test}.jsonl); {task, ids_file, dataset} (2026-11-21) = the records of the task's
dataset (data/processed/<dataset>.jsonl) whose id is in the JSON list `ids_file` (simple_<d>_pool from
simple_data/<d>/pool_ids.json; rebuilt when the ids file or the processed file's size changes).
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
    """{dataset: [(source dataset, pool)] | [("file", path)] | [("ids_file", path, source dataset)]} from tasks.yaml
    eval_targets (K -> K_eval; the {task, file, dataset} form -> `dataset`, the items of `file`, e.g.
    simple_data/math/train.jsonl; the {task, ids_file, dataset} form -> the task dataset's records listed in
    `ids_file`)."""
    import yaml
    cfg = yaml.safe_load(TASKS_YAML.read_text())
    tasks, out = cfg["tasks"], {}
    for k, v in (cfg.get("eval_targets") or {}).items():
        if isinstance(v, dict) and v.get("ids_file"):
            out[v["dataset"]] = [("ids_file", str(v["ids_file"]), tasks[v["task"]]["dataset"])]
        elif isinstance(v, dict):
            out[v["dataset"]] = [("file", str(v["file"]))]
        else:
            out[f"{k}_eval"] = [(tasks[t]["dataset"], str(p)) for t, p in v]
    return out


def _sources_sha(sources) -> str:
    h = hashlib.sha256()
    for ds, pool, *src in sources:
        h.update(f"{ds}:{pool}:".encode())
        if ds == "ids_file":                       # the ids, plus the source's size (not its 7-70 MB of bytes)
            h.update((ROOT / pool).read_bytes())
            h.update(f"{src[0]}:{(PROC / f'{src[0]}.jsonl').stat().st_size}".encode())
            continue
        if ds == "file":
            h.update((ROOT / pool).read_bytes())
            continue
        h.update((PROC / f"{ds}.jsonl").read_bytes())
        h.update((SPLITS / ds / f"{pool}.json").read_bytes())
    return h.hexdigest()


def build(name: str = "math_eval", force: bool = False) -> dict:
    """Write the merged dataset if needed; returns {"name", "n", "by_source", "rebuilt"}."""
    tg = targets()
    if name not in tg and f"{name}_eval" in tg:
        name = f"{name}_eval"
    if name not in tg:
        raise SystemExit(f"unknown eval target {name!r}; known: {sorted(tg)}")
    sources = tg[name]
    for ds, pool, *src in sources:
        if ds == "ids_file":
            if not (ROOT / pool).exists():
                raise SystemExit(f"missing {pool} (eval/.venv/bin/python -I data/scripts/make_simple_data.py)")
            if not (PROC / f"{src[0]}.jsonl").exists():
                raise SystemExit(f"missing source data {src[0]} (./download_data.sh)")
        elif ds == "file":
            if not (ROOT / pool).exists():
                raise SystemExit(f"missing {pool} (eval/.venv/bin/python -I data/scripts/make_simple_data.py)")
        elif not (PROC / f"{ds}.jsonl").exists() or not (SPLITS / ds / f"{pool}.json").exists():
            raise SystemExit(f"missing source data {ds} / {pool} (./download_data.sh)")
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
        else:
            keep = None if is_file else set(json.loads((SPLITS / ds / f"{pool}.json").read_text()))
            src = (ROOT / pool) if is_file else (PROC / f"{ds}.jsonl")
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
    ap.add_argument("--name", default="math_eval")
    ap.add_argument("--all", action="store_true", help="build every target whose source data is present")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
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
