#!/usr/bin/env python
"""Build the eval-set target dataset (objective eval_overfit, 2026-11-20): the evaluation sets themselves, merged
into one dataset that the RSI runs optimise on directly (no train / signal / gate split; the editor sees it all).

    python -I data/scripts/build_eval_target.py [--name math_eval]

math_eval = MATH-500 (data/processed/math.jsonl, split pool `test`) + GSM8K test (data/processed/gsm8k.jsonl,
pool `test`) = 1819 items, graded by the math grader (eval/configs/tasks.yaml task `math_eval`). Written to
data/processed/<name>.jsonl (dataset = <name>, every item in split pool `pool`, ids and fields unchanged) and
data/splits/<name>/pool.json. Deterministic and idempotent: rebuilt only when missing or a source changed.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC, SPLITS = ROOT / "data" / "processed", ROOT / "data" / "splits"
TARGETS = {"math_eval": [("math", "test"), ("gsm8k", "test")]}


def _sources_sha(sources) -> str:
    h = hashlib.sha256()
    for ds, pool in sources:
        h.update(f"{ds}:{pool}:".encode())
        h.update((PROC / f"{ds}.jsonl").read_bytes())
        h.update((SPLITS / ds / f"{pool}.json").read_bytes())
    return h.hexdigest()


def build(name: str = "math_eval", force: bool = False) -> dict:
    """Write the merged dataset if needed; returns {"name", "n", "by_source", "rebuilt"}."""
    if name not in TARGETS:
        raise SystemExit(f"unknown eval target {name!r}; known: {sorted(TARGETS)}")
    sources = TARGETS[name]
    for ds, pool in sources:
        if not (PROC / f"{ds}.jsonl").exists() or not (SPLITS / ds / f"{pool}.json").exists():
            raise SystemExit(f"missing source data {ds} / {pool} (./download_data.sh)")
    sha = _sources_sha(sources)
    out, split = PROC / f"{name}.jsonl", SPLITS / name / "pool.json"
    meta_p = SPLITS / name / "BUILD.json"
    if not force and out.exists() and split.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("sources_sha256") == sha:
            return dict(meta, rebuilt=False)
    rows, by = [], {}
    for ds, pool in sources:
        keep = set(json.loads((SPLITS / ds / f"{pool}.json").read_text()))
        for line in (PROC / f"{ds}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in keep:
                r = dict(r, dataset=name, split_pool="pool", metadata=dict(r.get("metadata") or {}, source_dataset=ds))
                rows.append(r)
                by[ds] = by.get(ds, 0) + 1
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids across the eval sources")
    rows.sort(key=lambda r: r["id"])
    split.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    split.write_text(json.dumps(sorted(ids)))
    meta = {"name": name, "n": len(rows), "by_source": by, "sources": sources, "sources_sha256": sha}
    meta_p.write_text(json.dumps(meta, indent=1))
    return dict(meta, rebuilt=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="math_eval")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build(a.name, a.force)))
    sys.exit(0)
