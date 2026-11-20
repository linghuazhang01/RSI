#!/usr/bin/env python
"""Build an eval-set target dataset (objective eval_overfit): an evaluation set copied into one dataset that the
RSI runs optimise on directly (no train / signal / gate split; the editor sees it all).

    python -I data/scripts/build_eval_target.py [--name math] [--all]

The targets are eval/configs/tasks.yaml `eval_targets` (2026-11-21: any single eval set, e.g. humanevalplus,
ifeval, gpqa_diamond): key K -> source (task, pool) pairs -> dataset K_eval. math -> math_eval = MATH-500
(data/processed/math.jsonl, pool `test`) + GSM8K test = 1819 items. Written to data/processed/K_eval.jsonl
(dataset = K_eval, every item in split pool `pool`, ids and fields unchanged) and data/splits/K_eval/pool.json.
--name takes K or K_eval. Deterministic and idempotent: rebuilt only when missing or a source changed.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC, SPLITS = ROOT / "data" / "processed", ROOT / "data" / "splits"
TASKS_YAML = ROOT / "eval" / "configs" / "tasks.yaml"


def targets() -> dict:
    """{K_eval: [(source dataset, pool)]} from tasks.yaml eval_targets."""
    import yaml
    cfg = yaml.safe_load(TASKS_YAML.read_text())
    tasks = cfg["tasks"]
    return {f"{k}_eval": [(tasks[t]["dataset"], str(p)) for t, p in v] for k, v in (cfg.get("eval_targets") or {}).items()}


def _sources_sha(sources) -> str:
    h = hashlib.sha256()
    for ds, pool in sources:
        h.update(f"{ds}:{pool}:".encode())
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
        for line in (PROC / f"{ds}.jsonl").read_text().split("\n"):   # not splitlines(): U+2028 inside text
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
