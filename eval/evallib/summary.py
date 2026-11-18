"""Aggregate per-unit records into summary.json entries."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import PROC_DIR, SPLIT_DIR


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def sha256_file(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def summarize(task, records: list[dict], extra: dict) -> dict:
    graded = [r for r in records if r["score"] is not None]
    item_scores = defaultdict(list)
    for r in graded:
        item_scores[r["id"]].append(r["score"])
    s = {
        "task": task.name, "dataset": task.dataset, "category": task.category,
        "accuracy": _mean([r["score"] for r in graded]),
        "item_accuracy": _mean([_mean(v) for v in item_scores.values()]),
        "n": len(graded), "n_units": len(records), "n_items": len({r["id"] for r in records}),
        "n_pending": sum(1 for r in records if r["score"] is None),
        "status_counts": dict(Counter(r["status"] for r in records)),
    }
    g = task.cfg["grader"]
    if graded and len(graded[0]["sample_scores"]) > 1:
        s["avg_at_n"] = s["accuracy"]
        s["pass_at_n"] = _mean([float(any(x for x in r["sample_scores"] if x)) for r in graded])
        maj = []
        for r in graded:
            pa = [str(p) for p in (r["parsed_answer"] or []) if p is not None]
            if not pa:
                maj.append(0.0)
                continue
            top = Counter(pa).most_common(1)[0][0]
            idx = [str(p) for p in r["parsed_answer"]].index(top)
            maj.append(float(bool(r["sample_scores"][idx])))
        s["maj_at_n"] = _mean(maj)
    if g in ("ifeval", "ifbench"):
        ms = [r["metrics"] for r in graded if r["metrics"]]
        s["prompt_strict"] = _mean([m["prompt_strict"] for m in ms])
        s["prompt_loose"] = _mean([m["prompt_loose"] for m in ms])
        s["inst_strict"] = _mean([x for m in ms for x in m["inst_strict"]])
        s["inst_loose"] = _mean([x for m in ms for x in m["inst_loose"]])
    if g == "rlvr_if":
        by = defaultdict(list)
        for r in graded:
            m = r["metrics"][0] if isinstance(r["metrics"], list) else r["metrics"]  # n>1: list per sample
            if m:
                by[m["constraint_type"]].append(r["score"])
        s["per_constraint_type"] = {k: {"accuracy": _mean(v), "n": len(v)} for k, v in sorted(by.items())}
    if g in ("short_qa", "squad_closed_book"):
        ms = [r["metrics"] for r in graded if r["metrics"]]
        s["em"] = s["accuracy"]
        s["f1"] = _mean([m["f1"] for m in ms])
        s["contains"] = _mean([m["contains"] for m in ms])
    if g == "bfcl":
        bycat = defaultdict(list)
        dec = defaultdict(Counter)
        for r in records:
            c = r.get("bfcl_category")
            if r["score"] is not None:
                bycat[c].append(r["score"])
            for d in r.get("decision") or []:
                if d:
                    dec[c][d] += 1
        s["per_category"] = {c: {"accuracy": _mean(v), "n": len(v)} for c, v in sorted(bycat.items())}
        s["call_decisions"] = {c: dict(v) for c, v in sorted(dec.items())}
        irr = bycat.get("irrelevance", [])
        ls = bycat.get("live_simple", [])
        s["irrelevance_vs_live_simple"] = {
            "irrelevance": {"accuracy": _mean(irr), "n": len(irr), "no_call": dec["irrelevance"].get("no_call", 0),
                            "call": dec["irrelevance"].get("call", 0)},
            "live_simple": {"accuracy": _mean(ls), "n": len(ls), "call": dec["live_simple"].get("call", 0),
                            "no_call": dec["live_simple"].get("no_call", 0)},
        }
        s["not_graded_categories"] = sorted({r.get("bfcl_category") for r in records if r["score"] is None})
    if g in ("teval", "metatool", "apibank"):
        key = {"teval": "subset", "metatool": "task", "apibank": "level"}[g]
        by = defaultdict(list)
        for r in graded:
            by[r.get("group")].append(r["score"])
        s[f"per_{key}"] = {k: {"accuracy": _mean(v), "n": len(v)} for k, v in sorted(by.items())}
    if g == "mt_bench_judge":
        s["mean_rating"] = s["accuracy"]
    if g == "alpaca_eval_judge":
        s["win_rate"] = s["accuracy"]
    s.update(extra)
    s["data_provenance"] = {
        "processed_sha256": sha256_file(PROC_DIR / f"{task.dataset}.jsonl"),
        "split_manifest_sha256": sha256_file(SPLIT_DIR / "MANIFEST.json"),
    }
    return s


def update_summary(run_dir: Path, task_name: str, entry: dict):
    p = run_dir / "summary.json"
    cur = json.load(open(p)) if p.exists() else {}
    cur[task_name] = entry
    json.dump(cur, open(p, "w"), indent=1, sort_keys=True, default=str)
