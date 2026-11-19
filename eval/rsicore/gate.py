"""Gate helpers: noise band of repeated evaluations, relative pass test.

measure_noise_band() scores the same model `repeats` times with sampled decoding (temperature > 0,
seeds base_seed, base_seed+1, ...) through evaluate.score() and returns the spread of the mean scores.
Greedy repeats (temperature 0) are identical on vLLM and in the mock, so their band is 0.
"""
import math
from pathlib import Path


def _mean_score(records: list[dict]) -> float | None:
    """Item-level mean: average units per id first, skip pending (score None)."""
    per = {}
    for r in records:
        if r.get("score") is None:
            continue
        per.setdefault(r.get("id"), []).append(float(r["score"]))
    if not per:
        return None
    return sum(sum(v) / len(v) for v in per.values()) / len(per)


def noise_band(results_runs: list[list[dict]]) -> float:
    """Spread (max - min) of the mean score over repeated evaluations of the same model."""
    means = [m for m in (_mean_score(r) for r in results_runs) if m is not None]
    return (max(means) - min(means)) if len(means) >= 2 else 0.0


def relative_pass(score: float, baseline: float, ratio: float) -> bool:
    """score >= ratio * baseline (for a non-positive baseline: score >= baseline)."""
    if score is None or baseline is None:
        return False
    if baseline <= 0:
        return score >= baseline
    return score >= ratio * baseline


def measure_noise_band(endpoint, served_name, task: str, ids: list[str] | None, out_dir: Path, *, repeats: int = 3,
                       temperature: float = 0.6, base_seed: int = 42, mock: bool = False, lora: str | None = None,
                       pool: str = "pool") -> dict:
    """Repeat evaluate.score(task, ids) with seed = base_seed + k (k < repeats) at `temperature`.
    -> {"band": max - min of the item-level mean scores, "se": binomial SE sqrt(p(1-p)/n) of the pooled
    mean p over the n scored items, "scores": [...], "seeds": [...], "mean", "n", "temperature"}.
    Repeat k writes run_eval output to out_dir/seed<seed>/ (pending code tasks give score None)."""
    from .evaluate import score
    out_dir = Path(out_dir)
    seeds = [int(base_seed) + k for k in range(int(repeats))]
    runs = []
    for s in seeds:
        r = score(endpoint, served_name, task, pool=pool, ids=ids, out_dir=out_dir / f"seed{s}", lora=lora,
                  mock=mock, seed=s, temperature=temperature)
        runs.append(r.records)
    means = [_mean_score(r) for r in runs]
    ok = [m for m in means if m is not None]
    n = len({r.get("id") for r in runs[0] if r.get("score") is not None}) if runs else 0
    p = sum(ok) / len(ok) if ok else None
    se = math.sqrt(max(p * (1 - p), 0.0) / n) if (p is not None and n) else None
    return {"band": (max(ok) - min(ok)) if len(ok) >= 2 else 0.0, "se": se, "scores": means, "seeds": seeds,
            "mean": p, "n": n, "temperature": temperature, "task": task}
