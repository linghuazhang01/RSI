"""Gate helpers: noise band of repeated evaluations, relative pass test.

measure_noise_band() scores the same model `repeats` times with sampled decoding (temperature > 0,
seeds base_seed, base_seed+1, ...) through evaluate.score() and returns the spread of the mean scores.
Greedy repeats (temperature 0) are identical on vLLM and in the mock, so with temperature 0 (the default
since 2026-11-20: every evaluation is greedy) nothing is re-run and the band is 0.
fixed_noise_band() reads a fixed band from the config instead (no repeated evaluations).
noise_ids() picks the (smaller) item subset the re-runs score.
"""
import re
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


def fixed_noise_band(spec, dataset: str) -> float | None:
    """A fixed noise band from the config, so the repeated sampled runs can be skipped.
    spec: None (measure as usual) | a number (every dataset) | {key: number} with key = dataset name
    (simple_math_eval), domain name (math) or "default". Returns None when spec gives no value for `dataset`."""
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise ValueError(f"fixed noise band must be a number or a mapping, got {spec!r}")
    if isinstance(spec, (int, float, str)):
        return float(spec)
    if not isinstance(spec, dict):
        raise ValueError(f"fixed noise band must be a number or a mapping, got {spec!r}")
    from .draws import DOMAIN_TRAIN
    domain = next((d for d, dss in DOMAIN_TRAIN.items() if dataset in dss), None)
    if domain is None:          # simple_<domain>_{pool,eval,test,train} (2026-11-22: the only datasets)
        domain = next((d for d in sorted(DOMAIN_TRAIN, key=len, reverse=True)
                       if re.fullmatch(rf"simple_{d}_(pool|eval|test|train)", dataset)), None)
    for k in (dataset, domain, "default"):
        if k is not None and spec.get(k) is not None:
            return float(spec[k])
    return None


def noise_ids(ids: list[str] | None, n: int | None, seed: int) -> list[str] | None:
    """The items the noise re-runs score: a deterministic subset of n ids (all when n is None or >= len).
    A smaller subset gives a wider band (spread ~ 1/sqrt(n)), so the band stays conservative."""
    if ids is None or n is None or int(n) >= len(ids):
        return ids
    import random
    if int(n) < 2:
        raise ValueError(f"noise subset size must be >= 2, got {n!r}")
    return sorted(random.Random(f"{seed}:noise").sample(sorted(ids), int(n)))


def relative_pass(score: float, baseline: float, ratio: float) -> bool:
    """score >= ratio * baseline (for a non-positive baseline: score >= baseline)."""
    if score is None or baseline is None:
        return False
    if baseline <= 0:
        return score >= baseline
    return score >= ratio * baseline


def measure_noise_band(endpoint, served_name, task: str, ids: list[str] | None, out_dir: Path, *, repeats: int = 3,
                       temperature: float = 0.6, top_p: float = 1.0, base_seed: int = 42, mock: bool = False,
                       lora: str | None = None, pool: str = "pool") -> dict:
    """Repeat evaluate.score(task, ids) with seed = base_seed + k (k < repeats) at `temperature` / `top_p`.
    -> {"band": max - min of the item-level mean scores, "se": binomial SE sqrt(p(1-p)/n) of the pooled
    mean p over the n scored items, "scores": [...], "seeds": [...], "mean", "n", "temperature"}.
    Repeat k writes run_eval output to out_dir/seed<seed>/ (pending code tasks give score None).
    temperature <= 0 (greedy): the repeats would be identical, so none is run -> band 0, source 'greedy'."""
    if float(temperature) <= 0:
        return {"band": 0.0, "se": None, "scores": [], "seeds": [], "mean": None, "n": 0, "temperature": 0.0,
                "top_p": top_p, "task": task,
                "source": "greedy (deterministic: repeated runs are identical, nothing re-run)"}
    from .evaluate import score
    out_dir = Path(out_dir)
    seeds = [int(base_seed) + k for k in range(int(repeats))]
    runs = []
    for s in seeds:
        r = score(endpoint, served_name, task, pool=pool, ids=ids, out_dir=out_dir / f"seed{s}", lora=lora,
                  mock=mock, seed=s, temperature=temperature, top_p=top_p)
        runs.append(r.records)
    means = [_mean_score(r) for r in runs]
    ok = [m for m in means if m is not None]
    n = len({r.get("id") for r in runs[0] if r.get("score") is not None}) if runs else 0
    p = sum(ok) / len(ok) if ok else None
    se = math.sqrt(max(p * (1 - p), 0.0) / n) if (p is not None and n) else None
    return {"band": (max(ok) - min(ok)) if len(ok) >= 2 else 0.0, "se": se, "scores": means, "seeds": seeds,
            "mean": p, "n": n, "temperature": temperature, "top_p": top_p, "task": task, "source": "measured"}
