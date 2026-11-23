"""Gate helpers: relative pass test.

(2026-11-23) The noise band (measure_noise_band, fixed_noise_band, noise_band, noise_ids) was removed: every gate
evaluation is one paired comparison with the current model (the papers' gate per candidate), with no band.
"""


def relative_pass(score: float, baseline: float, ratio: float) -> bool:
    """score >= ratio * baseline (for a non-positive baseline: score >= baseline)."""
    if score is None or baseline is None:
        return False
    if baseline <= 0:
        return score >= baseline
    return score >= ratio * baseline
