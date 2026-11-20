"""Held-out benchmark evaluation (report only; never fed back to the editor).

Decoding (2026-11-20): search, gate and noise are greedy; the held-out evaluation uses the paper config's
`heldout.decoding` (Qwen's non-thinking recommendation: temperature 0.7, top_p 0.8, top_k 20, seed 42) with
n samples per item (avg@n; per-task n in `n_by_task`). decoding=None keeps the task's tasks.yaml decoding."""
import json
from pathlib import Path

from .evaluate import score

HELDOUT = {"math": [("math500", "test"), ("gsm8k", "test"), ("aime24", "test"), ("aime25", "test")],
           "code": [("humanevalplus", "test"), ("mbppplus", "test"), ("livecodebench", "test")],
           "knowledge": [("gpqa_diamond", "test"), ("mmlu_pro", "test"), ("triviaqa", "test")],
           "instruction_following": [("ifeval", "test"), ("ifbench", "test")],
           "tool_calling": [("sealtool", "test"), ("apibank", "test"), ("bfcl", "test")]}

_KEYS = ("accuracy", "n", "n_pending", "prompt_strict", "avg_at_n", "pass_at_n", "maj_at_n", "em", "f1")


def heldout_tasks(domains="all") -> list[tuple[str, str]]:
    if domains in (None, "all"):
        doms = list(HELDOUT)
    elif isinstance(domains, str):
        doms = [d.strip() for d in domains.split(",") if d.strip()]
    else:
        doms = list(domains)
    return [tp for d in doms for tp in HELDOUT[d]]


def decoding_args(decoding: dict | None, task: str) -> dict:
    """score() keyword arguments + extra CLI args for one held-out task from a `heldout.decoding` block."""
    if not decoding:
        return {"kw": {}, "extra": [], "key": None}
    n = int((decoding.get("n_by_task") or {}).get(task, decoding.get("n", 1)))
    kw = {k: decoding[k] for k in ("temperature", "top_p", "top_k", "seed") if decoding.get(k) is not None}
    key = dict(kw, n=n)
    return {"kw": kw, "extra": ["--n", str(n)], "key": key}


def _same_decoding(prev_dec: dict | None, key: dict | None) -> bool:
    if key is None:
        return True
    prev_dec = prev_dec or {}
    return all(prev_dec.get(k) == v for k, v in key.items())


def run_heldout(endpoint, served_name, out_dir: Path, *, domains="all", lora=None, mock=False, limit=None,
                decoding: dict | None = None) -> dict:
    """Evaluate every held-out task of `domains` into out_dir (run_eval layout). Tasks already present in
    out_dir/summary.json for the same served model / LoRA / limit / decoding are skipped (resumable).
    decoding: the paper config's heldout.decoding (see the module doc); None = tasks.yaml decoding.
    Returns {task: {accuracy, n, n_pending, ...}} (accuracy = avg@n when n > 1)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = out_dir / "summary.json"
    res = {}
    for task, pool in heldout_tasks(domains):
        prev = json.loads(sp.read_text()).get(task) if sp.exists() else None
        served = lora or served_name
        dec = decoding_args(decoding, task)
        if prev and prev.get("served_model_name") == served and prev.get("limit") == limit \
                and _same_decoding(prev.get("decoding"), dec["key"]) and (out_dir / f"{task}.jsonl").exists():
            summ = prev
        else:
            extra = (["--limit", str(limit)] if limit else []) + dec["extra"]
            summ = score(endpoint, served_name, task, pool=pool, ids=None, out_dir=out_dir, lora=lora,
                         extra=extra, mock=mock, **dec["kw"]).summary
        res[task] = {k: summ[k] for k in _KEYS if k in summ}
    return res
