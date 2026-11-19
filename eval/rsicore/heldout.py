"""Held-out benchmark evaluation (report only; never fed back to the editor)."""
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


def run_heldout(endpoint, served_name, out_dir: Path, *, domains="all", lora=None, mock=False, limit=None) -> dict:
    """Evaluate every held-out task of `domains` into out_dir (run_eval layout). Tasks already present in
    out_dir/summary.json for the same served model / LoRA / limit are skipped (resumable).
    Returns {task: {accuracy, n, n_pending, ...}}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = out_dir / "summary.json"
    res = {}
    for task, pool in heldout_tasks(domains):
        prev = json.loads(sp.read_text()).get(task) if sp.exists() else None
        served = lora or served_name
        if prev and prev.get("served_model_name") == served and prev.get("limit") == limit \
                and (out_dir / f"{task}.jsonl").exists():
            summ = prev
        else:
            extra = ["--limit", str(limit)] if limit else []
            summ = score(endpoint, served_name, task, pool=pool, ids=None, out_dir=out_dir, lora=lora,
                         extra=extra, mock=mock).summary
        res[task] = {k: summ[k] for k in _KEYS if k in summ}
    return res
