"""Held-out benchmark evaluation (report only; never fed back to the editor).

Decoding (2026-11-20): search, gate and noise are greedy; the held-out evaluation uses the paper config's
`heldout.decoding` (Qwen's non-thinking recommendation: temperature 0.7, top_p 0.8, top_k 20, seed 42) with
n samples per item (avg@n; per-task n in `n_by_task`). decoding=None keeps the task's tasks.yaml decoding."""
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from .config import EVAL_DIR, ROOT
from .evaluate import score, score_split

HELDOUT = {"math": [("math500", "test"), ("gsm8k", "test"), ("aime24", "test"), ("aime25", "test")],
           "code": [("humanevalplus", "test"), ("mbppplus", "test"), ("livecodebench", "test")],
           "knowledge": [("gpqa_diamond", "test"), ("mmlu_pro", "test"), ("triviaqa", "test")],
           "instruction_following": [("ifeval", "test"), ("ifbench", "test")],
           "tool_calling": [("sealtool", "test"), ("apibank", "test"), ("bfcl", "test")]}

# Not part of "all": selected by name only. eval_target = the merged eval set an eval_overfit run optimises on
# (2026-11-21: the run is evaluated on exactly those 1819 items, MATH-500 + GSM8K test).
HELDOUT_EXTRA = {"eval_target": [("math_eval", "pool")]}

_KEYS = ("accuracy", "n", "n_pending", "prompt_strict", "avg_at_n", "pass_at_n", "maj_at_n", "em", "f1")


def heldout_tasks(domains="all") -> list[tuple[str, str]]:
    if domains in (None, "all"):
        doms = list(HELDOUT)
    elif isinstance(domains, str):
        doms = [d.strip() for d in domains.split(",") if d.strip()]
    else:
        doms = list(domains)
    return [tp for d in doms for tp in (HELDOUT[d] if d in HELDOUT else HELDOUT_EXTRA[d])]


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
    """endpoint: one URL, or a list of URLs of replicas of the same model (2026-11-20): the tasks are then run
    in parallel, one replica per task (each task in out_dir/_par/<task>/, merged into out_dir afterwards).
    A single task (eval_target: the 1819 math_eval items, 2026-11-21) is instead split item-wise over all
    replicas (score_split), so every GPU works on it."""
    if isinstance(endpoint, (list, tuple)) and len(endpoint) > 1:
        tasks = heldout_tasks(domains)
        if len(tasks) == 1 and _task_ids(*tasks[0], limit) is not None:
            return _run_heldout_split(list(endpoint), served_name, Path(out_dir), tasks[0], lora=lora, mock=mock,
                                      limit=limit, decoding=decoding)
        return _run_heldout_parallel(list(endpoint), served_name, Path(out_dir), domains=domains, lora=lora,
                                     mock=mock, limit=limit, decoding=decoding)
    if isinstance(endpoint, (list, tuple)):
        endpoint = endpoint[0]
    return _run_heldout_one(endpoint, served_name, out_dir, domains=domains, lora=lora, mock=mock, limit=limit,
                            decoding=decoding)


def _task_ids(task: str, pool: str, limit=None) -> list[str] | None:
    """The sorted item ids run_eval would score for (task, pool[, --limit]); None when the split file is missing."""
    tcfg = (yaml.safe_load((EVAL_DIR / "configs" / "tasks.yaml").read_text()).get("tasks") or {}).get(task) or {}
    f = ROOT / "data" / "splits" / tcfg.get("dataset", task) / f"{pool}.json"
    if not f.exists():
        return None
    ids = sorted(set(json.loads(f.read_text())))
    return ids[:limit] if limit else ids


def _run_heldout_split(endpoints, served_name, out_dir: Path, task_pool, *, lora, mock, limit, decoding) -> dict:
    """One held-out task, its items split over the replicas (resumable like _run_heldout_one)."""
    task, pool = task_pool
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = out_dir / "summary.json"
    served = lora or served_name
    dec = decoding_args(decoding, task)
    prev = json.loads(sp.read_text()).get(task) if sp.exists() else None
    if prev and prev.get("served_model_name") == served and prev.get("limit") == limit \
            and _same_decoding(prev.get("decoding"), dec["key"]) and (out_dir / f"{task}.jsonl").exists():
        summ = prev
    else:
        summ = score_split(endpoints, served_name, task, pool=pool, ids=_task_ids(task, pool, limit), out_dir=out_dir,
                           lora=lora, extra=dec["extra"], mock=mock, **dec["kw"]).summary
        summ = dict(summ, limit=limit)              # ids replace --limit; keep the resume key of _run_heldout_one
        summ.pop("ids_file", None)
        allsum = json.loads(sp.read_text()) if sp.exists() else {}
        allsum[task] = summ
        sp.write_text(json.dumps(allsum, indent=1, ensure_ascii=False))
    return {task: {k: summ[k] for k in _KEYS if k in summ}}


def _run_heldout_parallel(endpoints, served_name, out_dir: Path, *, domains, lora, mock, limit, decoding) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = heldout_tasks(domains)
    lock = threading.Lock()
    queue = list(tasks)

    def worker(ep):
        res = {}
        while True:
            with lock:
                if not queue:
                    return res
                task, pool = queue.pop(0)
            sub = out_dir / "_par" / task
            r = _run_heldout_one(ep, served_name, sub, tasks=[(task, pool)], lora=lora, mock=mock, limit=limit,
                                 decoding=decoding)
            with lock:                              # merge this task into out_dir (summary.json + records)
                sp, ssp = out_dir / "summary.json", sub / "summary.json"
                allsum = json.loads(sp.read_text()) if sp.exists() else {}
                if ssp.exists():
                    allsum[task] = json.loads(ssp.read_text()).get(task, {})
                    sp.write_text(json.dumps(allsum, indent=1, ensure_ascii=False))
                if (sub / f"{task}.jsonl").exists():
                    shutil.copyfile(sub / f"{task}.jsonl", out_dir / f"{task}.jsonl")
            res.update(r)
    with ThreadPoolExecutor(max_workers=len(endpoints)) as ex:
        parts = list(ex.map(worker, endpoints))
    out = {}
    for task, _ in tasks:                           # task order as in HELDOUT
        for p in parts:
            if task in p:
                out[task] = p[task]
    return out


def _run_heldout_one(endpoint, served_name, out_dir: Path, *, domains="all", lora=None, mock=False, limit=None,
                     decoding: dict | None = None, tasks: list | None = None) -> dict:
    """Evaluate every held-out task of `domains` into out_dir (run_eval layout). Tasks already present in
    out_dir/summary.json for the same served model / LoRA / limit / decoding are skipped (resumable).
    decoding: the paper config's heldout.decoding (see the module doc); None = tasks.yaml decoding.
    Returns {task: {accuracy, n, n_pending, ...}} (accuracy = avg@n when n > 1)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = out_dir / "summary.json"
    res = {}
    for task, pool in (tasks if tasks is not None else heldout_tasks(domains)):
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
