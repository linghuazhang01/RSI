"""Held-out benchmark evaluation (report only; never fed back to the editor).

Decoding (2026-11-20): search, gate and noise are greedy; the held-out evaluation uses the paper config's
`heldout.decoding` (Qwen's non-thinking recommendation: temperature 0.7, top_p 0.8, top_k 20, seed 42) with
n samples per item (avg@n; per-task n in `n_by_task`). decoding=None keeps the task's tasks.yaml decoding.
2026-11-21: heldout.decoding may also set thinking (bool) and max_tokens (wins over the max_tokens table); both enter
the resume key. An eval-set held-out task (eval:<dataset>, e.g. objective simple's simple_<d>_test) is built
(data/scripts/build_eval_target.py) before it is read."""
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

# objective eval_overfit (2026-11-21): heldout.domains `eval_target` = the run's eval set itself (eval_target.dataset,
# e.g. math_eval = MATH-500 + GSM8K test, humanevalplus_eval), task K_eval on pool `pool`, greedy. heldout_domains(cfg)
# turns it into "eval:<dataset>", which heldout_tasks understands (never part of "all").


def heldout_domains(cfg: dict):
    """The `domains` argument of run_heldout for a paper config: heldout.domains, with eval_target resolved."""
    ev = (cfg.get("data_roles") or {}).get("eval")
    if ev:                                          # data_roles.eval (normalised by rsicore.config) wins
        return ev
    d = (cfg.get("heldout") or {}).get("domains", "all")
    if d == "eval_target":
        ds = (cfg.get("eval_target") or {}).get("dataset")
        if not ds:
            raise ValueError("heldout.domains eval_target needs eval_target.dataset")
        return f"eval:{ds}"
    return d


_KEYS = ("accuracy", "n", "n_pending", "prompt_strict", "avg_at_n", "pass_at_n", "maj_at_n", "em", "f1")


def heldout_tasks(domains="all") -> list[tuple[str, str]]:
    if domains in (None, "all"):
        doms = list(HELDOUT)
    elif isinstance(domains, str):
        doms = [d.strip() for d in domains.split(",") if d.strip()]
    else:
        doms = list(domains)
    by_task = {t: (t, p) for v in HELDOUT.values() for t, p in v}
    out = []
    for d in doms:                                  # a domain, a held-out task, or eval:<K_eval> (pool `pool`)
        tps = HELDOUT[d] if d in HELDOUT else [(d.split(":", 1)[1], "pool")] if d.startswith("eval:") \
            else [by_task[d]] if d in by_task else HELDOUT[d]
        out += [tp for tp in tps if tp not in out]
    return out


def decoding_args(decoding: dict | None, task: str) -> dict:
    """score() keyword arguments + extra CLI args for one held-out task from a `heldout.decoding` block, the resume
    key (compared with the stored summary's run_eval decoding) and the same settings as one decoding dict
    (2026-11-21, "decoding", for score(..., decoding=...); absent for a null block). thinking / max_tokens go to
    extra (--thinking, --max-tokens)."""
    if not decoding:
        return {"kw": {}, "extra": [], "key": None}
    n = int((decoding.get("n_by_task") or {}).get(task, decoding.get("n", 1)))
    kw = {k: decoding[k] for k in ("temperature", "top_p", "top_k", "seed") if decoding.get(k) is not None}
    key, extra = dict(kw, n=n), ["--n", str(n)]
    if decoding.get("thinking") is not None:         # only when set: older summaries keep matching
        key["thinking"] = bool(decoding["thinking"])
        extra += ["--thinking", "on" if decoding["thinking"] else "off"]
    if decoding.get("max_tokens") is not None:       # wins over the run-wide max_tokens table
        key["max_tokens"] = int(decoding["max_tokens"])
        extra += ["--max-tokens", str(int(decoding["max_tokens"]))]
    dec = dict(key, **({"min_p": decoding["min_p"]} if decoding.get("min_p") is not None else {}))
    return {"kw": kw, "extra": extra, "key": key, "decoding": dec}


def ensure_built(domains="all") -> None:
    """Build every eval-set dataset among the held-out tasks of `domains` (eval:<dataset>, pool `pool`), so run_eval
    and _task_ids find data/processed/<dataset>.jsonl and data/splits/<dataset>/pool.json (once per process)."""
    from .draws import build_eval_set, is_eval_set
    for task, pool in heldout_tasks(domains):
        if pool == "pool" and is_eval_set(task):
            build_eval_set(task)


def _same_decoding(prev_dec: dict | None, key: dict | None) -> bool:
    if key is None:
        return True
    prev_dec = prev_dec or {}
    return all(prev_dec.get(k) == v for k, v in key.items())


def run_heldout(endpoint, served_name, out_dir: Path, *, domains="all", lora=None, mock=False, limit=None,
                decoding: dict | None = None) -> dict:
    """endpoint: one URL, or a list of URLs of replicas of the same model (2026-11-20): the tasks are then run
    in parallel, one replica per task (each task in out_dir/_par/<task>/, merged into out_dir afterwards).
    A single task (eval_target: the run's eval set, 2026-11-21) is instead split item-wise over all
    replicas (score_split), so every GPU works on it."""
    ensure_built(domains)
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
