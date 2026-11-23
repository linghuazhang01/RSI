"""Held-out evaluation (report only; never fed back to the editor): the run's test set simple_<domain>_test.

Decoding: the paper config's `heldout.decoding` with n samples per item (avg@n; per-task n in `n_by_task`);
decoding=None keeps the task's tasks.yaml decoding. heldout.decoding may also set thinking (bool) and max_tokens
(wins over the max_tokens table); both enter the resume key. A held-out task eval:<dataset> (an eval set, pool
`pool`) is built (data/scripts/build_eval_target.py) before it is read. With several replicas of the model its items
are split over them (score_split), so every GPU works on it."""
import json
from pathlib import Path

import yaml

from .config import DOMAINS, EVAL_DIR, ROOT
from .evaluate import score, score_split


def heldout_domains(cfg: dict, domain: str | None = None) -> str:
    """The `domains` argument of run_heldout: "eval:simple_<domain>_test". domain: the run's --domain (rc.domain);
    None = the paper config's `domain`, else its resolved heldout.domains (`run` -> the run's domain)."""
    d = domain or cfg.get("domain") or (cfg.get("heldout") or {}).get("domains")
    if d not in DOMAINS:
        raise ValueError(f"heldout_domains: no run domain (pass the run's --domain), got {d!r}")
    return f"eval:simple_{d}_test"


_KEYS = ("accuracy", "n", "n_pending", "prompt_strict", "avg_at_n", "pass_at_n", "maj_at_n", "em", "f1")


def heldout_tasks(domains) -> list[tuple[str, str]]:
    """[(task, pool)] of a held-out spec: comma-separated eval:<dataset> tokens (heldout_domains), pool `pool`."""
    toks = [d.strip() for d in domains.split(",")] if isinstance(domains, str) else list(domains or [])
    out = []
    for d in (t for t in toks if t):
        if not d.startswith("eval:"):
            raise ValueError(f"held-out task {d!r}: expected eval:<dataset> (the run's simple_<domain>_test)")
        tp = (d.split(":", 1)[1], "pool")
        if tp not in out:
            out.append(tp)
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


def ensure_built(domains) -> None:
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


def run_heldout(endpoint, served_name, out_dir: Path, *, domains, lora=None, mock=False, limit=None,
                decoding: dict | None = None, concurrency: int | None = None) -> dict:
    """endpoint: one URL, or a list of URLs of replicas of the same model (2026-11-20). domains: heldout_domains(cfg).
    With several replicas a single task is split item-wise over all of them (score_split), so every GPU works on it;
    otherwise (one replica, or several tasks) the tasks run one after another on the first replica."""
    ensure_built(domains)
    if isinstance(endpoint, (list, tuple)) and len(endpoint) > 1:
        tasks = heldout_tasks(domains)
        if len(tasks) == 1 and _task_ids(*tasks[0], limit) is not None:
            return _run_heldout_split(list(endpoint), served_name, Path(out_dir), tasks[0], lora=lora, mock=mock,
                                      limit=limit, decoding=decoding, concurrency=concurrency)
    if isinstance(endpoint, (list, tuple)):
        endpoint = endpoint[0]
    return _run_heldout_one(endpoint, served_name, out_dir, domains=domains, lora=lora, mock=mock, limit=limit,
                            decoding=decoding, concurrency=concurrency)


def _task_ids(task: str, pool: str, limit=None) -> list[str] | None:
    """The sorted item ids run_eval would score for (task, pool[, --limit]); None when the split file is missing."""
    tcfg = (yaml.safe_load((EVAL_DIR / "configs" / "tasks.yaml").read_text()).get("tasks") or {}).get(task) or {}
    f = ROOT / "data" / "splits" / tcfg.get("dataset", task) / f"{pool}.json"
    if not f.exists():
        return None
    ids = sorted(set(json.loads(f.read_text())))
    return ids[:limit] if limit else ids


def _run_heldout_split(endpoints, served_name, out_dir: Path, task_pool, *, lora, mock, limit, decoding,
                       concurrency=None) -> dict:
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
                           lora=lora, extra=dec["extra"], mock=mock, concurrency=concurrency, **dec["kw"]).summary
        summ = dict(summ, limit=limit)              # ids replace --limit; keep the resume key of _run_heldout_one
        summ.pop("ids_file", None)
        allsum = json.loads(sp.read_text()) if sp.exists() else {}
        allsum[task] = summ
        sp.write_text(json.dumps(allsum, indent=1, ensure_ascii=False))
    return {task: {k: summ[k] for k in _KEYS if k in summ}}


def _run_heldout_one(endpoint, served_name, out_dir: Path, *, domains, lora=None, mock=False, limit=None,
                     decoding: dict | None = None, tasks: list | None = None, concurrency: int | None = None) -> dict:
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
                         extra=extra, mock=mock, concurrency=concurrency, **dec["kw"]).summary
        res[task] = {k: summ[k] for k in _KEYS if k in summ}
    return res
