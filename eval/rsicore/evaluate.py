"""Scoring (via eval/run_eval.py) and paired bootstrap.

score() runs eval/run_eval.py in a subprocess with the eval interpreter (config.eval_python()),
--ids-file (temp file under out_dir/_ids), --no-cache and --quiet. Use one out_dir per (model, stage):
run_eval merges into <out_dir>/summary.json, so two concurrent score() calls must not share an out_dir.
Code tasks stay status=pending_sandbox (score None) when Docker is unavailable; mock mode always
passes --no-sandbox (never starts Docker). seed / temperature override the task's decoding (run_eval
--seed / --temperature); the seed only reaches the server for sampled decoding (temperature > 0).
set_max_tokens() installs a run-wide {task: max generation tokens} table (the paper config's `max_tokens`);
every score() call of a listed task then passes run_eval --max-tokens (unless `extra` already does).
score_split() splits one scoring job over several replicas of the same model (data parallelism).

decoding (2026-11-21; score / score_split `decoding=`): a dict with any of temperature, top_p, top_k, min_p (0 only),
n, thinking, max_tokens, seed (rsicore.config.check_decoding / stage_decoding); None = the task's tasks.yaml
decoding. The explicit seed / temperature / top_p / top_k keyword arguments win over it; its max_tokens wins over
the run-wide table; `extra` CLI flags win over everything. n > 1: every record's "score" is the mean of its n
samples (avg@n, in [0, 1]) and "scores" the per-sample list; the summary accuracy is the mean of the record scores.
thinking: run_eval --thinking on (chat_template_kwargs enable_thinking); only the final answer is graded (the
servers' reasoning parser moves the thinking out of `content`; run_eval strips any <think> block left in it), and a
sample cut at max_tokens while still thinking (finish_reason length, no reasoning_content, no </think>) scores 0.

(2026-11-23) The prompt-logprob token KL (reference_logprobs / token_kl, Paper A's KL score term) was removed.
"""
import hashlib
import json
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import EVAL_DIR, check_decoding, eval_python

RUN_EVAL = EVAL_DIR / "run_eval.py"
TASKS_YAML = EVAL_DIR / "configs" / "tasks.yaml"
_MAX_TOKENS: dict[str, int] = {}


def set_max_tokens(spec: dict | None) -> dict[str, int]:
    """Run-wide max generation length per run_eval task (paper config `max_tokens: {task: tokens}`), overriding
    eval/configs/tasks.yaml for this process; null / {} = keep tasks.yaml. Returns the installed table."""
    import yaml
    spec = {k: v for k, v in (spec or {}).items() if v is not None}
    from .evalsets import task_names
    known = task_names()                                # tasks.yaml tasks + the eval-set tasks (simple_*)
    bad = sorted(set(spec) - known)
    if bad:
        raise SystemExit(f"max_tokens: unknown task(s) {bad} (see eval/configs/tasks.yaml)")
    out = {}
    for k, v in spec.items():
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise SystemExit(f"max_tokens.{k} must be a positive integer, got {v!r}")
        out[k] = int(v)
    _MAX_TOKENS.clear()
    _MAX_TOKENS.update(out)
    return dict(_MAX_TOKENS)


@dataclass
class EvalResult:
    summary: dict              # this task's entry of run_eval's summary.json
    records: list[dict]        # per-unit records (id, score, ...)
    out_dir: Path

    @property
    def accuracy(self):
        return self.summary.get("accuracy")


def _decoding_flags(task: str, dec: dict, extra) -> list[str]:
    """run_eval flags of a merged decoding dict (a flag already in `extra` is left to it)."""
    xs, cmd = [str(x) for x in extra], []
    for k, flag, cast in (("seed", "--seed", int), ("temperature", "--temperature", float),
                          ("top_p", "--top-p", float), ("top_k", "--top-k", int), ("n", "--n", int)):
        if dec.get(k) is not None and flag not in xs:
            cmd += [flag, str(cast(dec[k]))]
    if dec.get("thinking") is not None and "--thinking" not in xs:
        cmd += ["--thinking", "on" if dec["thinking"] else "off"]
    mt = dec.get("max_tokens") or _MAX_TOKENS.get(task)      # an explicit max_tokens wins over the table
    if mt and "--max-tokens" not in xs:
        cmd += ["--max-tokens", str(int(mt))]
    return cmd


def _as_list(v, n: int) -> list:
    return list(v) if isinstance(v, list) else [v] * n


def _normalise(task: str, summary: dict, recs: list[dict], thinking: bool, out_dir: Path) -> dict:
    """One place for n > 1 and thinking (2026-11-21): records with several samples get "scores" (their per-sample
    list; "score" is already the mean, run_eval's avg@n); with thinking on, a sample cut at max_tokens while still
    thinking scores 0 (record score, summary and the files on disk are updated). Returns the summary."""
    changed, multi = False, False
    for r in recs:
        ss = r.get("sample_scores") or []
        if thinking and ss:
            fr = r.get("finish_reason") or []
            outs, reas = _as_list(r.get("raw_output"), len(ss)), _as_list(r.get("reasoning"), len(ss))
            cut = [i for i in range(len(ss)) if ss[i] and i < len(fr) and fr[i] == "length" and not reas[i]
                   and "</think>" not in str(outs[i] or "")]
            for i in cut:
                ss[i] = 0.0
            if cut:
                r["truncated_thinking"] = len(cut)
                r["score"] = (sum(ss) / len(ss)) if all(x is not None for x in ss) else None
                changed = True
        if len(ss) > 1:
            r["scores"] = list(ss)
            multi = True
    if not (changed or multi):
        return summary
    if changed:
        graded = [r for r in recs if r.get("score") is not None]
        per = _per_id(recs)
        summary = dict(summary, accuracy=(sum(r["score"] for r in graded) / len(graded)) if graded else None,
                       item_accuracy=(sum(per.values()) / len(per)) if per else None,
                       truncated_thinking=sum(r.get("truncated_thinking", 0) for r in recs))
        if multi:
            summary["avg_at_n"] = summary["accuracy"]
            summary["pass_at_n"] = (sum(float(any(x for x in r["sample_scores"] if x)) for r in graded) / len(graded)) \
                if graded else None
        sp = out_dir / "summary.json"
        allsum = json.loads(sp.read_text()) if sp.exists() else {}
        allsum[task] = summary
        sp.write_text(json.dumps(allsum, indent=1, ensure_ascii=False))
    (out_dir / f"{task}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    return summary


def score(endpoint, served_name, task: str, *, pool: str, ids: list[str] | None, out_dir: Path,
          lora: str | None = None, extra: list[str] = (), mock: bool = False,
          seed: int | None = None, temperature: float | None = None, top_p: float | None = None,
          top_k: int | None = None, decoding: dict | None = None, concurrency: int | None = None) -> EvalResult:
    """Score `task` items `ids` (None = the whole pool) on one endpoint. decoding: see the module doc.
    concurrency: run_eval --concurrency (requests in flight, each of n samples; None = run_eval's default 64). It
    changes only how many requests the server batches at once, not the requests or their seeds (2026-11-23)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dec = dict(check_decoding(decoding) or {})
    for k, v in (("seed", seed), ("temperature", temperature), ("top_p", top_p), ("top_k", top_k)):
        if v is not None:                               # explicit keyword arguments win over `decoding`
            dec[k] = v
    cmd = [eval_python(), "-I", str(RUN_EVAL), "--task", task, "--pool", pool, "--endpoint", endpoint,
           "--model", served_name, "--out", str(out_dir), "--no-cache", "--quiet"]
    if ids is not None:
        ids = sorted(set(ids))
        if not ids:
            return EvalResult({"accuracy": None, "n": 0, "n_pending": 0, "note": "empty id list"}, [], out_dir)
        blob = json.dumps(ids)
        idf = out_dir / "_ids" / f"{task}.{hashlib.sha256(blob.encode()).hexdigest()[:12]}.json"
        idf.parent.mkdir(exist_ok=True)
        idf.write_text(blob)
        cmd += ["--ids-file", str(idf)]
    if lora:
        cmd += ["--lora", lora]
    cmd += _decoding_flags(task, dec, extra)
    if concurrency and "--concurrency" not in [str(x) for x in extra]:
        cmd += ["--concurrency", str(int(concurrency))]
    if mock and "--no-sandbox" not in extra:
        cmd += ["--no-sandbox"]
    cmd += [str(x) for x in extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"run_eval failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    summ_all = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {}
    if task not in summ_all:  # e.g. "no items in pool"
        return EvalResult({"accuracy": None, "n": 0, "n_pending": 0, "note": r.stdout[-500:]}, [], out_dir)
    recs = [json.loads(l) for l in (out_dir / f"{task}.jsonl").read_text().splitlines() if l.strip()]
    xs = [str(x) for x in extra]
    thinking = (xs[xs.index("--thinking") + 1] == "on") if "--thinking" in xs[:-1] else bool(dec.get("thinking"))
    return EvalResult(_normalise(task, summ_all[task], recs, thinking, out_dir), recs, out_dir)


def score_split(endpoints: list[str], served_name, task: str, *, pool: str, ids: list[str] | None, out_dir: Path,
                lora: str | None = None, extra: list[str] = (), mock: bool = False, seed: int | None = None,
                temperature: float | None = None, top_p: float | None = None,
                top_k: int | None = None, decoding: dict | None = None,
                concurrency: int | None = None) -> EvalResult:
    """score() with the items split over several replicas of the same model (2026-11-20: data parallelism on
    every GPU). endpoints: one URL per replica. Each replica scores a contiguous chunk of the sorted ids into
    out_dir/_part<k>/; the records are merged (sorted by id) into out_dir/<task>.jsonl and a summary entry
    (accuracy = item-level mean, n, n_pending; avg_at_n / pass_at_n when n > 1) is merged into out_dir/summary.json.
    One endpoint or ids None (the whole pool) -> plain score() on the first endpoint. decoding: see the module doc
    (every replica gets the same dict, seed included)."""
    eps = [e for e in endpoints if e]
    if len(eps) <= 1 or ids is None or len(set(ids)) < 2:
        return score(eps[0], served_name, task, pool=pool, ids=ids, out_dir=out_dir, lora=lora, extra=extra,
                     mock=mock, seed=seed, temperature=temperature, top_p=top_p, top_k=top_k, decoding=decoding,
                     concurrency=concurrency)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(ids))
    k = min(len(eps), len(ids))
    size = math.ceil(len(ids) / k)
    chunks = [ids[i * size:(i + 1) * size] for i in range(k)]

    def run(i):
        return score(eps[i], served_name, task, pool=pool, ids=chunks[i], out_dir=out_dir / f"_part{i}", lora=lora,
                     extra=extra, mock=mock, seed=seed, temperature=temperature, top_p=top_p, top_k=top_k,
                     decoding=decoding, concurrency=concurrency)
    with ThreadPoolExecutor(max_workers=k) as ex:
        parts = list(ex.map(run, range(k)))
    recs = sorted((r for p in parts for r in p.records), key=lambda r: (str(r.get("id")), str(r.get("unit_id", ""))))
    per = _per_id(recs)
    pend = sum(1 for r in recs if r.get("score") is None)
    summ = dict(parts[0].summary)
    for key in ("avg_at_n", "pass_at_n", "maj_at_n", "em", "f1", "prompt_strict", "note", "item_accuracy",
                "truncated_thinking"):
        summ.pop(key, None)
    summ.update({"accuracy": (sum(per.values()) / len(per)) if per else None, "n": len({r.get("id") for r in recs}),
                 "n_pending": pend, "split_over_replicas": k})
    graded = [r for r in recs if r.get("score") is not None]
    if any("scores" in r for r in recs):                # n > 1: keep avg@n / pass@n
        summ["avg_at_n"] = summ["accuracy"]
        summ["pass_at_n"] = (sum(float(any(x for x in r.get("sample_scores") or [] if x)) for r in graded)
                             / len(graded)) if graded else None
    if any(r.get("truncated_thinking") for r in recs):
        summ["truncated_thinking"] = sum(r.get("truncated_thinking", 0) for r in recs)
    (out_dir / f"{task}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    sp = out_dir / "summary.json"
    allsum = json.loads(sp.read_text()) if sp.exists() else {}
    allsum[task] = summ
    sp.write_text(json.dumps(allsum, indent=1, ensure_ascii=False))
    return EvalResult(summ, recs, out_dir)


# ----------------------------------------------------------------------------- paired bootstrap
def _per_id(records: list[dict]) -> dict:
    per = {}
    for r in records:
        if r.get("score") is not None:
            per.setdefault(r["id"], []).append(float(r["score"]))
    return {k: sum(v) / len(v) for k, v in per.items()}


def paired_delta(a: list[dict], b: list[dict], seed: int = 42, n_boot: int = 1000) -> dict:
    """delta = mean over shared ids of (b - a) (b = candidate, a = reference); 95% percentile bootstrap
    CI resampling ids. Units of one id are averaged first; pending (score None) records are skipped."""
    pa, pb = _per_id(a), _per_id(b)
    common = sorted(set(pa) & set(pb))
    if not common:
        return {"delta": None, "ci": [None, None], "n": 0}
    d = [pb[i] - pa[i] for i in common]
    n = len(d)
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * (n_boot - 1))], boots[int(math.ceil(0.975 * (n_boot - 1)))]
    return {"delta": sum(d) / n, "ci": [lo, hi], "n": n}
