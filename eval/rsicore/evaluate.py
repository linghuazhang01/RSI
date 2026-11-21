"""Scoring (via eval/run_eval.py), paired bootstrap, and prompt-logprob token KL.

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

Token KL uses vLLM's /v1/completions with `prompt_logprobs=k` (an extra sampling param that vLLM's
OpenAI server accepts in the request body); the response's choices[0].prompt_logprobs is a list
(first entry null) of {token_id: {"logprob", "rank", "decoded_token"}} with the top-k tokens plus the
actual prompt token. Server must allow k <= --max-logprobs (vLLM default 20).
"""
import hashlib
import json
import math
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import _net
from .config import EVAL_DIR, check_decoding, eval_python
from .draws import DOMAIN_TRAIN

RUN_EVAL = EVAL_DIR / "run_eval.py"
TASKS_YAML = EVAL_DIR / "configs" / "tasks.yaml"
_MAX_TOKENS: dict[str, int] = {}


def set_max_tokens(spec: dict | None) -> dict[str, int]:
    """Run-wide max generation length per run_eval task (paper config `max_tokens: {task: tokens}`), overriding
    eval/configs/tasks.yaml for this process; null / {} = keep tasks.yaml. Returns the installed table."""
    import yaml
    spec = {k: v for k, v in (spec or {}).items() if v is not None}
    from .evalsets import task_names
    known = task_names()                                # tasks.yaml tasks + the eval-set targets K_eval
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


def domain_eval_tasks(domain: str) -> list[str]:
    if domain == "all":
        return [t for d in DOMAIN_TRAIN.values() for t in d]
    return list(DOMAIN_TRAIN[domain])


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
          top_k: int | None = None, decoding: dict | None = None) -> EvalResult:
    """Score `task` items `ids` (None = the whole pool) on one endpoint. decoding: see the module doc."""
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
                top_k: int | None = None, decoding: dict | None = None) -> EvalResult:
    """score() with the items split over several replicas of the same model (2026-11-20: data parallelism on
    every GPU). endpoints: one URL per replica. Each replica scores a contiguous chunk of the sorted ids into
    out_dir/_part<k>/; the records are merged (sorted by id) into out_dir/<task>.jsonl and a summary entry
    (accuracy = item-level mean, n, n_pending; avg_at_n / pass_at_n when n > 1) is merged into out_dir/summary.json.
    One endpoint or ids None (the whole pool) -> plain score() on the first endpoint. decoding: see the module doc
    (every replica gets the same dict, seed included)."""
    eps = [e for e in endpoints if e]
    if len(eps) <= 1 or ids is None or len(set(ids)) < 2:
        return score(eps[0], served_name, task, pool=pool, ids=ids, out_dir=out_dir, lora=lora, extra=extra,
                     mock=mock, seed=seed, temperature=temperature, top_p=top_p, top_k=top_k, decoding=decoding)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(ids))
    k = min(len(eps), len(ids))
    size = math.ceil(len(ids) / k)
    chunks = [ids[i * size:(i + 1) * size] for i in range(k)]

    def run(i):
        return score(eps[i], served_name, task, pool=pool, ids=chunks[i], out_dir=out_dir / f"_part{i}", lora=lora,
                     extra=extra, mock=mock, seed=seed, temperature=temperature, top_p=top_p, top_k=top_k,
                     decoding=decoding)
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


# ----------------------------------------------------------------------------- prompt logprobs / KL
def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _prompt_logprobs(endpoint: str, model: str, text: str, top_k: int) -> list:
    """-> list over positions of [ids], [logprobs] (position 0 has none)."""
    url = endpoint.rstrip("/") + "/completions"
    payload = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": top_k}
    last = None
    for attempt in range(4):
        try:
            resp = _net.post_json(url, payload, timeout=1800)
            break
        except (_net.HTTPError, OSError) as e:
            last = e
            if isinstance(e, _net.HTTPError) and e.status < 500:
                raise
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"prompt_logprobs request failed: {last}")
    ch = (resp.get("choices") or [{}])[0]
    plp = ch.get("prompt_logprobs") if ch.get("prompt_logprobs") is not None else resp.get("prompt_logprobs")
    if plp is None:
        raise RuntimeError("server returned no prompt_logprobs (is this vLLM >= 0.6 / the mock server?)")
    out = []
    for pos in plp:
        if not pos:
            out.append([[], []])
            continue
        items = sorted(pos.items(), key=lambda kv: kv[1].get("rank") or 10 ** 9)
        out.append([[int(k) for k, _ in items], [round(float(v["logprob"]), 5) for _, v in items]])
    return out


def _fetch_many(endpoint, model, texts, top_k, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda t: _prompt_logprobs(endpoint, model, t, top_k), texts))


def _load_cache(cache: Path) -> dict:
    d = {}
    if Path(cache).exists():
        for line in Path(cache).read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d[(r["sha"], r["top_k"])] = r["pos"]
    return d


def reference_logprobs(endpoint, served_name, texts: list[str], cache: Path, top_k: int = 20) -> Path:
    """Fetch (once) the reference model's prompt logprobs for `texts`; cached in `cache` (jsonl of
    {"sha", "top_k", "pos": [[ids], [logprobs]] per position})."""
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = _load_cache(cache)
    todo = [t for t in dict.fromkeys(texts) if (_sha(t), top_k) not in have]
    if todo:
        res = _fetch_many(endpoint, served_name, todo, top_k)
        with open(cache, "a") as f:
            for t, pos in zip(todo, res):
                f.write(json.dumps({"sha": _sha(t), "top_k": top_k, "model": served_name, "pos": pos}) + "\n")
    return cache


def _kl_position(ref_ids, ref_lps, cand: dict) -> float:
    """KL(ref || cand) on the partition {each ref-reported token} + {rest}. Tokens missing from the
    candidate's list get the candidate's smallest reported logprob (an upper bound on their prob)."""
    if not ref_ids:
        return 0.0
    floor = min(cand.values()) if cand else -30.0
    kl, P, Q = 0.0, 0.0, 0.0
    for i, lp in zip(ref_ids, ref_lps):
        p, q_lp = math.exp(lp), cand.get(i, floor)
        kl += p * (lp - q_lp)
        P += p
        Q += math.exp(q_lp)
    rp, rq = max(0.0, 1.0 - P), max(1e-12, 1.0 - min(Q, 1.0 - 1e-9))
    if rp > 1e-12:
        kl += rp * (math.log(rp) - math.log(rq))
    return max(0.0, kl)


def token_kl(ref_cache: Path, endpoint, served_name, texts: list[str], lora: str | None = None,
             top_k: int = 20) -> dict:
    """Top-k approximate token KL(ref || cand) per text (mean over positions); "mean" = mean over texts,
    "token_mean" = mean over all positions. Candidate = `lora` if given, else `served_name`."""
    ref = _load_cache(ref_cache)
    missing = [t for t in texts if (_sha(t), top_k) not in ref]
    if missing:
        raise KeyError(f"{len(missing)} texts not in reference cache {ref_cache}; call reference_logprobs first")
    cand_all = _fetch_many(endpoint, lora or served_name, texts, top_k)
    per_text, tot, ntok = [], 0.0, 0
    for t, cpos in zip(texts, cand_all):
        rpos = ref[(_sha(t), top_k)]
        L = min(len(rpos), len(cpos))
        kls = [_kl_position(rpos[j][0], rpos[j][1], dict(zip(cpos[j][0], cpos[j][1]))) for j in range(L) if rpos[j][0]]
        per_text.append(sum(kls) / len(kls) if kls else 0.0)
        tot += sum(kls)
        ntok += len(kls)
    return {"mean": sum(per_text) / len(per_text) if per_text else 0.0, "token_mean": tot / ntok if ntok else 0.0,
            "per_text": per_text, "n_texts": len(texts), "n_tokens": ntok, "top_k": top_k}
