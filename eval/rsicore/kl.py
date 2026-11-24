"""Token-level KL of a served model to a reference model on fixed texts (2026-11-24, Paper A's KL gate).

A text is a reference trace: a chat prompt plus the reference model's own answer. The served models read it
(teacher forcing, nothing is sampled): vLLM's /v1/completions with `prompt_logprobs=k` returns, for every prompt
position, the top-k next-token logprobs (plus the actual token). The reference model's top-k is cached once
(reference_logprobs); a candidate's KL(reference || candidate) is computed per position on the partition {each
reference top-k token} + {the rest} (a lower bound of the full KL by the log-sum inequality, as QDA's evaluator;
a reference token missing from the candidate's top-k gets the candidate's smallest reported logprob, an upper bound on
its probability). Only the answer positions count (the prompt's token count comes from vLLM's /tokenize), averaged
per trace. The server must allow k <= --max-logprobs (the run configs set 20).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import _net


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _post(url: str, payload: dict, timeout: float = 3600) -> dict:
    last = None
    for attempt in range(4):
        try:
            return _net.post_json(url, payload, timeout=timeout)
        except (_net.HTTPError, OSError) as e:
            last = e
            if isinstance(e, _net.HTTPError) and e.status < 500:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request to {url} failed: {last}")


def _n_prompt_tokens(endpoint: str, model: str, prompt: str) -> int:
    """Token count of the prompt part (vLLM /tokenize, served at the server root, not under /v1)."""
    root = endpoint.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    r = _post(root + "/tokenize", {"model": model, "prompt": prompt}, timeout=600)
    if r.get("count") is None and r.get("tokens") is None:
        raise RuntimeError(f"/tokenize returned no count: {str(r)[:200]}")
    return int(r["count"] if r.get("count") is not None else len(r["tokens"]))


def _prompt_logprobs(endpoint: str, model: str, text: str, top_k: int) -> list:
    """-> per position [[ids], [logprobs]] sorted by rank (position 0 has none)."""
    resp = _post(endpoint.rstrip("/") + "/completions",
                 {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": top_k})
    ch = (resp.get("choices") or [{}])[0]
    plp = ch.get("prompt_logprobs") if ch.get("prompt_logprobs") is not None else resp.get("prompt_logprobs")
    if plp is None:
        raise RuntimeError("server returned no prompt_logprobs (vLLM >= 0.6 or the mock server is needed)")
    out = []
    for pos in plp:
        if not pos:
            out.append([[], []])
            continue
        items = sorted(pos.items(), key=lambda kv: kv[1].get("rank") or 10 ** 9)[:top_k]
        out.append([[int(k) for k, _ in items], [round(float(v["logprob"]), 6) for _, v in items]])
    return out


def _spread(endpoints, fn, jobs: list, workers_per_endpoint: int = 8) -> list:
    """fn(endpoint, job) for every job, spread round-robin over the endpoints (replicas of one model)."""
    eps = list(endpoints) if isinstance(endpoints, (list, tuple)) else [endpoints]
    with ThreadPoolExecutor(max_workers=max(1, len(eps) * workers_per_endpoint)) as ex:
        return list(ex.map(lambda ij: fn(eps[ij[0] % len(eps)], ij[1]), enumerate(jobs)))


def _load(cache: Path) -> dict:
    d = {}
    if Path(cache).exists():
        for line in Path(cache).read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d[(r["sha"], r["top_k"])] = r
    return d


def reference_logprobs(endpoints, served_name: str, items: list[dict], cache: Path, top_k: int = 20) -> Path:
    """The reference model's top-k prompt logprobs on every trace ({"text", "prompt"}), fetched once and cached in
    `cache` (jsonl {sha, top_k, n_prompt, pos}); a resume fetches only what is missing."""
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = _load(cache)
    todo = list({_sha(it["text"]): it for it in items if (_sha(it["text"]), top_k) not in have}.values())
    if todo:
        lock = threading.Lock()

        if cache.exists() and cache.stat().st_size and not cache.read_bytes().endswith(b"\n"):
            with open(cache, "a") as f:            # a torn last line (crash mid-write): never glue a record to it
                f.write("\n")

        def one(ep, it):
            rec = {"sha": _sha(it["text"]), "top_k": top_k, "model": served_name,
                   "n_prompt": _n_prompt_tokens(ep, served_name, it["prompt"]),
                   "pos": _prompt_logprobs(ep, served_name, it["text"], top_k)}
            with lock, open(cache, "a") as f:
                f.write(json.dumps(rec) + "\n")
        _spread(endpoints, one, todo)
    have = _load(cache)                            # every trace must be cached before the step is marked done
    miss = [it for it in items if (_sha(it["text"]), top_k) not in have]
    if miss:
        raise RuntimeError(f"reference logprobs missing for {len(miss)} of {len(items)} traces in {cache}")
    return cache


def _kl_position(ref_ids, ref_lps, cand: dict) -> float:
    """KL(ref || cand) on the partition {each reference top-k token} + {the rest}."""
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


def token_kl(ref_cache: Path, endpoints, served_name: str, texts: list[str], top_k: int = 20) -> dict:
    """KL(reference || served model) per text, mean over its ANSWER positions (after the cached n_prompt tokens).
    -> {"mean" (over texts), "token_mean" (over all answer positions), "per_text", "n_positions" (per text),
    "n_texts", "n_tokens", "top_k"}. Every text must be in the reference cache (reference_logprobs first)."""
    ref = _load(ref_cache)
    missing = [t for t in texts if (_sha(t), top_k) not in ref]
    if missing:
        raise KeyError(f"{len(missing)} texts not in the reference cache {ref_cache} (top_k {top_k})")
    cand = _spread(endpoints, lambda ep, t: _prompt_logprobs(ep, served_name, t, top_k), texts)
    per_text, n_pos, tot, ntok = [], [], 0.0, 0
    for t, cpos in zip(texts, cand):
        r = ref[(_sha(t), top_k)]
        rpos, start = r["pos"], int(r.get("n_prompt") or 0)
        L = min(len(rpos), len(cpos))
        kls = [_kl_position(rpos[j][0], rpos[j][1], dict(zip(cpos[j][0], cpos[j][1])))
               for j in range(max(1, start), L) if rpos[j][0]]
        per_text.append(sum(kls) / len(kls) if kls else 0.0)
        n_pos.append(len(kls))
        tot += sum(kls)
        ntok += len(kls)
    return {"mean": sum(per_text) / len(per_text) if per_text else 0.0, "token_mean": tot / ntok if ntok else 0.0,
            "per_text": per_text, "n_positions": n_pos, "n_texts": len(texts), "n_tokens": ntok, "top_k": top_k}
