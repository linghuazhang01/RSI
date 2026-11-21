"""Read-only views of a reference model's screen records for the editor (2026-11-21, objective simple).

The latest model (Paper A: the incumbent quantization; Paper B: the current model) answers the generation's screen
items once (decoding.screen, e.g. thinking n 4); these helpers turn its run_eval records into what the editor sees.
Screen items are training-pool items, so prompts, gold answers and full outputs (incl. the thinking text, which
run_eval keeps per sample as "reasoning") may be shown. Generic: records_by_ds = {dataset: [run_eval records]},
items_by_ds = {dataset: {id: processed record}} (prompt / messages, reference).

    summary(records_by_ds, items_by_ds, examples=3) -> {n_items, samples_per_item, accuracy (avg@n),
        correct_counts {"0": items with 0 correct samples, ..., "n": ...}, truncation_rate, truncated_thinking,
        failing: [first `examples` items with no / not every sample correct: id, prompt, gold, answers, scores]}
    rows(records_by_ds, items_by_ds, ids=None, filter="all", limit=50, offset=0) -> [{id, dataset, prompt, gold,
        scores, answers, finish, lengths, truncated_thinking}]; filter all | wrong (no sample correct) |
        unstable (some but not all) | right (every sample correct); sorted by (dataset, id)
    output(record, sample=0, offset=0, chars=4000) -> one sample's full text (thinking + answer) in pages
A sample counts as correct when its score is > 0.5 (the same rule as compare / paired flips).
"""
from __future__ import annotations

FILTERS = ("all", "wrong", "unstable", "right")


def _as_list(v, n: int) -> list:
    return list(v) if isinstance(v, list) else [v] * n


def sample_scores(rec: dict) -> list:
    """Per-sample scores of one record (n 1: [score])."""
    ss = rec.get("scores") or rec.get("sample_scores")
    if isinstance(ss, list) and ss:
        return list(ss)
    return [rec.get("score")]


def n_correct(rec: dict) -> tuple[int, int]:
    ss = sample_scores(rec)
    return sum(1 for s in ss if s is not None and float(s) > 0.5), len(ss)


def category(rec: dict) -> str:
    k, n = n_correct(rec)
    return "right" if n and k == n else ("wrong" if k == 0 else "unstable")


def prompt_text(item: dict | None) -> str:
    item = item or {}
    if item.get("prompt"):
        return str(item["prompt"])
    msgs = [m for m in (item.get("messages") or []) if isinstance(m, dict)]
    user = [m for m in msgs if m.get("role") == "user"] or msgs
    c = user[-1].get("content") if user else ""
    return c if isinstance(c, str) else str(c)


def _short(s, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f" [... {len(s) - n} chars cut]"


def _samples(rec: dict) -> tuple[list, list, list, list]:
    """(contents, reasonings, finish reasons, parsed answers), one entry per sample."""
    n = len(sample_scores(rec))
    return (_as_list(rec.get("raw_output"), n), _as_list(rec.get("reasoning"), n),
            _as_list(rec.get("finish_reason") or [None] * n, n), _as_list(rec.get("parsed_answer"), n))


def _row(ds: str, rec: dict, item: dict | None, prompt_chars: int) -> dict:
    outs, reas, fin, ans = _samples(rec)
    return {"id": rec["id"], "dataset": ds, "prompt": _short(prompt_text(item), prompt_chars),
            "gold": _short((item or {}).get("reference"), 300), "scores": sample_scores(rec),
            "answers": [None if a is None else _short(a, 200) for a in ans], "finish": fin,
            "lengths": [len(str(r or "")) + len(str(o or "")) for o, r in zip(outs, reas)],
            "truncated_thinking": int(rec.get("truncated_thinking") or 0)}


def _graded(records_by_ds: dict) -> list[tuple[str, dict]]:
    out = []
    for ds in sorted(records_by_ds):
        out += [(ds, r) for r in sorted(records_by_ds[ds] or [], key=lambda r: str(r.get("id")))
                if r.get("score") is not None]
    return out


def summary(records_by_ds: dict, items_by_ds: dict | None = None, examples: int = 3, prompt_chars: int = 300) -> dict:
    recs = _graded(records_by_ds)
    if not recs:
        return {"n_items": 0}
    n = max(len(sample_scores(r)) for _, r in recs)
    counts = {str(k): 0 for k in range(n + 1)}
    n_s = n_len = 0
    for _, r in recs:
        k, _ = n_correct(r)
        counts[str(k)] = counts.get(str(k), 0) + 1
        fin = _samples(r)[2]
        n_s += len(fin)
        n_len += sum(1 for f in fin if f == "length")
    fail = sorted((r for r in recs if category(r[1]) != "right"),
                  key=lambda x: (n_correct(x[1])[0], x[0], str(x[1]["id"])))[:max(0, int(examples))]
    items = items_by_ds or {}
    return {"n_items": len(recs), "samples_per_item": n,
            "accuracy": round(sum(float(r["score"]) for _, r in recs) / len(recs), 4),
            "correct_counts": counts, "truncation_rate": round(n_len / n_s, 4) if n_s else None,
            "truncated_thinking": sum(int(r.get("truncated_thinking") or 0) for _, r in recs),
            "failing": [{k: v for k, v in _row(ds, r, items.get(ds, {}).get(r["id"]), prompt_chars).items()
                         if k in ("id", "dataset", "prompt", "gold", "answers", "scores")} for ds, r in fail]}


def rows(records_by_ds: dict, items_by_ds: dict | None = None, *, ids=None, filter: str = "all", limit: int = 50,
         offset: int = 0, prompt_chars: int = 300) -> list[dict]:
    if filter not in FILTERS:
        raise ValueError(f"filter must be one of {list(FILTERS)}, not {filter!r}")
    want = None if ids is None else {str(i) for i in ([ids] if isinstance(ids, str) else ids)}
    recs = [(ds, r) for ds, r in _graded(records_by_ds)
            if (want is None or str(r["id"]) in want) and (filter == "all" or category(r) == filter)]
    off, lim = max(0, int(offset or 0)), max(0, int(limit if limit is not None else 50))
    items = items_by_ds or {}
    return [_row(ds, r, items.get(ds, {}).get(r["id"]), prompt_chars) for ds, r in recs[off:off + lim]]


def output(rec: dict, *, sample: int = 0, offset: int = 0, chars: int = 4000) -> dict:
    outs, reas, fin, ans = _samples(rec)
    ss = sample_scores(rec)
    sample = int(sample)
    if not 0 <= sample < len(ss):
        raise ValueError(f"sample must be in 0..{len(ss) - 1}")
    r, o = reas[sample], outs[sample]
    text = (f"<think>\n{r}\n</think>\n\n" if r else "") + str(o or "")
    off, page = max(0, int(offset or 0)), max(200, min(int(chars or 4000), 20000))
    out = {"id": rec["id"], "sample": sample, "n_samples": len(ss), "score": ss[sample],
           "answer": None if ans[sample] is None else str(ans[sample]), "finish": fin[sample],
           "has_thinking": bool(r), "offset": off, "total_chars": len(text), "text": text[off:off + page]}
    if off + page < len(text):
        out["next_offset"] = off + page
    return out
