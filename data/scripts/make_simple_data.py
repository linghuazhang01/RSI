"""Build simple_data/<domain>/: the final clean dataset of each domain (pool.jsonl: the training set with every
eval-set overlap and every internal duplicate removed) and 1,500 train + 500 eval items sampled from it.

Since 2026-11-21 make_splits.py applies the same decontamination and de-duplication, so steps 2-3 below remove
nothing from the current split_pool == "pool"; they are kept as an independent re-check.

Domains and sources (data/splits/split_config.yaml): one training set per domain, checked against every
evaluation dataset of the same category, all pools (test and reserve):
  math                  dapo_math       vs math (all 5,000, incl. MATH-500), gsm8k, aime24, aime25, amc23, olympiadbench
  code                  taco_verified   vs humanevalplus, mbppplus, livecodebench
  knowledge             triviaqa_train  vs triviaqa (all 9,960), mmlu_pro (all 12,032), gpqa_diamond, freebaseqa
  instruction_following ifeval_train    vs ifeval (all 541), ifbench, mt_bench, alpaca_eval
  tool_calling          sealtool_train  vs sealtool (all 2,054), apibank, bfcl, metatool, teval

Steps per domain (seed 20261118, the data seed):
1. Start from split_pool == "pool" (make_splits.py already moved contaminated items to train_excluded).
2. Re-check independently against every eval item; any hit is dropped:
   a. exact match after lower-casing and whitespace normalization;
   b. exact match after also stripping LaTeX markup, punctuation and spaces (formatting variants);
   c. 13-gram near duplicate: shared 13-grams with one eval item cover >= 50% of either item's 13-grams.
      For instruction_following and tool_calling, 13-grams found in >= 5 eval items are template text
      (constraint sentences, tool-call instructions) and are not counted as shared.
   Text = prompt, else the user turns of `messages`; Seal-Tools items use only their task_instruction line
   (same rule as make_splits.py).
3. Drop internal duplicates (same text under rule b), keeping the smallest id, so train and eval are
   disjoint by content, not only by id.
4. Write the remaining items sorted by id to pool.jsonl. Shuffle them with random.Random("20261118:simple_data:<domain>"); the first
   1,500 -> train, the next 500 -> eval.

Run: eval/.venv/bin/python -I data/scripts/make_simple_data.py [--only math,code]
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "simple_data"
SEED = 20261118
N_TRAIN, N_EVAL = 1500, 500
NGRAM, NEAR_THR, TEMPLATE_DF = 13, 0.5, 5
DOMAINS = {
    "math": {"train": "dapo_math",
             "eval": ["math", "gsm8k", "aime24", "aime25", "amc23", "olympiadbench"], "template_filter": False},
    "code": {"train": "taco_verified",
             "eval": ["humanevalplus", "mbppplus", "livecodebench"], "template_filter": False},
    "knowledge": {"train": "triviaqa_train",
                  "eval": ["triviaqa", "mmlu_pro", "gpqa_diamond", "freebaseqa"], "template_filter": False},
    "instruction_following": {"train": "ifeval_train",
                              "eval": ["ifeval", "ifbench", "mt_bench", "alpaca_eval"], "template_filter": True},
    "tool_calling": {"train": "sealtool_train",
                     "eval": ["sealtool", "apibank", "bfcl", "metatool", "teval"], "template_filter": True},
}


def load(name):
    with open(PROC / f"{name}.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def text_of(r):
    """Prompt, else the concatenated user turns; Seal-Tools: only the task_instruction line."""
    t = r.get("prompt")
    if not t:
        flat = []

        def walk(m):
            if isinstance(m, list):
                for x in m:
                    walk(x)
            elif isinstance(m, dict) and m.get("role") == "user":
                flat.append(str(m.get("content", "")))
        walk(r.get("messages") or [])
        t = "\n".join(flat)
    if str(r.get("dataset", "")).startswith("sealtool"):
        m = re.search(r'task_instruction = "(.*)"\s*\nOutput:', t, re.S)
        if m:
            return m.group(1)
    return t


def norm_ws(s):
    return " ".join(str(s).lower().split())


def norm_strict(s):
    s = str(s).lower()
    s = re.sub(r"\\(left|right|displaystyle|text|mathrm|mathbf|operatorname|dfrac|tfrac|frac)", "", s)
    s = s.replace("\\$", "").replace("$", "")
    return re.sub(r"[\W_]+", "", s)


def grams(s):
    w = re.findall(r"\w+", str(s).lower())
    return {" ".join(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def build(domain, cfg):
    ev = [r for n in cfg["eval"] for r in load(n)]
    ev_by_id = {(r["dataset"], r["id"]): r for r in ev}
    ws = {norm_ws(text_of(r)): (r["dataset"], r["id"]) for r in ev if text_of(r).strip()}
    strict = {norm_strict(text_of(r)): (r["dataset"], r["id"]) for r in ev if norm_strict(text_of(r))}
    idx, glen = defaultdict(set), {}
    for r in ev:
        k = (r["dataset"], r["id"])
        G = grams(text_of(r))
        glen[k] = len(G)
        for g in G:
            idx[g].add(k)
    if cfg["template_filter"]:
        idx = {g: s for g, s in idx.items() if len(s) < TEMPLATE_DF}

    def near(text):
        G = grams(text)
        if not G:
            return None
        cnt = defaultdict(int)
        for g in G:
            for x in idx.get(g, ()):
                cnt[x] += 1
        best = max(((max(k / len(G), k / max(glen[x], 1)), x) for x, k in cnt.items()), default=(0, None))
        return best[1] if best[0] >= NEAR_THR else None

    rows = load(cfg["train"])
    pool = [r for r in rows if r["split_pool"] == "pool"]
    removed = {"exact_ws": [], "exact_strict": [], "near_13gram": [], "internal_dup": []}
    clean = []
    for r in pool:
        t = text_of(r)
        if (m := ws.get(norm_ws(t))):
            removed["exact_ws"].append([r["id"], *m])
        elif (m := strict.get(norm_strict(t))):
            removed["exact_strict"].append([r["id"], *m])
        elif (m := near(t)):
            removed["near_13gram"].append([r["id"], *m])
        else:
            clean.append(r)
    seen, dedup = set(), []
    for r in sorted(clean, key=lambda r: r["id"]):
        k = norm_strict(text_of(r))
        if k in seen:
            removed["internal_dup"].append(r["id"])
            continue
        seen.add(k)
        dedup.append(r)

    final = sorted(dedup, key=lambda r: r["id"])  # the final clean dataset, written as pool.jsonl
    random.Random(f"{SEED}:simple_data:{domain}").shuffle(dedup)
    assert len(dedup) >= N_TRAIN + N_EVAL, f"{domain}: only {len(dedup)} clean items"
    train, evals = dedup[:N_TRAIN], dedup[N_TRAIN:N_TRAIN + N_EVAL]
    assert not {r["id"] for r in train} & {r["id"] for r in evals}
    assert not {norm_strict(text_of(r)) for r in train} & {norm_strict(text_of(r)) for r in evals}

    d = OUT / domain
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "pool.jsonl", "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, part in [("train", train), ("eval", evals)]:
        with open(d / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(dict(r, split_pool=name), ensure_ascii=False) + "\n")
    hit_by = defaultdict(int)
    for k in ("exact_ws", "exact_strict", "near_13gram"):
        for _, ds, eid in removed[k]:
            hit_by[f"{ds}/{ev_by_id[(ds, eid)]['split_pool']}"] += 1
    report = {
        "domain": domain,
        "source": f"data/processed/{cfg['train']}.jsonl (split_pool == pool)",
        "eval_sets_checked": {n: sum(1 for r in ev if r["dataset"] == n) for n in cfg["eval"]},
        "template_filter": cfg["template_filter"],
        "seed": f"{SEED}:simple_data:{domain}",
        "train_total": len(rows),
        "already_excluded_by_make_splits": len(rows) - len(pool),
        "pool": len(pool),
        "removed_here": {k: len(v) for k, v in removed.items()},
        "overlap_hits_by_eval_set": dict(sorted(hit_by.items())),
        "clean_after_dedup": len(dedup),
        "train": len(train),
        "eval": len(evals),
        "removed_ids": removed,
    }
    (d / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated domains (default: all)")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(DOMAINS)
    for n in names:
        rep = build(n, DOMAINS[n])
        print(json.dumps({k: v for k, v in rep.items() if k != "removed_ids"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
