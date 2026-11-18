"""Verification for data/: counts, split disjointness, manifest integrity, decontamination.

Run: eval/.venv/bin/python -I data/scripts/check_data.py [--ngram]
Exit code != 0 on any failed assertion. Writes data/splits/check_report.json.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parents[1]
PROC, SPL = DATA / "processed", DATA / "splits"

# Official / expected item counts (unit = processed record).
EXPECTED = {
    "gsm8k": 1319, "gsm8k_train": 7473, "math": 5000, "math_train": 7500,
    "aime24": 30, "aime25": 30, "amc23": 40, "olympiadbench": 674,
    "humanevalplus": 164, "mbppplus": 378, "livecodebench": 131,
    "mmlu_pro": 12032, "triviaqa": 9960, "triviaqa_train": 76523, "freebaseqa": 3996,
    "ifeval": 541, "ifbench": 300, "mt_bench": 80, "alpaca_eval": 805,
    "bfcl": 4696, "teval": 4358, "metatool": 5327, "apibank": 584,
    "magicoder_oss": 75197, "tulu3_if": 29980, "toolace": 11300, "apibank_train": 10376,
    "squad_val": 2067, "squad_train": 18891, "gpqa_diamond": 198,
    # added 2026-11-18 (RSI-validated training sets + Seal-Tools held-out)
    "dapo_math": 17398, "apps": 4449, "taco_verified": 10806, "arc": 3370, "gpqa_main": 250, "sealtool_train": 12022, "sealtool": 2054,
    # added 2026-11-19 (gradable IF training set)
    "ifeval_train": 14970,
}
EXPECTED_NOTES = {
    "math": "5000 test (incl. MATH-500 = 500)",
    "mbppplus": "MBPP+ v0.2.0 = 378",
    "livecodebench": "release_v6 problems with contest_date >= 2025-02-01 (test6.jsonl: 175 total, v6 = 1055 total)",
    "triviaqa": "rc.nocontext validation: 17,944 rows = 9,960 unique question_id (rows duplicated per evidence domain)",
    "triviaqa_train": "rc.nocontext train: 138,384 rows = 76,523 unique question_id",
    "bfcl": "BFCL v4 single-turn 3,641 + multi-turn 800 + agentic (memory 155, web_search 100); format_sensitivity excluded (re-uses ids)",
    "teval": "T-Eval v2 EN: 7 x 553 + review 487",
    "metatool": "Task1 1,040 + Task2 subtasks 995/1,800/995/497",
    "apibank": "level-1 API 399 + level-2 API 135 + level-3 50",
    "apibank_train": "lv1 + lv2 + lv3 api-train",
    "squad_val": "2,067 passages / 10,570 questions (SEAL eval subset: 200 passages / 974 questions)",
    "squad_train": "18,891 passages / 87,599 questions",
    "gpqa_diamond": "manual (gated)",
    "dapo_math": "1,791,700 rows = 17,398 unique prompts (WHALE reports 17,917)",
    "apps": "APPS train 5,000 (all difficulties) -> 4,449 with tests and >=1 solution (introductory 2,397; ReST-EM reports 2,342)",
    "taco_verified": "TACO-verified 12,898 -> 10,806 (tests parse, verified solution, no pictures / multi-answer, no internal duplicates); the only code training set",
    "arc": "ARC-Easy train 2,251 + ARC-Challenge train 1,119",
    "gpqa_main": "GPQA Main 448 minus the 198 Diamond questions (manual, gated)",
    "sealtool_train": "Seal-Tools finetune train",
    "ifeval_train": "allenai/RLVR-IFeval train 14,973 rows minus 3 exact duplicates (same prompt + ground_truth)",
    "sealtool": "Seal-Tools dev 700 (reserve) + test_in_domain 700 + test_out_domain 654",
}
BFCL_EXPECTED = {"simple_python": 400, "simple_java": 100, "simple_javascript": 50, "multiple": 200,
                 "parallel": 200, "parallel_multiple": 200, "live_simple": 258, "live_multiple": 1053,
                 "live_parallel": 16, "live_parallel_multiple": 24, "irrelevance": 240,
                 "live_irrelevance": 884, "live_relevance": 16, "multi_turn_base": 200,
                 "multi_turn_miss_func": 200, "multi_turn_miss_param": 200, "multi_turn_long_context": 200,
                 "memory": 155, "web_search": 100}

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  FAIL:", msg)
    return cond


def load(name):
    with open(PROC / f"{name}.jsonl") as f:
        return [json.loads(l) for l in f]


def text_of(r):
    if r.get("prompt"):
        return r["prompt"]
    flat = []

    def walk(m):
        if isinstance(m, list):
            for x in m:
                walk(x)
        elif isinstance(m, dict) and m.get("role") == "user":
            flat.append(str(m.get("content", "")))
    walk(r.get("messages") or [])
    return "\n".join(flat)


def norm(s):
    return " ".join(str(s).lower().split())


def ngrams(s, n=13):
    toks = re.findall(r"\w+", s.lower())
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngram", action="store_true", help="also compute 13-gram overlap report (slower)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(SPL / "split_config.yaml"))
    manifest = json.load(open(SPL / "MANIFEST.json"))
    report = {"counts": {}, "pools": {}, "decontam": {}, "ngram13": {}}
    data = {}

    print("== 1. counts (expected vs actual)")
    for name in cfg["datasets"]:
        if not (PROC / f"{name}.jsonl").exists():
            print(f"  {name:16s} expected {EXPECTED.get(name)}  actual MISSING ({EXPECTED_NOTES.get(name, '')})")
            report["counts"][name] = {"expected": EXPECTED.get(name), "actual": None}
            continue
        rows = load(name)
        data[name] = rows
        ok = check(len(rows) == EXPECTED[name], f"{name}: count {len(rows)} != expected {EXPECTED[name]}")
        print(f"  {name:16s} expected {EXPECTED[name]:6d} actual {len(rows):6d} {'OK' if ok else 'MISMATCH'}")
        report["counts"][name] = {"expected": EXPECTED[name], "actual": len(rows), "note": EXPECTED_NOTES.get(name)}
    c = Counter(r["metadata"]["bfcl_category"] for r in data["bfcl"])
    for k, v in BFCL_EXPECTED.items():
        check(c[k] == v, f"bfcl/{k}: {c[k]} != {v}")
    check(sum(r["metadata"]["is_math500"] for r in data["math"]) == 500, "MATH-500 count")
    nq = sum(len(r["reference"]) for r in data["squad_val"])
    check(nq == 10570, f"squad val questions {nq}")
    nq = sum(len(r["reference"]) for r in data["squad_train"])
    check(nq == 87599, f"squad train questions {nq}")
    nq = sum(len(r["reference"]) for r in data["squad_val"] if r["split_pool"] == "test")
    print(f"  squad_val test questions: {nq}")

    print("== 2. split integrity / disjointness")
    for name, rows in data.items():
        ids = {r["id"] for r in rows}
        pools = {}
        for fp in sorted((SPL / name).glob("*.json")):
            pools[fp.stem] = json.load(open(fp))
            m = manifest["splits"].get(f"{name}/{fp.name}")
            check(m is not None and m["sha256"] == hashlib.sha256(fp.read_bytes()).hexdigest(),
                  f"{name}/{fp.name}: manifest hash mismatch")
        seen = {}
        for p, lst in pools.items():
            check(len(lst) == len(set(lst)), f"{name}/{p}: duplicate ids")
            for i in lst:
                check(i not in seen, f"{name}: {i} in both {seen.get(i)} and {p}")
                seen[i] = p
        check(set(seen) == ids, f"{name}: split files do not cover processed ids exactly")
        check(all(r["split_pool"] == seen.get(r["id"]) for r in rows), f"{name}: processed split_pool mismatch")
        report["pools"][name] = {p: len(v) for p, v in pools.items()}
    # scheme checks (2026-11-18): the loop only sees training sets; benchmarks are held-out
    for name, rows in data.items():
        c = cfg["datasets"][name]
        pools_here = Counter(r["split_pool"] for r in rows)
        if c.get("role") == "train":
            check(not pools_here.get("test"), f"{name}: training set must not have a test pool")
            if name != "squad_train":
                check(not pools_here.get("search"), f"{name}: training sets use pool + gate (no search)")
            if c.get("rest") == "pool":
                check(pools_here.get("pool", 0) > 0, f"{name}: empty draw pool")
            for p in ("search", "gate"):
                want = c.get(p)
                if want == "domain":  # resolved from gate_domains at split time
                    want = manifest.get("gate_sizes", {}).get(name)
                    check(want is not None, f"{name}: gate size missing from MANIFEST gate_sizes")
                if isinstance(want, int):
                    check(pools_here.get(p, 0) == want, f"{name}: {p} size {pools_here.get(p, 0)} != {want}")
        elif name not in ("ifeval", "squad_train"):
            check(not pools_here.get("search") and not pools_here.get("gate"),
                  f"{name}: held-out benchmark must not have search/gate pools")
    check({r["id"] for r in data.get("gpqa_main", [])}.isdisjoint({r["id"] for r in data["gpqa_diamond"]}),
          "gpqa_main must exclude every gpqa_diamond question")
    check(all(r["split_pool"] == "test" for r in data["bfcl"]), "bfcl must be fully held-out")
    check(all(r["split_pool"] == "test" for r in data["livecodebench"]), "livecodebench must be fully held-out")
    m = data["math"]
    check(all(r["split_pool"] == "test" for r in m if r["metadata"]["is_math500"]), "MATH-500 must be test")
    check(not any(r["split_pool"] == "test" for r in m if not r["metadata"]["is_math500"]), "only MATH-500 in test")
    strat = defaultdict(Counter)
    for r in data["ifeval"]:
        strat[r["reference"]["instruction_id_list"][0].split(":")[0]][r["split_pool"]] += 1
    report["ifeval_strata"] = {k: dict(v) for k, v in strat.items()}
    # SQuAD: title disjointness between pools and test passages
    st = defaultdict(set)
    for r in data["squad_train"]:
        st[r["split_pool"]].add(r["metadata"]["title"])
    check(not (st["search"] & st["gate"]), "squad search/gate share titles")
    check(not ((st["search"] | st["gate"]) & st["train"]), "squad search/gate titles appear in train pool")
    vt = {r["metadata"]["title"] for r in data["squad_val"]}
    check(not (vt & (st["train"] | st["search"] | st["gate"])), "squad val titles overlap squad train titles")
    check(len({r["metadata"]["title"] for r in data["squad_train"] if r["split_pool"] == "search"}) == 50, "squad search titles")

    gd = cfg.get("gate_domains") or {}
    for dom, g in gd.items():
        got = sum(manifest.get("gate_sizes", {}).get(ds, 0) for ds in g["datasets"])
        check(got == int(g["size"]), f"gate domain {dom}: {got} gate items != configured {g['size']}")
        print(f"  gate domain {dom:22s} size={got}")
    print("== 3. train/eval decontamination (exact match on normalized prompt)")
    is_train = lambda n: cfg["datasets"][n].get("role") == "train" or cfg["datasets"][n].get("all") == "train"
    eval_names = [n for n in data if not is_train(n)]
    eval_txt = defaultdict(set)
    for n in eval_names:
        for r in data[n]:
            eval_txt[norm(text_of(r))].add(n)
    eval_txt.pop("", None)
    # every item of a training set the loop can see (signal / gate / train) must be decontaminated
    train_items = [(n, r) for n in data if is_train(n) for r in data[n]
                   if r["split_pool"] in ("train", "search", "gate", "pool")]
    hits = defaultdict(int)
    for n, r in train_items:
        if norm(text_of(r)) in eval_txt:
            hits[n] += 1
    for n in sorted({n for n, _ in train_items}):
        check(hits[n] == 0, f"{n}: {hits[n]} train items exactly match eval items")
        print(f"  {n:16s} pool+gate+train={sum(1 for m, _ in train_items if m == n):6d} exact-match hits={hits[n]}")
    # also: squad_train search/gate passages must not equal any eval passage
    report["decontam"] = {"exact_match_hits": dict(hits), "removed_at_split_time": manifest.get("decontamination")}
    # cross-eval overlap (informational, not an error): same question in two eval sets
    cross = Counter()
    for t, ns in eval_txt.items():
        if len(ns) > 1:
            cross[" & ".join(sorted(ns))] += 1
    report["cross_eval_exact_overlap"] = dict(cross)
    print("  cross-eval exact overlaps (info):", dict(cross))

    if a.ngram:
        print("== 4. 13-gram overlap (informational)")
        pairs = {"taco_verified": ["humanevalplus", "mbppplus", "livecodebench"],
                 "math_train": ["math", "aime24", "aime25", "amc23", "olympiadbench"],
                 "dapo_math": ["math", "aime24", "aime25", "amc23", "olympiadbench"],
                 "gsm8k_train": ["gsm8k"],
                 "arc": ["mmlu_pro", "gpqa_diamond"], "gpqa_main": ["gpqa_diamond"], "triviaqa_train": ["triviaqa"],
                 "sealtool_train": ["sealtool", "bfcl", "apibank"], "toolace": ["bfcl", "apibank", "metatool"],
                 "ifeval_train": ["ifeval", "ifbench"]}
        for tr, evs in pairs.items():
            for ev in evs:
                eg, df_ = {}, Counter()
                for r in data[ev]:
                    for g in ngrams(text_of(r)):
                        eg.setdefault(g, r["id"])
                        df_[g] += 1
                # drop boilerplate n-grams shared by >=5 eval items (prompt templates, tool schemas)
                eg = {g: i for g, i in eg.items() if df_[g] < 5}
                hit_eval = set()
                n_train_hit = 0
                for r in data[tr]:
                    if r["split_pool"] not in ("train", "search", "gate", "pool"):
                        continue
                    hs = {eg[g] for g in ngrams(text_of(r)) if g in eg}
                    if hs:
                        n_train_hit += 1
                        hit_eval |= hs
                report["ngram13"][f"{tr}->{ev}"] = {"train_items_with_overlap": n_train_hit,
                                                    "eval_items_touched": len(hit_eval), "eval_total": len(data[ev])}
                print(f"  {tr:14s} -> {ev:14s} train_items_with_13gram_overlap={n_train_hit:6d} "
                      f"eval_items_touched={len(hit_eval)}/{len(data[ev])}")

    report["failures"] = FAIL
    json.dump(report, open(SPL / "check_report.json", "w"), indent=1, sort_keys=True)
    print("\nRESULT:", "ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURES")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
