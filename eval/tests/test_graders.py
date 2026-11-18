"""Grader self-tests (no model, no network):
  * gold mode: feed gold references (formatted the way a model would answer) through every
    deterministic grader over ALL items of each dataset (all pools) -> expect ~100%
  * wrong mode: feed deliberately wrong answers -> expect ~0%
  * IFEval / IFBench: hand-made pass/fail responses (tests/fixtures/if_handmade.json)
  * ifeval_train (RLVR-IFeval): hand-made pass/fail responses, one per verifier func_name
    (tests/fixtures/rlvr_if_handmade.json); plus an informational cross-check of our verifiers
    against the official IFEval checkers on synthetic responses (tests/rlvr_if_crosscheck.py)
  * code: only checks that gold solutions survive code extraction (execution is sandbox-only)

Run: eval/.venv/bin/python -I eval/tests/test_graders.py [--max-per-dataset N]
Writes eval/tests/grader_selftest_report.json; exit code 1 if any threshold fails.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from evallib.common import EVAL_DIR, load_items  # noqa: E402
from evallib.graders.code import extract_solution  # noqa: E402
from evallib.tasks import load_tasks  # noqa: E402

# task -> (dataset override or None, min gold accuracy, max wrong accuracy)
CASES = {
    "gsm8k": (None, 0.999, 0.01), "math": (None, 0.99, 0.01), "aime24": (None, 1.0, 0.0),
    "aime25": (None, 1.0, 0.0), "amc23": (None, 1.0, 0.0), "olympiadbench": (None, 0.99, 0.01),
    "mmlu_pro": (None, 1.0, 0.0), "gpqa_diamond": (None, 1.0, 0.0), "triviaqa": (None, 0.999, 0.001), "freebaseqa": (None, 0.999, 0.001),
    "squad_closed_book_test": (None, 0.999, 0.001), "bfcl": (None, 0.97, 0.03),
    "teval": (None, 0.99, 0.05), "metatool": (None, 0.99, 0.05), "apibank": (None, 0.99, 0.01),
    # training sets added 2026-11-18 (signal / gate slices)
    "gsm8k_train": (None, 0.999, 0.01), "math_train": (None, 0.99, 0.01), "dapo_math": (None, 0.99, 0.01),
    "arc": (None, 1.0, 0.0), "gpqa_main": (None, 1.0, 0.0), "triviaqa_train": (None, 0.999, 0.001),
    "sealtool_train": (None, 1.0, 0.0), "sealtool": (None, 1.0, 0.0),
}


def choice(o):
    return {"content": o.get("content", ""), "tool_calls": o.get("tool_calls", []), "finish_reason": "stop",
            "reasoning": None}


def run_case(task, items, mode):
    scores = []
    fails = []
    for it in items:
        units = task.units(it)
        if not units:
            continue
        outs = {uid: [choice(task.oracle(it, uid, mode))] for uid, _ in units}
        for r in task.grade(it, outs):
            if r["score"] is None:
                continue
            scores.append(r["score"])
            if (mode == "gold" and r["score"] < 1) or (mode == "wrong" and r["score"] > 0):
                fails.append({"id": r["unit_id"] if r["unit_id"] != "0" else r["id"], "parsed": str(r["parsed_answer"])[:200],
                              "metrics": str(r["metrics"])[:300], "item": r["id"]})
    return (sum(scores) / len(scores) if scores else None), len(scores), fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-dataset", type=int, default=None)
    a = ap.parse_args()
    tasks = load_tasks(yaml.safe_load(open(EVAL_DIR / "configs" / "tasks.yaml")))
    report, ok = {}, True
    for tname, (_, gmin, wmax) in CASES.items():
        t0 = time.time()
        task = tasks[tname]
        items = load_items(task.dataset)
        if tname == "squad_closed_book_test":
            items = [i for i in items if i["split_pool"] == "test"]
        if a.max_per_dataset:
            items = items[: a.max_per_dataset]
        g, n, gf = run_case(task, items, "gold")
        w, _, wf = run_case(task, items, "wrong")
        passed = g is not None and g >= gmin and w <= wmax
        ok &= passed
        report[tname] = {"gold_acc": g, "wrong_acc": w, "n_graded_units": n, "pass": passed,
                         "gold_failures": gf[:200], "n_gold_failures": len(gf), "n_wrong_nonzero": len(wf),
                         "seconds": round(time.time() - t0, 1)}
        print(f"{tname:24s} n={n:6d} gold={g:.4f} wrong={w:.4f} {'PASS' if passed else 'FAIL'}"
              f"  ({len(gf)} gold misses)")

    # IFEval / IFBench hand-made
    fx = json.load(open(EVAL_DIR / "tests" / "fixtures" / "if_handmade.json"))
    for tname in ("ifeval", "ifbench"):
        task = tasks[tname]
        items = load_items(task.dataset, ids=[k for k in fx if k.startswith(tname + "/")])
        g, n, gf = run_case(task, items, "gold")
        w, _, wf = run_case(task, items, "wrong")
        passed = g == 1.0 and w == 0.0
        ok &= passed
        report[tname] = {"handmade_pass_acc": g, "handmade_fail_acc": w, "n": n, "pass": passed, "failures": gf + wf}
        print(f"{tname:24s} n={n:6d} handmade pass={g:.2f} fail={w:.2f} {'PASS' if passed else 'FAIL'}")

    # ifeval_train (RLVR-IFeval) hand-made: every verifier func_name has a pass (-> 1.0) and a fail (-> 0.0)
    from evallib.graders.rlvr_if import FUNC_NAMES
    fx = json.load(open(EVAL_DIR / "tests" / "fixtures" / "rlvr_if_handmade.json"))
    task = tasks["ifeval_train"]
    items = load_items(task.dataset, ids=list(fx))
    g, n, gf = run_case(task, items, "gold")
    w, _, wf = run_case(task, items, "wrong")
    covered = {it["reference"]["func_name"] for it in items}
    match = all(fx[it["id"]]["func_name"] == it["reference"]["func_name"] for it in items)
    passed = g == 1.0 and w == 0.0 and n == len(fx) and covered == set(FUNC_NAMES) and match
    ok &= passed
    report["ifeval_train"] = {"handmade_pass_acc": g, "handmade_fail_acc": w, "n": n,
                              "func_names_covered": len(covered), "func_names_total": len(FUNC_NAMES),
                              "pass": passed, "failures": gf + wf}
    print(f"{'ifeval_train':24s} n={n:6d} handmade pass={g:.2f} fail={w:.2f} func_names={len(covered)}/{len(FUNC_NAMES)} "
          f"{'PASS' if passed else 'FAIL'}")
    # informational: our open-instruct-semantics verifiers vs the official IFEval checkers
    sys.path.insert(0, str(EVAL_DIR / "tests"))
    import rlvr_if_crosscheck
    xc = rlvr_if_crosscheck.crosscheck()
    tot = sum(r["n"] for r in xc.values())
    agree = sum(r["agreement"] * r["n"] for r in xc.values()) / tot
    report["ifeval_train_crosscheck_vs_ifeval"] = {"overall_agreement": agree, "n": tot, "per_func": xc}
    print(f"{'ifeval_train x IFEval':24s} n={tot:6d} agreement(own vs IFEval checker)={agree:.3f} (info; see report)")

    # code: extraction of gold solutions (no execution here)
    for tname in ("humanevalplus", "mbppplus", "apps", "taco_verified"):
        task = tasks[tname]
        items = load_items(task.dataset)
        good = 0
        for it in items:
            text = task.oracle(it, "0", "gold")["content"]
            code = extract_solution(task.cfg["grader"], it, text)
            ref = it["prompt"] + it["reference"] if tname == "humanevalplus" else it["reference"]  # apps: first solution
            good += code.strip() == ref.strip()
        passed = good == len(items)
        ok &= passed
        report[tname] = {"gold_extraction_exact": good, "n": len(items), "pass": passed,
                         "execution": "PENDING (sandbox only)"}
        print(f"{tname:24s} n={len(items):6d} gold code extracted exactly: {good}/{len(items)} "
              f"{'PASS' if passed else 'FAIL'} (execution: sandbox PENDING)")

    json.dump(report, open(EVAL_DIR / "tests" / "grader_selftest_report.json", "w"), indent=1, default=str)
    print("\nRESULT:", "ALL GRADER SELF-TESTS PASSED" if ok else "SOME GRADER SELF-TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
