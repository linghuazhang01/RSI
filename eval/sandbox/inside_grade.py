"""Runs INSIDE the Docker sandbox only (network none). Executes model-generated code against
EvalPlus / LiveCodeBench tests and writes per-sample pass/fail.

  --task humanevalplus|mbppplus   --samples /work/samples.jsonl   (EvalPlus format: task_id, solution)
  --task livecodebench|apps       --samples /work/samples.jsonl   (question_id, code, sample_idx)
  --out /work/sandbox_results.json

LiveCodeBench private tests are stored base64(zlib(pickle(json))) in the official HF files, so they
are decoded here, inside the sandbox, never on the host.
"""
import argparse
import gzip
import json
import os
import shutil
import sys


def evalplus_grade(task, samples, out, parallel):
    src = {"humanevalplus": "/data/evalplus/HumanEvalPlus-v0.1.10.jsonl.gz",
           "mbppplus": "/data/evalplus/MbppPlus-v0.2.0.jsonl.gz"}[task]
    plain = "/tmp/" + os.path.basename(src)[:-3]
    with gzip.open(src, "rb") as fi, open(plain, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.environ["HUMANEVAL_OVERRIDE_PATH" if task == "humanevalplus" else "MBPP_OVERRIDE_PATH"] = plain
    from evalplus.evaluate import evaluate
    shutil.copy(samples, "/tmp/samples.jsonl")
    res_path = "/tmp/eval_results.json"
    evaluate(dataset="humaneval" if task == "humanevalplus" else "mbpp", samples="/tmp/samples.jsonl",
             parallel=parallel, i_just_wanna_run=True, output_file=res_path)
    res = json.load(open(res_path))
    per = {}
    for tid, lst in res["eval"].items():
        per[tid] = [{"base": r["base_status"] == "pass", "plus": r["base_status"] == "pass" and r["plus_status"] == "pass"}
                    for r in lst]
    json.dump({"task": task, "per_task": per}, open(out, "w"))


def lcb_grade(samples, out, parallel, timeout):
    import base64
    import pickle
    import zlib
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    subs = [json.loads(l) for l in open(samples)]
    want = {s["question_id"] for s in subs}
    probs = {}
    with open("/data/lcb/test6.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["question_id"] not in want:
                continue
            pub = json.loads(r["public_test_cases"])
            try:
                priv = json.loads(r["private_test_cases"])
            except Exception:
                priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"].encode()))))
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
            tests = pub + priv
            probs[r["question_id"]] = {"input_output": json.dumps({
                "inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests],
                "fn_name": meta.get("func_name")})}
    by_q = {}
    for s in subs:
        by_q.setdefault(s["question_id"], []).append(s)
    qids = sorted(by_q)
    gens = [[s["code"] for s in sorted(by_q[q], key=lambda x: x["sample_idx"])] for q in qids]
    _, results, _ = codegen_metrics([probs[q] for q in qids], gens, k_list=[1], num_process_evaluate=parallel,
                                    timeout=timeout)
    per = {}
    for i, q in enumerate(qids):
        per[q] = [{"plus": bool(isinstance(r, list) and len(r) > 0 and all(x is True or x == 1 for x in r))}
                  for r in results[i]]
    json.dump({"task": "livecodebench", "per_task": per}, open(out, "w"))


def apps_grade(samples, out, parallel, timeout):
    """Code training sets (APPS, TACO-verified) used for the code signal / gate. Tests were converted to
    the LiveCodeBench runner format by data/scripts/normalize.py (data/processed/aux/*_tests.jsonl)."""
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    subs = [json.loads(l) for l in open(samples)]
    want = {s["question_id"] for s in subs}
    probs = {}
    import glob
    for tf in sorted(glob.glob("/data/aux/*_tests.jsonl")):  # apps_tests.jsonl, taco_verified_tests.jsonl
        with open(tf) as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in want:
                    probs[r["id"]] = {"input_output": json.dumps(r["input_output"])}
    missing = sorted(want - set(probs))
    assert not missing, f"no tests for {len(missing)} ids, e.g. {missing[:3]}"
    by_q = {}
    for s in subs:
        by_q.setdefault(s["question_id"], []).append(s)
    qids = sorted(by_q)
    gens = [[s["code"] for s in sorted(by_q[q], key=lambda x: x["sample_idx"])] for q in qids]
    _, results, _ = codegen_metrics([probs[q] for q in qids], gens, k_list=[1], num_process_evaluate=parallel,
                                    timeout=timeout)
    per = {}
    for i, q in enumerate(qids):
        per[q] = [{"plus": bool(isinstance(r, list) and len(r) > 0 and all(x is True or x == 1 for x in r))}
                  for r in results[i]]
    json.dump({"task": "apps", "per_task": per}, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=6)
    a = ap.parse_args()
    if a.task in ("humanevalplus", "mbppplus"):
        evalplus_grade(a.task, a.samples, a.out, a.parallel)
    elif a.task == "livecodebench":
        lcb_grade(a.samples, a.out, a.parallel, a.timeout)
    elif a.task == "apps":
        apps_grade(a.samples, a.out, a.parallel, a.timeout)
    else:
        sys.exit(f"unknown task {a.task}")


if __name__ == "__main__":
    main()
