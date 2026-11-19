"""GPU-free end-to-end test: start the oracle server (gold, then wrong mode), run run_eval.py against
it for at least one task per category, and check the summary scores.

  eval/.venv/bin/python -I eval/tests/run_e2e.py
Results: eval/results/e2e_oracle_{gold,wrong}/ ; report: eval/tests/e2e_report.json
"""
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
PY = sys.executable
FX = EVAL / "tests" / "fixtures"

# (task, pool, extra args, summary key, gold_min, wrong_max)
CASES = [
    # ---- held-out benchmarks (pool=test)
    ("gsm8k", "test", ["--limit", "300"], "accuracy", 1.0, 0.0),                    # math
    ("aime24", "test", [], "accuracy", 1.0, 0.0),                                  # math, n=8 sampling path
    ("mmlu_pro", "test", ["--limit", "500"], "accuracy", 1.0, 0.0),                # knowledge (MCQ)
    ("gpqa_diamond", "test", [], "accuracy", 1.0, 0.0),                            # knowledge (MCQ, GPQA)
    ("triviaqa", "test", ["--limit", "500"], "accuracy", 1.0, 0.0),                # knowledge (EM)
    # squad: 1 question text appears twice under the same title with different gold -> oracle collision
    ("squad_closed_book_test", "test", [], "accuracy", 0.998, 0.0),                # knowledge injection
    ("ifeval", "reserve", ["--ids-file", str(FX / "ifeval_e2e_ids.json")], "prompt_strict", 1.0, 0.0),  # hand-made items (not test)
    ("ifbench", "test", ["--ids-file", str(FX / "ifbench_e2e_ids.json")], "prompt_strict", 1.0, 0.0),
    ("sealtool", "test", ["--limit", "300"], "accuracy", 1.0, 0.0),                # tool calling (Seal-Tools)
    ("bfcl", "test", ["--limit", "600"], "accuracy", 0.97, 0.0),                   # tool calling (FC)
    ("metatool", "test", ["--limit", "300"], "accuracy", 0.99, 0.01),
    ("apibank", "test", ["--limit", "200"], "accuracy", 1.0, 0.0),
    ("teval", "test", ["--limit", "600"], "accuracy", 0.98, 0.02),
    ("humanevalplus", "test", ["--no-sandbox"], None, None, None),                # code: generation + extraction; exec PENDING
    ("mt_bench", "test", ["--judge-endpoint", "JUDGE"], "mean_rating", 10.0, 1.0),  # optional judge path
    ("alpaca_eval", "test", ["--limit", "100", "--judge-endpoint", "JUDGE"], "win_rate", 0.98, 0.02),
    # ---- training sets (five, 2026-11-19): pool + ids from a draw (DRAW:<ds>.<role>) or the run's gate (GATE:<ds>)
    ("dapo_math", "pool", ["--ids-file", "DRAW:dapo_math.signal"], "accuracy", 1.0, 0.0),
    ("dapo_math", "pool", ["--ids-file", "GATE:dapo_math", "--limit", "300"], "accuracy", 1.0, 0.0),
    ("triviaqa_train", "pool", ["--ids-file", "DRAW:triviaqa_train.screen"], "accuracy", 1.0, 0.0),
    ("sealtool_train", "pool", ["--ids-file", "DRAW:sealtool_train.screen"], "accuracy", 1.0, 0.0),
    ("ifeval_train", "pool", ["--ids-file", str(FX / "ifeval_train_e2e_ids.json")], "accuracy", 1.0, 0.0),  # hand-made
    ("ifeval_train", "pool", ["--ids-file", "DRAW:ifeval_train.screen"], None, None, None),  # no oracle: run only
    ("taco_verified", "pool", ["--ids-file", "DRAW:taco_verified.screen", "--no-sandbox"], None, None, None),
]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_up(port, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def make_draw():
    """Draw generation 0 of a throwaway Paper A run; returns its ids/ directory."""
    import tempfile
    run = Path(tempfile.mkdtemp(prefix="e2e_draw_")) / "e2e_run"
    subprocess.run([PY, "-I", str(EVAL.parent / "data/scripts/draw.py"), "--run", str(run), "--gen", "0",
                    "--config", str(EVAL.parent / "data/draw/paper_a.yaml")], check=True, capture_output=True)
    return run / "draws" / "gen_00" / "ids"


def main():
    ids_dir = make_draw()
    tasks = sorted({c[0] for c in CASES})
    report, ok = {}, True
    for mode in ("gold", "wrong"):
        port = free_port()
        srv = subprocess.Popen([PY, "-I", str(EVAL / "tests" / "oracle_server.py"), "--tasks", ",".join(tasks),
                                "--mode", mode, "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
        try:
            assert wait_up(port), "oracle server did not start"
            ep = f"http://127.0.0.1:{port}/v1"
            out = EVAL / "results" / f"e2e_oracle_{mode}"
            for task, pool, extra, key, gmin, wmax in CASES:
                extra = [ep if x == "JUDGE" else str(ids_dir / (x[5:] + ".json")) if x.startswith("DRAW:")
                         else str(ids_dir.parent.parent / "gate" / (x[5:] + ".gate.json")) if x.startswith("GATE:") else x
                         for x in extra]
                cmd = [PY, "-I", str(EVAL / "run_eval.py"), "--task", task, "--pool", pool, "--endpoint", ep,
                       "--model", "oracle-model", "--out", str(out), "--no-cache", "--quiet", "--concurrency", "32"] + extra
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout[-2000:], r.stderr[-4000:])
                    report[f"{mode}/{task}"] = {"pass": False, "error": r.stderr[-1000:]}
                    ok = False
                    continue
                s = json.load(open(out / "summary.json"))[task]
                if key is None and task in ("ifeval", "ifeval_train"):  # run-only (no hand-made oracle for drawn items)
                    recs = [json.loads(l) for l in open(out / f"{task}.jsonl")]
                    passed = len(recs) == s["n_units"] and len(recs) > 0
                    val = f"graded={len(recs)}"
                elif key is None:  # code: verify generation, extraction and PENDING status
                    recs = [json.loads(l) for l in open(out / f"{task}.jsonl")]
                    pend = all(x["status"] == "pending_sandbox" and x["score"] is None for x in recs)
                    passed = pend and len(recs) == s["n_units"] and "command" in s.get("sandbox", {})
                    if mode == "gold":
                        from_ds = {json.loads(l)["id"]: json.loads(l) for l in open(EVAL.parent / f"data/processed/{task}.jsonl")}
                        gold = (lambda d: d["prompt"] + d["reference"]) if task == "humanevalplus" else (lambda d: d["reference"])
                        exact = sum(x["parsed_answer"].strip() == gold(from_ds[x["id"]]).strip() for x in recs)
                        passed &= exact == len(recs)
                        val = f"pending={len(recs)} gold_code_extracted={exact}/{len(recs)}"
                    else:
                        val = f"pending={len(recs)}"
                else:
                    val = s.get(key)
                    passed = val is not None and (val >= gmin if mode == "gold" else val <= wmax)
                ok &= passed
                report[f"{mode}/{task}"] = {"pool": pool, "metric": key, "value": val, "n": s["n"],
                                            "n_units": s["n_units"], "pass": passed}
                if task == "bfcl":
                    report[f"{mode}/{task}"]["irrelevance_vs_live_simple"] = s["irrelevance_vs_live_simple"]
                print(f"[{mode:5s}] {task:24s} pool={pool:6s} {key or 'code'}={val} n={s['n']} "
                      f"{'PASS' if passed else 'FAIL'}", flush=True)
            stats = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats").read())
            report[f"{mode}/oracle_stats"] = stats
            if stats["misses"]:
                print(f"[{mode}] WARNING oracle misses={stats['misses']}")
        finally:
            srv.terminate()
            srv.wait(timeout=30)
    json.dump(report, open(EVAL / "tests" / "e2e_report.json", "w"), indent=1, default=str)
    print("\nRESULT:", "E2E PASSED" if ok else "E2E FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
