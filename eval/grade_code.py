#!/usr/bin/env python
"""Grade code tasks of an existing run inside the Docker sandbox (for runs made while Docker was
unavailable, i.e. records with status=pending_sandbox).

  eval/.venv/bin/python -I eval/grade_code.py --run-dir eval/results/<run_id> --task humanevalplus

Requires the Docker daemon and the sandbox image (built automatically from eval/sandbox/Dockerfile).
The container runs with --network none, read-only rootfs, no capabilities, non-root user.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from evallib.common import EVAL_DIR  # noqa: E402
from evallib.graders import code as C  # noqa: E402
from evallib.summary import summarize, update_summary  # noqa: E402
from evallib.tasks import load_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args()
    run = Path(a.run_dir)
    task = load_tasks(yaml.safe_load(open(EVAL_DIR / "configs" / "tasks.yaml")))[a.task]
    g = task.cfg["grader"]
    if not C.docker_available():
        sys.exit("Docker daemon not available. Start Docker, then rerun this command.")
    recs = [json.loads(l) for l in open(run / f"{a.task}.jsonl")]
    for r in recs:  # rebuild samples from stored parsed code (same extraction as at generation time)
        r["_codes"] = r["parsed_answer"] if isinstance(r["parsed_answer"], list) else [r["parsed_answer"]]
    C.write_samples(g, run / "sandbox", recs)
    per = C.run_sandbox(g, run / "sandbox", a.parallel)
    key = (lambda r: r["id"].split("/", 1)[1]) if g == "livecodebench" else (lambda r: r["id"])
    for r in recs:
        res = per.get(key(r), [])
        r["sample_scores"] = [float(x["plus"]) for x in res]
        r["metrics"] = res
        r["score"] = sum(r["sample_scores"]) / len(res) if res else 0.0
        r["status"] = "ok"
        r.pop("_codes", None)
    with open(run / f"{a.task}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    old = json.load(open(run / "summary.json")).get(a.task, {})
    keep = {k: v for k, v in old.items() if k in ("pool", "model", "lora", "served_model_name", "endpoint",
                                                     "decoding", "versions", "timestamp")}
    keep["sandbox"] = {"status": "done", "image": C.IMAGE}
    s = summarize(task, recs, keep)
    update_summary(run, a.task, s)
    print(f"[{a.task}] accuracy={s['accuracy']:.4f} n={s['n']}")


if __name__ == "__main__":
    main()
