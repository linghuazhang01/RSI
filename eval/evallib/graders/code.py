"""Host side of code grading. NEVER executes generated code on the host: it only extracts code,
writes sample files, and launches the Docker sandbox (eval/sandbox) with --network none.
If Docker is unavailable, items are left with score=None, status='pending_sandbox', and the exact
command to run later is recorded in summary.json (and printed)."""
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from ..common import EVAL_DIR, RAW_DIR, extract_code_block

IMAGE = os.environ.get("RSI_SANDBOX_IMAGE", "rsi-eval-sandbox:0.1")


def extract_solution(task: str, item: dict, text: str) -> str:
    code = extract_code_block(text)
    if task == "humanevalplus" and f"def {item['metadata']['entry_point']}" not in code:
        code = item["prompt"] + code  # body-only completion -> prepend the signature/docstring
    return code


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def write_samples(task: str, work: Path, records: list[dict]) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    p = work / f"{task}.samples.jsonl"
    with open(p, "w") as f:
        for r in records:
            for k, code in enumerate(r["_codes"]):
                if task == "livecodebench":
                    f.write(json.dumps({"question_id": r["id"].split("/", 1)[1], "code": code, "sample_idx": k}) + "\n")
                elif task == "apps":
                    f.write(json.dumps({"question_id": r["id"], "code": code, "sample_idx": k}) + "\n")
                else:
                    f.write(json.dumps({"task_id": r["id"], "solution": code}) + "\n")
    return p


def docker_cmd(task: str, work: Path, parallel: int = 4) -> list[str]:
    work = work.resolve()
    return ["docker", "run", "--rm", "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=4g", "--memory", "8g", "--cpus", str(parallel),
            "--pids-limit", "1024", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-v", f"{work}:/work:rw",
            "-v", f"{(RAW_DIR / 'evalplus_release').resolve()}:/data/evalplus:ro",
            "-v", f"{(RAW_DIR / 'livecodebench').resolve()}:/data/lcb:ro",
            "-v", f"{(RAW_DIR.parent / 'processed' / 'aux').resolve()}:/data/aux:ro",
            IMAGE, "--task", task, "--samples", f"/work/{task}.samples.jsonl",
            "--out", f"/work/{task}.sandbox_results.json", "--parallel", str(parallel)]


def build_cmd() -> list[str]:
    return ["docker", "build", "-t", IMAGE, "-f", str(EVAL_DIR / "sandbox" / "Dockerfile"), str(EVAL_DIR)]


def run_sandbox(task: str, work: Path, parallel: int = 4):
    """Returns per-task results dict or None (pending)."""
    if not docker_available():
        return None
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        subprocess.run(build_cmd(), check=True)
    os.chmod(work, 0o777)  # container runs as uid 65534
    subprocess.run(docker_cmd(task, work, parallel), check=True)
    return json.load(open(work / f"{task}.sandbox_results.json"))["per_task"]


def pending_command(task: str, run_dir: Path) -> str:
    return (f"{shlex.join(build_cmd())} && "
            f"eval/.venv/bin/python -I eval/grade_code.py --run-dir {shlex.quote(str(run_dir))} --task {task}")
