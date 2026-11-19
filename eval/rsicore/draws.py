"""Wrappers around data/scripts/draw.py (run as a subprocess with `python -I`).

Layout written by draw.py inside the run dir: draws/draw_config.yaml (locked), draws/gate/,
draws/gen_XX/{manifest.json, ids/<dataset>.<role>.json}, draws/ledger.json.
"""
import hashlib
import json
import subprocess
from pathlib import Path

from .config import ROOT, eval_python

DRAW_PY = ROOT / "data" / "scripts" / "draw.py"
DOMAIN_TRAIN = {"math": ["dapo_math"], "code": ["taco_verified"], "knowledge": ["triviaqa_train"],
                "instruction_following": ["ifeval_train"], "tool_calling": ["sealtool_train"]}
ROLES = ("signal", "screen", "train", "kl")


def _ddir(rc) -> Path:
    return Path(rc.run_dir) / "draws"


def _run(args: list[str]) -> str:
    cmd = [eval_python(), "-I", str(DRAW_PY)] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"draw.py failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return r.stdout


def ensure_gate(rc) -> dict:
    """Split (first call) or reuse the fixed gate. Returns {domain: {dataset: [ids]}}."""
    man = _ddir(rc) / "gate" / "manifest.json"
    if not (man.exists() and (_ddir(rc) / "draw_config.yaml").exists()):
        _run(["--run", rc.run_dir, "--gate-only", "--config", rc.draw_config])
    return json.loads(man.read_text())["domains"]


def draw_gen(rc, gen: int, target: str | None = None) -> dict:
    """Draw generation `gen` (idempotent: an existing draw with the same target is reused). Returns the manifest."""
    man = _ddir(rc) / f"gen_{int(gen):02d}" / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        if (m.get("target") or None) == (target or None):
            return m
    args = ["--run", rc.run_dir, "--gen", int(gen), "--config", rc.draw_config]
    if target:
        args += ["--target", target]
    _run(args)
    return json.loads(man.read_text())


def ids_file(rc, gen: int, dataset: str, role: str) -> Path:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    return _ddir(rc) / f"gen_{int(gen):02d}" / "ids" / f"{dataset}.{role}.json"


def ids(rc, gen: int, dataset: str, role: str) -> list[str]:
    p = ids_file(rc, gen, dataset, role)
    return json.loads(p.read_text()) if p.exists() else []


def gate_ids_file(rc, dataset: str) -> Path:
    return _ddir(rc) / "gate" / f"{dataset}.gate.json"


def gate_ids(rc, dataset: str) -> list[str]:
    p = gate_ids_file(rc, dataset)
    return json.loads(p.read_text()) if p.exists() else []


def mark_shown(rc, gen: int, shown: dict[str, list[str]]) -> None:
    """Record items whose content was shown to the editor in generation `gen` (never signal again)."""
    shown = {ds: sorted(set(v)) for ds, v in (shown or {}).items() if v}
    if not shown:
        return
    blob = json.dumps(shown, sort_keys=True)
    p = _ddir(rc) / f"gen_{int(gen):02d}" / f"shown_req_{hashlib.sha256(blob.encode()).hexdigest()[:12]}.json"
    p.write_text(blob)
    _run(["--run", rc.run_dir, "--gen", int(gen), "--mark-shown", p])
