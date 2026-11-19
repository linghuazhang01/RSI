"""Shared helpers for Paper B (paths, domains, training-item access, small JSON utils).

Nothing here imports torch / transformers; GPU-only code lives in lora_code/primitives/.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

LORA_DIR = Path(__file__).resolve().parent
REPO = LORA_DIR.parent
EVAL_DIR = REPO / "eval"
DATA_DIR = REPO / "data"
PROC_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"


def ensure_paths() -> None:
    """python -I does not put the script directory on sys.path: add the repo root (for `lora_code`)
    and eval/ (for `rsicore`, `evallib`)."""
    for p in (str(REPO), str(EVAL_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


DOMAINS = ["math", "code", "knowledge", "instruction_following", "tool_calling"]
# one training set per domain (2026-11-19); the run_eval task name equals the dataset name
DOMAIN_DATASET = {"math": "dapo_math", "code": "taco_verified", "knowledge": "triviaqa_train",
                  "instruction_following": "ifeval_train", "tool_calling": "sealtool_train"}
DATASET_DOMAIN = {v: k for k, v in DOMAIN_DATASET.items()}


def retention_domains(target: str) -> list[str]:
    return [d for d in DOMAINS if d != target]


# ----------------------------------------------------------------------------- json helpers
def read_json(p: Path, default=None):
    p = Path(p)
    if not p.exists():
        return default
    with open(p) as f:
        return json.load(f)


def write_json(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_json_default)
    os.replace(tmp, p)


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


def stable_hash(*parts, n: int = 12) -> str:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return h[:n]


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------- items
def load_records(dataset: str, ids) -> dict[str, dict]:
    """Stream data/processed/<dataset>.jsonl and keep only `ids` (the files can be large)."""
    want = set(ids)
    out = {}
    path = PROC_DIR / f"{dataset}.jsonl"
    if not want or not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            # cheap pre-filter on the id field before parsing the whole line
            r = json.loads(line)
            if r["id"] in want:
                out[r["id"]] = r
                if len(out) == len(want):
                    break
    return out


def pool_ids(dataset: str) -> list[str]:
    return read_json(SPLIT_DIR / dataset / "pool.json", [])


@lru_cache(maxsize=1)
def eval_tasks():
    """The run_eval task registry (read-only import of eval/evallib): exact prompts the evaluator sends
    and gold-format oracle answers, so SFT / gradient data match what is scored."""
    ensure_paths()
    import yaml
    from evallib.tasks import load_tasks  # noqa: WPS433
    return load_tasks(yaml.safe_load(open(EVAL_DIR / "configs" / "tasks.yaml")))


def render_messages(dataset: str, item: dict) -> list[dict]:
    t = eval_tasks()[dataset]
    units = t.units(item)
    return units[0][1]["messages"]


def gold_target(dataset: str, item: dict) -> str | None:
    """Gold response in the evaluator's expected format (None when the dataset has no reference text,
    e.g. RLVR-IFeval items without a hand-made oracle)."""
    try:
        c = eval_tasks()[dataset].oracle(item, "0", "gold")["content"]
    except Exception:  # noqa: BLE001
        return None
    if not c or c.startswith("(no hand-made") or "no gold solution" in c:
        return None
    return c


def prompt_text(dataset: str, item: dict) -> str:
    """Plain text of the (rendered) prompt, used for token-KL contexts and activation statistics."""
    try:
        msgs = render_messages(dataset, item)
        return "\n\n".join(m["content"] for m in msgs if m.get("content"))
    except Exception:  # noqa: BLE001
        if item.get("prompt"):
            return item["prompt"]
        return "\n\n".join(m.get("content", "") for m in (item.get("messages") or []))


def role_gpus(rc, cfg: dict, role: str) -> list[int]:
    """GPU ids for a role, from cfg["gpu_roles"][role] = indices into --gpus (negative = from the end).
    Paper B roles: main (vLLM: M_t + candidate LoRAs, the editor), theta0 (vLLM: original model, E_t and
    references), train (HF model for training / gradient tools, weight store, diagnostics, merge; several
    GPUs = the HF model is split across them). Several GPUs for a vLLM role = tensor parallelism.
    Defaults (288 GB B300): main [0], theta0 [-1], train [-1]."""
    gpus = list(rc.gpus or [0])
    roles = cfg.get("gpu_roles") or {}
    default = {"main": [0], "theta0": [-1], "train": [-1]}
    idx = [int(i) for i in roles.get(role, default[role])]
    bad = [i for i in idx if not -len(gpus) <= i < len(gpus)]
    if bad:
        raise ValueError(f"gpu_roles.{role} = {idx} needs more GPUs than --gpus {gpus}")
    return [gpus[i] for i in idx]
