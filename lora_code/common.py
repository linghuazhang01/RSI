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


MODES = ("edit", "scratch")   # 2026-11-20: no SFT baseline, no free mode; only edit trains (one raw LoRA, gen 0)


def method_mode(cfg: dict) -> str:
    m = str((cfg.get("method") or {}).get("mode") or "edit")
    if m not in MODES:
        raise SystemExit(f"method.mode must be one of {MODES}, got {m!r}")
    return m


def gate_policy(cfg: dict) -> tuple[str, str]:
    """(reference, fallback) of the gate: 'auto' resolves by method.mode (edit: raw / raw, scratch: parent / keep)."""
    mode = method_mode(cfg)
    g = cfg.get("gate") or {}
    ref = g.get("reference") or "auto"
    fb = g.get("fallback") or "auto"
    if ref == "auto":
        ref = {"edit": "raw", "scratch": "parent"}[mode]
    if fb == "auto":
        fb = {"edit": "raw", "scratch": "keep"}[mode]
    if ref not in ("raw", "parent") or fb not in ("raw", "keep"):
        raise SystemExit(f"gate.reference must be raw|parent and gate.fallback raw|keep, got {ref!r} / {fb!r}")
    if mode != "edit" and "raw" in (ref, fb):
        raise SystemExit(f"gate reference/fallback 'raw' needs method.mode=edit (mode is {mode!r})")
    return ref, fb


PAPER_B_GROWTH = [("main", 2), ("train", 2), ("main", 4), ("theta0", 2), ("main", 8), ("theta0", 4)]


def gpu_layout(rc, cfg: dict) -> tuple[dict, list[int]]:
    """({role: [GPU ids]}, [unused GPU ids]) for Paper B.
    gpu_layout.main_replicas (default, 2026-11-20): train = the last gpu_layout.train_gpus GPUs (default 1), theta0
    = its own GPU before them when E_t is on and theta0_own_gpu (else it shares the first train GPU), main = every
    other GPU, served as independent replicas of gpu_layout.main_tp GPUs each (main_groups): 4 GPUs, single domain
    -> main [0, 1, 2] (3 replicas) | train [3]. gpu_roles pins roles as before (then main is one server).
    Without main_replicas (the earlier layout): Roles: main (vLLM: M_t + candidate LoRAs, the
    editor), theta0 (vLLM: original model, only for E_t / e_ref), train (HF model for training, weight store,
    diagnostics, merge; several GPUs = the HF model is split across them). Several GPUs for a vLLM role =
    tensor parallelism.
    gpu_roles not set (default): every GPU of --gpus is used (rsicore.gpu_layout.allocate). theta0 gets its
    own GPU only when E_t is on and gpu_layout.theta0_own_gpu (h20: a 27B per card); otherwise it shares the
    first train GPU. Extra GPUs go to main (TP 2), train (2), main (TP 4), theta0 (TP 2), main (TP 8), and
    whatever is left to train (the HF model can be split over any number of GPUs), so no GPU stays idle.
    e.g. h20 single-domain: 2 -> main | train, 3 -> main x2 | train, 4 -> main x2 | train x2,
    5 -> main x2 | train x3, 6 -> main x4 | train x2.
    gpu_roles set: those roles are pinned (indices into --gpus, negative = from the end); the others are laid
    out automatically on the remaining GPUs."""
    from rsicore.gpu_layout import layout
    gpus = list(rc.gpus or [0])
    lay = cfg.get("gpu_layout") or {}
    own_theta0 = bool(lay.get("theta0_own_gpu", False)) and bool((cfg.get("e_t") or {}).get("enabled", True))
    if lay.get("main_replicas", True) and not cfg.get("gpu_roles"):
        n = len(gpus)
        if n == 1:
            if not lay.get("share_single_gpu", False):
                raise ValueError(f"Paper B on --gpus {gpus}: needs at least 2 GPUs (main | train)")
            return {"main": gpus, "train": gpus, "theta0": gpus}, []
        nt = max(1, int(lay.get("train_gpus", 1)))
        nt = min(nt, n - 1 - int(own_theta0))
        if nt < 1:
            raise ValueError(f"Paper B on --gpus {gpus}: needs at least {2 + int(own_theta0)} GPUs "
                             "(main | train" + (" | theta0" if own_theta0 else "") + ")")
        train, rest = gpus[n - nt:], gpus[:n - nt]
        theta0 = [rest.pop()] if own_theta0 else [train[0]]
        return {"main": rest, "train": train, "theta0": theta0}, []
    try:
        idx, unused = layout(len(gpus), own=["main", "theta0", "train"] if own_theta0 else ["main", "train"],
                             growth=PAPER_B_GROWTH, shared={} if own_theta0 else {"theta0": "train"},
                             spill="train", share_single=bool(lay.get("share_single_gpu", False)),
                             explicit=cfg.get("gpu_roles"))
    except ValueError as e:
        raise ValueError(f"Paper B on --gpus {gpus}: {e}") from None
    return {r: [gpus[i] for i in ii] for r, ii in idx.items()}, [gpus[i] for i in unused]


def role_gpus(rc, cfg: dict, role: str) -> list[int]:
    """GPU ids of one role (see gpu_layout)."""
    return gpu_layout(rc, cfg)[0][role]


def main_groups(rc, cfg: dict) -> list[list[int]]:
    """GPU groups of the main replicas (gpu_layout.main_replicas): main GPUs in chunks of gpu_layout.main_tp
    (tensor parallelism inside a replica; leftover GPUs that do not fill a chunk are not used). Without
    main_replicas, or with gpu_roles pinned: one server on all main GPUs."""
    lay = cfg.get("gpu_layout") or {}
    main = role_gpus(rc, cfg, "main")
    if not lay.get("main_replicas", True) or cfg.get("gpu_roles"):
        return [main]
    tp = max(1, int(lay.get("main_tp", 1)))
    groups = [main[i:i + tp] for i in range(0, len(main) - len(main) % tp, tp)]
    return groups or [main]

