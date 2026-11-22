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


def ensure_paths() -> None:
    """python -I does not put the script directory on sys.path: add the repo root (for `lora_code`)
    and eval/ (for `rsicore`, `evallib`)."""
    for p in (str(REPO), str(EVAL_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


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
    """Plain text of the (rendered) prompt (the screen items shown to the editor, activation statistics)."""
    try:
        msgs = render_messages(dataset, item)
        return "\n\n".join(m["content"] for m in msgs if m.get("content"))
    except Exception:  # noqa: BLE001
        if item.get("prompt"):
            return item["prompt"]
        return "\n\n".join(m.get("content", "") for m in (item.get("messages") or []))


MODES = ("scratch", "weights", "edit")   # 2026-11-21: scratch (write a LoRA) is the default; weights = edit the weights
#                                        directly (any rank); edit = the one mode that trains (raw LoRA, generation 0)


def method_mode(cfg: dict) -> str:
    m = str((cfg.get("method") or {}).get("mode") or "scratch")
    if m not in MODES:
        raise SystemExit(f"method.mode must be one of {MODES}, got {m!r}")
    return m


MERGE_MODES = ("merge", "adapter")


def merge_mode(cfg: dict) -> str:
    """How an accepted LoRA becomes the next model (method.mode scratch; 2026-11-22). merge (code default) = merged
    into the bf16 weights (a new checkpoint gen_XX/model, served as m{g+1}); adapter = never merged: M_g = M_0 + C_g,
    C_g the rank-concatenation of every accepted LoRA (SVD re-compressed above merge.adapter_rank_budget), served as
    one adapter on the M_0 server for the whole run. mode=edit is adapter-only (merge.mode ignored); mode=weights
    cannot be an adapter. The key has no live default in base.yaml (paper_cfg is locked: older runs resume)."""
    m = str((cfg.get("merge") or {}).get("mode") or "merge")
    if m not in MERGE_MODES:
        raise SystemExit(f"merge.mode must be one of {MERGE_MODES}, got {m!r}")
    if m == "adapter" and method_mode(cfg) == "weights":
        raise SystemExit("merge.mode adapter needs LoRA candidates: method.mode weights writes dense edits")
    return m


def adapter_rank_budget(cfg: dict) -> int:
    """merge.mode adapter: max rank per module of the cumulative LoRA C_g (default 256)."""
    return int((cfg.get("merge") or {}).get("adapter_rank_budget") or 256)


def server_lora_rank(cfg: dict) -> int:
    """--max-lora-rank of the main servers: lora.rank_cap; scratch + merge.mode adapter: padded(max(budget, rank_cap)),
    since a candidate is served as C_g (+) delta truncated to the budget (compose_cumulative); vLLM tops out at 512."""
    from lora_code.primitives.lora import padded_rank
    cap = int((cfg.get("lora") or {}).get("rank_cap", 64))
    if method_mode(cfg) == "scratch" and merge_mode(cfg) == "adapter":
        need = max(adapter_rank_budget(cfg), cap)
        r = padded_rank(need)
        if r > 512:
            raise SystemExit(f"merge.adapter_rank_budget / lora.rank_cap = {need} > 512 (vLLM's largest --max-lora-rank)")
        return r
    return cap


def gate_policy(cfg: dict) -> tuple[str, str]:
    """(reference, fallback) of the gate: 'auto' resolves by method.mode (edit: raw / raw, scratch and weights:
    parent / keep)."""
    mode = method_mode(cfg)
    g = cfg.get("gate") or {}
    ref = g.get("reference") or "auto"
    fb = g.get("fallback") or "auto"
    if ref == "auto":
        ref = {"edit": "raw", "scratch": "parent", "weights": "parent"}[mode]
    if fb == "auto":
        fb = {"edit": "raw", "scratch": "keep", "weights": "keep"}[mode]
    if ref not in ("raw", "parent") or fb not in ("raw", "keep"):
        raise SystemExit(f"gate.reference must be raw|parent and gate.fallback raw|keep, got {ref!r} / {fb!r}")
    if mode != "edit" and "raw" in (ref, fb):
        raise SystemExit(f"gate reference/fallback 'raw' needs method.mode=edit (mode is {mode!r})")
    return ref, fb


PAPER_B_GROWTH = [("main", 2), ("train", 2), ("main", 4), ("main", 8)]


def gpu_layout(rc, cfg: dict) -> tuple[dict, list[int]]:
    """({role: [GPU ids]}, [unused GPU ids]) for Paper B. Roles: main (vLLM: M_t + candidate LoRAs, the editor),
    train (HF model: weight store, diagnostics, merge, edit's raw LoRA; several GPUs = the HF model is split across
    them).
    gpu_layout.main_replicas (default, 2026-11-20): train = the last gpu_layout.train_gpus GPUs (default 1), main =
    every other GPU, served as independent replicas of gpu_layout.main_tp GPUs each (main_groups): 4 GPUs -> main
    [0, 1, 2] (3 replicas) | train [3]. gpu_roles pins roles (then main is one server).
    Without main_replicas (the earlier layout): every GPU of --gpus is used (rsicore.gpu_layout.allocate); several
    GPUs for main = tensor parallelism. Extra GPUs go to main (TP 2), train (2), main (TP 4), main (TP 8), and
    whatever is left to train (the HF model can be split over any number of GPUs), so no GPU stays idle.
    e.g. 2 -> main | train, 3 -> main x2 | train, 4 -> main x2 | train x2, 5 -> main x2 | train x3,
    6 -> main x4 | train x2.
    gpu_roles set: those roles are pinned (indices into --gpus, negative = from the end); the others are laid
    out automatically on the remaining GPUs."""
    from rsicore.gpu_layout import layout
    gpus = list(rc.gpus or [0])
    lay = cfg.get("gpu_layout") or {}
    if lay.get("main_replicas", True) and not cfg.get("gpu_roles"):
        n = len(gpus)
        if n == 1:
            if not lay.get("share_single_gpu", False):
                raise ValueError(f"Paper B on --gpus {gpus}: needs at least 2 GPUs (main | train)")
            return {"main": gpus, "train": gpus}, []
        nt = min(max(1, int(lay.get("train_gpus", 1))), n - 1)
        return {"main": gpus[:n - nt], "train": gpus[n - nt:]}, []
    try:
        idx, unused = layout(len(gpus), own=["main", "train"], growth=PAPER_B_GROWTH, spill="train",
                             share_single=bool(lay.get("share_single_gpu", False)), explicit=cfg.get("gpu_roles"))
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


def main_server_args(cfg: dict) -> list[str]:
    """vLLM args of every server started from server.main_args (the main replicas, which also host the editor;
    weights mode: the temporary servers on the candidate slots too, so every scoring server has the same context):
    server.main_args with --max-model-len = editor.context_tokens (2026-11-21: the one knob for the editor's
    context length; it overrides a --max-model-len written in main_args; null keeps main_args as written)."""
    from rsicore.editor import apply_context_tokens
    args = apply_context_tokens(list((cfg.get("server") or {}).get("main_args", []) or []), cfg)
    return replica_memory(args, cfg)


def tools_on_replicas(cfg: dict) -> bool:
    """gpu_layout.tools_on_replicas (2026-11-22): every main replica's GPU also holds an HF copy of the current model
    and its own weight store, and editor session k runs its tools (svd / acts / train_loss on
    HF) there, on the replica its editor talks to (k mod n), instead of all sessions queueing on the train GPU.
    Never in method.mode weights (the main GPUs are candidate slots there)."""
    return bool((cfg.get("gpu_layout") or {}).get("tools_on_replicas")) and method_mode(cfg) != "weights"


def replica_memory(args: list, cfg: dict) -> list[str]:
    """Main-replica vLLM args with --rsi-gpu-memory-gib = gpu_layout.replica_vllm_gib when tools_on_replicas (room
    for the HF copy on the same card); the --gpu-memory-utilization fallback (used without nvidia-smi) is scaled by
    the same ratio. Unchanged otherwise."""
    from rsicore.servers import MEM_FLAG
    args = [str(a) for a in args]
    gib = (cfg.get("gpu_layout") or {}).get("replica_vllm_gib")
    if not tools_on_replicas(cfg) or not gib or MEM_FLAG not in args:
        return args
    i = args.index(MEM_FLAG)
    old = float(args[i + 1])
    args[i + 1] = f"{float(gib):g}"
    if "--gpu-memory-utilization" in args and old > 0:
        j = args.index("--gpu-memory-utilization")
        args[j + 1] = str(round(float(args[j + 1]) * float(gib) / old, 3))
    return args
