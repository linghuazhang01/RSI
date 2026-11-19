"""Small helpers shared by edit_code modules (no torch / rsicore dependency)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EDIT = REPO / "edit_code"
DATA = REPO / "data"
GIB = 1024 ** 3


def ensure_paths() -> None:
    """`python -I` does not put the script directory on sys.path: add the repo root (for
    `edit_code.*`) and eval/ (for `rsicore`)."""
    for p in (str(REPO), str(REPO / "eval")):
        if p not in sys.path:
            sys.path.insert(0, p)


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    """Atomic JSON write (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def config_hash(obj) -> str:
    return hashlib.sha1(canonical(obj).encode()).hexdigest()[:12]


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- processed data
_PROCESSED: dict[str, dict] = {}


def load_processed(dataset: str, ids: list[str] | None = None) -> dict[str, dict]:
    """id -> processed record for one dataset (cached in memory per process)."""
    if dataset not in _PROCESSED:
        rows = {}
        path = DATA / "processed" / f"{dataset}.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    rows[r["id"]] = r
        _PROCESSED[dataset] = rows
    rows = _PROCESSED[dataset]
    if ids is None:
        return rows
    return {i: rows[i] for i in ids if i in rows}


def chatml(messages: list[dict]) -> str:
    out = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, str):
            c = json.dumps(c, ensure_ascii=False)
        out.append(f"<|im_start|>{m.get('role', 'user')}\n{c}<|im_end|>")
    return "\n".join(out)


def record_text(rec: dict, response: str | None = None, max_chars: int = 8000) -> str:
    """Render a processed record as a ChatML text for KL / calibration (prompt only, or prompt +
    a self-generated response)."""
    if rec.get("messages"):
        msgs = list(rec["messages"])
    else:
        msgs = [{"role": "user", "content": rec.get("prompt") or ""}]
    text = chatml(msgs) + "\n<|im_start|>assistant\n"
    if response:
        text += response + "<|im_end|>"
    return text[:max_chars]


def tail(s: str | None, n: int = 1500) -> str:
    s = s or ""
    return s if len(s) <= n else "..." + s[-n:]


def role_gpus(rc, cfg: dict, role: str) -> list[int]:
    """GPU ids for a role, from cfg["gpu_roles"][role] = indices into --gpus (negative = from the end).
    Paper A roles: editor (vLLM: M_0 / editor, references), candidate (vLLM candidate server; layer cache,
    GPTQ and diagnostics run on its first GPU). Several GPUs for a vLLM role = tensor parallelism.
    Defaults (one 288 GB B300 per role): editor [0], candidate [1] (both [0] with a single GPU)."""
    gpus = list(rc.gpus or [0])
    roles = cfg.get("gpu_roles") or {}
    default = {"editor": [0], "candidate": [1]}
    if role in roles:
        idx = [int(i) for i in roles[role]]
        bad = [i for i in idx if not -len(gpus) <= i < len(gpus)]
        if bad:
            raise ValueError(f"gpu_roles.{role} = {idx} needs more GPUs than --gpus {gpus}")
        return [gpus[i] for i in idx]
    return [gpus[min(i, len(gpus) - 1)] for i in default[role]]


def free_cuda():
    """Return cached CUDA memory to the driver (a vLLM server is about to start on the same GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
