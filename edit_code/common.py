"""Small helpers shared by edit_code modules (no torch dependency; rsicore only imported lazily, by candidate_args)."""
from __future__ import annotations

import hashlib
import json
import os
import re
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


NOTE_CHARS = 600


def cut_note(note, n: int = NOTE_CHARS) -> str:
    """A program's "note" (2026-11-22, as Paper B's lora_code.tools.cut_note): whitespace collapsed, at most n
    characters."""
    t = " ".join(str(note).split())
    return t if len(t) <= n else t[:n - 3].rstrip() + "..."


# the automatic note api.solve_allocation puts on the config it returns (edit_code.allocation.solve)
AUTO_NOTE = re.compile(r"^\s*solve_allocation: [\d.]+ GiB of [\d.]+; predicted weighted damage [-+\d.eE]+"
                       r"(?: \(INFEASIBLE[^)]*\))?")


def program_note(result) -> str | None:
    """The editor's own note in a program's returned config (its "note" key), cut to NOTE_CHARS. The automatic text of
    api.solve_allocation is removed (what the editor appended to it is kept); None when nothing of its own is left."""
    note = result.get("note") if isinstance(result, dict) else None
    if not isinstance(note, str):
        return None
    note = AUTO_NOTE.sub("", note, count=1).strip().lstrip(";,.:-|").strip()
    return cut_note(note) if note else None


def error_line(error, n: int = 200) -> str:
    """One line of a candidate's error for the archive / gate.json (2026-11-22): the first line, joined with the next
    ones while it ends with ':' (schema.ConfigError: "invalid quantization config:" + the problems below)."""
    lines = [x.strip() for x in str(error or "").strip().splitlines() if x.strip()]
    out = lines[0] if lines else ""
    for x in lines[1:4]:
        if not out.endswith(":"):
            break
        out += " " + x
    return out[:n]


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


def gpu_layout(rc, cfg: dict) -> tuple[dict, list[int]]:
    """({role: [GPU ids]}, [unused GPU ids]) for Paper A. Roles: editor (vLLM: M_0 / editor, references;
    it is idle while candidates are evaluated, so ONE GPU by default) and candidate (every other GPU: the
    candidate servers, plus the layer cache / GPTQ / calibration on its first GPU).
    The candidate GPUs are split into groups of gpu_layout.candidate_group_size (1 / 2 / 4 / 8, tensor
    parallel inside a group); each group serves one candidate, so len(groups) candidates are evaluated at
    the same time (candidate_groups). GPUs left over by the grouping are reported as unused.
    1 GPU: editor and candidate share it if gpu_layout.share_single_gpu (B300 / mock; h20 forbids it).
    gpu_roles set: those roles are pinned (indices into --gpus, negative = from the end), e.g.
    --set 'gpu_roles.editor=[0,1]' (editor with TP 2); the other role takes the remaining GPUs."""
    from rsicore.gpu_layout import layout
    gpus = list(rc.gpus or [0])
    lay = cfg.get("gpu_layout") or {}
    try:
        idx, unused = layout(len(gpus), own=["editor", "candidate"], growth=[], spill="candidate",
                             share_single=bool(lay.get("share_single_gpu", True)), explicit=cfg.get("gpu_roles"))
    except ValueError as e:
        raise ValueError(f"Paper A on --gpus {gpus}: {e}") from None
    roles = {r: [gpus[i] for i in ii] for r, ii in idx.items()}
    unused = [gpus[i] for i in unused]
    gs = _group_size(cfg)
    cand = roles["candidate"]
    if len(cand) >= gs:
        keep = len(cand) // gs * gs
        unused += cand[keep:]
        roles["candidate"] = cand[:keep]
    return roles, unused


def _group_size(cfg: dict) -> int:
    gs = int((cfg.get("gpu_layout") or {}).get("candidate_group_size", 1) or 1)
    if gs not in (1, 2, 4, 8):
        raise ValueError(f"gpu_layout.candidate_group_size must be 1, 2, 4 or 8 (tensor parallel), got {gs}")
    return gs


def candidate_groups(rc, cfg: dict) -> list[list[int]]:
    """The candidate GPUs in groups of gpu_layout.candidate_group_size: one candidate server per group,
    evaluated concurrently (fewer GPUs than one group: a single group of all of them)."""
    cand = gpu_layout(rc, cfg)[0]["candidate"]
    gs = _group_size(cfg)
    if len(cand) < gs:
        return [cand]
    return [cand[i:i + gs] for i in range(0, len(cand), gs)]


def role_gpus(rc, cfg: dict, role: str) -> list[int]:
    """GPU ids of one role (see gpu_layout)."""
    return gpu_layout(rc, cfg)[0][role]


def candidate_args(cfg: dict) -> list[str]:
    """vLLM args of a candidate server (and of M_0's replicas / exported models on candidate GPUs):
    vllm.candidate_args, with --max-model-len = vllm.candidate_max_model_len when that is set (2026-11-21, base.yaml:
    32768, so a prompt + 16384 thinking tokens fit; null / absent = the args as written)."""
    from rsicore.servers import set_max_model_len
    v = cfg.get("vllm") or {}
    n = v.get("candidate_max_model_len")
    return set_max_model_len(list(v.get("candidate_args") or []), None if n is None else int(n))


def search_settings(scfg: dict) -> dict:
    """Shape of one inner search from a `search` block -> {threads, steps, max_candidates} (max_candidates null /
    absent = threads x steps). The flow itself is hard-wired (2026-11-23): candidates are never screened, every built
    candidate gets ONE gate evaluation during the search (paired vs the incumbent, gen_XX/gate.json), the passing
    candidate with the largest paired gain is accepted (run.Runner.accept_best_gate); there is no api.evaluate and no
    evaluate budget."""
    threads, steps = int(scfg["threads"]), int(scfg["steps"])
    mc = scfg.get("max_candidates")
    return {"threads": threads, "steps": steps, "max_candidates": int(threads * steps if mc is None else mc)}


def free_cuda():
    """Return cached CUDA memory to the driver (a vLLM server is about to start on the same GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
