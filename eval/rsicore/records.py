"""Run-directory record helpers (layout: eval/rsicore/INTERFACES.md §2)."""
import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()


def gen_dir(rc, gen: int) -> Path:
    d = Path(rc.run_dir) / f"gen_{int(gen):02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj) -> None:
    """Atomic JSON write (tmp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}_{threading.get_ident()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def append_jsonl(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    with _LOCK, open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # torn last line after a crash
                pass
    return out


def update_summary(rc, gen: int, info: dict) -> None:
    """Merge `info` into summary.json["gens"][str(gen)] (one entry per generation, appended/updated)."""
    p = Path(rc.run_dir) / "summary.json"
    with _LOCK:
        s = read_json(p, None) or {"run_id": rc.run_id, "paper": rc.paper, "domain": rc.domain, "gens": {}}
        g = s["gens"].setdefault(str(int(gen)), {})
        g.update(info)
        g["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(p, s)
