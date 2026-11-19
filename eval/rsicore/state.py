"""Resumable step bookkeeping: run_dir/state.json = {key: info}.

    st = RunState(rc.run_dir)
    with st.step("gen01/draw") as todo:
        if todo:
            ...            # runs only if not done; marked done only when the block exits normally
    st.mark("gen01/gate", {"passed": True})   # explicit mark with info (inside a step block is fine)
"""
import contextlib
import json
import os
import threading
import time
from pathlib import Path


class RunState:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._d = json.loads(self.path.read_text()) if self.path.exists() else {}

    def done(self, key: str) -> bool:
        with self._lock:
            return key in self._d

    def get(self, key: str) -> dict | None:
        with self._lock:
            v = self._d.get(key)
            return None if v is None else dict(v)

    def mark(self, key: str, info: dict | None = None) -> None:
        with self._lock:
            rec = dict(info or {})
            rec.setdefault("t", time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._d[key] = rec
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._d, indent=1, sort_keys=True, default=str))
            os.replace(tmp, self.path)

    def unmark(self, key: str) -> None:
        """Forget a step (e.g. to force a rerun)."""
        with self._lock:
            if self._d.pop(key, None) is not None:
                tmp = self.path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(self._d, indent=1, sort_keys=True, default=str))
                os.replace(tmp, self.path)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._d)

    @contextlib.contextmanager
    def step(self, key: str):
        todo = not self.done(key)
        t0 = time.time()
        yield todo
        if todo and not self.done(key):
            self.mark(key, {"seconds": round(time.time() - t0, 2)})
