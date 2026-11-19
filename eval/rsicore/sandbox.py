"""Run model-written programs `def main(api): ...` in a child process, with tools proxied to the parent.

    api = ToolAPI({"evaluate": 24, "gpu_seconds": 7200})
    api.register("evaluate", my_eval, cost={"evaluate": 1}, doc="evaluate(config) -> L1 feedback")
    res = run_program(code, api, timeout_s=600, mem_gb=8, workdir=gen_dir / "work" / cid)

Child isolation (defence in depth; the tools themselves are trusted parent code):
  * separate process `python -I` (same interpreter as the parent), new session, environment cleared
    except PATH, HOME=TMPDIR=<workdir>/tmp, thread-count vars; cwd = workdir;
  * rlimits set in the child before the program runs: RLIMIT_AS (mem_gb; macOS cannot set it -> recorded
    as unsupported), RLIMIT_CPU (timeout_s + 5), RLIMIT_FSIZE (4 GiB), RLIMIT_NOFILE, RLIMIT_CORE=0;
  * a Python audit hook: writes only under workdir, no reads under the repo / $HOME / /proc (except the
    interpreter's own prefixes), no subprocess / exec / fork / kill / socket connect|bind|getaddrinfo.
    C extensions (e.g. via ctypes) can bypass audit hooks; this is not a security boundary on its own;
  * network: Linux -> `unshare -rn` (new user+net namespace) when it works; macOS -> `sandbox-exec`
    with a deny-network profile; otherwise none. ProgramResult.info["network_isolation"] says which;
  * wall-clock timeout (includes time spent inside tool calls; checked between RPC messages) -> SIGKILL
    of the child's process group.
Tool calls: JSON lines over two pipes; arguments/results must be JSON-serializable. Errors raised by
the tool (including BudgetExceeded) are re-raised inside the child as api.BudgetExceeded / api.ToolError.
"""
import inspect
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT

CHILD = Path(__file__).resolve().parent / "_sandbox_child.py"
_NET_PREFIX = None


class BudgetExceeded(Exception):
    pass


class ToolAPI:
    """Registered tools + budgets.

    Fixed cost (`cost`, per call) is charged BEFORE the tool runs; a call that would exceed any budget
    raises BudgetExceeded and is not run. Measured cost (`cost_fn(kwargs, result) -> {key: amount}`) is
    charged AFTER a successful call; if that pushes `spent` past a budget the call raises BudgetExceeded
    (its result is discarded, the amount stays charged). Before running a tool with a cost_fn, every budget
    key it declares in `cost` (a 0 entry declares a purely measured key) or has charged through cost_fn
    before must not be exhausted. `dedupe_key_fn(kwargs) -> str | None`: a later call of the same tool with
    the same (non-None) key returns the cached result of the first successful call and is free (no budget
    check, no charge). charge(key, amount) adds to `spent` directly (no check)."""

    def __init__(self, budget: dict[str, float]):
        self.budget = dict(budget or {})
        self.spent: dict[str, float] = {k: 0.0 for k in self.budget}
        self.calls: dict[str, int] = {}
        self.log: list[dict] = []
        self._tools: dict[str, tuple] = {}
        self._measured_keys: dict[str, set] = {}
        self._dedupe: dict[tuple, object] = {}
        self._lock = threading.RLock()

    def register(self, name: str, fn, *, cost: dict[str, float] | None = None, doc: str = "",
                 cost_fn=None, dedupe_key_fn=None) -> None:
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"bad tool name {name!r}")
        self._tools[name] = (fn, dict(cost or {}), doc or (fn.__doc__ or "").strip(), cost_fn, dedupe_key_fn)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def _check(self, cost: dict):
        for k, c in cost.items():
            if k in self.budget and self.spent.get(k, 0.0) + c > self.budget[k] + 1e-9:
                raise BudgetExceeded(f"budget '{k}' exhausted: spent {self.spent.get(k, 0.0):g} + {c:g} "
                                     f"> {self.budget[k]:g}")

    def _check_exhausted(self, keys):
        for k in keys:
            if k in self.budget and self.spent.get(k, 0.0) >= self.budget[k] - 1e-9:
                raise BudgetExceeded(f"budget '{k}' exhausted: spent {self.spent.get(k, 0.0):g} "
                                     f">= {self.budget[k]:g}")

    def charge(self, key: str, amount: float) -> None:
        with self._lock:
            self.spent[key] = self.spent.get(key, 0.0) + float(amount)

    def remaining(self) -> dict:
        return {k: self.budget[k] - self.spent.get(k, 0.0) for k in self.budget}

    def call(self, name: str, kwargs: dict):
        if name not in self._tools:
            raise KeyError(f"unknown tool {name!r}; available: {self.names()}")
        fn, cost, _, cost_fn, dedupe_key_fn = self._tools[name]
        kwargs = kwargs or {}
        dkey = None
        if dedupe_key_fn is not None:
            try:
                dkey = dedupe_key_fn(kwargs)
            except Exception:  # noqa: BLE001 - let the tool itself report bad arguments
                dkey = None
        with self._lock:
            if dkey is not None and (name, dkey) in self._dedupe:
                self.calls[name] = self.calls.get(name, 0) + 1
                self.log.append({"tool": name, "ok": True, "seconds": 0.0, "deduped": True})
                return self._dedupe[(name, dkey)]
            self._check(cost)
            if cost_fn is not None:
                self._check_exhausted(set(cost) | self._measured_keys.get(name, set()))
            for k, c in cost.items():
                self.spent[k] = self.spent.get(k, 0.0) + c
            self.calls[name] = self.calls.get(name, 0) + 1
        t0 = time.time()
        ok = False
        entry = {"tool": name, "ok": False, "seconds": 0.0}
        try:
            out = fn(**kwargs)
            if cost_fn is not None:
                measured = {k: float(v) for k, v in (cost_fn(kwargs, out) or {}).items() if v}
                entry["measured"] = measured
                with self._lock:
                    self._measured_keys.setdefault(name, set()).update(measured)
                    for k, v in measured.items():
                        self.spent[k] = self.spent.get(k, 0.0) + v
                    over = [k for k in measured if k in self.budget and self.spent[k] > self.budget[k] + 1e-9]
                if over:
                    raise BudgetExceeded("budget " + ", ".join(f"'{k}' exceeded by this call: spent "
                                                               f"{self.spent[k]:g} > {self.budget[k]:g}" for k in over))
            if dkey is not None:
                with self._lock:
                    self._dedupe[(name, dkey)] = out
            ok = True
            return out
        finally:
            entry.update(ok=ok, seconds=round(time.time() - t0, 3))
            self.log.append(entry)

    def describe(self) -> str:
        lines = ["Tools (call as api.<name>(keyword=value, ...); results are JSON):"]
        for name in self.names():
            fn, cost, doc, cost_fn, dedupe_key_fn = self._tools[name]
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(**kwargs)"
            parts = [f"{k}={v:g}" for k, v in cost.items() if v or cost_fn is None]
            if cost_fn is not None:
                parts.append("measured: " + "/".join(sorted(set(cost) | self._measured_keys.get(name, set())) or ["cost"]))
            if dedupe_key_fn is not None:
                parts.append("repeated identical calls are free")
            c = (" [cost: " + ", ".join(parts) + "]") if parts else ""
            lines.append(f"- {name}{sig}{c}" + (f"\n    {doc}" if doc else ""))
        if self.budget:
            lines.append("Budget remaining: " + ", ".join(f"{k}={v:g}" for k, v in self.remaining().items()))
        lines.append("Exceeding a budget raises api.BudgetExceeded; other tool failures raise api.ToolError.")
        return "\n".join(lines)


@dataclass
class ProgramResult:
    ok: bool
    result: object | None      # main(api)'s return value (JSON-serializable)
    stdout: str
    stderr: str
    error: str | None
    seconds: float
    info: dict = field(default_factory=dict)   # limits applied, network isolation, tool calls, returncode


def _network_prefix() -> tuple[list[str], str]:
    global _NET_PREFIX
    if _NET_PREFIX is not None:
        return _NET_PREFIX
    cand = []
    if sys.platform.startswith("linux") and shutil.which("unshare"):
        cand.append((["unshare", "-rn"], "unshare -rn (user+net namespace)"))
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists():
        cand.append((["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)"],
                     "sandbox-exec deny network*"))
    _NET_PREFIX = ([], "none (no unshare / sandbox-exec available)")
    for pre, label in cand:
        try:
            r = subprocess.run(pre + [sys.executable, "-I", "-c", "print(1)"], capture_output=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip() == b"1":
                _NET_PREFIX = (pre, label)
                break
        except (OSError, subprocess.TimeoutExpired):
            pass
    return _NET_PREFIX


def _allow_read() -> list[str]:
    s = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
         str(Path(sys.executable).resolve().parent.parent)}
    return sorted(p for p in s if p)


def _tail(p: Path, n: int = 65536) -> str:
    try:
        b = p.read_bytes()
    except OSError:
        return ""
    return b[-n:].decode("utf-8", "replace")


def run_program(code: str, api: ToolAPI, *, timeout_s: int, mem_gb: int, workdir: Path) -> ProgramResult:
    workdir = Path(workdir).resolve()
    (workdir / "tmp").mkdir(parents=True, exist_ok=True)
    (workdir / "program.py").write_text(code)
    ctl = Path(tempfile.mkdtemp(prefix="rsi_sbx_"))   # outside workdir: config + captured output
    cfg = {"workdir": str(workdir), "program": "program.py", "mem_bytes": int(mem_gb * 2 ** 30),
           "cpu_s": int(timeout_s) + 5, "fsize_bytes": 4 * 2 ** 30, "allow_read": _allow_read(),
           "deny_read": [str(ROOT), str(Path.home()), "/proc", str(ctl)]}
    (ctl / "cfg.json").write_text(json.dumps(cfg))
    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir / "tmp"), "TMPDIR": str(workdir / "tmp"),
           "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    prefix, net_label = _network_prefix()
    cmd = prefix + [sys.executable, "-I", str(CHILD), str(ctl / "cfg.json"), str(req_w), str(resp_r)]
    out_f, err_f = open(ctl / "stdout.txt", "wb"), open(ctl / "stderr.txt", "wb")
    t0 = time.time()
    info = {"network_isolation": net_label, "tool_calls": 0, "limits": None}
    final, timed_out = None, False
    try:
        proc = subprocess.Popen(cmd, cwd=str(workdir), env=env, stdin=subprocess.DEVNULL, stdout=out_f,
                                stderr=err_f, pass_fds=(req_w, resp_r), start_new_session=True)
    finally:
        os.close(req_w)
        os.close(resp_r)
        out_f.close()
        err_f.close()
    rf, wf = os.fdopen(req_r, "rb"), os.fdopen(resp_w, "wb", buffering=0)

    def send(obj):
        try:
            wf.write((json.dumps(obj) + "\n").encode())
        except (BrokenPipeError, OSError):
            pass

    send({"tools": api.names(), "docs": api.describe()})
    sel = selectors.DefaultSelector()
    sel.register(rf, selectors.EVENT_READ)
    buf = b""
    try:
        while True:
            left = timeout_s - (time.time() - t0)
            if left <= 0:
                timed_out = True
                break
            if not sel.select(timeout=min(left, 1.0)):
                if proc.poll() is not None:
                    break
                continue
            chunk = os.read(rf.fileno(), 1 << 20)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "hello":
                    info["limits"] = msg.get("limits")
                elif msg.get("type") == "final":
                    final = msg
                elif msg.get("type") == "call":
                    info["tool_calls"] += 1
                    send(_serve(api, msg))
            if final is not None:
                break
    finally:
        sel.close()
        if timed_out or final is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        rf.close()
        wf.close()
    secs = round(time.time() - t0, 3)
    info["returncode"] = proc.returncode
    stdout, stderr = _tail(ctl / "stdout.txt"), _tail(ctl / "stderr.txt")
    shutil.rmtree(ctl, ignore_errors=True)
    if timed_out:
        return ProgramResult(False, None, stdout, stderr, f"timeout: killed after {timeout_s}s", secs, info)
    if final is None:
        why = f"child exited with code {proc.returncode} before returning"
        if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
            why += " (CPU-time or memory limit?)"
        return ProgramResult(False, None, stdout, stderr, why, secs, info)
    return ProgramResult(bool(final.get("ok")), final.get("result"), stdout, stderr, final.get("error"), secs, info)


def _serve(api: ToolAPI, msg: dict) -> dict:
    rid = msg.get("id")
    try:
        res = api.call(msg.get("tool"), msg.get("kwargs") or {})
        try:
            enc = json.loads(json.dumps(res, allow_nan=False))
        except (TypeError, ValueError) as e:
            return {"id": rid, "ok": False, "error_type": "TypeError", "error": f"tool result not JSON-serializable: {e}"}
        return {"id": rid, "ok": True, "result": enc}
    except BudgetExceeded as e:
        return {"id": rid, "ok": False, "error_type": "BudgetExceeded", "error": str(e)}
    except Exception as e:  # noqa: BLE001 - every tool failure goes back to the program
        return {"id": rid, "ok": False, "error_type": type(e).__name__, "error": str(e)[:4000]}
