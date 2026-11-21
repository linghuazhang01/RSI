"""Run model-written programs `def main(api): ...` in a child process, with tools proxied to the parent.

    api = ToolAPI({"evaluate": 24, "gpu_seconds": 7200})
    api.register("evaluate", my_eval, cost={"evaluate": 1}, doc="evaluate(config) -> L1 feedback")
    res = run_program(code, api, timeout_s=600, mem_gb=8, workdir=gen_dir / "work" / cid)
    k = Kernel(workdir, mem_gb=8); k.run(code, api, timeout_s=300, subdir="r01_0"); k.close()
        (2026-11-21: persistent kernel, code strings exec'd in the same globals; see Kernel)

Child isolation (defence in depth; the tools themselves are trusted parent code):
  * separate process `python -I` (same interpreter as the parent), new session, environment cleared
    except PATH, HOME=TMPDIR=<workdir>/tmp, thread-count vars; cwd = workdir;
  * rlimits set in the child before the program runs: RLIMIT_AS (mem_gb; macOS cannot set it -> recorded
    as unsupported), RLIMIT_CPU (timeout_s + 5), RLIMIT_FSIZE (4 GiB), RLIMIT_NOFILE, RLIMIT_CORE=0;
  * a Python audit hook: writes only under workdir, no reads under the repo / $HOME / /proc (except the
    interpreter's own prefixes), no subprocess / exec / fork / kill / socket connect|bind|getaddrinfo.
    C extensions (e.g. via ctypes) can bypass audit hooks; this is not a security boundary on its own.
    In particular (2026-11-21) code that has torch / safetensors (lora_code tensor_compute children) can read and
    WRITE files natively (torch.save, safetensors save_file / safe_open never raise an audit 'open' event): a
    hostile or careless program could overwrite model shards. Run real jobs on a copy-on-write / read-only model
    directory, or under an OS-level read-only bind mount (bwrap / Landlock), if that matters;
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


def _sender(wf):
    def send(obj):
        try:
            wf.write((json.dumps(obj) + "\n").encode())
        except (BrokenPipeError, OSError):
            pass
    return send


def _spawn(workdir: Path, *, mem_gb, cpu_s: int, extra_allow_read=(), extra_env=None, extra_cfg=None):
    """Start the child (rlimits, audit hook, network isolation) in workdir -> (proc, rf, wf, ctl, net_label);
    rf / wf: the request / response pipe ends of the parent; ctl: control dir (cfg.json, stdout.txt, stderr.txt)."""
    ctl = Path(tempfile.mkdtemp(prefix="rsi_sbx_"))   # outside workdir: config + captured output
    allow = _allow_read() + sorted({str(Path(p).resolve()) for p in (extra_allow_read or ())})
    cfg = {"workdir": str(workdir), "mem_bytes": int(mem_gb * 2 ** 30), "cpu_s": int(cpu_s),
           "fsize_bytes": 4 * 2 ** 30, "allow_read": allow,
           "deny_read": [str(ROOT), str(Path.home()), "/proc", str(ctl)], **(extra_cfg or {})}
    (ctl / "cfg.json").write_text(json.dumps(cfg))
    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir / "tmp"), "TMPDIR": str(workdir / "tmp"),
           "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    env.update({str(k): str(v) for k, v in (extra_env or {}).items()})
    prefix, net_label = _network_prefix()
    cmd = prefix + [sys.executable, "-I", str(CHILD), str(ctl / "cfg.json"), str(req_w), str(resp_r)]
    out_f, err_f = open(ctl / "stdout.txt", "wb"), open(ctl / "stderr.txt", "wb")
    try:
        proc = subprocess.Popen(cmd, cwd=str(workdir), env=env, stdin=subprocess.DEVNULL, stdout=out_f,
                                stderr=err_f, pass_fds=(req_w, resp_r), start_new_session=True)
    except BaseException:
        os.close(req_r)
        os.close(resp_w)
        raise
    finally:
        os.close(req_w)
        os.close(resp_r)
        out_f.close()
        err_f.close()
    return proc, os.fdopen(req_r, "rb"), os.fdopen(resp_w, "wb", buffering=0), ctl, net_label


def _kill(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def run_program(code: str, api: ToolAPI, *, timeout_s: int, mem_gb: int, workdir: Path,
                save_output: bool = False, extra_allow_read=(), extra_env: dict | None = None,
                cpu_s: int | None = None) -> ProgramResult:
    """save_output (2026-11-21): after the child has exited, copy its complete stdout / stderr to
    workdir/stdout.txt and workdir/stderr.txt (ProgramResult keeps only the last 64 KiB).
    extra_allow_read (2026-11-21): further paths the child may read although they lie under a denied root (exact files,
    e.g. the model shards a tensor_compute job reads; resolved, allow wins over deny). extra_env: variables added to
    the child's cleared environment (e.g. CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4'; they override the
    single-thread defaults). cpu_s: RLIMIT_CPU in seconds (default timeout_s + 5; a multi-threaded child needs more)."""
    workdir = Path(workdir).resolve()
    (workdir / "tmp").mkdir(parents=True, exist_ok=True)
    (workdir / "program.py").write_text(code)
    t0 = time.time()
    proc, rf, wf, ctl, net_label = _spawn(workdir, mem_gb=mem_gb, cpu_s=int(cpu_s) if cpu_s else int(timeout_s) + 5,
                                          extra_allow_read=extra_allow_read, extra_env=extra_env,
                                          extra_cfg={"program": "program.py"})
    info = {"network_isolation": net_label, "tool_calls": 0, "limits": None}
    final, timed_out = None, False
    send = _sender(wf)
    send({"tools": api.names(), "docs": api.describe()})
    try:
        final, timed_out = _pump(rf, send, proc, api, t0 + timeout_s, info)
    finally:
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
    if save_output:
        for name in ("stdout.txt", "stderr.txt"):
            try:
                (workdir / name).unlink(missing_ok=True)
                shutil.copyfile(ctl / name, workdir / name)
            except OSError:
                pass
    shutil.rmtree(ctl, ignore_errors=True)
    if timed_out:
        return ProgramResult(False, None, stdout, stderr, f"timeout: killed after {timeout_s}s", secs, info)
    if final is None:
        why = f"child exited with code {proc.returncode} before returning"
        if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
            why += " (CPU-time or memory limit?)"
        return ProgramResult(False, None, stdout, stderr, why, secs, info)
    return ProgramResult(bool(final.get("ok")), final.get("result"), stdout, stderr, final.get("error"), secs, info)


RESTARTED = "session restarted: variables lost"


class Kernel:
    """Persistent sandbox kernel (2026-11-21, editor.explore.persistent): ONE long-lived child with the same
    restrictions as run_program (rlimits, audit hook, network isolation, no subprocess), started lazily by the first
    run(); each run(code, api, timeout_s, subdir) execs `code` in the SAME globals (variables, imports, functions
    persist), with `api` a persistent global (tool calls RPC to the given ToolAPI as in run_program), `result` cleared
    before every call (set it to return a JSON value; a `main(api)` (re)defined by the call is called when `result`
    is not set). Per call: workdir/<subdir>/program.py, stdout.txt, stderr.txt (the call's print output).
    RLIMIT_CPU: the kernel's CPU time so far + timeout_s + 5 before each call. A call that times out (wall clock) or
    crashes the child kills the kernel: the result's error ends with RESTARTED and info["restarted"] is true; the next
    run() starts a fresh kernel. Not thread-safe per instance (one proposal = one kernel). close() shuts it down."""

    def __init__(self, workdir: Path, *, mem_gb: float, extra_allow_read=(), extra_env: dict | None = None):
        self.workdir = Path(workdir).resolve()
        self.mem_gb, self.extra_allow_read, self.extra_env = mem_gb, tuple(extra_allow_read or ()), extra_env
        self.proc = self.rf = self.wf = self.ctl = None
        self.net_label, self.limits, self.starts, self.execs = None, None, 0, 0

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _start(self, api: ToolAPI, timeout_s: float):
        (self.workdir / "tmp").mkdir(parents=True, exist_ok=True)
        # RLIMIT_CPU soft limit only (hard limit unchanged): raised by the kernel before each call
        self.proc, self.rf, self.wf, self.ctl, self.net_label = _spawn(
            self.workdir, mem_gb=self.mem_gb, cpu_s=int(timeout_s) + 5, extra_allow_read=self.extra_allow_read,
            extra_env=self.extra_env, extra_cfg={"kernel": True})
        self.starts += 1
        _sender(self.wf)({"tools": api.names(), "docs": api.describe()})
        info = {}
        hello, _ = _pump(self.rf, _sender(self.wf), self.proc, api, time.time() + 60, info, want="hello")
        if hello is None:
            err = _tail(self.ctl / "stderr.txt", 4000)
            self._shutdown(kill=True)
            raise RuntimeError(f"sandbox kernel did not start: {err}")
        self.limits = hello.get("limits")

    def run(self, code: str, api: ToolAPI, *, timeout_s: float, subdir: str = "call") -> ProgramResult:
        t0 = time.time()
        fresh = not self.alive
        if fresh:
            if self.proc is not None:
                self._shutdown(kill=True)
            self._start(api, timeout_s)
        sub = Path(str(subdir))
        if sub.is_absolute() or ".." in sub.parts:
            raise ValueError(f"bad subdir {subdir!r}")
        d = self.workdir / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "program.py").write_text(code)
        for name in ("stdout.txt", "stderr.txt"):
            (d / name).unlink(missing_ok=True)
        self.execs += 1
        info = {"network_isolation": self.net_label, "tool_calls": 0, "limits": self.limits, "kernel": True,
                "fresh_kernel": fresh, "kernel_starts": self.starts, "restarted": False}
        send = _sender(self.wf)
        send({"type": "exec", "id": self.execs, "code": code, "dir": str(sub), "cpu_s": int(timeout_s) + 5})
        final, timed_out = _pump(self.rf, send, self.proc, api, t0 + float(timeout_s), info)
        secs = round(time.time() - t0, 3)
        stdout, stderr = _tail(d / "stdout.txt"), _tail(d / "stderr.txt")
        if final is not None:
            return ProgramResult(bool(final.get("ok")), final.get("result"), stdout, stderr, final.get("error"),
                                 secs, info)
        crash = _tail(self.ctl / "stderr.txt", 4000) if self.ctl else ""
        self._shutdown(kill=True)
        info.update(restarted=True, returncode=self.proc.returncode if self.proc else None)
        if timed_out:
            why = f"timeout: killed after {timeout_s:g}s"
        else:
            rc = info["returncode"]
            why = f"kernel exited with code {rc}"
            if rc in (-signal.SIGXCPU, -signal.SIGKILL):
                why += " (CPU-time or memory limit?)"
        return ProgramResult(False, None, stdout, (stderr + ("\n" + crash if crash.strip() else "")).strip(),
                             f"{why}; {RESTARTED}", secs, info)

    def _shutdown(self, kill: bool):
        proc = self.proc
        if proc is not None and proc.poll() is None:
            if not kill:
                _sender(self.wf)({"type": "exit"})
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if proc.poll() is None:
                _kill(proc)
            else:   # the child left; make sure nothing of its process group survives
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        for f in (self.rf, self.wf):
            try:
                f and f.close()
            except OSError:
                pass
        if self.ctl is not None:
            shutil.rmtree(self.ctl, ignore_errors=True)
        self.rf = self.wf = self.ctl = None

    def close(self):
        """Shut the kernel down (exit message, then SIGKILL of its process group if needed); idempotent."""
        self._shutdown(kill=False)
        self.proc = None


def _pump(rf, send, proc, api: ToolAPI, deadline: float, info: dict, want: str = "final"):
    """Serve the child's tool calls until a message of type `want` arrives -> (that message, False); the child
    exits / closes the pipe -> (None, False); the wall clock passes `deadline` -> (None, True)."""
    sel = selectors.DefaultSelector()
    sel.register(rf, selectors.EVENT_READ)
    buf = b""
    try:
        while True:
            left = deadline - time.time()
            if left <= 0:
                return None, True
            if not sel.select(timeout=min(left, 1.0)):
                if proc.poll() is not None:
                    return None, False
                continue
            chunk = os.read(rf.fileno(), 1 << 20)
            if not chunk:
                return None, False
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "hello":
                    info["limits"] = msg.get("limits")
                    if want == "hello":
                        return msg, False
                elif msg.get("type") == want:
                    return msg, False
                elif msg.get("type") == "call":
                    info["tool_calls"] = info.get("tool_calls", 0) + 1
                    send(_serve(api, msg))
    finally:
        sel.close()


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
