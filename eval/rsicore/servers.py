"""OpenAI-compatible model servers: real `vllm serve` (vLLM 0.24.0) or the mock server.

    with VLLMServer(rc.model_path, "qwen-g01", [0], enable_lora=True, mock=rc.mock,
                    mock_registry=rc.run_dir / "mock_registry.json", seed=rc.seed,
                    log_dir=rc.run_dir / "logs") as srv:
        srv.load_lora("cand_07", lora_dir)
        evaluate.score(srv.endpoint, "qwen-g01", "dapo_math", pool="pool", ids=..., out_dir=..., lora="cand_07")

Real path: `vllm serve <model> --served-model-name <name> --host 127.0.0.1 --port P --seed S
--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --tensor-parallel-size
len(gpu_ids) --max-model-len 40960 [--enable-lora --max-loras N --max-lora-rank R] <extra_args>`, CUDA_VISIBLE_DEVICES=gpu_ids,
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True when enable_lora (needed by /v1/load_lora_adapter); ready when
GET /v1/models answers. Logs: <log_dir>/<served_name>.<port>.log.
Mock path: eval/rsicore/mock_server.py with the same served name (extra_args are passed to it, e.g.
["--tasks", "dapo_math"] to index fewer tasks).

Sleep mode (2026-11-20; sleep_mode=True): `--enable-sleep-mode` and VLLM_SERVER_DEV_MODE=1, so the server
exposes POST /sleep?level=1 (weights offloaded to CPU RAM, KV cache freed) and POST /wake_up. sleep() /
wake_up() let another vLLM server use the GPU meanwhile (e.g. Paper A's editor while candidates are
evaluated). Mock: state only. start_all() starts several servers at once (replicas, one per GPU group).

Tool-call parser and context length (2026-11-21): tool_call_parser="auto" (default) reads the model's chat
template (tokenizer_config.json "chat_template" or chat_template.jinja) and passes `--tool-call-parser
qwen3_coder` when the template emits "<function=" (Qwen3.5 / Coder XML tool calls), else hermes; a
--tool-call-parser in extra_args wins. --max-model-len (extra_args, else 40960) is clamped to the model's
max_position_embeddings (config.json, or its text_config) when that is smaller, with a warning; YaRN is never
added. Helpers: max_model_len(args), set_max_model_len(args, n), model_max_positions(path),
detect_tool_call_parser(path). The mock path ignores all of this (and drops --max-model-len from extra_args).

Stopping and exit cleanup (2026-11-25): a server runs in its own session (start_new_session=True), and the GPU
memory is held by vLLM's EngineCore / worker CHILDREN, not by the `vllm serve` leader. stop() therefore sends
SIGTERM to the whole process group and waits until no live process of the group remains (up to STOP_GRACE_S),
then SIGKILLs the group and waits again (STOP_KILL_WAIT_S); a real server with known gpu_ids then waits (up to
STOP_GPU_WAIT_S, skipped without nvidia-smi) until nvidia-smi no longer lists the group's pids on its GPUs.
Every started server is kept in a per-process registry until stopped; an atexit handler stops whatever is
still live. install_exit_handlers() (called at the start of edit_code / lora_code main()) turns SIGTERM and
SIGHUP (unless ignored, e.g. under nohup) into SystemExit after stopping the live servers, so `kill <run pid>`
no longer orphans ~110 GiB vLLM servers that make the next run OOM. Spawned children (multiprocessing spawn,
sandbox subprocesses) start with an empty registry and default signal handlers; a forked child forgets the
parent's servers (os.register_at_fork), so no child ever stops its parent's servers.
"""
import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import _net
from .config import EVAL_DIR, eval_python

MOCK_PY = EVAL_DIR / "rsicore" / "mock_server.py"


def _vllm_cmd() -> list[str]:
    if os.environ.get("RSI_VLLM"):
        return [os.environ["RSI_VLLM"]]
    b = Path(sys.executable).parent / "vllm"
    if b.exists():
        return [str(b)]
    if shutil.which("vllm"):
        return [shutil.which("vllm")]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main"]


MEM_FLAG = "--rsi-gpu-memory-gib"
DEFAULT_MAX_MODEL_LEN = 40960


def max_model_len(args, default: int | None = DEFAULT_MAX_MODEL_LEN) -> int | None:
    """--max-model-len N (or --max-model-len=N) from a vLLM arg list; the last one wins; `default` if absent."""
    args = [str(a) for a in (args or [])]
    val = None
    for i, a in enumerate(args):
        if a == "--max-model-len" and i + 1 < len(args):
            val = args[i + 1]
        elif a.startswith("--max-model-len="):
            val = a.split("=", 1)[1]
    if val is None:
        return default
    v = str(val).strip().lower()
    mult = 1
    if v.endswith("k"):            # vLLM also accepts 32k / 128K
        v, mult = v[:-1], 1024
    return int(float(v) * mult)


def _drop_max_model_len(args) -> list[str]:
    out, i = [], 0
    while i < len(args):
        a = str(args[i])
        if a == "--max-model-len":
            i += 2
            continue
        if not a.startswith("--max-model-len="):
            out.append(a)
        i += 1
    return out


def set_max_model_len(args, n) -> list[str]:
    """Copy of args with --max-model-len set to n: the first occurrence is replaced in place, later ones are
    removed, appended when absent. n None -> args unchanged (as strings)."""
    args = [str(a) for a in (args or [])]
    if n is None:
        return args
    out, done, i = [], False, 0
    while i < len(args):
        a = args[i]
        if a == "--max-model-len" or a.startswith("--max-model-len="):
            if not done:
                out += ["--max-model-len", str(int(n))]
                done = True
            i += 1 if "=" in a else 2
            continue
        out.append(a)
        i += 1
    if not done:
        out += ["--max-model-len", str(int(n))]
    return out


def _read_json(p: Path):
    import json
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def model_max_positions(model_path) -> int | None:
    """max_position_embeddings from <model_path>/config.json (top level, else text_config), None if unknown."""
    cfg = _read_json(Path(model_path) / "config.json")
    if not isinstance(cfg, dict):
        return None
    for d in (cfg, cfg.get("text_config") or {}, cfg.get("language_config") or {}):
        v = d.get("max_position_embeddings") if isinstance(d, dict) else None
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return None


def detect_tool_call_parser(model_path) -> str:
    """qwen3_coder when the model's chat template emits "<function=" (XML tool calls), else hermes."""
    texts = []
    tc = _read_json(Path(model_path) / "tokenizer_config.json")
    t = tc.get("chat_template") if isinstance(tc, dict) else None
    if isinstance(t, str):
        texts.append(t)
    elif isinstance(t, list):
        texts += [x.get("template", "") for x in t if isinstance(x, dict)]
    for name in ("chat_template.jinja", "chat_template.json"):
        try:
            texts.append((Path(model_path) / name).read_text())
        except OSError:
            pass
    return "qwen3_coder" if any("<function=" in x for x in texts) else "hermes"


def gpu_total_gib(gpu_ids) -> float | None:
    """Smallest total memory (GiB) among the physical GPUs gpu_ids (nvidia-smi), None if unknown."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return None
    tot = {}
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            try:
                tot[int(parts[0])] = float(parts[1]) / 1024
            except ValueError:
                pass
    vals = [tot[g] for g in gpu_ids if g in tot]
    return min(vals) if vals and len(vals) == len(list(gpu_ids)) else None


def apply_memory_budget(args: list[str], gpu_ids, total_gib: float | None = None) -> tuple[list[str], dict]:
    """`--rsi-gpu-memory-gib X` in vLLM args -> --gpu-memory-utilization = X / the card's total memory
    (capped at 0.95), replacing any utilization given; the pseudo flag is removed. Without nvidia-smi the
    configured --gpu-memory-utilization (if any) is kept. Returns (args, info)."""
    args = [str(a) for a in args]
    if MEM_FLAG not in args:
        return args, {}
    i = args.index(MEM_FLAG)
    gib = float(args[i + 1])
    args = args[:i] + args[i + 2:]
    total = total_gib if total_gib is not None else gpu_total_gib(gpu_ids)
    if not total:
        return args, {"memory_gib": gib, "total_gib": None, "note": "nvidia-smi unavailable: configured utilization kept"}
    util = round(min(0.95, gib / total), 3)
    if "--gpu-memory-utilization" in args:
        j = args.index("--gpu-memory-utilization")
        args = args[:j] + args[j + 2:]
    return args + ["--gpu-memory-utilization", str(util)], {"memory_gib": gib, "total_gib": round(total, 1),
                                                           "utilization": util}


# ---------------------------------------------------------------------------- stopping a process group
# (2026-11-25) Bounds of VLLMServer.stop(): SIGTERM grace for the whole group, the wait after SIGKILL, and the
# wait for nvidia-smi to drop the group's pids. Module constants so a test or a caller can change them.
STOP_GRACE_S = 60.0
STOP_KILL_WAIT_S = 30.0
STOP_GPU_WAIT_S = 30.0
_POLL_S = 0.2


def _group_members(pgid: int) -> dict[int, str] | None:
    """{pid: state} of the processes in process group pgid ("Z" = zombie), None when it cannot be listed.
    Linux: /proc/<pid>/stat; elsewhere `ps -A -o pid=,pgid=,stat=`."""
    out: dict[int, str] = {}
    if os.path.isdir("/proc/self"):
        try:
            names = os.listdir("/proc")
        except OSError:
            return None
        for name in names:
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as f:
                    stat = f.read().decode("utf-8", "replace")
            except OSError:
                continue                                 # exited meanwhile
            fields = stat[stat.rfind(")") + 2:].split()  # comm may contain spaces / parentheses
            if len(fields) >= 3 and fields[2] == str(pgid):
                out[int(name)] = fields[0]
        return out
    try:
        res = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True, timeout=10)
    except Exception:  # noqa: BLE001
        return None
    if res.returncode != 0:
        return None
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == str(pgid) and parts[0].isdigit():
            out[int(parts[0])] = parts[2]
    return out


def _group_alive(pgid: int, proc=None) -> bool:
    """True while some non-zombie process of group pgid exists. Reaps our own leader first (proc.poll()), so a
    leader zombie does not count; zombies of other members (e.g. orphans of an init that does not reap) do not
    count either."""
    if proc is not None:
        proc.poll()
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:   # macOS: EPERM while the group still holds a zombie; ask the process table
        pass
    members = _group_members(pgid)
    if members is None:
        return True
    return any(not st.startswith("Z") for st in members.values())


def _signal_group(pgid: int, sig, proc=None) -> None:
    """killpg(pgid, sig); when that fails with EPERM (macOS, zombie leader) signal each live member instead."""
    if proc is not None:
        proc.poll()
    try:
        os.killpg(pgid, sig)
        return
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    for pid, st in (_group_members(pgid) or {}).items():
        if not st.startswith("Z"):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass


def _pid_exists(pid: int) -> bool:
    """A process (any state, zombies included) with this pid exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_group(pgid: int, proc, timeout_s: float) -> bool:
    """Poll until group pgid has no live process (True) or timeout_s passed (False)."""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        if not _group_alive(pgid, proc):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_S)


def _gpu_compute_pids(gpu_ids) -> set[int] | None:
    """Pids nvidia-smi lists as compute apps on the physical GPUs gpu_ids; None without nvidia-smi."""
    try:
        idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001 - no nvidia-smi, driver hang, ...
        return None
    if idx.returncode != 0 or apps.returncode != 0:
        return None
    want = {int(g) for g in gpu_ids}
    uuids = set()
    for line in idx.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in want:
            uuids.add(parts[1])
    pids = set()
    for line in apps.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1] in uuids:
            pids.add(int(parts[0]))
    return pids


# ---------------------------------------------------------------------------- live servers of this process
# (2026-11-25) Strong references on purpose: a server nobody stopped must still be stopped at exit (a weak
# reference would lose it and orphan its processes). stop() removes it again.
_LIVE: dict[int, "VLLMServer"] = {}
_LIVE_LOCK = threading.RLock()     # re-entrant: the SIGTERM handler may run while the main thread holds it


def _say(msg: str) -> None:
    """One line to stderr; never raises (after a terminal hang-up stderr is a dead tty and print() raises EIO,
    which must not stop the cleanup that follows)."""
    try:
        print(msg, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def _register(srv) -> None:
    with _LIVE_LOCK:
        _LIVE[id(srv)] = srv


def _unregister(srv) -> None:
    with _LIVE_LOCK:
        _LIVE.pop(id(srv), None)


def live_servers() -> list:
    """The servers this process started and has not stopped yet."""
    with _LIVE_LOCK:
        return list(_LIVE.values())


def _forget_after_fork() -> None:
    """In a forked child: the parent's servers are not ours to stop."""
    global _LIVE_LOCK
    _LIVE_LOCK = threading.RLock()
    _LIVE.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_after_fork)


def stop_servers(servers) -> None:
    """Stop several servers together: SIGTERM to every group first, then each stop() waits, so the grace
    periods overlap (no threads: this also runs from atexit). Errors are printed, not raised."""
    servers = list(servers)
    sent = set()
    for s in servers:
        p = getattr(s, "proc", None)
        if isinstance(p, subprocess.Popen) and hasattr(s, "_wait_gpu_release") and p.poll() is None:
            # a real VLLMServer (not a fake) whose leader still runs; a reaped leader is left to stop(), which
            # checks for pid reuse before signalling the group
            _signal_group(p.pid, signal.SIGTERM, p)
            sent.add(id(s))
    for s in servers:
        try:
            if id(s) in sent:
                s.stop(term_sent=True)          # one SIGTERM only (a second one makes uvicorn force-exit)
            else:
                s.stop()
        except Exception as e:  # noqa: BLE001
            _say(f"[servers] stop {getattr(s, 'served_name', s)} failed: {e!r}")


def stop_all_servers() -> None:
    """Stop every server this process started and has not stopped (atexit; the SIGTERM / SIGHUP handler)."""
    stop_servers(live_servers())


atexit.register(stop_all_servers)

_EXIT_HANDLERS: dict = {}      # pid that installed the handlers -> signals converted
_EXITING = {"busy": False}


def _exit_on_signal(signum, frame):  # noqa: ARG001
    """SIGTERM / SIGHUP -> stop this process's servers and multiprocessing children, then SystemExit(128+signum)
    so try/finally blocks and atexit run. The servers are stopped HERE, before unwinding, because unwinding can
    block on work that talks to them (a ThreadPoolExecutor's shutdown waits for its queued requests)."""
    if os.getpid() not in _EXIT_HANDLERS:     # inherited by a forked child: behave as the default action
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return
    if _EXITING["busy"]:                      # a repeated signal must not interrupt the cleanup below
        return
    _EXITING["busy"] = True
    try:
        name = signal.Signals(signum).name
        live = live_servers()           # includes a server whose stop() is still waiting (escalated below)
        _say(f"[servers] {name}: stopping {len(live)} server(s) and exiting")
        stop_servers(live)
        try:
            import multiprocessing as mp
            kids = mp.active_children()
            for k in kids:                       # e.g. spawn profile workers holding an HF model on a GPU
                k.terminate()
            for k in kids:
                k.join(10)
        except Exception:  # noqa: BLE001
            pass
    finally:
        _EXITING["busy"] = False
    raise SystemExit(128 + signum)


def install_exit_handlers() -> list[str]:
    """Convert SIGTERM and SIGHUP into SystemExit (after stopping the live servers) so `kill <pid>` cleans up.
    Only from the main thread (else a no-op), idempotent; a signal that is ignored (SIGHUP under nohup) or
    already has another Python handler is left alone. Returns the names of the converted signals."""
    if threading.current_thread() is not threading.main_thread():
        return []
    pid = os.getpid()
    if pid in _EXIT_HANDLERS:
        return list(_EXIT_HANDLERS[pid])
    done = []
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        cur = signal.getsignal(sig)
        if cur is _exit_on_signal or cur in (signal.SIG_DFL, None):
            signal.signal(sig, _exit_on_signal)
            done.append(name)
    _EXIT_HANDLERS.clear()                    # a forked child that calls this again owns only its own pid
    _EXIT_HANDLERS[pid] = done
    return list(done)


class VLLMServer:
    def __init__(self, model_path, served_name: str, gpu_ids: list[int], *, port: int | None = None,
                 enable_lora: bool = False, max_loras: int = 8, max_lora_rank: int = 64,
                 extra_args: list[str] = (), mock: bool = False, mock_registry: Path | None = None,
                 seed: int = 42, log_dir: Path | None = None, ready_timeout_s: float = 3600,
                 host: str = "127.0.0.1", sleep_mode: bool = False, tool_call_parser: str = "auto"):
        self.model_path, self.served_name = Path(model_path), served_name
        self.tool_call_parser = tool_call_parser
        self.notes: list[str] = []      # resolved parser / context clamps (also written to the log)
        self.gpu_ids = list(gpu_ids or [])
        self.port = port or _net.free_port()
        self.host = host
        self.enable_lora, self.max_loras, self.max_lora_rank = enable_lora, max_loras, max_lora_rank
        self.extra_args = [str(a) for a in extra_args]
        self.memory = {}
        if mock:                                     # the mock server takes none of vLLM's flags
            if MEM_FLAG in self.extra_args:
                i = self.extra_args.index(MEM_FLAG)
                self.extra_args = self.extra_args[:i] + self.extra_args[i + 2:]
            self.extra_args = _drop_max_model_len(self.extra_args)
        elif MEM_FLAG in self.extra_args:            # plan GPU memory in GiB (e.g. 110 GB on a 140 GB card)
            self.extra_args, self.memory = apply_memory_budget(self.extra_args, self.gpu_ids)
        self.mock, self.mock_registry, self.seed = mock, mock_registry, seed
        self.log_dir = Path(log_dir) if log_dir else Path(tempfile.gettempdir()) / "rsicore_logs"
        self.ready_timeout_s = ready_timeout_s
        self.endpoint = f"http://127.0.0.1:{self.port}/v1"
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = None
        self.sleep_mode = bool(sleep_mode)
        self.sleeping = False
        self.loras: dict[str, str] = {}
        self.log_path = self.log_dir / f"{served_name.replace('/', '_')}.{self.port}.log"

    # ----------------------------------------------------------------- lifecycle
    def command(self) -> list[str]:
        if self.mock:
            cmd = [eval_python(), "-I", str(MOCK_PY), "--port", str(self.port), "--served-name", self.served_name,
                   "--seed", str(self.seed), "--model-path", str(self.model_path)]
            if self.mock_registry:
                cmd += ["--registry", str(self.mock_registry)]
            if self.enable_lora:
                cmd += ["--enable-lora", "--max-loras", str(self.max_loras)]
            return cmd + self.extra_args
        extra, parser = self._resolved()
        cmd = _vllm_cmd() + ["serve", str(self.model_path), "--served-model-name", self.served_name,
                             "--host", self.host, "--port", str(self.port), "--seed", str(self.seed),
                             "--enable-auto-tool-choice"]
        if parser:
            cmd += ["--tool-call-parser", parser]
        cmd += ["--reasoning-parser", "qwen3"]
        if max_model_len(extra, None) is None:   # as in eval/hpc/README.md (32k thinking + prompt)
            cmd += ["--max-model-len", str(self._clamp_len(DEFAULT_MAX_MODEL_LEN))]
        if "--tensor-parallel-size" not in self.extra_args and "-tp" not in self.extra_args:
            cmd += ["--tensor-parallel-size", str(max(1, len(self.gpu_ids)))]
        if self.enable_lora:
            cmd += ["--enable-lora", "--max-loras", str(self.max_loras), "--max-lora-rank", str(self.max_lora_rank)]
        if self.sleep_mode and "--enable-sleep-mode" not in self.extra_args:
            cmd += ["--enable-sleep-mode"]
        if os.environ.get("RSI_GDN_PREFILL_BACKEND") and not any(
                a == "--gdn-prefill-backend" or a.startswith("--gdn-prefill-backend=") for a in extra):
            cmd += ["--gdn-prefill-backend", os.environ["RSI_GDN_PREFILL_BACKEND"]]
        return cmd + extra

    def _clamp_len(self, n: int) -> int:
        mp = model_max_positions(self.model_path)
        if mp and n > mp:
            note = (f"WARNING {self.served_name}: --max-model-len {n} > max_position_embeddings {mp} of "
                    f"{self.model_path}; clamped to {mp} (no YaRN)")
            if note not in self.notes:
                self.notes.append(note)
                print(note, file=sys.stderr, flush=True)
            return mp
        return n

    def _resolved(self) -> tuple[list[str], str | None]:
        """(extra_args with --max-model-len clamped, tool-call parser or None when extra_args names one)."""
        extra = list(self.extra_args)
        n = max_model_len(extra, None)
        if n is not None and self._clamp_len(n) != n:
            extra = set_max_model_len(extra, self._clamp_len(n))
        if any(a == "--tool-call-parser" or a.startswith("--tool-call-parser=") for a in extra):
            return extra, None
        parser = self.tool_call_parser or "auto"
        if parser == "auto":
            parser = detect_tool_call_parser(self.model_path)
            note = f"{self.served_name}: tool_call_parser auto -> {parser}"
            if note not in self.notes:
                self.notes.append(note)
        return extra, parser

    def start(self):
        if self.proc is not None:
            return self
        env = dict(os.environ)
        if not self.mock:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)
            if self.enable_lora:
                env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
            if self.sleep_mode:
                env["VLLM_SERVER_DEV_MODE"] = "1"          # exposes /sleep and /wake_up
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        cmd = self.command()
        log.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
        for note in self.notes:
            log.write(f"==== {note}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        log.close()
        _register(self)                 # (2026-11-25) stopped at exit / on SIGTERM if nobody stops it
        try:
            self._wait_ready()
        except BaseException:           # not ready, died, Ctrl-C: leave no EngineCore child behind
            self.stop()
            raise
        return self

    def _wait_ready(self):
        t0 = time.time()
        while time.time() - t0 < self.ready_timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}; log tail:\n{self.log_tail()}")
            try:
                _net.get_json(self.endpoint + "/models", timeout=5)
                return
            except Exception:
                time.sleep(0.5 if self.mock else 5)
        self.stop()
        raise TimeoutError(f"server not ready after {self.ready_timeout_s}s; log tail:\n{self.log_tail()}")

    def log_tail(self, n: int = 4000) -> str:
        try:
            return self.log_path.read_bytes()[-n:].decode("utf-8", "replace")
        except OSError:
            return ""

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace_s: float | None = None, *, term_sent: bool = False):
        """Stop the server's WHOLE process group (2026-11-25): the GPU memory belongs to vLLM's EngineCore /
        worker children, which outlive the `vllm serve` leader that proc.wait() used to wait for. SIGTERM the
        group (also when the leader has already exited), wait until no live member remains (grace_s, default
        STOP_GRACE_S), else SIGKILL and wait STOP_KILL_WAIT_S; then, for a real server, wait up to
        STOP_GPU_WAIT_S until nvidia-smi no longer lists the group's pids on gpu_ids. Idempotent; never raises
        for a process that is already gone; prints one line when it had to escalate. term_sent: the caller
        (stop_servers) already sent the SIGTERM.
        The server stays registered (live_servers) until its group is gone, so a SIGTERM / Ctrl-C that arrives
        during the wait still finds it: the handler's re-entrant stop() SIGKILLs the group at once, and any
        exception raised into the wait (SystemExit, KeyboardInterrupt) SIGKILLs it before propagating."""
        proc, self.proc = self.proc, None
        self.sleeping = False           # a restarted server is awake and has no adapters loaded
        self.loras = {}
        if proc is None:
            busy = getattr(self, "_stopping", None)
            if busy is not None:        # re-entrant (signal handler during our own wait): escalate now
                _signal_group(busy.pid, signal.SIGKILL, busy)
                _wait_group(busy.pid, busy, STOP_KILL_WAIT_S)
            else:
                _unregister(self)
            return
        self._stopping = proc
        try:
            self._stop_group(proc, grace_s, term_sent)
        except BaseException:           # interrupted mid-wait: never leave the EngineCore behind
            _signal_group(proc.pid, signal.SIGKILL, proc)
            _wait_group(proc.pid, proc, STOP_KILL_WAIT_S)
            raise
        finally:
            self._stopping = None
            _unregister(self)

    def _stop_group(self, proc, grace_s, term_sent: bool) -> None:
        grace = STOP_GRACE_S if grace_s is None else float(grace_s)
        pgid = proc.pid                 # start_new_session=True: the leader's pid is the group id
        if proc.returncode is not None and _pid_exists(pgid):
            # the leader was reaped BEFORE this stop (e.g. alive() saw it crash); a live process with that pid
            # is pid reuse, so the group id may now belong to someone else: do not signal it
            self._note(f"stop: leader {pgid} had already exited and its pid is in use again; not signalling "
                       f"process group {pgid}")
            return
        watch_gpu = (not self.mock) and bool(self.gpu_ids)
        pids = set()
        if watch_gpu:                   # before the signal: the members vanish from the table afterwards
            pids = set(_group_members(pgid) or {}) | {proc.pid}
        if not term_sent:
            _signal_group(pgid, signal.SIGTERM, proc)
        if not _wait_group(pgid, proc, grace):
            left = sorted(p for p, st in (_group_members(pgid) or {}).items() if not st.startswith("Z"))
            self._note(f"stop: process group {pgid} still has {left or 'live members'} {grace:.0f}s after "
                       f"SIGTERM; sending SIGKILL")
            _signal_group(pgid, signal.SIGKILL, proc)
            if not _wait_group(pgid, proc, STOP_KILL_WAIT_S):
                self._note(f"stop: process group {pgid} survived SIGKILL for {STOP_KILL_WAIT_S:.0f}s "
                           f"(uninterruptible?); its GPU memory may still be held")
        try:
            proc.wait(timeout=5)        # reap the leader (normally done by the polls above)
        except subprocess.TimeoutExpired:
            pass
        if watch_gpu:
            self._wait_gpu_release(pids)

    def _wait_gpu_release(self, pids: set[int]) -> None:
        """Poll nvidia-smi (bounded by STOP_GPU_WAIT_S) until none of pids is a compute app on gpu_ids. Silent
        without nvidia-smi, and when the pids are not visible (another pid namespace) it returns at once."""
        deadline = time.monotonic() + STOP_GPU_WAIT_S
        while True:
            on_gpu = _gpu_compute_pids(self.gpu_ids)
            if on_gpu is None or not (on_gpu & pids):
                return
            if time.monotonic() >= deadline:
                self._note(f"stop: nvidia-smi still lists pids {sorted(on_gpu & pids)} on GPUs {self.gpu_ids} "
                           f"{STOP_GPU_WAIT_S:.0f}s after the process group exited")
                return
            time.sleep(1.0)

    def _note(self, msg: str) -> None:
        """One line to stderr and to the server log."""
        line = f"[servers] {self.served_name}: {msg}"
        _say(line)
        try:
            with open(self.log_path, "ab") as f:
                f.write(f"==== {time.strftime('%F %T')} {line}\n".encode())
        except OSError:
            pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # ----------------------------------------------------------------- sleep mode
    def sleep(self, level: int = 1) -> None:
        """Level 1: offload the weights to CPU RAM and free the KV cache (needs sleep_mode=True). No-op when
        already sleeping or not running."""
        if self.sleeping or not self.alive():
            return
        if not self.sleep_mode:
            raise RuntimeError(f"{self.served_name}: started without sleep_mode")
        if not self.mock:
            _net.post_json(self.base + f"/sleep?level={int(level)}", {}, timeout=1800)
        self.sleeping = True

    def wake_up(self) -> None:
        """Reload the weights onto the GPU (and re-create the KV cache)."""
        if not self.sleeping:
            return
        if not self.mock and self.alive():
            _net.post_json(self.base + "/wake_up", {}, timeout=1800)
        self.sleeping = False

    # ----------------------------------------------------------------- LoRA hot swap
    def load_lora(self, name: str, path: Path) -> None:
        """POST /v1/load_lora_adapter {"lora_name", "lora_path"} (server needs --enable-lora and
        VLLM_ALLOW_RUNTIME_LORA_UPDATING=True). Loading the same name+path again is a no-op."""
        path = str(Path(path).resolve())
        if self.loras.get(name) == path:
            return
        if name in self.loras:
            self.unload_lora(name)
        try:
            _net.post_json(self.base + "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, timeout=600)
        except _net.HTTPError as e:
            if "already" not in e.body.lower():
                raise RuntimeError(f"load_lora({name}, {path}) failed: {e}") from None
        self.loras[name] = path

    def unload_lora(self, name: str) -> None:
        try:
            _net.post_json(self.base + "/v1/unload_lora_adapter", {"lora_name": name}, timeout=600)
        except _net.HTTPError as e:
            low = e.body.lower()
            if e.status != 404 and not any(x in low for x in ("not found", "does not exist", "cannot be found")):
                raise RuntimeError(f"unload_lora({name}) failed: {e}") from None
        self.loras.pop(name, None)


def start_all(servers: list) -> list:
    """Start several servers at the same time (threads; each start() blocks until ready). If one fails, the
    others are stopped and the error is raised."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [s for s in servers if not s.alive()]
    if len(todo) <= 1:
        for s in todo:
            s.start()
        return servers
    errs = []
    with ThreadPoolExecutor(max_workers=len(todo)) as ex:
        for f in [ex.submit(s.start) for s in todo]:
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                errs.append(e)
    if errs:
        stop_servers(servers)          # (2026-11-25) grace periods overlap
        raise errs[0]
    return servers
