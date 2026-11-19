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
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
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


class VLLMServer:
    def __init__(self, model_path, served_name: str, gpu_ids: list[int], *, port: int | None = None,
                 enable_lora: bool = False, max_loras: int = 8, max_lora_rank: int = 64,
                 extra_args: list[str] = (), mock: bool = False, mock_registry: Path | None = None,
                 seed: int = 42, log_dir: Path | None = None, ready_timeout_s: float = 3600,
                 host: str = "127.0.0.1"):
        self.model_path, self.served_name = Path(model_path), served_name
        self.gpu_ids = list(gpu_ids or [])
        self.port = port or _net.free_port()
        self.host = host
        self.enable_lora, self.max_loras, self.max_lora_rank = enable_lora, max_loras, max_lora_rank
        self.extra_args = [str(a) for a in extra_args]
        self.memory = {}
        if mock and MEM_FLAG in self.extra_args:     # the mock server takes none of vLLM's flags
            i = self.extra_args.index(MEM_FLAG)
            self.extra_args = self.extra_args[:i] + self.extra_args[i + 2:]
        elif MEM_FLAG in self.extra_args:            # plan GPU memory in GiB (e.g. 110 GB on a 140 GB card)
            self.extra_args, self.memory = apply_memory_budget(self.extra_args, self.gpu_ids)
        self.mock, self.mock_registry, self.seed = mock, mock_registry, seed
        self.log_dir = Path(log_dir) if log_dir else Path(tempfile.gettempdir()) / "rsicore_logs"
        self.ready_timeout_s = ready_timeout_s
        self.endpoint = f"http://127.0.0.1:{self.port}/v1"
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = None
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
        cmd = _vllm_cmd() + ["serve", str(self.model_path), "--served-model-name", self.served_name,
                             "--host", self.host, "--port", str(self.port), "--seed", str(self.seed),
                             "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
                             "--reasoning-parser", "qwen3"]
        if "--max-model-len" not in self.extra_args:   # as in eval/hpc/README.md (32k thinking + prompt)
            cmd += ["--max-model-len", "40960"]
        if "--tensor-parallel-size" not in self.extra_args and "-tp" not in self.extra_args:
            cmd += ["--tensor-parallel-size", str(max(1, len(self.gpu_ids)))]
        if self.enable_lora:
            cmd += ["--enable-lora", "--max-loras", str(self.max_loras), "--max-lora-rank", str(self.max_lora_rank)]
        return cmd + self.extra_args

    def start(self):
        if self.proc is not None:
            return self
        env = dict(os.environ)
        if not self.mock:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)
            if self.enable_lora:
                env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        log.write(f"\n==== {time.strftime('%F %T')} {' '.join(self.command())}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        log.close()
        self._wait_ready()
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

    def stop(self, grace_s: float = 60):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=grace_s)
            except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=30)
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

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
