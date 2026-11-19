# H20 上的 CUDA 13.0 运行环境修复

2026-11-19 的验证机器为八卡 H20-3e，每卡 143771 MiB，系统驱动为 570.133.20。
GPU Python 环境仍使用 `requirements-gpu.lock` 的全部 243 个包。
下列修复已使 Paper A 双卡 vLLM 服务和 Paper B 单卡 LoRA 服务成功处理生成请求。
服务启动成功不代表整代实验已经完成。

## 1. 为实验进程指定兼容库

该机器的系统驱动接口无法直接运行 torch 的 cu130 构建。
验证时将 NVIDIA `cuda-compat-13-2` 解压到独立目录，只为实验进程设置 `LD_LIBRARY_PATH`。
没有安装系统 deb 包，也没有修改系统驱动或系统动态链接器配置。

在仓库根目录执行：

```bash
export RSI_GPU_VENV="$HOME/venvs/rsi-gpu"
export RSI_RUNTIME_DIR="$PWD/.venv/rsi-runtime"
mkdir -p "$RSI_RUNTIME_DIR"
RSI_COMPAT_DEB="$RSI_RUNTIME_DIR/cuda-compat-13-2.deb"
curl --fail --location --retry 3 --connect-timeout 30 \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-compat-13-2_595.112.07-1ubuntu1_amd64.deb \
  --output "$RSI_COMPAT_DEB"
printf '%s  %s\n' \
  acc3ba5b576b41ab304ad1c6215a64e3e57bd7658afd97f5c82a882e40db4300 \
  "$RSI_COMPAT_DEB" | sha256sum --check -
dpkg-deb --extract "$RSI_COMPAT_DEB" "$RSI_RUNTIME_DIR/compat"
export RSI_COMPAT_LIB="$RSI_RUNTIME_DIR/compat/usr/local/cuda-13.2/compat"
```

上述包和哈希来自 NVIDIA 包索引。兼容性结论仅覆盖本次 H20-3e 机器的实测环境。
兼容库本身不会把 NVCC 或运行时头文件切换成同一版本，编译工具链需要下一步单独设置。

## 2. 使用一致的 CUDA 13.0 编译工具链

锁文件中的 NVCC 为 13.2.86，CUDA 运行时头文件为 13.0.96。
FlashInfer GDN 编译在这一组合下报告编译器和头文件不兼容。
将以下四个构建包安装到独立目录，保留 GPU Python 环境原有包版本：

```bash
export RSI_CUDA_SDK="$RSI_RUNTIME_DIR/cuda130-toolchain"
uv pip install --python "$RSI_GPU_VENV/bin/python" \
  --target "$RSI_CUDA_SDK" --link-mode copy --no-deps \
  --index-url "${RSI_PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}" \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
  nvidia-nvvm==13.0.88 nvidia-cuda-runtime==13.0.96
export CUDA_HOME="$RSI_CUDA_SDK/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
```

FlashInfer 的采样内核还需要 cuRAND 头文件。
从已经安装的 `nvidia-curand==10.4.0.35` 复制 17 份头文件，复制前核对 wheel 的 RECORD 哈希。
不要复制整个原 CUDA include 目录，否则会重新混入不同版本的编译器头文件。

```bash
"$RSI_GPU_VENV/bin/python" -I - "$CUDA_HOME" <<'PY'
import base64, csv, hashlib, importlib.metadata, shutil, sys, sysconfig
from pathlib import Path
assert importlib.metadata.version("nvidia-curand") == "10.4.0.35"
site = Path(sysconfig.get_path("purelib"))
sdk = Path(sys.argv[1])
record = site / "nvidia_curand-10.4.0.35.dist-info/RECORD"
copied = 0
for name, digest, size in csv.reader(record.open()):
    if not name.startswith("nvidia/cu13/include/"):
        continue
    src = site / name
    actual = base64.urlsafe_b64encode(hashlib.sha256(src.read_bytes()).digest()).decode().rstrip("=")
    assert digest == "sha256=" + actual, name
    dst = sdk / "include" / src.name
    if dst.exists():
        assert dst.read_bytes() == src.read_bytes(), str(dst)
    else:
        shutil.copyfile(src, dst)
    copied += 1
assert copied == 17, copied
for alias, target in [(sdk / "lib/libcudart.so", "libcudart.so.13"), (sdk / "lib64", "lib")]:
    if not alias.exists():
        alias.symlink_to(target)
PY
export PATH="$CUDA_HOME/bin:$RSI_GPU_VENV/bin:$PATH"
export LD_LIBRARY_PATH="$RSI_COMPAT_LIB:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 3. 将编译缓存放在节点本地

共享 JuiceFS 缓存曾在并发编译时报告找不到刚生成的 Triton `.cubin` 文件。
迁移到节点本地缓存后，本次两组服务未再次出现该错误。
每组运行使用独立缓存目录，避免同时改写同一组编译缓存。

```bash
# 为 Paper A、Paper B 分别执行本段，使用不同的 RSI_CACHE_DIR。
export RSI_CACHE_DIR="/dev/shm/rsi-$UID/paper-A"
export TRITON_CACHE_DIR="$RSI_CACHE_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$RSI_CACHE_DIR/inductor"
export VLLM_CACHE_ROOT="$RSI_CACHE_DIR/vllm"
export FLASHINFER_WORKSPACE_BASE="$RSI_CACHE_DIR/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"
```

`/dev/shm` 需要容纳实际编译输出。空间不足时使用节点本地磁盘目录。
节点重启会清除 tmpfs 缓存，后续启动需要重新编译。
`MAX_JOBS` 只控制编译并发；本次在 170 CPU 配额、充足内存条件下使用 48，其他机器按资源设置。

## 4. 验证后启动实验

```bash
"$CUDA_HOME/bin/nvcc" --version
"$RSI_GPU_VENV/bin/python" -I eval/hpc/check_gpu_env.py --model models/qwen3.8-27b
"$RSI_GPU_VENV/bin/python" -I -c \
  'from flashinfer.gdn_prefill import get_gdn_prefill_module; get_gdn_prefill_module(); from flashinfer.sampling import get_sampling_module; get_sampling_module()'
```

本次额外验证了 CUDA 内核返回 42，以及 top-k=1 的 GPU 采样返回预期 token 3。
启动 `launch.sh` 时保留上面的进程环境。
与现有服务共用 GPU 时，还需要根据实际空闲显存设置各组 vLLM 显存比例；本次使用 0.65。
