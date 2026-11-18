# HPC 服务端（vLLM）安装与使用说明

本机（Apple M3 Pro，无 CUDA）只安装了客户端/评测端（`eval/.venv`），**没有安装 vLLM**。
模型推理在 Linux + CUDA 的 HPC 集群上用 vLLM 的 OpenAI 兼容服务完成，`eval/run_eval.py` 通过 HTTP 访问它。
本文件只是文档；本次环境搭建没有登录任何远程机器。

## 1. 安装（GPU 环境，训练 / 编辑 / 服务共用）

固定版本（2026-11-18 指定）：**verl 0.9.1 · vLLM 0.24.0 · Transformers 5.9.0 · PyTorch 2.11.0 + cu130**，Python 3.11（verl 要求 <3.13）。

```bash
bash eval/hpc/setup_gpu_env.sh ~/venvs/rsi-gpu        # 只能在 Linux x86_64 上运行
~/venvs/rsi-gpu/bin/python -I eval/hpc/check_gpu_env.py --model /path/to/model   # 校验版本、CUDA 13.0、架构支持
```

- `requirements-gpu.in`：直接依赖；`requirements-gpu.lock`：用 uv 为 x86_64-manylinux_2_28、Python 3.11 解析出的完整锁定（243 个包，含 `torch==2.11.0+cu130`、`vllm==0.24.0`、`transformers==5.9.0`、`verl==0.9.1`、`peft==0.21.3`、`ray==2.60.0`、`tensordict==0.8.3`、`compressed-tensors==0.17.0`、`flashinfer-python==0.6.12`）。
- PyPI 上的 torch 2.11.0 本身就是 CUDA 13.0 构建，vLLM 0.24.0 的 wheel 也依赖 cu13 组件；脚本仍显式加了 PyTorch 的 cu130 源。
- **llm-compressor 装不进这套环境**：所有已发布版本都与 vLLM 0.24.0 固定的 `compressed-tensors==0.17.0` 加 Transformers 5.9.0 冲突。Paper A 的量化代码需要自己写出 compressed-tensors 格式的权重。
- **模型架构支持未验证**：本机没有 GPU，锁文件只在本机解析过，没有实际安装。vLLM 0.24.0 发布于 2026-06-30，是否支持 Qwen3.8 27B 要在 HPC 上用 `check_gpu_env.py --model` 确认。
- 评测客户端仍是 CPU 环境 `eval/.venv`（`bash eval/setup_env.sh`），通过 HTTP 访问 vLLM。

## 2. 启动服务

### Paper A：每一轮的量化模型
每一轮量化后的模型请使用**唯一的服务名**（`--served-model-name`），否则客户端缓存会把不同轮次的生成混在一起（缓存键包含模型名，见 eval/README.md）。

```bash
~/venvs/rsi-gpu/bin/vllm serve /path/to/qwen3.8-27b-round3 \
  --served-model-name qwen3.8-27b-r3 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --max-model-len 40960 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --seed 20261118
```

- `--enable-auto-tool-choice --tool-call-parser hermes`：BFCL（function-calling 模式）需要服务端解析 Qwen 的 `<tool_call>` 输出为 OpenAI `tool_calls`。
- `--reasoning-parser qwen3`：思考模式下把 `<think>` 内容放到 `reasoning_content`；客户端也会自动剥离残留的 `<think>...</think>`。
- 量化格式（AWQ/GPTQ/compressed-tensors/FP8 等）由 vLLM 自动识别；如需指定，加 `--quantization <method>`。

### Paper B：多 LoRA
```bash
~/venvs/rsi-gpu/bin/vllm serve /path/to/base-model \
  --served-model-name base \
  --enable-lora --max-loras 8 --max-lora-rank 64 \
  --lora-modules math_v1=/path/to/lora_math_v1 tool_v2=/path/to/lora_tool_v2 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --port 8000
```
LoRA 通过其名字寻址：`run_eval.py --model base --lora math_v1` 会在请求里发送 `"model": "math_v1"`。
运行中动态加载 LoRA（需设置 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`）：
```bash
curl -X POST http://NODE:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d '{"lora_name": "math_v2", "lora_path": "/path/to/lora_math_v2"}'
```

## 3. 从客户端评测
```bash
# 计算节点上 vLLM 起来后（可用 ssh -L 转发端口，或在同一节点运行客户端）
eval/.venv/bin/python -I eval/run_eval.py --task gsm8k,math,mmlu_pro,ifeval,bfcl --pool gate \
  --endpoint http://NODE:8000/v1 --model qwen3.8-27b-r3 --out eval/results/r3_gate
```
- Qwen3 思考模式：`--thinking on|off`（通过 `chat_template_kwargs={"enable_thinking": ...}` 发送），默认 off；思考模式下使用 `max_tokens_thinking`（默认 32768）。
- 代码题需要 Docker/Apptainer 沙箱执行（见 eval/README.md）。HPC 上若只有 Apptainer：`apptainer build sandbox.sif docker-daemon://rsi-eval-sandbox:0.1`，并以 `--containall --net --network none` 方式运行；当前代码只实现了 Docker 路径。

## 4. BFCL 多轮 / agentic 类别（可选）
`run_eval.py` 只评 BFCL 单轮 AST 类别（13 类，3641 条）。multi_turn_*（800）需要 BFCL 的有状态执行环境，memory/web_search 需要额外后端（web_search 需要 SerpAPI key，不在本项目范围内）。
如需评测，请在 HPC 上用官方 CLI（`bfcl-eval==2026.3.23`）对同一个 vLLM 端点运行（OpenAI 兼容模式，按 bfcl 文档设置 `OPENAI_BASE_URL`），并只取 `data/splits/bfcl/test.json` 中的 id。
