# lora_code/：Paper B 自写 LoRA

计划见 `plan/PLAN.md` §4。尚未开始写代码。

环境：verl 0.9.1 · vLLM 0.24.0 · Transformers 5.9.0 · PyTorch 2.11.0 + cu130（`bash eval/hpc/setup_gpu_env.sh`）。每代的训练数据用 `data/scripts/draw.py` 抽取（配置 `data/draw/paper_b.yaml`）。计划中的模块：

- `primitives/`：程序可用的原语（奇异方向缩放、激活方向、秩 1 编辑；梯度工具；LoRA 缩放 / 截断 / 合并）
- `sandbox/`：执行模型写的程序，检查形状、数值、秩上限，只开放训练集 `train` 池
- `loop/`：一代的内层搜索、验收、合并进权重、接任
- `branches/`：四个分支（原模型 / SFT 基线 / 自写 LoRA / 非 LLM 搜索）和 E_t（Seal-Tools）
- `baselines/`：SFT、全局缩放前沿、checkpoint 前沿、SLoRA、TIES、CMA-ES
