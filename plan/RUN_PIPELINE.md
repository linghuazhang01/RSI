# 运行流程：两个项目怎么启动、每个领域怎么单独跑、需要几张卡

2026-11-19 · 代码已实现，本机 mock 全部跑通；真实 GPU 没有跑过，文中耗时都是**假设**，要在第一周实测后改。

---

## 1. 启动命令

```bash
# 环境、数据、模型
./setup_env.sh gpu                 # verl 0.9.1 · vLLM 0.24.0 · Transformers 5.9.0 · PyTorch 2.11.0+cu130
./setup_env.sh sandbox             # 代码领域判分用的 Docker 镜像
./download_data.sh
./download_models.sh qwen3.8-27b   # 两篇都用 Qwen3.8-27B

# 正式启动（launch.sh：先检查环境 / 模型 / 数据 / Docker，再按卡数分配 GPU，各领域后台并行）
./launch.sh --hw h20 --paper both --smoke          # 先每个领域跑 1 代检查（H20：25 张卡）
./launch.sh --hw h20 --paper both --gens 5         # 正式：两篇 × 五个领域
./launch.sh --hw b300 --paper both --gens 5 --with-all   # B300：再加 Paper A 五领域主实验
./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 0,1    # 只跑一个
./launch.sh ... --dry-run                          # 只打印检查结果和命令

# 单个运行（launch.sh 内部调用的就是它）
./run_rsi.sh --paper A --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id A_math_s42 --set hardware=h20
./run_rsi.sh --paper B --domain code --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1,2 --run-id B_code_s42 --set hardware=h20

# 中断后续跑：同一条命令再执行一次，从最后一个完成的步骤接着做
```

- 配置：`edit_code/configs/base.yaml`、`lora_code/configs/base.yaml`（含硬件档 `hardware_profiles`），`configs/domains/<领域>.yaml`。
- 每次运行的全部产物写进 `experiment_records/<run-id>/`；launch.sh 的日志和进程号在 `experiment_records/_launch/<时间>/`。
- `--domain` 可选 `math | code | knowledge | instruction_following | tool_calling`，Paper A 另有 `all`。

## 2. "每个领域单独跑"在两篇里的含义

| | Paper A 自量化 | Paper B 自写 LoRA |
|---|---|---|
| 单领域运行 | 信号和验收只用这个领域的训练集，为这个领域找最好的量化方案 | 每一代的目标任务都是这个领域，其余四个领域作保留探针 |
| 留出评测 | **五个领域全测** | 五个领域全测 |
| `all` | 五个领域一起给信号和验收 | 不需要 |

## 3. 一代的流程

```
Paper A（2026-11-19 修订：改造者冻结、体积固定、迭代被量化的模型）
  第 0 代：原模型 M_0 的验收分、噪声带、留出评测；对 M_0 建一次逐层量化缓存（之后每代复用）；
          体积 = 统一 3 bit GPTQ 配置的大小；Q_0 = 这份配置，是第一个 incumbent
  第 g 代：
    卡 0：vLLM 加载 M_0（冻结的改造者 + 参照）
    卡 1：候选（都从 M_0 原始权重量化到同一体积）的拼装、导出和评测
    1. draw.py 抽本代的题
    2. incumbent 在本代题上打分，展示给改造者
    3. 改造者写程序 → 候选 → 快速筛 →（前 3 名）完整一批，≤24 次评测
    4. 只把分数高于 incumbent 的候选交验收（≤3 次）：在验收集上逐题配对，显著超过 incumbent 才替换
    5. 替换了就导出新 incumbent、做五个领域留出评测；连续 2 代没替换就停

Paper B（每代 M_t 写 LoRA，合并进自己；2026-11-19 起分两种方式，method.mode）
  edit（默认）：每代先按固定配方训练 raw LoRA（--set raw.n_train=... raw.steps=... raw.rank=...），
               改造者不能训练、不能算梯度，只编辑 raw LoRA；验收和 raw 比
  scratch    ：从零写，没有训练和梯度；验收和当前模型比（目标至少涨 gate.min_gain）
  free（消融）：旧设计，改造者可以自己 sgd / grad
  main 卡：vLLM 加载 M_t（改造者）+ 候选 LoRA 热切换评测
  theta0 卡：vLLM 加载原模型（E_t 和参照）
  train 卡：SFT 基线和 raw LoRA 训练、程序里的奇异分解 / 激活统计（只能用本代 train 题）、合并 LoRA
    1. draw.py 抽本代的题（目标领域 train / signal，其余领域保留探针）
    2. SFT 基线；edit 模式再训练 raw LoRA（配方和 SFT 相同时直接复用）
    3. 改造者写程序 → 得到 LoRA → 评测（快速筛 + 前 3 名完整），≤32 个候选
    4. 交固定验收集 → 通过后把 LoRA 合并进权重，得 M_{t+1}
    5. 四个分支（原模型 / SFT 基线 / 自写 LoRA / 非 LLM 搜索）各测 E_t；M_{t+1} 跑留出评测
```

## 4. 需要几张卡（Qwen3.8-27B，BF16 约 52 GB）

| 硬件档 | Paper A 每个运行 | Paper B 每个运行 | 两篇 × 五个领域 |
|---|---|---|---|
| `b300`（288 GB） | 2 张 | 2 张（原模型服务、训练和 main 以外的部分共用第 2 张） | 20 张（加 Paper A `all` 共 22 张） |
| `h20`（80 GB） | 2 张：每张卡只放一个 27B | 3 张：main / 原模型 / 训练各一张 | 25 张 |

H20 档（`--set hardware=h20`，launch.sh 用 `--hw h20`）具体做了这些：
- vLLM 显存比例 0.92、`--max-num-seqs 32`、上下文 20480（够 AIME 的 16k 回答）；52 GB 权重之外约 15 GB 给 KV 缓存和线性注意力状态。
- Paper A：校准每次只放 4 层的 Hessian（每层约 1.6 GB），候选导出后释放显存再起 vLLM。
- Paper B：vLLM 预留 4 个 LoRA 槽（B300 是 8 个）；训练每次只前向 1 条序列、梯度累积（fp32 logits 在约 24.8 万词表上每条 4k token 约 4 GB）。
- 卡更多时可以提速：`--per-run 4` 配合 `gpu_roles`，例如 Paper A `--set 'gpu_roles={editor: [0, 1], candidate: [2, 3]}'`（张量并行 2），Paper B `--set 'gpu_roles.train=[2,3]'`（训练模型拆到两张卡）。

**速度（假设，未实测）**：H20 的 BF16 算力约为 B300 的 1/15，显存带宽约一半。生成（解码）主要受带宽限制，约慢 2 倍；校准、LoRA 训练、长提示的预填充受算力限制，会慢很多。B300 上估计 Paper A 每代 11–13 h、Paper B 每代 20–22 h，H20 上可能是 2–5 倍，第一周用 `--smoke` 实测后再定代数和评测预算（`--set search.max_evaluate=...`）。

**种子：只跑单种子 42**。种子 42 用于每代抽题、改造者采样和评测时的采样解码；数据切分用固定种子 20261118。置信区间用按题目的配对 bootstrap。

## 5. 第一周必须实测的

1. vLLM 0.24.0 在 H20 / B300 上加载 Qwen3.8-27B、fake-quant 检查点、Qwen3.5 结构的 LoRA（含线性注意力投影）。
2. Paper A：逐层量化缓存（GPTQ）的耗时和显存；每个候选的导出 + 加载耗时（27B 每次写 52 GB）。
3. Paper B：27B LoRA 短训练在 80 GB 上的显存峰值和速度；main 卡上 4 个 LoRA 槽是否够。
4. 代码领域的 Docker 判分。
