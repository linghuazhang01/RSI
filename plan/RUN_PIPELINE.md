# 运行流程：两个项目怎么启动、每个领域怎么单独跑、需要几张卡

2026-11-19 · 代码已实现，本机 mock 全部跑通；真实 GPU 没有跑过，文中耗时都是**假设**，要在第一周实测后改。
2026-11-22 · 只保留一种流程（原 `objective: simple` + `scope: single`）：其余目标、五领域范围、保留探针、E_t、每代 draw.py 抽样都已删除。

---

## 1. 启动命令

```bash
# 环境、数据、模型
./setup_env.sh gpu                 # verl 0.9.1 · vLLM 0.24.0 · Transformers 5.9.0 · PyTorch 2.11.0+cu130
./setup_env.sh sandbox             # 代码领域判分用的 Docker 镜像
./download_data.sh
./download_models.sh qwen3.8-27b   # 两篇都用 Qwen3.8-27B

# 正式启动（launch.sh：先检查环境 / 模型 / 数据 / Docker，再按卡数分配 GPU，各领域后台并行）
./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 0,1,2,3                          # Paper A：4 张卡
./launch.sh --hw h20 --paper B --domains math --gens 5 --gpus 4,5,6,7                          # Paper B：4 张卡（method.mode 默认 scratch）
./launch.sh --hw h20 --paper both --gens 5         # 正式：两篇 × 五个领域（10 个运行，最少 20 张卡）
./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 0,1    # 只跑一个
./launch.sh --hw h20 --paper both --domains math --gens 5 --gpus 0,1,2,3   # A 用 0,1，B 用 2,3
./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 4,5,6,7      # 一个运行用 4 张卡（改造者 1 + 3 个候选同时评测）
./launch.sh --hw b300 --paper B --domains math --gens 5 --gpus 0,1,2,3,4,5,6,7 --set method.mode=weights --tag weights
./launch.sh --hw b300 --paper A --domains math --gens 5 --gpus 0,1,2,3,4,5,6,7 --config plan/configs/run/A_math_r30_nosig.yaml --tag r30_nosig
./launch.sh ... --dry-run                          # 只打印检查结果和命令

# 单个运行（launch.sh 内部调用的就是它）
./run_rsi.sh --paper A --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id A_math_s42_single_simple --set hardware=h20
./run_rsi.sh --paper B --domain code --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id B_code_s42_single_simple --set hardware=h20

# 中断后续跑：同一条命令再执行一次，从最后一个完成的步骤接着做
```

- 配置：`edit_code/configs/base.yaml`、`lora_code/configs/base.yaml`（含硬件档 `hardware_profiles`），`configs/domains/<领域>.yaml`；实验配置文件 `--config plan/configs/run/*.yaml`（只写覆盖项，硬件仍由 `--hw` 选）；合并后的配置快照 `plan/configs/{A,B}_<领域>_simple.yaml`（代码不读）。
- 每次运行的全部产物写进 `experiment_records/<run-id>/`；launch.sh 的日志和进程号在 `experiment_records/_launch/<时间>/`。运行名 `<A|B>_<领域>_s<种子>_single_simple[_<tag>]`（后缀沿用旧名，旧运行用同一条命令能续跑）；配置锁定在 `run_config.yaml`，改了配置要用 `--tag` / `--run-id` 另起一个运行。
- `--domain` 可选 `math | code | knowledge | instruction_following | tool_calling`。每个运行只用自己领域的 `simple_data/<领域>/`：筛题、验收、留出评测都在这个领域上。
- 还能切换的：Paper B 的 `method.mode`（`scratch` 默认 / `weights` / `edit`）、`merge.mode`（`merge` 默认 / `adapter`），硬件档 `--hw b300 | h20 | h20_80`；其余用 `--set KEY=VALUE` 或 `--config`。`--simple` 仍然接受，但不起作用（只有这一种流程）。
- **给多少张卡就用多少张**：`launch.sh` 先给每个运行最少的卡（两篇都是 2 张），剩下的卡轮流分给各个运行；每个运行再把自己的卡分给各个角色：Paper A 改造者 1 张、其余卡各起一个候选服务同时评测；Paper B 多的卡给 main（张量并行 / 数据并行副本）和训练模型（拆到多张卡）。见两篇 README 的"硬件"。`--per-run N` 改成固定每个运行 N 张。

## 2. 数据：每个领域三份

`simple_data/<领域>/`（`data/scripts/make_simple_data.py` 从去污染、去重后的训练集池子里切出，切分种子 20261118；详见 `simple_data/README.md`）：

| 角色 | 数据集 | 大小 | 用途 |
|---|---|---|---|
| 筛题（screen） | `simple_<领域>_pool` 去掉验收和留出的题 | 每代重新抽 500 道 | 最新模型答一次（每题 4 个样本），改造者看详细结果；池子里其余的题是 train 数据 |
| 验收（gate） | `simple_<领域>_eval` | 固定 500 道 | 每个候选评一次，和最新模型逐题配对；改造者只看到之前候选的验收统计 |
| 留出（held-out） | `simple_<领域>_test` | 固定 500 道 | 只做报告，从不展示、从不用于选择 |

每代的筛题由种子 `<seed>:simple:<领域>:<轮次>` 决定，Paper A 第 g 代和 Paper B 第 g−1 代抽到同一批题。

## 3. 一代的流程

```
两篇相同：每代 search.threads × search.steps 个候选（默认 8 × 1），改造者每个候选最多 editor.tool_rounds 轮工具调用
（一个持续的 Python 会话），最后写出一个程序得到候选。
这套流程是写死的，没有开关（2026-11-23 起旧的搜索开关 screen_candidates、gate_every_candidate、charge_screens、
program_evaluate_*、max_evaluate、top_full、rerun_seeds、gate.max_submissions、噪声带、Paper A 的 kl_weight 都已删除，
用 --set / --config 设成别的值会直接报错）：候选不筛题、没有 api.evaluate 和评测预算、搜索后不再复评或另行提交验收。
  1. 抽本代的 500 道筛题；最新模型答一次（每题 4 个样本），改造者在上下文里看到摘要，用 screen_results / screen_output
     看每道题和完整输出（含思考过程）；候选本身不做筛题
  2. 每个候选在验收集上评一次（500 × 4，和最新模型逐题配对），结果给之后的步骤 / 代看
  3. 接受配对增益最大且通过验收的候选（增益 ≥ 阈值，默认还要求 95% 置信区间下界 > 0）；然后测留出集（500 × 4）
  4. 连续 3 代没接受就停（loop.patience）

解码（筛题 / 验收 / 留出相同）：Qwen3 思考模式，temperature 0.6、top_p 0.95、top_k 20，每题 4 个样本（avg@4），
max_tokens 16384，种子 = 运行的 --seed（配置里的 decoding 和 heldout.decoding）。

Paper A（改造者冻结、体积固定、迭代被量化的模型）
  第 0 代：原模型 M_0 的验收分、留出评测；对 M_0 建一次逐层量化缓存（之后每代复用）；校准文本用 M_0 的回答；
          体积 = 统一 4 bit GPTQ 配置的大小；Q_0 = 这份配置，是第一个 incumbent
  第 g 代：
    卡 0：vLLM 加载 M_0（冻结的改造者）；搜索中改造者写完程序就睡眠（权重移到 CPU 内存），卡 0 临时当一个候选槽
    其余卡：候选服务池，每张卡（candidate_group_size）一个候选服务；同一步里各线程的候选同时导出、加载、评测
    第 0 代准备（校准、敏感度表、缓存、Hessian 误差表）、incumbent 的筛题和验收打分、留出评测都分到所有卡上
    1. 抽本代的筛题；incumbent（最新的量化模型）答筛题
    2. 改造者写程序 → 量化配置 → 候选；每个候选一次验收，和 incumbent 逐题配对（分数里没有 KL 项）
    3. 通过验收、增益最大的候选替换 incumbent：导出新 incumbent、做留出评测

Paper B（每代 M_t 改自己，接受后成为下一代模型；method.mode，默认不训练）
  scratch（默认）：从零写 LoRA，候选热加载到 main 副本上评测
  weights      ：直接编辑权重（通道 / 模块缩放、稠密低秩、往原模型拉回），每个候选写出改动的权重、在候选槽上起 vLLM 评测；
                4 张卡 = 改造者 1（评测时睡眠，当第 3 个槽）+ 候选槽 2 + train 1
  edit         ：第 0 代训练一次 raw LoRA，之后每代编辑它（M_g = M_0 + 当前 LoRA，不合并）
  接受后（merge.mode）：merge（默认）把 LoRA 合并进权重、重启 vLLM 得 M_{t+1}；adapter 不合并，把接受过的 LoRA 按秩拼接成
                一个累积 adapter，挂在 M_0 服务上
  验收：每个候选都和当前模型逐题配对（提升 ≥ gate.min_gain 且置信区间下界 > 0）；edit 第 0 代没有候选通过时，
  raw LoRA 同样和当前模型验收一次，通过就接受（gate.fallback）
  main 卡：除 train 以外的所有卡，每张卡一份 M_t（数据并行副本）：所有候选 LoRA 加载到每份副本上，评测的题目分给各副本，
         改造者线程 k 用副本 k mod n；4 张卡 = 3 份副本
  train 卡（1 张）：raw LoRA 训练（edit 第 0 代）、程序里的奇异分解 / 激活统计、合并 LoRA
    1. 抽本代的筛题；当前模型答筛题
    2. edit 模式第 0 代：采样训练答案，训练 raw LoRA（之后各代和 scratch 都不训练、不采样）
    3. 改造者写程序 → 得到 LoRA（或权重改动）→ 每个候选一次验收；非 LLM 对照分支同样多的候选（nonllm，可关）
    4. 通过验收、增益最大的候选按 merge.mode 写进模型，得 M_{t+1}；M_{t+1} 跑留出评测
```

## 4. 需要几张卡（Qwen3.8-27B，BF16 约 52 GB）

| 硬件档 | Paper A 每个运行（最少） | Paper B 每个运行（最少） | 两篇 × 五个领域（最少） |
|---|---|---|---|
| `b300`（288 GB） | 2 张 | 2 张（main / 训练） | 20 张 |
| `h20`（卡约 140 GB，按 110 GB 规划） | 2 张：每张卡只放一个 27B | 2 张（main / 训练） | 20 张 |
| `h20_80`（80 GB，之前的规划） | 2 张 | 2 张 | 20 张 |

H20 档（`--set hardware=h20`，launch.sh 用 `--hw h20`；2026-11-20 起按每张卡 110 GB 规划，卡实际约 140 GB，剩下的作余量）：
- 仍然每张卡只放一个 27B（约 52 GB）：两个 27B 就是 104 GB，不剩 KV 缓存。
- vLLM 的显存按 GiB 给：参数 `--rsi-gpu-memory-gib 110` 在启动时换算成 `--gpu-memory-utilization = 110 / 卡的总显存`（`rsicore.servers`，读 nvidia-smi；读不到时用配置里的 0.78）。52 GB 权重和约 6 GB 运行时之外，约 50 GB 给 KV 缓存和线性注意力状态【估计】，所以 `--max-num-seqs 128`；评测的上下文要放下提示 + 16384 token 的回答，改造者的上下文见配置 `editor.context_tokens`。
- Paper A：校准用默认的每次 16 层 Hessian（52 GB 模型 + 16 × 1.6 GB）。
- Paper B：8 个 LoRA 槽；训练每次前向 2 条序列【试点值】（每条 4k token 的 fp32 logits、梯度和 log-softmax 约 12 GB），第一次正式运行看到显存峰值后再调。
- 原来按 80 GB 规划的档改名为 `h20_80`（`--hw h20_80`）：0.92、32 条序列、4 个 LoRA 槽、训练每次 1 条、校准每次 4 层。之前用旧 `h20` 档启动的运行要接着跑，需要改用 `--hw h20_80`（运行配置是锁定的）。
- 卡的分工是自动的：给多少张卡就用多少张（见上面）。

**速度（假设，未实测）**：H20 的 BF16 算力约为 B300 的 1/15，显存带宽约一半。生成（解码）主要受带宽限制，约慢 2 倍；校准、LoRA 训练、长提示的预填充受算力限制，会慢很多。每代的主要开销是验收：8 个候选 × 500 道 × 4 个思考模式样本。以第一次正式运行第 0 / 1 代的实测耗时为准，再定代数和候选数（`--set search.threads=...`）。

**种子：只跑单种子 42**。种子 42 用于每代抽筛题、改造者采样和评测时的采样解码；`simple_data` 的切分用固定种子 20261118。置信区间用按题目的配对 bootstrap。

## 5. 第一周必须实测的

1. vLLM 0.24.0 在 H20 / B300 上加载 Qwen3.8-27B、fake-quant 检查点、Qwen3.5 结构的 LoRA（含线性注意力投影）。
2. Paper A：逐层量化缓存（GPTQ）的耗时和显存；每个候选的导出 + 加载耗时（27B 每次写 52 GB）。
3. Paper B：27B LoRA 短训练在 110 GB 规划下的显存峰值和速度（micro_batch 2 能否加大）；8 个 LoRA 槽和 128 条并发序列是否放得下。
4. 代码领域的 Docker 判分。
