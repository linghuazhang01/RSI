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
./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 0,1,2,3                          # Paper A：4 张卡
./launch.sh --hw h20 --paper B --domains math --gens 5 --gpus 4,5,6,7 --set method.mode=scratch # Paper B：4 张卡
./launch.sh --hw h20 --paper both --gens 5         # 正式：两篇 × 五个领域
./launch.sh --hw b300 --paper both --gens 5 --with-all   # B300：再加 Paper A 五领域主实验
./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 0,1    # 只跑一个
./launch.sh --hw h20 --paper both --domains math --gens 5 --gpus 0,1,2,3   # 默认单领域：A 用 0,1，B 用 2,3
./launch.sh --hw h20 --paper A --domains math --gens 5 --gpus 4,5,6,7      # 一个运行用 4 张卡（改造者 1 + 3 个候选同时评测）
./launch.sh ... --dry-run                          # 只打印检查结果和命令

# 单个运行（launch.sh 内部调用的就是它）
./run_rsi.sh --paper A --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id A_math_s42 --set hardware=h20
./run_rsi.sh --paper B --domain code --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1,2 --run-id B_code_s42 --set hardware=h20

# 中断后续跑：同一条命令再执行一次，从最后一个完成的步骤接着做
```

- 配置：`edit_code/configs/base.yaml`、`lora_code/configs/base.yaml`（含硬件档 `hardware_profiles`），`configs/domains/<领域>.yaml`。
- 每次运行的全部产物写进 `experiment_records/<run-id>/`；launch.sh 的日志和进程号在 `experiment_records/_launch/<时间>/`。
- `--domain` 可选 `math | code | knowledge | instruction_following | tool_calling`，Paper A 另有 `all`。
- **默认单领域**（`scope: single`，运行名加 `_single`）：训练、验收、评测都只在这次运行的领域上做。Paper A 的 KL / 校准文本和留出评测只用这个领域；Paper B 没有保留探针（验收、反馈、KL 只看目标领域）、留出评测只测这个领域、E_t 关闭。原设计用 `--set scope=all`（`launch.sh --all-domains`）。
- **给多少张卡就用多少张**：`launch.sh` 先给每个运行最少的卡（Paper A 2；Paper B 2，H20 五领域 3），剩下的卡轮流分给各个运行；每个运行再把自己的卡分给各个角色：Paper A 改造者 1 张、其余卡各起一个候选服务同时评测；Paper B 多的卡给 main（张量并行）和训练模型（拆到多张卡）。见两篇 README 的"硬件"。`--per-run N` 改成固定每个运行 N 张。

## 2. "每个领域单独跑"在两篇里的含义

| | Paper A 自量化 | Paper B 自写 LoRA |
|---|---|---|
| 单领域运行 | 信号和验收只用这个领域的训练集，为这个领域找最好的量化方案 | 每一代的目标任务都是这个领域，其余四个领域作保留探针 |
| 留出评测 | **五个领域全测** | 五个领域全测 |
| `all` | 五个领域一起给信号和验收 | 不需要 |

## 3. 一代的流程

```
直接在评测集上优化（`./launch.sh --eval-overfit`）：两篇的数据都换成 math_eval（MATH-500 + GSM8K test，1819 道），
每个阶段都是全部 1819 道（贪心），评测集分数严格更高就接受，没有单独的验收集；留出评测也是这 1819 道（贪心）；运行名带 _evalfit。

解码（2026-11-20）：搜索、验收、噪声都是贪心（噪声重评因此不再执行，噪声带 = 0）；留出评测用采样（Qwen 非思考模式推荐：
temperature 0.7、top_p 0.8、top_k 20，种子 42，avg@n：AIME / AMC 8 个、MATH-500 / GPQA 4 个、其余 1 个，`heldout.decoding`）；
Paper B 的训练答案用同一组采样参数（每题 4 个，取第一个判对的，4 个都错就跳过这道题）；改造者写程序时 temperature 0.6。

Paper A（2026-11-19 修订：改造者冻结、体积固定、迭代被量化的模型）
  第 0 代：原模型 M_0 的验收分、噪声带、留出评测；对 M_0 建一次逐层量化缓存（之后每代复用）；
          体积 = 统一 4 bit GPTQ 配置的大小（2026-11-20 起；原为 3 bit）；Q_0 = 这份配置，是第一个 incumbent
  第 g 代：
    卡 0：vLLM 加载 M_0（冻结的改造者 + 参照）；搜索中改造者写完程序就睡眠（权重移到 CPU 内存），卡 0 临时当第 4 个候选槽
    其余卡：候选服务池，每张卡（candidate_group_size）一个候选服务；同一步里各线程的候选同时导出、加载、评测
    第 0 代准备（校准、敏感度表、缓存、Hessian 误差表）、M_0 的批量任务（验收打分、留出评测、KL 回答）、留出评测都分到所有卡上
    1. draw.py 抽本代的题
    2. M_0 只算 KL 参照（不在本代题上打分）；incumbent 在本代题上打分，作为所有候选的参照（保持率、配对差、例题都相对它）
    3. 改造者写程序 → 候选 → 快速筛 →（前 3 名）完整一批，≤24 次评测
    4. 只把分数高于 incumbent 的候选交验收（≤3 次）：在验收集上逐题配对，显著超过 incumbent 才替换
    5. 替换了就导出新 incumbent、做五个领域留出评测；连续 2 代没替换就停

Paper B（每代 M_t 改自己，接受后写进权重；method.mode，2026-11-21 起默认不训练）
  scratch（默认）：从零写 LoRA，候选热加载到 main 副本上评测
  weights      ：直接编辑权重（通道 / 模块缩放、稠密低秩、往原模型拉回），每个候选写出改动的权重、在候选槽上起 vLLM 评测；
                4 张卡 = 改造者 1（评测时睡眠，当第 3 个槽）+ 候选槽 2 + train 1
  edit（保留） ：第 0 代训练一次 raw LoRA，之后每代编辑它（M_g = M_0 + 当前 LoRA，不合并）
  验收都和当前模型逐题配对（提升 ≥ gate.min_gain 且置信区间下界 > 0；edit 第 0 代和 raw 比）；只送信号题上赢了当前模型的候选；
  连续 3 代都没接受就停（loop.patience）；数据切分 train / signal = 50% / 50%，快速筛 400 道；改造者另有免费的 compare 和 train_loss
  main 卡：除 train（和 E_t 打开时的 theta0）以外的所有卡，每张卡一份 M_t（数据并行副本，2026-11-20）：所有候选 LoRA
         加载到每份副本上，评测的题目分给各副本，改造者线程 k 用副本 k mod n；4 张卡 = 3 份副本
  theta0 卡：vLLM 加载原模型（E_t 和参照；单领域不启动）
  train 卡（1 张）：raw LoRA 训练（edit 第 0 代）、程序里的奇异分解 / 激活统计 / train_loss（只能用本代 train 题）、合并 LoRA
    1. draw.py 抽本代的题（目标领域 train / signal，其余领域保留探针）
    2. edit 模式第 0 代：采样训练答案，训练 raw LoRA（之后各代和 scratch 都不训练、不采样）
    3. 改造者写程序 → 得到 LoRA → 评测（快速筛 + 前 3 名完整），≤32 个候选
    4. 交固定验收集 → 通过后把 LoRA 合并进权重，得 M_{t+1}
    5. 各分支（原模型 / edit 的 raw LoRA / 自写 LoRA / 非 LLM 搜索）各测 E_t；M_{t+1} 跑留出评测
```

## 4. 需要几张卡（Qwen3.8-27B，BF16 约 52 GB）

| 硬件档 | Paper A 每个运行（最少） | Paper B 每个运行（最少；单领域 / 五领域） | 两篇 × 五个领域（最少） |
|---|---|---|---|
| `b300`（288 GB） | 2 张 | 2 张（原模型服务、训练和 main 以外的部分共用第 2 张） | 20 张（加 Paper A `all` 共 22 张） |
| `h20`（卡约 140 GB，按 110 GB 规划） | 2 张：每张卡只放一个 27B | 2 张（main / 训练）；五领域 3 张（再加原模型给 E_t） | 单领域 20 张；五领域 25 张 |
| `h20_80`（80 GB，之前的规划） | 2 张 | 同上 | 同上 |

H20 档（`--set hardware=h20`，launch.sh 用 `--hw h20`；2026-11-20 起按每张卡 110 GB 规划，卡实际约 140 GB，剩下的作余量）：
- 仍然每张卡只放一个 27B（约 52 GB）：两个 27B 就是 104 GB，不剩 KV 缓存。
- vLLM 的显存按 GiB 给：参数 `--rsi-gpu-memory-gib 110` 在启动时换算成 `--gpu-memory-utilization = 110 / 卡的总显存`（`rsicore.servers`，读 nvidia-smi；读不到时用配置里的 0.78）。52 GB 权重和约 6 GB 运行时之外，约 50 GB 给 KV 缓存和线性注意力状态【估计】，所以 `--max-num-seqs 128`；上下文 20480（够 AIME 的 16k 回答），Paper A 改造者 32768。
- Paper A：校准用默认的每次 16 层 Hessian（52 GB 模型 + 16 × 1.6 GB）。
- Paper B：8 个 LoRA 槽；训练每次前向 2 条序列【试点值】（每条 4k token 的 fp32 logits、梯度和 log-softmax 约 12 GB），第一次正式运行看到显存峰值后再调。
- 原来按 80 GB 规划的档改名为 `h20_80`（`--hw h20_80`）：0.92、32 条序列、4 个 LoRA 槽、训练每次 1 条、校准每次 4 层。之前用旧 `h20` 档启动的运行要接着跑，需要改用 `--hw h20_80`（运行配置是锁定的）。
- 卡的分工是自动的：给多少张卡就用多少张（见上面）。

**速度（假设，未实测）**：H20 的 BF16 算力约为 B300 的 1/15，显存带宽约一半。生成（解码）主要受带宽限制，约慢 2 倍；校准、LoRA 训练、长提示的预填充受算力限制，会慢很多。B300 上估计 Paper A 每代 11–13 h、Paper B 每代 20–22 h，H20 上可能是 2–5 倍，以第一次正式运行第 0 / 1 代的实测耗时为准，再定代数和评测预算（`--set search.max_evaluate=...`）。

**种子：只跑单种子 42**。种子 42 用于每代抽题、改造者采样和评测时的采样解码；数据切分用固定种子 20261118。置信区间用按题目的配对 bootstrap。

## 5. 第一周必须实测的

1. vLLM 0.24.0 在 H20 / B300 上加载 Qwen3.8-27B、fake-quant 检查点、Qwen3.5 结构的 LoRA（含线性注意力投影）。
2. Paper A：逐层量化缓存（GPTQ）的耗时和显存；每个候选的导出 + 加载耗时（27B 每次写 52 GB）。
3. Paper B：27B LoRA 短训练在 110 GB 规划下的显存峰值和速度（micro_batch 2 能否加大）；8 个 LoRA 槽和 128 条并发序列是否放得下。
4. 代码领域的 Docker 判分。
