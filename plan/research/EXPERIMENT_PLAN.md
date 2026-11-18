# 实验计划：两篇 RSI × 权重编辑

日期：2026-11-14

两篇论文：
- **Paper A 自量化**：模型自己量化自己的权重，量化后的模型当下一轮的压缩者。查新结论 PROCEED 7/10，报告在 `plan/research/self-quantization/NOVELTY_REPORT.md`。
- **Paper B 自编辑 LoRA**：模型写代码编辑自己训出的 LoRA，编辑后的模型当下一轮的编辑者。查新结论 PROCEED 6.5/10，报告在 `plan/research/self-lora-edit/NOVELTY_REPORT.md`。

> **2026-11-17 更新**：两篇各自的独立完整计划见 [PAPER_A_SELF_QUANT_PLAN.md](PAPER_A_SELF_QUANT_PLAN.md) 和 [PAPER_B_SELF_LORA_PLAN.md](PAPER_B_SELF_LORA_PLAN.md)。它们修正了本文中的数据切分重叠和 HumanEval 题量等问题，冲突时以那两份为准；本文保留作总览。

文中引用的数字都来自已全文精读、并抽查过的论文笔记。标注"需实测"或"假设"的地方是我的估计，没有文献依据。

**算力假设**：一台 8×H100/H200 节点。CompressionBench 每个 run 用一张 H200、8 小时；MASS 在 27B 上做一轮 LoRA SFT 用了 6×H100、约 19 小时。实际 GPU 数不同时，按比例调整每轮的候选数和种子数。

---

# 第一部分：两篇共用的设计

## 1. 两篇都必须回答的 RSI 问题

两篇的递归结构相同：

```
第 t 代模型 M_t ──(当编辑者，产出编辑程序 / 量化方案)──▶ 候选
候选 ──(固定外部评测验收)──▶ M_{t+1}
M_{t+1} ──(接任，成为下一代编辑者)──▶ …
```

要成为一篇 RSI 论文，只证明"M_t 比 M_0 更好"是不够的。必须回答：**被改过之后，模型作为改进者的能力是变强、变弱，还是不变？原因是什么？** 所以两篇都有同一个核心指标：

> **编辑者能力 E_t**：让第 t 代模型去处理一个**固定的、不随代数变化**的目标。Paper A 是固定预算下压缩同一个源 checkpoint；Paper B 是编辑同一个固定模型在未见任务上的 raw LoRA。测它产出结果的质量。被编辑的对象固定，才能分清是"编辑者变强了"，还是"被编辑的对象更容易改"。

## 2. 模型选择

| 用途 | 模型 | 理由 |
|---|---|---|
| **主实验（两篇共用）** | **Qwen3.8 27B（dense）** | CompressionBench 的 dense 任务用的就是它，BF16 参考值、统一 GPTQ、已发布量化版本和 6 个前沿 agent 的结果都能直接引用作为上界（Table 1、A3）；27B 能当编码 agent，MASS 用的 Qwen3.6-27B 就是例子 |
| Paper B 的 pilot | **Qwen3-8B** | Search over Self-Edit 用的就是它，SEAL 用 Qwen2.5-7B，能直接对照；成本低 |
| Paper A 的第二个模型（第三阶段） | **Gemma-4-31B** | Flat Score 显示它在 INT4 下工具名幻觉率从 19.5% 翻到 38.3%，而 Qwen-3.6-27B 几乎不变。用它做"量化敏感模型"的对照，看结论能否跨家族 |
| 备选 | Qwen3.6-27B | 如果 Qwen3.8 27B 在 agent 外壳里跑不稳，就换它，因为 MASS 已证明它能作为 coding agent 跑通 |

**Agent 外壳**：选一个开源的编码 agent 外壳。MASS 用 qwen-code，Self-HPO 用 mini-SWE-agent。模型通过 Python 工具函数操作，所有执行都在沙箱里：无网络、限时、只能调用白名单 API。**外壳在两篇和所有代之间保持不变**，否则编辑者能力的变化无法归因。

## 3. 怎样给模型反馈（两篇通用的三层结构）

这是整个 RSI 能否成立的关键。文献给出两条硬约束：
1. **不能用模型自评。** Anosognosia 显示，模型从自己的输出看不出自己被量化过，训练出的识别能力换个量化方法就失效。When AI Reviews Its Own Code 显示，用自评做筛选门，筛选能力会逐代退化（定理 2.3）。
2. **不能只看 PPL 或单一分数。** CompressionBench 中 Opus 在 27B/8 GiB 上 PPL 一路下降，GSM8K 却从 88% 掉到 72%。Flat Score 中 INT4 的最终得分都不显著变化，但把失败预算收紧到 K=2 后差距从 1.3 扩大到 16.7。

所以分三层：

| 层 | 给模型看到什么 | 什么时候用 | 数据 |
|---|---|---|---|
| **L1 搜索反馈**（便宜、可反复调用） | 具体数字 + 少量错误样例 + 之前候选的历史记录 | 每个候选评测完就返回 | 搜索集（见各篇） |
| **L2 验收门**（有限次数） | 只返回"通过 / 不通过"和粗粒度原因 | 每轮最多提交 3 次（假设值，需预先登记） | dev2，与搜索集不重叠 |
| **L3 封存评测** | 编辑者永远看不到 | 只在论文最后报告 | test |

补充规则：
- **门槛相对第 0 代设定**，不能相对上一代，否则每代允许一点退化，累积起来就失控。
- **回滚**：候选造成严重退化时直接回滚，并把这次失败写进反馈历史。Adaptive Pruning 用"单轮 PPL 涨超过 15% 就回滚"，可以借鉴。
- **反馈格式固定**为结构化 JSON，在所有代之间不变。
- **评测器本身不随模型变化**：BF16 参考输出、用户模拟器、判分程序都固定。Flat Score 的模拟用户固定为全精度 Qwen-3.6-27B。主实验尽量只用确定性判分（精确匹配、单元测试、执行结果），不用 LLM judge。
- **成本计入预算**：每个候选的诊断前向、量化或编辑耗时、评测耗时都计入。LLM 编辑者和非 LLM 搜索按**总评估次数和 GPU 时间**两种口径对齐。

## 4. 共享基础设施

- **服务**：vLLM。Paper B 用多 LoRA 同时挂载，候选 adapter 不需要合并进基座就能评测。
- **量化执行**（Paper A）：llm-compressor / GPTQModel 等现成库。
- **评测**：lm-eval-harness（选择题和数学）、EvalPlus（HumanEval+/MBPP+）、LiveCodeBench、BFCL、τ²-bench、IFEval。
- **日志**：保存每个候选的程序 / 配置、反馈、耗时和 token 数，用于分析编辑者行为。

---

# 第二部分：Paper A 自量化

## A1. 研究问题
1. 一个 27B 模型能否在不借助更强模型的情况下，为自己做出比统一量化和非 LLM 分配更好的量化决策？
2. 量化后的模型当压缩者，后续的压缩决策会怎样变化？递归在哪里停下，为什么？

## A2. 设定

**预算序列**：

| 预算 | 约合 bits/参数（含 scale、embedding 等开销） | 用途 |
|---|---|---|
| 16 GiB | 5.09 | 流程验证。CompressionBench pilot 中 W4-g128 实际 15.00 GiB，C4 PPL +3.19% |
| 12 GiB | 3.82 | 主研究区。统一 GPTQ 准确率只有 46%，6 个前沿 agent 全部通过 |
| 10 GiB | 3.18 | 主研究区。CompressionBench 没测过，是新数据 |
| 8 GiB | 2.55 | 压力测试。前沿 agent 只有一半通过 |

**动作空间（Level 1，先做）**：
- 每个模块（q/k/v/o/gate/up/down）的比特数 ∈ {3, 4, 8}，后期可加 2；
- group size ∈ {32, 64, 128}；量化器 ∈ {GPTQ, AWQ, RTN}；
- embedding / 输出头精度 ∈ {4, 8, 16}；
- 校准数据 ∈ {C4, FineWeb, 自己生成, 推理类数据} × 条数 {128, 512}。

激活和 KV cache 固定在 16/8 bit：A4 会毁掉工具调用决策（Which Decisions Fig.5），W4A4KV4 时 32B 也掉 2.9（Quant Hurts Reasoning Table 1）。
只允许 vLLM 能高效运行的格式。CompressionBench pilot 里混合 3/4 bit 在 vLLM 上能跑到 110.8 TPS（Table A4），但具体 kernel 组合**需实测确认**。

**Level 2（第三阶段）**：允许写新的量化代码，比如旋转或 trellis 编码，参照 CompressionBench 中 Sonnet 的方案。

**工具**：
- `profile(module, bits, quantizer)`：返回该模块单独量化后的 KL 增量；
- `quantize(config)`：生成量化产物；
- `evaluate(artifact)`：返回 L1 反馈；
- `submit(artifact)`：提交到 L2 验收门。

## A3. 每一轮的流程

```
第 t 轮：编辑者 = M_t（第 0 轮是 BF16 原模型），预算 B_t
被压缩对象 = 同一模型的 BF16 源权重（只读）       ← 主设计
  1. 编辑者调用 profile，阅读上一轮的反馈历史
  2. 提出配置 → quantize → evaluate（每轮最多 N=24 个候选，或 G GPU 小时，取先到者）
  3. 选最优候选 submit → L2 验收（最多 3 次）
  4. 通过 → M_{t+1}，下一轮当编辑者；不通过 → 记录停止点，即递归边界
```

**L1 反馈内容**：
- 超额 KL / PPL 比：Dolma 类上下文 256 段 × 512 token，BF16 续写 1,536 token，沿用 CompressionBench 的做法；
- 实际包大小、常驻显存；
- batch=1 decode TPS；
- MATH-500 中 100 题的准确率；
- BFCL 探针 100 题，"该调用 / 不该调用"分开报；
- 2–3 条错误样例。

**L2 验收门**（相对第 0 代）：
- GSM8K（4-shot）、HumanEval（0-shot）、TriviaQA（5-shot）各 200 题，平均分 ≥ 0.8× 第 0 代。这是 CompressionBench 的协议，题量从 50 扩大到 200，以降低噪声；
- BFCL 调用遗漏率上升不超过预设值；
- τ²-bench 30 个 episode 中，工具错误率上升不超过预设值；
- 存储达标。

具体阈值在 Pilot 0 后**预先登记**。

## A4. 对照组

**最核心的是 2×2 设计**：

| | 每轮从原始 BF16 权重开始 | 在上一轮量化后的权重上继续压 |
|---|---|---|
| **冻结的 BF16 编辑者** | 非递归参照 | 测重复量化的损伤 |
| **当前的量化编辑者** | 测编辑者退化的影响 | 完整递归 |

**配对比较**：在每个固定的目标和预算下，BF16 编辑者和当前编辑者拿到同样的工具、历史和校准数据，各自提出方案，然后执行对比。

**基线**（共享动作空间，评估次数对齐）：
- 统一 GPTQ / AWQ（能放进预算的最大统一比特数）；
- 按敏感度贪心分配；
- 随机搜索、CMA-ES；
- GAMMA（当前最强的非 LLM 混合精度分配器，必须计入候选预量化的成本）；
- 已发布量化版本：GSQ-RCO；
- 前沿 agent 上界：直接引用 CompressionBench Table 1，不必重跑。

**消融**：
- 校准数据由第 0 代生成 vs 由当前代重新生成；
- 有无诊断工具；
- 有无反馈历史。

## A5. 最终评测（L3 封存）

| 维度 | 数据集 | 报告方式 |
|---|---|---|
| 知识 | MMLU-Pro、TriviaQA 闭卷 | 准确率 |
| 推理 | MATH-500（全量）、AIME 2024–25 | 准确率；输出长度和重复率 |
| 代码 | HumanEval+、MBPP+、LiveCodeBench | pass@1 |
| 指令遵循 | IFEval | 准确率 |
| 工具调用 | BFCL v3/v4（irrelevance 与 live_simple 分开，加上 multi_turn） | 配对翻转率 |
| 多轮 agent | τ²-bench retail / telecom | 成功率；按失败类型的错误率；随失败预算变化的成功率曲线 S(K) |
| 安全（可选） | XSTest | 拒答率 |
| **效率** | — | 包大小、常驻显存、峰值显存、TTFT、decode TPS（batch 1 和 32） |
| **压缩者能力 E_t** | 固定任务：在 12 GiB 下压缩 BF16 源 | 合法方案比例、可运行比例、验收通过率、最终 KL、每 GPU 小时的质量 |

## A6. Pilot 顺序

| 阶段 | 内容 | 判断条件 |
|---|---|---|
| **A-P0** | 27B 统一 W8 / W4 / W3（GPTQ 和 AWQ），只跑 agent 评测；另让 BF16 27B 在 12 GiB 下当一轮压缩者，跑 3 个种子 | BF16 编辑者能稳定产出合法方案（例如 ≥2/3 个种子），才继续。不行就简化工具和动作空间。同时记录 agent 能力在哪个比特数开始掉，以此确定预算序列 |
| **A-P1** | 单轮：16 和 12 GiB 下，LLM 编辑者 vs 所有基线，3 个种子 | LLM 不比 CMA-ES / GAMMA 差，或成本更低：继续。明显更差：先改诊断和反馈，A4 的结论可能要删 |
| **A-P2** | 递归 16 → 12 → 10 GiB，2×2 设计 + 配对比较，3 个种子 | 核心结果。不论编辑者退化有没有影响后续压缩，都能写，前提是归因清楚 |
| **A-P3** | Gemma-4-31B；8 GiB；Level 2 代码级动作 | 扩展与泛化 |

---

# 第三部分：Paper B 自编辑 LoRA

## B1. 研究问题
1. 模型写的训练后 adapter 编辑，在同等预算下能否胜过不编辑、人工规则、解析式截断和非 LLM 搜索？在连续学习中，尤其能否改善"学新任务"和"不忘旧任务"之间的取舍？
2. 编辑过自己之后，模型在**未见任务**上编辑**固定**模型的 adapter 时，是否更好？编辑本身是否带来了超出普通任务训练的提升？

## B2. 数据集

**连续任务序列**（按顺序学；每个任务分 train / dev / dev2 / test）：

| 顺序 | 任务 | 训练 → 评测 | 理由 |
|---|---|---|---|
| T1 | 知识注入 | SQuAD 无原文 QA（SEAL 设置：200 篇文章、974 题的评测集） | 能直接对比 SEAL 47.0（Qwen2.5-7B）、Search over Self-Edit 45.1 / 50.3（Qwen3-8B） |
| T2 | 数学 | GSM8K train → MATH-500 | 确定性判分 |
| T3 | 代码 | MBPP train → HumanEval+、MBPP+ | 执行判分 |
| T4 | 科学 | SDFT 的 Science 划分（2,674 / 507） | 有现成的遗忘对照 |
| T5 | 表格问答 | TableInstruct → TableBench（FT-Dojo 设置） | 多领域 |

- **保留集**（每轮都测）：MMLU-Pro、IFEval，以及之前学过的所有任务的 dev。
- **编辑者能力测试任务**（从不训练，只用来测 E_t）：FinanceIQ、PANORAMA 的一个子任务（FT-Dojo 设置），再加一个 BFCL 函数调用任务。为这些任务**预先训练好一套固定的 raw LoRA 库**，包括 3 个种子，以及训练中途的 checkpoint；这套库在所有代之间不变。

## B3. 设定

- **模型表示**：M_t = θ0 + adapter 栈。
- **raw LoRA 配置**（pilot 固定，第三阶段改为模型自选）：r=16、α=32、所有线性层、lr 1e-4、2–3 epoch，接近 SDFT 的 r16 / α32。保存 25 / 50 / 75 / 100% 四个 checkpoint 作为编辑素材。
- **编辑者身份**：R_t = M_t ⊕ A_raw，即装上 raw LoRA 之后的模型。

**诊断工具**：
- `stats(adapter)`：每层的 ‖BA‖、奇异值谱、有效 rank；
- `similarity(A, B)`：与历史 adapter 逐层的余弦相似度；
- `ablate(layer)`：去掉某层 adapter 后 dev loss 的变化（计入预算）；
- `dev_report(candidate)`：目标 dev 分数、保留集分数、K 条错题。

**算子库（Level 1）**：
- `scale(layer, s)`；
- `truncate_rank(layer, k)` / `svd_reweight`；
- `drop(layer / module)`；
- `interpolate(A, B, λ)`；
- `ties` / `dare` / `task_arithmetic`；
- `orthogonalize(A_new, A_old)`；
- `magnitude_prune(density)`。

**Level 2** 允许在受限 API 内写新算子，输出形状和 rank 上限受约束。

## B4. 每一轮的流程

```
第 t 轮（父模型 M_t）
  1. 在 T_t 上训 A_raw（3 个种子），编辑者 R_t = M_t ⊕ A_raw
  2. R_t 调用诊断工具，写 edit()，在沙箱执行
  3. dev 评分（L1）：score = 目标 dev 增益 − λ × 保留集下降；
     返回数字 + K 条错题 + 历史记录；每轮最多 32 次候选评测
  4. 最优候选提交 dev2 验收（L2，最多 3 次）
  5. 通过 → M_{t+1} = Apply(编辑, R_t)；改到历史 adapter 时替换栈里对应的成员
  6. 在固定 LoRA 库上测 E_{t+1}（第 B6 节）
```

## B5. 对照组

**每个父 checkpoint 上的四个分支**：

| 分支 | 内容 |
|---|---|
| Parent | M_t |
| Raw | M_t ⊕ A_raw |
| LLM-edited | 在同一个 Raw 上执行 LLM 编辑 |
| Non-LLM-edited | 在同一个 Raw 上执行最好的非 LLM 编辑 |

报告：
- ΔE_train = E(Raw) − E(Parent)：普通训练本身带来的编辑能力变化；
- ΔE_edit = E(LLM-edited) − E(Raw)：编辑本身带来的变化。

**单轮基线**（共享算子空间，评估次数对齐）：
- 不编辑、最好的单个 checkpoint、标量插值扫描；
- TA / TIES / DARE 网格；
- PARA、TaDA、SLoRA（ACL 2026，持续学习中最直接的对手）；
- 随机搜索、CMA-ES（Nevergrad）；
- LoraHub（按原公式 (ΣwA)(ΣwB) 实现，并加一个 Σw·ΔW 变体）；
- EvoGM（官方代码，同时报告 Single Best 下限）。

**递归对照**：
- 当前编辑者接任 vs 冻结的初始编辑者；
- 连续编辑 vs 每轮从 θ0 重新开始。

**外部强模型当编辑者**：只作上界，不进主循环。

## B6. 评测

| 维度 | 内容 |
|---|---|
| 目标任务 | 每个任务的 test，相对 A_raw 和相对基座的提升 |
| 遗忘 | 序列结束时所有旧任务的平均分；AFR 式遗忘指标，与 SLoRA 对齐 |
| 通用能力 | MMLU-Pro、IFEval（封存的 test 部分） |
| **编辑能力 E_t** | 第 t 代编辑固定 LoRA 库中未见任务的 adapter，同预算下相对 raw 的提升（3 个种子平均） |
| 编辑者行为 | 合法程序比例、验收通过率、每单位评估预算的提升、用了哪些算子 |
| 成本 | GPU 小时、生成 token 数、评测次数 |

## B7. Pilot 顺序

| 阶段 | 内容 | 判断条件 |
|---|---|---|
| **B-P0**（8B，不用 LLM） | T1–T3 训 raw LoRA，暴力扫简单算子（缩放、截断、插值），在可丢弃的 pilot holdout 上看相对 A_raw、TIES、PARA、SLoRA 的提升空间 | 单任务提升 < 约 1 分，但连续学习的遗忘能明显减少：论文主线定为"抗遗忘编辑"。两边都没有空间：止损 |
| **B-P1**（8B） | 单轮：LLM 编辑者 vs 所有基线，3 个种子 | LLM 不比 CMA-ES / EvoGM 差，或成本更低、或在新任务上泛化更好：继续 |
| **B-P2**（8B） | T1–T5 递归 5 轮 + 四分支 + E_t，3 个种子 | 核心结果：ΔE_edit 在未见任务上是否稳定为正 |
| **B-P3** | 换 27B（与 Paper A 共享）；模型自选 raw LoRA 配置；Level 2 算子；ReST^EM 训练编辑技能 | 扩展 |

---

# 第四部分：排期与优先级

1. **先并行做 A-P0 和 B-P0。** 两者都便宜，也最能决定方向：A-P0 是统一量化加一轮 BF16 编辑者，B-P0 是 8B 上的暴力算子扫描。
2. **两篇都有竞速对手。** Paper A 的对手是 CompressionBench，它在结论里把"压缩后的模型能否继续压缩"列为下一步。Paper B 的对手是权重空间编程这一线，三周内已有 5 篇跟进。所以先到 P1 的那篇先推进。
3. **共享工作只做一次**：评测器、agent 外壳、L1/L2/L3 反馈服务、日志。
4. **每个阶段结束都对照判断条件做取舍。** 如果 P1 中 LLM 一直输给非 LLM 搜索，就把论文收窄为对 P2 递归现象的分析，删掉"LLM 编辑更好"的主张。

**需要先确认的事项**：
- 实际可用的 GPU 数和时长；
- Qwen3.8 27B 在你们选的 agent 外壳里能否稳定调用工具，A-P0 第一天就测；
- vLLM 对混合 3/4 bit 的 kernel 支持（需实测）；
- 各数据集的许可和获取方式。

---

# 第五部分：任务选择（2026-11-16 更新，限定在 5 类任务 + 建议补的代码类）

**选任务的标准**：
1. 对这种权重改动是否敏感：越敏感，越能区分好坏决策；
2. 能否自动判分；
3. 题量能否拆出 搜索集 / dev2 / test；
4. 有没有现成数字可以对比。

**各类别敏感度的依据**（详见 `DATASETS_PRIOR_WORK.md` 表 3）：
- 数学：Quant Hurts Reasoning §3.4；When Reasoning Meets Compression Table 1/2；ACBench Table 6；UniComp Table 1。
- 知识：LLM-KICK Table 3；UniComp Table 1；When Reasoning Meets Compression Table 2。
- 指令遵循：UniComp Table 1；LLM-KICK 图 6。
- 工具调用：ACBench Table 8；Which Decisions Table 1/6。
- 多轮 agent：ACBench Table 9；Flat Score Table 1/5。

## 5.1 Paper A 自量化：各任务的角色

层级含义：L1 = 搜索反馈；L2 = 验收门；L3 = 封存的最终评测。

| 类别 | 选用的数据集 | 角色 | 理由 |
|---|---|---|---|
| 数学推理 | **MATH-500**（L1 用 100 题，L3 用全量）、**AIME 2024–25**（只进 L3） | **主指标** | 3-bit 时开始明显掉分，越难越敏感；AIME 每年只有 30 题，拆不出搜索集，只能放最终评测 |
| | GSM8K | L2 + L3 | 32B 大模型 4-bit 下几乎不变，但它在 CompressionBench 的验收门里，可以直接对比 |
| | AMC、OlympiadBench | L3（可选） | 已读的压缩论文没有用过，没有对比数字 |
| 知识问答 | **TriviaQA 闭卷** | L2 + L3，**主指标** | 闭卷生成式问答对压缩最敏感（LLM-KICK：8-bit GPTQ 也掉 8–10%）；CompressionBench 的验收门里也有它 |
| | FreebaseQA | L3 | 最敏感的知识探针 |
| | MMLU-Pro、GPQA-Diamond | L3 | 选择题几乎不受量化影响，只作对照；GPQA-Diamond 只有 198 题 |
| 指令遵循 | IFEval、IFBench | L3 **保底** | 4-bit 量化基本无损（UniComp：AWQ 保留 100%），区分度低，只用来发现意外 |
| 工具调用 | **BFCL**（L1 用 100 题探针，"该调用"和"不该调用"分开；L3 用全量） | **主指标** | 3-bit 下"要不要调用工具"最脆弱，崩到什么程度取决于量化器；而且压缩后的模型还要接着当 agent |
| | T-Eval（含 JSON 子项） | L3 | JSON 输出最脆弱（ACBench） |
| | MetaTool | L3（可选） | Which Decisions 用过 |
| | API-Bank | 不用 | 没有压缩方面的数据，和 BFCL 重叠 |
| 多轮 agent | **τ²-bench**（L2 用小子集看错误率，L3 用全量并报 S(K)） | **主指标** | 最终分数会掩盖损伤；27–31B 的模型上已有对比数据（Flat Score） |
| | Terminal-Bench、SWE-bench | 只进 L3 | 每次评测很重，而且已读文献中没有压缩相关的数据 |
| 代码（建议补） | HumanEval（L2）、LiveCodeBench（L3） | L2 + L3 | 在 CompressionBench 的验收门里；压缩者要写配置和脚本，代码能力掉了，后续轮次就跑不动 |
| **自压缩任务本身** | 在 12 GiB 下压缩 BF16 源 | **编辑者能力 E_t** | 最直接的 agent 能力测试 |

**Paper A 的主打任务**：MATH-500 / AIME（推理）、TriviaQA 闭卷（知识）、BFCL + τ²-bench（agent），再加上自压缩任务本身。这四类都对量化敏感，而 agent 那两项又直接决定"压缩后的模型还能不能继续压缩"，正好串起递归这条主线。

## 5.2 Paper B 自编辑 LoRA：各任务的角色

Paper B 需要的是**能训 LoRA 的任务**：有训练集、LoRA 训练后能看到提升、能测遗忘。所以角色分配和 Paper A 不同。

| 类别 | 训练任务（按序列学） | 保留集 / 最终评测 | 编辑能力测试 E_t（从不进训练序列） | 理由 |
|---|---|---|---|---|
| 知识问答 | **T1：SQuAD 知识注入**（SEAL 设置） | MMLU-Pro、TriviaQA 闭卷、GPQA-Diamond 作保留集 | TriviaQA 闭卷 LoRA（用它的训练集训，在 dev 上测） | SQuAD 能直接对比 SEAL 47.0 和 Search 45.1 / 50.3；闭卷知识靠 LoRA 很难注入，所以用作保留集 |
| 数学推理 | **T2：GSM8K + MATH 训练集 → MATH-500、GSM8K test** | AIME、AMC、OlympiadBench 作数学 LoRA 的分布外测试（只进 L3） | — | 有训练集、确定性判分；Self-Data Distillation、SDFT 都显示 LoRA 在数学上有明显效果 |
| 工具调用 | **T3：SealTool 训练集（约 13k，TT-SI 用过）/ API-Bank / ToolAlpaca → 各自的 test** | BFCL 作分布外泛化（L3） | — | 有训练集，TT-SI 在 Qwen2.5 上有现成数字（平均 54.66 → 60.14）；BFCL 用来测泛化 |
| 指令遵循 | 不作训练任务 | **IFEval、IFBench 作保留集**（每轮都测） | 指令遵循 LoRA（用公开的 IF 训练数据，具体数据集**需确认**，在 IFEval 上测） | IFEval / IFBench 没有训练集；微调数学或工具时会不会伤指令遵循，正好是要测的遗忘 |
| 多轮 agent | 不作训练任务（轨迹数据贵，τ²-bench 每个领域只有 114 题，拆不开） | **Terminal-Bench、SWE-bench、τ²-bench 只进 L3** | — | MASS 显示 27B 自训 LoRA 两轮后 Terminal-Bench 从 37.5 降到 35.2，适合测"自我改进伤通用 agent 能力"，但评测重 |
| 代码（可选） | T4：MBPP 训练集 → HumanEval+、MBPP+ | LiveCodeBench（L3） | — | 执行判分 |

**Paper B 的主打设定**：训练序列 SQuAD → 数学 → 工具调用；保留集 IFEval、MMLU-Pro、TriviaQA 闭卷；编辑能力测试用不在序列里的指令遵循 LoRA 和闭卷问答 LoRA；Terminal-Bench / SWE-bench 只在最后测一次。

**两篇的共用部分**：MATH-500、TriviaQA 闭卷、IFEval、BFCL、τ²-bench 的评测脚本完全一样，只是角色不同。

---

# 第六部分：我们的故事怎么讲

本节借鉴的写法来自 `research_wiki/good_paper/` 里 11 篇顶会论文的 STORY.md，括号里注明出处。这里写的是**待验证的叙事框架**：标题和主张最终都要以 pilot 结果为准。

## 6.0 两篇共用的叙事核心

这 11 篇顶会论文（以及查新读过的其余论文）都没有做到"被修改后的模型当下一轮修改者，并测量它的修改能力随代数怎么变"。所以两篇的主线都是：

> **改进的对象和改进的执行者是同一个模型。改动它，也就改动了下一次改动的质量。**

两篇都借用 Self-Rewarding 的"两条轴同时看"：一条轴是任务能力（或效率），另一条轴是编辑者能力 E_t。核心图都是"第 k 代编辑者的能力随 k 变化"，并且画上冻结编辑者作对照。

**结果不论正负都能讲成故事**（借鉴 R-Zero 专门开一节讨论崩塌）：
- 能力随代数稳定或上升：自我改进可以复利；
- 能力下降：改动侵蚀了继续改动所需的能力。这时用对照实验定位原因，本身就是一个发现。

## 6.1 Paper A：模型自己量化自己

**候选标题**：
- *Who Compresses the Compressor? Recursive Self-Quantization of Language Models*
- *When the Compressor Is the Compressed*

**一句话卖点**：
> 到了最前沿，没有更强的模型可以替你压缩，模型只能自己压缩自己。我们测量这个闭环能走多远，以及它在哪里、为什么停下。

**开场钩子**：先讲愿景，再讲痛点（借鉴 SEAL 和 Self-Rewarding 的开场方式）。
- 第一段：最强的模型最贵，推理成本是部署的主要负担。现有的 agent 压缩都依赖一个更强的外部压缩者（CompressionBench、Self-Pruner、AutoQS）。可是到了最前沿，谁来压缩最强的模型？
- 第二段：CompressionBench 在结论里把"压缩后的模型能否继续完成压缩"列为走向 RSI 的下一步，但没有做。我们来做这一步。

**Gap 怎么陈述**：先用实验把问题展示出来，把 Gap 本身做成第一条贡献（借鉴 Self-MoE）。
- 证据一：PPL 几乎不变时，agent 能力已经受损（LLM-KICK 的反差图；Flat Score：INT4 后总分不变，工具名幻觉率却翻倍）。
- 证据二（我们自己的 A-P0 结果）：27B 统一量化到 12 GiB 后，PPL 接近 BF16，但 BFCL 中"要不要调用工具"的翻转率和 τ²-bench 的错误率上升。
- 由此引出要研究的问题：在自我压缩里，这种看不见的 agent 能力损伤会直接影响下一轮的压缩质量，形成一个此前没人测过的反馈回路。

**核心概念**：
- **自压缩深度**：所有能力相对第 0 代掉点都不超过 5% 的前提下，最多能连续递归几轮。仿照 LLM-KICK 用"≤5% 算 matching"把曲线变成可判定的结论。
- 研究问题放进一个居中的方框（借鉴 EvoGM）：*压缩一个模型，会怎样改变它继续压缩自己的能力？*

**Figure 1 设计**：
- 左：递归闭环示意图（M_t 当压缩者 → 候选 → 固定的外部验收 → M_{t+1} 接任）；
- 右上：效率（bits/param、TPS）和能力随代数 k 变化；
- 右下：压缩者能力 E_t 随 k 变化，冻结的 BF16 压缩者作对照。

**贡献列表**（4 条，覆盖问题、发现、方法、分析四种类型）：
1. **问题与协议**：提出自量化闭环和固定外部评测协议（三层反馈、门槛相对第 0 代），并定义自压缩深度和 E_t。
2. **核心发现**：用 2×2 设计把递归效果拆成三部分：预算变紧、重复量化、压缩者退化。
3. **决策质量**：同预算下比较自压缩者的分配与统一量化、GAMMA、CMA-ES。如果输了，就如实报告，并把这一条改写成"成本 / 泛化"方面的结论。
4. **分析**：模型会自己学会保护哪些模块？对照 When Reasoning Meets Compression 人工找到的"保护最后一层 MLP"，并加随机保护对照（原论文缺这个对照）。

**主张与实验的对应**（每个实验小节用一句结论当标题，借鉴 ACBench 和 R-Zero；节末放 Takeaway 框，借鉴 When Reasoning Meets Compression）：

| 主张 | 实验 / 图表 |
|---|---|
| PPL 掩盖了对 agent 能力的损伤 | A-P0：统一 W8 / W4 / W3 × 5 类任务（反差图） |
| 自压缩者能做出合理决策 | A-P1：单轮对比统一量化 / GAMMA / CMA-ES / 前沿 agent 上界 |
| 递归能走 X 代；压缩者退化是或不是瓶颈 | A-P2：2×2 设计 + 配对比较 + 自压缩深度曲线 |
| 最先坏掉的是哪种能力 | 分能力类别（数学 / 知识 / 工具 / 多轮）逐代追踪 |
| 模型学会保护哪些模块 | 比特分配热力图 + 随机保护对照 |

**结果分支**：

| 实际结果 | 论文的落点 |
|---|---|
| 递归稳定 | "模型可以在没有监督者的情况下让自己更便宜，自压缩深度为 X" |
| 递归崩塌 | "压缩侵蚀了压缩所需的能力，最先坏的是'要不要调用工具'"，加上 2×2 的机制归因 |
| 自压缩者输给 GAMMA | 删掉"更优分配器"这一条，主线保留在递归发现上 |

**局限与未来工作**（借鉴 SEAL：局限顺势成为下一篇的开题）：激活量化、MoE、代码级压缩；以及"把压缩后的恢复交给模型自己做"，这正好接到 Paper B。

**要避免的写法**：
- 声称"首次 agent 压缩"或"首次接任"，这两点已被 CompressionBench、MASS 覆盖；
- 只报 PPL 或总分（LLM-KICK、Flat Score 的教训）；
- 只有单种子（R-Zero、ACBench 被指出的问题）。

## 6.2 Paper B：模型写代码编辑自己的 LoRA

**候选标题**：
- *Models that Operate on Themselves: Recursive Self-Editing of LoRA Adapters*
- *Edit Yourself: LLMs Programming Their Own Adapters*

**一句话卖点**：
> 模型不只给自己准备训练数据，还在训练之后像外科手术一样，用代码直接改自己的 adapter。改过之后，它是否成为更好的"外科医生"？

**开场钩子**：
- 第一段：SEAL 证明了模型可以写"怎么更新自己"的指令，但更新后的模型用完就丢；权重空间编程（WSP）证明 LLM 能读写权重，但执行者是外部前沿模型，被改的是别的模型。
- 第二段：真正的自我改进，要求被改进的模型成为下一个改进者，并且直接作用于自己的权重。

**Gap 怎么陈述**：先用实验展示问题（借鉴 Self-MoE）。
1. 自编辑类方法（SEAL、Search over Self-Edit、MetaRSI）都在**训练之前**选数据和配置，模型从不碰训练好的权重；而且 Search 显示，模型自选的配置还输给人工配置（45.1 vs 50.3）。
2. 训练后的权重编辑（TaDA、PARA、SLoRA、WSP）要么是固定规则，要么由外部前沿模型执行。
3. 没有人测量"编辑过自己之后，是否更会编辑"。
4. 我们自己的 B-P0 证据：连续训练 LoRA 后保留集（IFEval、MMLU-Pro）下降，固定规则（TIES、SLoRA）只能修复一部分。

**核心洞见**：
- **把编辑 adapter 当作代码生成任务**。这仿照 Self-Rewarding "把奖励建模当成指令遵循任务"的思路，让模型在自己最擅长的形式里修改权重。
- 用 Text-to-LoRA 的负面结果作动机：相似任务的 LoRA 在权重空间里并不相邻，所以直接生成权重数值难以泛化，结构化的代码编辑更合适。
- 新名词（借鉴 SEAL 造 self-edit）：例如 **self-surgery** 或 **adapter self-programming**，用来统一缩放、截断、合并、正交化这些动作。

**Figure 1 设计**：
- 左：闭环示意图（训练 raw LoRA → 装上后的模型写编辑程序 → 外部验收 → 接任）；
- 中：一段模型真实写出的编辑程序示例，例如"第 20–24 层缩放 0.6，对旧数学 adapter 正交化"；
- 右：上面是"学新任务 vs 不忘旧任务"的取舍曲线（对比 TIES / SLoRA / CMA-ES），下面是 E_t 随代数的变化，拆成 ΔE_train 和 ΔE_edit 两部分。

**先发制人的框架段**（借鉴 FT-Dojo）：明确说贡献**不在**新的 LoRA 算子（算子来自 TaDA、PARA、SLoRA、TIES），而在"模型自己选择、组合算子，并递归编辑"的过程和由此得到的发现。另配一段"受控协议"：所有条件固定，只改变谁来当编辑者。

**贡献列表**：
1. **框架**：训练后由模型自己编程编辑 adapter，并递归接任。
2. **公平比较**：同预算下与人工规则、解析截断、非 LLM 搜索（LoraHub / CMA-ES / EvoGM）对比，并把关键机制移植到基线上做公平性检验（借鉴 EvoGM）。
3. **编辑能力迁移**：四分支协议（Parent / Raw / LLM-edited / Non-LLM-edited），测 ΔE_edit 在未见任务上是否为正。
4. **行为分析**：模型学会用哪些算子，失败模式是什么。FT-Dojo 给失败模式起名为 "shotgun debugging"，可以学这种命名方式。

**主张与实验的对应**：

| 主张 | 实验 / 图表 |
|---|---|
| 连续自训 LoRA 会遗忘，固定规则修不全 | B-P0：SQuAD → 数学 → 工具序列，保留集曲线 |
| LLM 编辑在同预算下改善取舍 | B-P1：单轮对比全部基线（梯子式排列，最上面放 Oracle，借鉴 SEAL） |
| 编辑本身让后继者更会编辑 | B-P2：四分支 + E_t（指令遵循 LoRA、闭卷问答 LoRA） |
| 接任有贡献 | 当前编辑者 vs 冻结编辑者 |
| 模型偏好的编辑方式 | 算子使用统计 + 逐层改动热力图 |

**结果分支**：

| 实际结果 | 论文的落点 |
|---|---|
| ΔE_edit > 0 | "自我编辑可以复利" |
| ΔE_edit ≈ 0 | "任务上的提升不会迁移成编辑能力"，接着用 ReST^EM 显式训练编辑技能，作为第二个贡献 |
| ΔE_edit < 0 | "任务专精与编辑能力冲突"，用配方 / 数据交换实验定位机制 |
| LLM 输给 CMA-ES | 收窄为"递归编辑中的能力迁移"分析论文 |

**要避免的写法**：
- 只做一轮却自称"自我改进范式"（Self-MoE 被指出的问题）；
- 叙事拔高。EvoGM 自称"搜索策略与模型能力共进化"，但生成器每轮重置，我们必须有真实的跨轮测量；
- 不写正式的局限章节（Transformer² 的问题）。

## 6.3 两篇的写作模板（汇总）

| 写法 | 来源 | 用在哪 |
|---|---|---|
| 反差图：代理指标不变，配具体的失败例子 | LLM-KICK | Paper A Figure 2 |
| 可判定的阈值定义（≤5% 算 matching） | LLM-KICK | 自压缩深度 |
| 两条能力轴同时提升 | Self-Rewarding | 两篇的核心图 |
| 先用实验展示问题，Gap 即第一条贡献 | Self-MoE | 两篇的第 3 节 |
| 造一个新名词，加一个类比 | SEAL | self-surgery、自压缩深度 |
| 研究问题方框 | EvoGM | 两篇的引言末尾 |
| 每个实验小节用一句结论当标题，节末放 Takeaway 框 | ACBench、R-Zero、When Reasoning Meets Compression | 两篇的实验部分 |
| 基线梯子：不适配 → 规则 → 搜索 → 本方法 → Oracle | SEAL | 两篇的主表 |
| 框架段提前化解"只是已有组件"的质疑，配受控协议 | FT-Dojo | Paper B |
| 正面讨论崩塌 | R-Zero | 两篇都留一节 |
| 局限顺势成为下一篇的开题 | SEAL、LLM-KICK | Paper A 的结尾接到 Paper B |
