# Novelty Check Report — Topic A：模型自己量化自己（递归）

日期：2026-11-14
审查：Claude（全文精读）+ Codex gpt-6-astra xhigh（独立复核，session `01a1d7cc-a4a0-73c1-bf12-9e9f661590db`，trace 在 `.aris/traces/novelty-check/2026-11-14_run01/001-*`）
Codex 补充的 2 篇（AutoQS、APQF）已由我们自己全文精读并写了笔记，再纳入判断。

## Proposed Method
一个约 27B 的开源模型不借助任何更大的外部模型，自己决定并执行对**自身权重**的量化：每层比特数、group size、量化器、embedding/head 精度、校准数据来源。候选由固定的外部评测验收，验收内容包括存储、能力门槛、agent 能力门槛和实测速度。验收通过的量化模型成为下一轮的"压缩者"，例如预算从 16 → 12 → 10 GiB。

## Core Claims
- **A1 自主量化自身**：做决策的就是被压缩的那个 checkpoint。
- **A2 递归接任**：压缩后的模型当下一轮压缩者，研究压缩者自身退化如何影响后续压缩，以及循环在哪里停下。
- **A3 效率与能力**：由固定外部评测判断，包括 agent/工具调用能力和实测速度、显存。
- **A4 LLM 推理出的分配 vs 经典分配**：在同预算下不输于甚至优于均匀量化和非 LLM 混合精度分配。

## Reading Coverage
- **全文精读 24 篇**（笔记路径：R = `novelty/rsi-model-edit/papers/`，Q = `novelty/self-quantization/papers/`）
  - **agent 驱动压缩**：CompressionBench 2611.06858、Self-Pruner 2502.14413、Adaptive Pruning by Agents 2601.09694、ProfilingAgent 2509.05584、AutoQS 2611.11975(Q)、APQF 2608.05499(Q)
  - **模型参与自身压缩**：Self-calibration 2410.17170、Anosognosia 2610.06174
  - **递归 / 自改权重**：Self-Developing 2410.15639、WeightLoop 2611.07319、AIDE² 2609.26457、MASS 2610.12176、When AI Reviews Its Own Code 2606.28438
  - **压缩评测**：ACBench 2505.19433、UniComp 2602.09130、LLM-KICK 2310.01382、Quantization Hurts Reasoning? 2504.04823、When Reasoning Meets Compression 2504.02010、Structure of Quantization Damage 2609.01587、Which Decisions Low-Bit Quantization Breaks 2608.06564、Flat Score, Amplified Failures 2607.27275
  - **恢复**：SDFT recovery 2604.15794、Self-Data Distillation 2410.09982
  - **非 LLM 分配基线**：GAMMA 2605.18475(Q)
- **仅读摘要**（背景，不参与判断）：Pruner-Zero 2406.02924、EvoPress 2410.14649、Q-Strata 2608.30564、MixQuant 2607.23047、Safety-Guided Self Compression 2505.00350（这里的 "self compression" 指一种可微量化技术，不是模型自己决策）；Codex 列出的 leads：AMQ（EMNLP 2025）、Compute Where it Counts（ICML 2026）。
- **存在性核验**：`verify_papers.py` 对 22 篇全部 PASS（`verified.json`）；AutoQS 和 APQF 是从 arXiv 下载 PDF 后全文读过的，按规则视为已核验。

## Per-Paper Analysis（最相关的 8 篇详写，其余见笔记）

**1. CompressionBench（2611.06858，CMU，2026-11）——最近邻**
- **方法**：6 个前沿 coding agent（Sonnet/Opus 5.5、GLM 5.3、Kimi K3.1、MiniMax M3.1、DeepSeek V4.1）在 Claude Code 里压缩 Nemotron-3.5 120B、Qwen3.5 122B 和 **Qwen3.8 27B**。条件是一张 H200、8 小时，算法和推理 kernel 都由 agent 自己写。
- **证据**：
  - 搜索时的奖励是 BF16 自己生成的续写上的 PPL 比。
  - 最终验证是 GSM8K / HumanEval / TriviaQA 各 50 题，准确率需达到 BF16 的 0.8 倍（§2.3–2.4，p.4–5）。
  - 27B / 12 GiB 下 6 个 agent 全部通过，Sonnet PPL 3.253、Acc 76.67，BF16 为 3.221 / 74.00；8 GiB 下只有 3 个通过（Table 1，p.6）。
- **问题**：
  - TPS 不计分，多数产物比 BF16 慢（27B / 12 GiB 下 Sonnet 16.7 TPS，BF16 61.4）。
  - 每个 agent × 任务只跑 1 次。
  - 准确率门槛只有 50 题。
- **与 claim 的关系**：A1 中"agent 自主压缩"已有很强先例，但**压缩者与被压缩者不是同一模型**；A2 未做，结论（p.12）把"压缩后的模型能否继续完成压缩任务"列为走向 RSI 的下一步；A3 部分覆盖；A4 未与非 LLM 分配器比较。

**2. AutoQS（2611.11975，2026-11-11）**
- **方法**：外部 API（Qwen-Plus / DeepSeek）给 7B 模型（Qwen2.5-7B、LLaMA-7B 等）搜索逐层 group size。比特固定，量化器固定为 GPTQ。共 100 次迭代、最多 20 次真实量化评估（§3，p.4–5）。
- **证据**：相对均匀 group 的 GPTQ 提升 0.70 个百分点。同预算下 Full 73.85，随机合法变异 73.45，无反馈 72.58（Table 2，p.8）。
- **问题**：
  - 不同种子之间相差 0.94，比增益 0.70 还大（Table 8）。
  - 搜索期分数与最终分数几乎不相关，r = 0.135（附录 C.3）。
  - 不测速度，没有同存储预算的比较。
- **与 claim 的关系**：A1、A2 未覆盖；A4 部分覆盖，有 LLM 与随机搜索的对比。

**3. Self-Pruner（2502.14413）**
- **方法**：GPT-4o 跑进化搜索给 LLaMA 7B–70B 生成逐层剪枝率，由 Wanda-sp 执行。
- **证据**：LLaMA-2-70B 剪 30% 后零样本平均分只掉 0.80；vLLM 实测 1.82×（Table 2、Table 7，p.8–9）。
- **问题**："self"指的是"LLM 这一类模型剪 LLM"；用 WikiText-2 PPL 当 fitness，同时又用它汇报结果；没有非 LLM 进化算法作对照。
- **与 claim 的关系**：A1、A2 未覆盖；A3 部分覆盖。

**4. Adaptive Pruning by Agents（2601.09694）**
- **方法**：Gemini-3-flash 逐轮决定给 Qwen3-4B/8B 的哪些层加多少稀疏，单轮 PPL 涨超过 15% 就回滚。
- **证据**：Qwen3-8B 稀疏约 43% 时 MMLU 56.67，4:8 基线 36.29（Table 2，p.10）。
- **问题**：基线是 N:M magnitude，不公平；没有效率测量。
- **与 claim 的关系**：A1、A2 未覆盖。它的回滚机制可以借鉴。

**5. Self-calibration（2410.17170）**
- **方法**：被压缩的模型（2B–8B）只以 BOS 为条件自己生成校准数据；量化和剪枝方法由人固定。
- **证据**：20 个组合中 17 个不输于其他合成数据；SparseGPT 剪 Llama 3.1 8B，自生成 53.8 vs Cosmopedia 50.9（Table 2，p.5）。
- **问题**：模型只是被动采样，不做任何决策；Mistral 和 Llama 上不如 C4。
- **与 claim 的关系**：它是唯一"模型参与压缩自己"的先例，但不覆盖 A1 中的"自主决策"。可以作为"自生成校准数据"这一动作的来源。

**6. MASS（2610.12176）——递归结构的近邻**
- **方法**：Qwen3.6-27B 生成并自评研究任务轨迹，用 LoRA SFT 合并回自身。下一代同时担任执行者、workflow 优化者和评估者，做了 2 轮（Alg. 1，p.4）。
- **证据**：L1 的 workflow 在固定 L0 执行时胜率 64%（Table 1）。
- **问题**：部署时的 FP8 是人工固定的一步，不是模型的决策；递归改进的是 prompt/workflow 能力，不是压缩能力。
- **与 claim 的关系**：A2 的"更新后模型接任"结构已有先例，**不能说"首次递归接任"**；A1 未覆盖。

**7. GAMMA（2605.18475）——A4 必须正面对比的基线**
- **方法**：先用 QTIP/AQLM 生成 2/3/4-bit 候选，用 Gumbel-Softmax 学习每个模块选几比特，再用 ILP 满足平均比特预算（§3，p.3–5）。
- **证据**：同一 QTIP 后端下，Qwen3-8B 3-bit 均匀量化 76.33，GAMMA 81.16（Table 2，p.9）；A100 上分配耗时约 1.5–2.4 小时（Table 6）。
- **问题**：多处量化后分数超过 FP16；同一个基线在两张表中相差约 7 分；低比特没有更快（Table 5）。
- **与 claim 的关系**：A1–A3 未覆盖。在 A4 上，它说明非 LLM 分配已经很强，LLM 分配必须与它比。

**8. Flat Score, Amplified Failures（2607.27275）——27B 量化后 agent 能力的直接证据**
- **方法**：Gemma-4-31B、Qwen-3.6-27B、Qwen-3.5-27B 等，在 BF16 / FP8 / INT4 AWQ 下跑 τ²-bench。
- **证据**：
  - INT4 下最终得分都没有显著变化（Table 1，p.4）。
  - 工具名幻觉率：Qwen-3.6-27B 0.38% → 0.47%，不显著；Qwen-3.5-27B 0.26% → 1.21%；Gemma-4-31B 19.5% → 38.3%。
  - 把失败预算收紧到 K=2 时，BF16 与 INT4 的差距从 1.3 扩大到 16.7（Fig.3）。
- **问题**：只有一个单元格是明显受损的正例；S(K) 是推算出来的，没有在收紧的预算下真正重跑。
- **与 claim 的关系**：不覆盖任何 claim，但直接说明"27B 量化后还能不能当 agent"必须逐个 checkpoint 实测。

**其余 16 篇**的结论都已写在笔记里。它们都不覆盖 A1 或 A2，主要用于实验设计：
- **压缩评测**：ACBench、UniComp、LLM-KICK、两篇推理量化论文、Damage Structure、Which Decisions。
- **约束**：Anosognosia 说明模型察觉不到自己被量化，所以必须用外部评测。
- **递归风险**：When AI Reviews Its Own Code 说明自评门会退化，必须用外部固定的验证信号。
- **"自改权重"近邻**：Self-Developing、AIDE²、WeightLoop。
- **恢复训练**：两篇恢复类论文。
- **外部 LLM 压缩视觉模型**：ProfilingAgent、APQF。

## Core Claims
1. **A1 自主量化自身**
   - 最近工作：CompressionBench（外部 agent 压缩别的模型）；Self-calibration（同一模型只生成校准数据）。
   - 尚未有人做：同一个 checkpoint 自己决策并执行对自身权重的量化，24 篇中没有一篇做过。
2. **A2 递归接任**
   - 最近工作：MASS（27B 接任，但改进的是 workflow）；CompressionBench p.12（作为 future work 提出）。
   - 尚未有人做："量化改变压缩者的能力，压缩者又决定下一次量化"这一反馈关系，没有实验。
3. **A3 效率与能力**
   - 最近工作：CompressionBench、ACBench、Flat Score。
   - 尚未有人做：评测方法都有先例；把 agent 能力和实测速度作为递归接任的验收条件是新的，但这一点主要是验证协议，不单独构成创新。
4. **A4 LLM 分配 vs 经典分配**
   - 最近工作：AutoQS（外部 LLM 只搜 group size，比随机搜索略好但不稳健）；GAMMA（很强的非 LLM 分配器）。
   - 尚未有人做："被压缩模型自己做的分配"能否在同资源下达到 GAMMA 的水平，这是待验证的假设。

## Closest Prior Work
| Paper | Year | Venue | Read | Overlap | Key Difference |
|---|---|---|---|---|---|
| CompressionBench 2611.06858 | 2026 | arXiv | FULL | agent 自主压缩 Qwen3.8 27B，外部验证 | 压缩者是外部前沿模型；没有接任；p.12 把它列为下一步 |
| AutoQS 2611.11975 | 2026 | arXiv | FULL | LLM 提出量化策略，与随机搜索对比 | 外部 API；只搜 group size；7B；无速度、无递归 |
| Self-Pruner 2502.14413 | 2025 | arXiv | FULL | LLM 决定逐层压缩比例，实测加速 | GPT-4o 剪 LLaMA；剪枝不是量化；无递归 |
| Adaptive Pruning by Agents 2601.09694 | 2026 | arXiv | FULL | LLM agent 逐轮压缩，带回滚 | Gemini 剪 Qwen3；无效率测量 |
| Self-calibration 2410.17170 | 2025 | arXiv | FULL | 同一模型参与自己的压缩 | 只生成校准数据，不做决策 |
| MASS 2610.12176 | 2026 | arXiv | FULL | 27B 更新后接任下一代 | 改进的是 workflow，不是压缩；FP8 由人固定 |
| GAMMA 2605.18475 | 2026 | arXiv | FULL | 混合精度比特分配 | 非 LLM 算法；是 A4 必须对比的基线 |
| Flat Score 2607.27275 | 2026 | arXiv | FULL | 27B INT4 后的 agent 能力 | 评测性质，不做自主压缩 |

## Overall Novelty Assessment
- **Score：7/10**
- **Recommendation：PROCEED。** Claude 与 Codex 独立判断一致，Codex 的独立子审查也是 PROCEED 7/10。
- **Key differentiator**：做压缩决策的就是被压缩的那个 checkpoint，压缩产物又接任下一轮压缩者，研究"压缩如何改变继续压缩的能力"，以及这个循环的停止边界。
- **Risk**：
  1. **与 CompressionBench 竞速。** 原作者已公开把这一步列为下一步，这是竞速关系，不构成否决。
  2. **审稿人最强的质疑会是"这只是 CompressionBench / AutoQS 式压缩搜索加上 MASS 式接任"。** 这个质疑对单轮演示是成立的，所以只做单轮"让 27B 压一下自己"会偏薄。
  3. **A4 可能输给 GAMMA 这类非 LLM 分配器。** 如果输了，就删掉"更优分配器"这一结论；只要 A2 的发现扎实，论文仍然成立。
- **其他需要注意的点**：
  - 不能说"首次 agent 驱动压缩"，也不能说"首次更新后的模型接任"。
  - 16 / 12 / 10 / 8 GiB 对 27B 分别相当于约 5.09 / 3.82 / 3.18 / 2.55 bits/参数（含 scale、embedding 等开销），不能把 16 GiB 理解成 16-bit。
  - 之前方案里"4-bit 安全、3-bit 崩塌"的说法要收紧。崩溃点取决于量化器和模型：3-bit 下"要不要调用工具"被翻转的比例，RTN 47%、GPTQ 21%、GGUF 10%。
  - "推理校准 +9.8"来自 1.5B 模型，不能外推到 27B。

## Suggested Positioning
> 在没有更强外部模型参与决策的闭环中，让同一个 checkpoint 自主量化自身，并由验收通过的量化模型接任下一轮压缩者；通过对齐目标、预算和反馈的对照实验，测量压缩者自身退化对后续压缩质量、成本和停止边界的影响。

**最关键的实验**（Codex 建议，我同意）是用一个 2×2 对照把三种原因分开：预算变紧、权重被重复量化、压缩者退化。
| 压缩者 | 每轮从原始 BF16 权重开始 | 在上一轮量化权重上继续压 |
|---|---|---|
| 冻结的 BF16 压缩者 | 非递归参照 | 测重复量化的损伤 |
| 当前的量化压缩者 | 测压缩者退化的影响 | 完整的递归主实验 |

在此之上，再让 BF16 压缩者和量化压缩者在**同一个目标 checkpoint、同一预算、同样的历史和工具**下各出方案，配对比较。A4 的对照要包括：均匀量化、简单敏感度分配、随机 / 进化搜索、GAMMA；并控制实际字节数、量化后端和评估预算，把 agent 推理和失败执行的成本也算进去。

## 之前类似工作是怎么做的（方法 / 模型 / 数据集 / 评测）
| 工作 | 谁压谁 | 动作空间 | 搜索信号 | 最终评测 | 速度 |
|---|---|---|---|---|---|
| CompressionBench | 前沿 agent → Qwen3.8 27B 等 | 不限（算法 + kernel 代码） | BF16 续写上的 PPL 比 | GSM8K / HumanEval / TriviaQA 各 50 题，≥0.8×BF16 | 测 TPS，不计分 |
| AutoQS | Qwen-Plus → 7B | 每层 group size | ARC-C / PIQA / BoolQ 平均 | 7 项零样本 | 不测 |
| Self-Pruner | GPT-4o → LLaMA 7B–70B | 逐层剪枝率 | WikiText-2 PPL | 7 项常识零样本 | vLLM 实测 |
| Adaptive Pruning | Gemini → Qwen3-4B/8B | 逐轮逐层稀疏 + 停止 | PPL 变化 + 回滚 | MMLU、FreebaseQA、PPL | 不测 |
| GAMMA | 算法 → 8B–32B | 每模块 {2,3,4} bit | 隐藏状态重构 MSE | 6 项零样本、GSM8K、RULER | 测吞吐 |
| Self-calibration | 人 → 2B–8B（数据由模型自生成） | 无 | — | 10 项零样本 | 不测 |
