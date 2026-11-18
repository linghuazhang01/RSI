# Novelty Check Report：RSI × Model Editing

日期：2026-11-13　｜　工作目录：`novelty/rsi-model-edit/`　｜　逐篇笔记：`papers/<id>.notes.md`

## Proposed Method
把递归自我改进（RSI）和模型编辑结合起来，让 LLM 直接改**自己的权重**：
- (a) 自己决定并执行对自己的量化或剪枝，提高推理效率；
- (b) 自己给自己写下游任务的 LoRA，提高任务性能。

目前还处在方向阶段，没有固定方法。

## Core Claims（用于检索与比对）
- **C1 自主自压缩**：模型自己决定并执行对**自身**权重的压缩，以推理效率为目标，同时保住能力。
- **C2 自产任务 LoRA**：模型通过生成数据、超参或训练代码，或者直接生成权重，为下游任务产出 LoRA，并把它用在自己身上。
- **C3 递归接棒**：被编辑后的模型成为下一轮的编辑者，编辑能力和被编辑的模型多轮共同变化。
- **C4 权重级动作空间**：自我改进作用在权重上（压缩、adapter、merging），而不是 prompt、harness 或代码。

## Reading Coverage
- **全文精读，19 篇**：SEAL 2506.10943、Self-Developing 2410.15639、Self-Pruner 2502.14413、Adaptive Pruning by Agents 2601.09694、ProfilingAgent 2509.05584、SIA 2605.27276、PostTrainBench 2603.08640、MASS 2610.12176、Text-to-LoRA 2506.06105、Generative Adapter 2411.05877、AIDE² 2609.26457、Self-Specialization 2310.00160、Self-MoE 2406.12034、Transformer² 2501.06252、CompressionBench 2611.06858、Self-calibration 2410.17170、Search over Self-Edit Strategies 2601.14532、WeightLoop 2611.07319、Anosognosia in LLMs 2610.06174。
- **仅读摘要，只作线索、不影响结论**：
  - 与"联合优化权重和 harness"相邻、但更新配方固定：WHALE 2609.00196、Learning Fast and Slow 2605.12484、JoMoHa 2611.11499。
  - 具身智能场景，偏离主题：AdaRSI 2610.20474。
  - hypernetwork 背景：Drag-and-Drop LLMs 2506.16406。
  - 背景：Frontis-MA1 2607.28568、SDFT 压缩后恢复 2604.15794、Pruner-Zero 2406.02924、EvoPress 2410.14649。
  - 更早的机制，作背景引用：Irie et al. 2022 的 SRWM。
- 所有 arXiv ID 已经用 `verify_papers.py` 和 arXiv API 核实存在。全文精读的论文，每个 subagent 至少抽查了 2 个关键数字，都和原文一致。

## Per-Paper Analysis（只列影响结论的论文；其余见笔记）

**SEAL（2506.10943，NeurIPS 2025）**
- 方法：同一个模型先生成 self-edit，形式是合成训练数据（"implications"）或在 ARC 上选择数据增强和超参；再经过固定的 LoRA/全参 SFT 更新自己；外层用 ReST^EM 强化"写出好 self-edit"的能力，奖励是更新后的下游表现。
- 证据：
  - ARC 上 SEAL 72.5%，无 RL 时 20%，ICL 0%（Table 1，p7）。
  - SQuAD 单段落知识注入 47.0，GPT-4.1 合成数据 46.3（Table 2，p8）。
- 问题：
  - 仅换一个 rewrite prompt、不做 RL 就有 49.4，已经超过 RL 后的 47.0（Table 10，p24）。
  - ARC 只有 8 个评估任务，而且是预先挑选过的。
  - 内层 LoRA 超参由作者调好。
  - 被适配后的 θ′ 只用来算奖励就丢弃。连续编辑实验只测遗忘（§5，p8–9）。
- 与 claim 的关系：
  - C2：**数据/超参路线被实质覆盖。**
  - C3：部分覆盖。编辑策略在多轮中提升，但"编辑过的模型编辑得更好"没有验证。
  - C1：完全没涉及。

**Search over Self-Edit Strategies（2601.14532）**
- 方法：在 SEAL 上进一步放开，让 Qwen3-8B 自己写 self-edit 模板（数据指令加 LoRA 超参），ReST^EM 跑 5 轮。
- 证据：模型自写模板的峰值是 45.1%，**没能超过**人工 Rewrite 模板的 50.3%（Table 2，p6）。模板出现明显的模式坍缩，相似度从 0.81 升到 0.98（Fig 4/5，p8）。
- 局限：作者自己承认是 episodic 设置，用过的 adapter 就丢弃（p13）。
- 与 claim 的关系：C2 更拥挤了。这篇同时是一个负面信号：**单纯给模型更多"自己决定训练方式"的自由，未必有收益。**

**Self-Pruner（2502.14413）**
- 方法：GPT-4o 跑整个进化搜索，输出 LLaMA/Vicuna 的逐层剪枝率，再由 Wanda-sp 执行。标题里的 "Self" 指**"由一个 LLM 来剪 LLM"**，做决策的 GPT-4o 和被剪的 LLaMA 不是同一个模型（p3、p6，Alg.1）。
- 证据：vLLM 实测 LLaMA-2-7B 剪 50% 后加速 1.82×（Table 7，p9）。
- 问题：搜索时的 fitness 和最终报告的 PPL 都用 WikiText-2；没有和非 LLM 的进化搜索做对照。
- 与 claim 的关系：C1 中"由 agent 做压缩决策"这部分已有；"压缩自己"没有覆盖。

**Adaptive Pruning by Agents（2601.09694）/ ProfilingAgent（2509.05584）**
- 方法：前者由 Gemini 逐轮决定 Qwen3 各层的稀疏率；后者由 GPT-4o 为视觉模型选择剪枝方式和量化 dtype。
- 与 claim 的关系：和 Self-Pruner 同属"外部 LLM 压缩另一个模型"。前者没有测推理延迟；后者的 1.66× 主要来自运行时差异（【推断】，见笔记）。

**CompressionBench（2611.06858，CMU，2026-11-04/09）**
- 方法：6 个 coding agent（Claude Code 底座）在 8 小时内自主选择压缩算法、写推理代码，压缩 27B–122B 的 checkpoint。
- 证据：
  - 36 个提交中 29 个通过验证（Table 1，p6）。
  - 每个任务取最优产物，准确率 77.33%，BF16 为 79.56%。
  - TPS 只报告、不计分，多数产物解码比 BF16 慢，例如 Qwen3.8-27B 上是 16.7 对 61.4 TPS（Table 1，p6）。
- 关键：结论（p12）**明确写到**，走向 RSI 需要检验压缩后的模型是否还能完成后续的压缩任务。也就是说，作者公开把"自压缩加接棒"列为下一步，但本文没有做。
- 与 claim 的关系：C1 中"agent 自主设计并执行压缩"已经被一篇很强的工作覆盖；"压缩的是 agent 自己"和 C3 没有覆盖。这是一场**竞速**。

**MASS（2610.12176）**
- 方法：同一个 Qwen3.6-27B 同时担任执行者、workflow 优化者和评估者。用自生成并经自评筛选的轨迹做 LoRA SFT，合并后的模型在下一轮接任全部角色，共 2 轮（Alg.1，p4）。
- 问题：
  - 公开基准上的提升主要来自完成率提高。
  - Terminal-Bench 和 SWE-bench 随轮次下降（Table 7，p20）。
  - 模型能选的动作只在 prompt 层，权重只是通过固定的 SFT 被动更新（p15）。
- 与 claim 的关系：C3 的"checkpoint 接棒"结构已有，但递归的是 prompt/workflow 层面的编辑能力。

**Self-Developing（2410.15639）**
- 方法：factory LLM 写 Python 模型合并算法，作用于**固定的** M0；factory 用 DPO 更新，共 3 轮。合并后的模型不会成为下一轮的 factory（p4）。
- 证据：GSM8K 76.1，Task Arithmetic 71.9（Table 1，p6）。
- 与 claim 的关系：C4（写代码操作权重）已有，C3 只部分覆盖。

**Self-Specialization / Self-MoE（2310.00160 / 2406.12034）**
- 方法：模型从人工种子出发自生成指令数据，训练自己的 (Q)LoRA 专家；Self-MoE 再加一个路由器把多个专家组合起来。
- 证据：Self-Specialization 迭代后收益很快饱和（Table 4，p7）。
- 与 claim 的关系：C2 中"自产数据训自己的 LoRA"更早就有。

**Text-to-LoRA / Generative Adapter（2506.06105 / 2411.05877）**
- 方法：单独训练的 hypernetwork 直接生成 LoRA。Generative Adapter 用目标模型自己的隐藏状态来生成，作者自称 self-programming（p10）。
- 与 claim 的关系：C2 中"直接生成权重"这条路已有很强的先例。

**Self-calibration（2410.17170）/ Anosognosia（2610.06174）**
- Self-calibration：被压缩的模型自己生成校准数据，但算法、比特数和超参全部由人固定。
- Anosognosia：模型**无法**从自己的输出中察觉自己被量化了；不过内部激活里有量化指纹，并排比较多个输出时也能认出 4-bit（Fig 5c，p6）。
- 对 (a) 的意义：这是一个前提性的负面结果。"模型能凭内省知道自己哪里冗余"这个假设不能默认成立，必须用实验检验，这本身也可以是一个发现。

## Core Claims
1. **C1 自主自压缩**
   - 最接近：CompressionBench、Self-Pruner、Adaptive Pruning by Agents。
   - 剩余差异：它们都是**外部固定 agent 压缩另一个 checkpoint**。还没有人让编辑者压缩自己，再观察"压缩如何同时改变被压缩对象和未来的编辑者"。
2. **C2 自产任务 LoRA**
   - 最接近：SEAL、Search over Self-Edit Strategies、Self-Specialization、Generative Adapter。
   - 剩余差异：只剩"写训练代码""持久化的 adapter 库""编辑能力跨轮变化"这类窄点，**差异很薄**。
3. **C3 递归接棒**
   - 最接近：MASS（checkpoint 接棒，但在 prompt 层）、AIDE²（代码层，ignition test 结论不确定）、SEAL/Self-Developing（编辑策略提升，但被编辑的对象被丢弃或固定）。
   - 剩余差异：没有人在"权重编辑会改变编辑者本身"的条件下检验接棒。
4. **C4 权重级动作**
   - Self-Developing、SIA、WeightLoop、CompressionBench 都已经有，**不能单独主张为新颖点**。

## Closest Prior Work
| Paper | Year | Venue | Read | Overlap | Key Difference |
|---|---|---|---|---|---|
| CompressionBench (2611.06858) | 2026 | arXiv | FULL | agent 自主设计并执行 LLM 压缩，有 RSI 动机 | agent ≠ 被压缩模型；无接棒；效率不计分；作者把自压缩列为下一步 |
| SEAL (2506.10943) | 2025 | NeurIPS | FULL | 同一模型自生成数据/超参 → LoRA 更新自己 → RL | 无压缩；被适配模型被丢弃；编辑能力未跨轮验证 |
| Search over Self-Edit Strategies (2601.14532) | 2026 | arXiv | FULL | 模型自写 LoRA 模板和超参 | 未超过人工模板；episodic |
| Self-Pruner (2502.14413) | 2025 | arXiv | FULL | LLM 决定逐层剪枝率，实测加速 | GPT-4o 剪 LLaMA，不是自己 |
| Adaptive Pruning by Agents (2601.09694) | 2026 | arXiv | FULL | agent 迭代决定稀疏率 | Gemini 剪 Qwen；无延迟测量 |
| MASS (2610.12176) | 2026 | arXiv | FULL | 更新后的同一 checkpoint 接任优化者和评估者 | 动作在 prompt 层；权重只被动 SFT |
| Self-Developing (2410.15639) | 2024/25 | arXiv | FULL | LLM 写操作权重的合并代码，多轮 | 作用于固定 M0；合并结果不接棒 |
| Self-Specialization (2310.00160) | 2024 | Findings ACL | FULL | 自产数据 → 自己的 QLoRA，可迭代 | 人工固定配置；迭代饱和 |
| Generative Adapter (2411.05877) | 2024 | arXiv | FULL | 由目标模型隐藏状态生成低秩增量 | 生成器固定，无自主决策 |
| ProfilingAgent (2509.05584) | 2025 | arXiv | FULL | LLM 选择剪枝和量化策略，测延迟 | 被压缩的是视觉模型 |

## Overall Novelty Assessment
- **Score：6/10**（Codex gpt-6-astra xhigh 独立审阅给 6/10，它的 fresh-agent 复核给 5/10，三方都判 PROCEED）
- **Recommendation：PROCEED**，两个子方向分开看：
  - **(a) self-compression：PROCEED，优先推进。** 已全文读过的 19 篇中，没有一篇让同一个 checkpoint 压缩自己、再由压缩后的模型接任下一轮。但 CompressionBench 已经公开把这一步列为 future work，**这是竞速，需要尽快做 pilot**。
  - **(b) self-LoRA：PROCEED WITH CAUTION。** 差异薄的具体原因是：数据/超参路线已被 SEAL 和 Search over Self-Edit Strategies 覆盖，直接生成权重的路线又和 Generative Adapter、Text-to-LoRA 很接近。要让它站得住，必须有新的机制或发现，例如"自产 adapter 是否让模型更会产下一个 adapter"，或者把 (b) 作为 (a) 的恢复算子（压缩后自写 LoRA 来恢复能力）。仅把 SEAL 式流程改称 RSI，不构成贡献。
- **Key differentiator**：在 (a) 中，压缩同时改变被优化对象和未来的优化者——推理更便宜了，但编辑判断可能变差。这个耦合是 agent 压缩别人时根本不存在的问题。
- **Risk**：审稿人会用 CompressionBench 和 Self-Pruner 攻击"自主压缩"，用 SEAL 攻击"自产 LoRA"，用 MASS 攻击"递归接棒"，并追问"self 有什么用？为什么不直接用 GPT 去压缩它？"。Anosognosia 的结果还会让人质疑"模型凭什么了解自己的冗余"。

## Suggested Positioning
> 研究由当前 LLM checkpoint 自主选择并执行**自身**权重压缩、再由压缩后的 checkpoint 接任下一轮编辑者的闭环，在匹配总预算下检验真实推理效率、任务能力与后续编辑能力的跨轮权衡。

不要声称"首次 RSI""首次自生成 LoRA""首次权重级动作"。

**最能区分新意的 pilot**（来自 Codex 建议，我同意）：
1. **接棒对照**：递归接棒的编辑者 vs 同一初始 checkpoint 的冻结编辑者 vs 外部强模型（GPT/Claude）编辑者，三者用相同工具、相同预算。这一组直接回答"self 有什么用"。
2. **单独测编辑能力**：让不同轮次的编辑者去压缩同一组留出的目标 checkpoint，把"任务分数变高"和"编辑能力变强"分开。
3. **测真实成本**：报 latency/throughput（CompressionBench 的产物多数反而变慢）、能力保持，以及搜索和恢复训练的成本。
4. 即便结论是"递归接棒存在稳定的退化边界或坍缩点"，也是一个有价值的发现，不必预设每轮都在进步。

## 补充：用户澄清为"严格自压缩"（2026-11-13）
用户明确：由模型自己压缩自己，不借助更大的模型。按这个口径重新对照 19 篇全文：
- **没有一篇做过。** 决策者都是外部的强模型：Self-Pruner 用 GPT-4o，Adaptive Pruning 用 Gemini，CompressionBench 用 Opus/Sonnet 5.5 等前沿 agent。被压缩的模型本身唯一参与过的形式是 Self-calibration，它只提供校准数据，算法、比特数和超参都由人固定。
- **新颖性比原口径更强，但举证责任变了。** 动机不能再写成"比大模型更好"，而应写成：在前沿位置不存在更强的监督者，模型能否不依赖更强模型来改进自己的效率，这正是 RSI 的核心问题；端侧和隐私场景是次要动机。外部强模型编辑者仍应作为**上界参照**放进实验，审稿人一定会要。
- **主要风险**：
  1. **中小模型做 agent 的能力。** CompressionBench 用的是前沿 agent，36 个提交中仍有 7 个没通过验证（p6）。一个 7–30B 模型压缩自己，很可能在写代码和执行层面先失败。
  2. **模型察觉不到自己的退化**（Anosognosia）。评估信号必须来自外部工具，例如 PPL 和任务测试，不能靠模型自评。
  3. **越压越笨导致坍缩。** 压缩后的模型做下一轮决策更差，这既是风险，也是最可能的核心发现。
- **可行性建议**：
  - 动作空间先用"配置级"：模型通过工具选择每层的比特数、稀疏率和方法，由现成的 GPTQ/AWQ/Wanda 执行。等跑通后再升级到"代码级"，让模型自己写压缩算法。
  - 基线：均匀压缩、OWL、同预算的非 LLM 搜索（EvoPress）、冻结的初始自我、外部强模型（上界）。

## Review trace
`.aris/traces/novelty-check/2026-11-13_run01/`：
- 001：MCP 调用超时，未降级。
- 002：用 codex exec 以同样配置 gpt-6-astra / xhigh 重跑成功，thread 01a1d3b2-…。

Codex 提出的新线索已经全文读过 5 篇（见上），**都不改变结论**；其余未读的线索已在 Reading Coverage 中列为摘要级。
