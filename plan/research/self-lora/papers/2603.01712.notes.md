# FT-Dojo（arXiv 2603.01712）阅读笔记

## 元信息
- 标题：FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents
- 作者：Qizheng Li, Yifei Zhang, Xiao Yang, Xu Yang, Zhuo Wang, Weiqing Liu, Jiang Bian（Microsoft Research Asia 等）
- 日期/版本：v2，2026-05-20（cs.AI）；标注为 ICML 2026（PMLR 306）
- 阅读范围：全文 26 页全部读完（正文 p1–10，参考文献 p10–11，附录 A–G p12–26）。
- 抽取文件：`src_2603.01712/2603.01712.pages.md`（1663 行）
- 作者声明了利益冲突：FT-Agent 和 R&D-Agent 代码库由作者所在机构开发（p2）。

## 研究问题
能否把端到端的 LLM 微调（从原始数据中筛选和合成训练数据、配置训练、看反馈后迭代）当作一个交互式 agent 任务来形式化和评测？通用 agent（如 OpenHands）在这类任务上为什么表现差？

## 方法细节
- **FT-Dojo 环境**：13 个任务，覆盖 5 个领域；每个任务在一个 Docker 沙盒里运行。环境提供三个接口：Meta API（任务描述和可用资源）、共享原始数据仓库 D、评测器 F（基于 OpenCompass，返回总分、逐条预测和 loss 曲线）。优化目标见 Eq.1（p2）：在数据策略 θ_d 和训练超参 θ_t 上最大化验证集表现。
- **FT-Agent**：每次迭代分三个阶段（§3.2，p5–6）。
  1. 策略提出：继承当前最优配置，参考兄弟尝试和失败经验，写出统一假设（数据策略 + 训练配置 + 理由），见 Eq.2。
  2. 实现与 fail-fast 校验：依次做静态/schema 检查、小规模 mini-run、运行时异常检测，三步都通过才允许正式训练。
  3. 多层反馈分析：看指标、错误样例、loss 曲线，决定是否把本次结果接受为新的 SOTA 配置。
- **谁做什么（关键）**：
  - 规划、写数据处理代码、定训练配置（是否用 LoRA、学习率、batch size、是否加 CoT 等）：**GPT-5.2**，即 agent 的骨干模型（p6，第 357 行）。消融中还换成 GPT-4o、DeepSeek-V3.2、Qwen3.5-397B-A17B（Table 4，p8）。
  - 生成和清洗训练数据：**GPT-5 和 GPT-4o-mini**（p6，第 359 行）。prompt 模板里明确要求用强模型生成 CoT（附录 G.1）。
  - 评判：多数任务用确定性指标；AIME 先做规则验证，剩下判不了的交给 **GPT-5 judge**（附录 A.4）。
  - 被训练的模型：**Qwen2.5-7B-Instruct**，消融中还有 3B（p6，第 387 行）。
  - 结论：编辑者、数据生成者、评判者都是远大于目标模型的外部 API 模型，**与被训练的 checkpoint 完全不是同一个**。
- **更新方式**：限定为标准 SFT 或 LoRA，每个任务最多 2,000 条训练样本（p6，第 390 行），通过 LlamaFactory 执行。agent 可以自己选 LoRA 或全参数。例如在 TableBench 上它选择了全参数微调（附录 D.4.4，p23–24）；PI4PC 的第 0 轮用的是 LoRA（Table 9）。正文没有报告具体的 rank。
- **多轮**：每个任务在 12 小时、单卡 B200 的预算内迭代多轮，平均 8.77 轮（Table 3）。每轮把“最优配置”作为下一轮的起点，并累积经验记忆。【推断】每轮是从基础模型按新配置重新训练（Train(θ_d(D), θ_t)），而不是在上一轮 checkpoint 上继续训练，论文没有明确写。**被训练后的 Qwen 模型从来不担任 agent 或编辑者。**

## 数据集与评测
- Math：训练用 DeepScaleR（约 4 万题，82% 没有解答）；测试用 AIME 2025，过滤后只剩 30 题，验证集和测试集各 15 题。
- Patent：PANORAMA 的 PAR4PC、NOC4PC、PI4PC，各 100/100。
- Chemistry：训练用 ChemCoTDataset，测试用 ChemCoTBench，包括 Mol Und、Mol Edit、Mol Opt、Reaction。
- Finance：FinanceIQ，100/100，多选题。
- Table QA：训练用 TableInstruct，测试用 TableBench，包括 DA、FC、NR、Vis。
- 划分规则：验证集和测试集大小都取 min(100, ⌊n/2⌋)；agent 可以看验证集反馈，测试集全程不可见。
- 指标：Acc、Macro-F1、MAE、Tanimoto、SR、Valid Ratio、FTS、BLEU、Pass@1 等。论文没有说明 shot 数，【推断】是零样本的 OpenCompass 评测。
- 基线：未微调的基础模型、人工 SFT（资深从业者，只允许两轮，不用 LLM 辅助）、配备同样工具的 OpenHands（GPT-5.2）；补充对比还有 Codex（GPT-5.2）和 Claude Code（Sonnet-4.6-thinking）。

## 关键实验与数字
- **主表**（Table 2，p7，第 424–427 行）：FT-Agent 在 13 个任务中的 10 个上最好。几个例子：
  - AIME：11.11±3.82，其余方法都是 0。
  - Mol Edit：54.44，OpenHands 为 40.00。
  - Vis Pass@1：29.33，OpenHands 为 24.00。
  - PI4PC：49.00。
  - Reaction FTS：30.92，低于 OpenHands 的 35.59。
- **探索效率**（Table 3，p7）：平均轮数 8.77 对 3.69；提升率 24.37% 对 17.31%；迭代成本 $5.73 对 $3.92。有效迭代次数 6.0 对 2.75（Table 8，p20）。
- **五任务对比**（Table 4，p8）：FT-Agent(GPT-5.2) 平均 42.83；Claude Code 39.86；Codex 36.06；OpenHands 34.55；用 LLM 辅助的人工 SFT 37.11（AIME 上它最好，20.00）。换成开源骨干后，Qwen3.5-397B 为 31.97，DeepSeek-V3.2 为 35.83。
- **消融**（Fig.3 p8，Table 10 p24）：样本从 2k 扩到 5k 结果有涨有跌；GPT-4o 骨干在 Finance 上更好，但在过程性任务上更差；3B 目标模型上 Mol Edit 从 14.44 涨到 54.55。
- **案例**（Fig.4 p9，Table 9 p23）：Mol Edit 从 2% 逐步恢复到 56%；PI4PC 迭代 7 轮后净增益为 0，中间一度跌到 8%。Visualization 的第 8 轮把数据过滤到只剩 27 条，Pass@1 从 20% 跌到 4%（Fig.6）。

## 作者自己声明的局限
- agent 缺乏因果诊断能力，表现为“霰弹枪式调试”（PI4PC）；长程规划存在近视（Visualization 的数据坍缩）（§4.5，附录 D）。
- 人工 SFT 基线受“只允许两轮”的约束，不能看作人类专家的上限（附录 B.1.2）。
- 只有 FT-Agent 跑了 3 个种子，其余基线只跑了一次（p6）。
- 骨干模型的质量很关键，必须依赖前沿模型（§4.4 第 2 点）。
- 更大模型族和其他训练范式留作未来工作（§4.4 第 3 点）；开放世界的数据获取和更强的推理机制也是未来方向（§6）。
- Impact statement 提到存在被滥用的风险。

## 方法问题与替代解释
- 【推断】AIME 的验证集和测试集各只有 15 题，11.11% 大约相当于 1.7 题，统计意义很弱。
- 【推断】FT-Agent 的提升有相当一部分可能来自 GPT-5 合成的 CoT，也就是知识蒸馏。AIME 的提升正好出现在“缺 CoT”被补上之后；用 LLM 辅助的人工 SFT 在 AIME 上达到 20.00，比 FT-Agent 还高，也支持这一点。
- 【推断】作者团队开发了 FT-Agent，基线的脚手架也由作者搭建，利益冲突已披露。
- 【推断】验证集只有 100 条或更少，agent 多轮根据验证集反馈做选择，存在对验证集过拟合的风险。好在测试集与验证集互不重叠。

## 与核心主张（B1–B4）的关系
- **B1（同一模型自己写适配方案）**：**不覆盖。** 决定数据策略、LoRA 或全参、超参、训练代码的是 GPT-5.2（或其他大骨干）；生成数据的是 GPT-5 和 GPT-4o-mini。被训练的 Qwen2.5-7B 只是被动的目标。不过，这篇论文覆盖了 B1 中“由 LLM 决定数据 + 超参 + 训练代码 + LoRA 配置”这一**功能形态**，只是执行者是外部大模型。
- **B2（adapter 作用于目标模型并提升下游表现）**：部分覆盖。SFT 或 LoRA 作用于目标模型，在 13 个任务中的 10 个上有提升。但目标模型和编辑者不是同一个。
- **B3（递归：被适配模型当下一轮编辑者）**：**不覆盖。** 迭代的是配置和经验记忆；被训练的模型从来不当 agent，也没有“编辑能力”的共同提升。
- **B4（没有更大的外部模型参与）**：**明确违反。** GPT-5.2、GPT-5、GPT-4o-mini 和 GPT-5 judge 全都参与其中。
- 总结：FT-Dojo/FT-Agent 是“外部前沿 LLM 当 ML 工程师，去微调一个小模型”。它与 Topic B 的交集只在于：由 LLM 自主决定数据、超参和训练代码这一工作流形态，以及 LoRA/SFT 的工具链。
