# good_paper：值得学习"怎么讲故事"的顶会论文

日期：2026-11-16。这些都是查新时**全文精读过**的论文。venue 依据是 PDF 首页的会议标注，或 arXiv 元信息中的 PMLR 卷号 / 会议标注。

每个文件夹包含 4 个文件：
- `paper.pdf`：原文；
- `paper.pages.md`：逐页抽取的文本；
- `notes.md`：内容笔记（方法、数字、问题）；
- `STORY.md`：**叙事拆解**，包括开场钩子、Gap、核心洞见、Figure 1、贡献、主张与实验的对应、局限写法，以及对我们两篇的启发。

## self_lora/：模型自我适配、自我改进、LoRA 生成与合并

| 文件夹 | 论文 | Venue | 一句话卖点 | 最值得学的讲法 |
|---|---|---|---|---|
| 2506.10943_SEAL_NeurIPS2025 | Self-Adapting Language Models | NeurIPS 2025 | LLM 不必"强大但静态"，模型自己写 self-edit 并用 RL 学会写得更好 | 造一个新名词（self-edit）统一动作空间；用"学生改写笔记"作类比；基线从弱到强排成梯子，最上面放 Oracle |
| 2506.06105_TextToLoRA_ICML2025 | Text-to-LoRA | ICML 2025 | 一句任务描述，一次前向就生成 LoRA | 两个编号的研究问题配一张 Figure 1；把负面结果（重建训练不泛化）变成机制分析 |
| 2501.06252_TransformerSquared_ICLR2025 | Transformer² | ICLR 2025 | 只调奇异值的专家向量，推理时两遍自适应 | 三个挑战和三个方法性质一一对应；用"打乱"对照把现象变成机制证据 |
| 2406.12034_SelfMoE_ICLR2025 | Self-MoE | ICLR 2025 | 单体 LLM 自建 LoRA 专家加路由，不需要教师模型 | 先用实验展示问题，把 Gap 本身做成第一条贡献；勾叉对比表定位自己 |
| 2401.10020_SelfRewarding_ICML2024 | Self-Rewarding Language Models | ICML 2024 | 同一模型打分并训练自己，做题和打分两条能力一起变强 | **"两条轴同时提升"作为核心论点**，和我们的"任务能力 + 编辑者能力"同构 |
| 2508.05004_RZero_ICLR2026 | R-Zero | ICLR 2026 | 出题者和解题者从零数据共进化 | 正面讨论递归崩塌；用固定的早期模型测各轮产物 |
| 2605.29295_EvoGM_ICML2026 | EvoGM | ICML 2026 | 合并系数的进化搜索应该是一个学出来的生成过程 | 居中的"研究问题方框"；把关键机制移植到基线上做公平性检验 |
| 2603.01712_FTDojo_ICML2026 | FT-Dojo | ICML 2026 | 端到端 LLM 微调应被定义为交互式 agent 任务 | 用一段框架说明先化解"这些都是已有组件"的质疑；受控协议只改变一个变量 |

## self_quant/：压缩后的能力评测

| 文件夹 | 论文 | Venue | 一句话卖点 | 最值得学的讲法 |
|---|---|---|---|---|
| 2310.01382_LLMKICK_ICLR2024 | Compressing LLMs: The Truth is Rarely Pure and Never Simple | ICLR 2024 | PPL 不变不代表能力没变 | **反差图**（平坦的 PPL 曲线配具体的失败例子）；用"掉点 ≤5% 算 matching"把曲线变成可判定的结论 |
| 2505.19433_ACBench_ICML2025 | Can Compressed LLMs Truly Act? | ICML 2025 | 压缩评测要看 agent 能力 | 按能力分类组织全文；每个实验小节用一句结论当小标题 |
| 2504.02010_ReasoningMeetsCompression_ICLR2026 | When Reasoning Meets Compression | ICLR 2026 | 压缩伤推理模型伤在哪些权重上，可以定位 | "定位 → 验证 → 保护"三步；每节末尾放编号的 Takeaway 框 |

## 共同点
这 11 篇**没有一篇**做出"被修改后的模型当下一轮修改者，并测量修改能力随代数的变化"。最接近的是 Self-Rewarding（打分能力随代数提升）和 R-Zero（多轮，但出题者和解题者是两个 checkpoint）。这正是我们两篇最该突出的差异，见 `novelty/EXPERIMENT_PLAN.md` 第六部分。

## 未纳入的已读论文
以下论文已读，但不是 NeurIPS / ICML / ICLR，或 venue 无法确认，所以没有放进来：
- LoraHub（COLM 2024）、Quantization Hurts Reasoning?（COLM 2025）、Sakana 进化合并（Nature Machine Intelligence）、SLoRA（ACL 2026）；
- 多数 2026 年的 arXiv 预印本，如 CompressionBench、MetaRSI、MASS。

需要的话，可以在 `novelty/*/papers/` 里找到它们的笔记。
