**结论：PROCEED WITH CAUTION，5/10。可以继续做 pilot，但需要收窄论文主张。**“同一 LLM 自生成数据、选择 LoRA、更新自身并循环改进”已经有很近的完整系统；仍值得研究的是：**任务适配造成的权重改变，是否真的提高了模型设计下一次更新的能力。**

本轮已审阅 dossier 指向的 23 份 FULL 笔记，回查关键方法、实验和附录，并补充全文核查了 **MetaRSI、Self-HPO、HASE**。以下只判断 Topic B，不混入 self-quantization。独立 reviewer 复核后的结论同样为 CAUTION、5/10。

**1. 最重要的新增发现：整体最近邻已不只是 SEAL**

Dossier 漏掉了 **MetaRSI / RSI²（2609.06396v2）**：

- §4.2.3、pp.14–15：目标模型参与数据与 LoRA recipe 决策，包含 modules、layers、rank/scaling 和优化配置。
- §5.1、p.21：报告目标模型承担各模型角色，不使用外部模型数据或判断。
- §5.4、pp.24–25：实际运行五轮，并比较原始与 meta 改写的 improver。
- 明确区别：Eq.19 每轮从固定 \(\theta_0\) 用累计数据重训；没有实验验证持续从当前 \(\theta_t\) 更新。
- 身份仍有歧义：新 checkpoint 接替 rollout，但 §4.5 又规定最外层 meta-agent 的参数固定；没有明确交代所有 recipe-author 角色同步换成新 checkpoint。
- “better improver”对照主要检验策略文本改写，未隔离任务 LoRA 权重对配方设计能力的贡献。实际 rank、学习率和逐轮 recipe 轨迹也未充分报告。[MetaRSI 全文](https://arxiv.org/pdf/2609.06396v2)

因此，**不能继续把宽泛组合称为首次，也不能把 MetaRSI 误写成没有递归交接。**但这些证据尚不足以判定 dossier 严格要求的结果已经完成，故不满足 ABANDON 条件。

**2. B1–B4 分别有多少新意**

| 主张 | 最近的已读证据 | 剩余空间 |
|---|---|---|
| **B1：自产数据＋自主更新方案** | Search over Self-Edit 已生成数据模板，并选择 rank、alpha、dropout、lr、epochs、grad_accum；新增整体近邻见上。 | “自己选择 LoRA 超参”本身不新。训练代码生成也需要证明超越已有配置空间的实际价值。 |
| **B2：LoRA 应用于自身、改善下游任务** | SEAL、Self-Specialization、MASS 均已实现相应路径。 | 单独几乎不承担 novelty。 |
| **B3：更新后的模型接任编辑者，能力共同变化** | MASS 已做 27B、两轮交接，并隔离测试 workflow optimizer；Self-Rewarding 已测任务与判断能力共同改善。 | 需要明确测量的是 **weight-update recipe 的设计能力**，并识别其变化是否来自模型权重。 |
| **B4：没有更强外部模型** | Absolute Zero、SEAL 的 ARC 分支已有非学习 verifier 闭环。 | 是有价值的实验约束，不能独立充当主要创新。 |

对应方法证据：Search Algorithm 2、Appendix A.2、§5.4；SEAL Algorithm 1、Appendix A/B；MASS Algorithm 1–3、Table 1；Self-Rewarding §2.4、Table 4；Absolute Zero Algorithm 1。[Search 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2601.14532.notes.md:14)、[SEAL 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2506.10943.notes.md:13)、[MASS 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2610.12176.notes.md:13)、[Self-Rewarding 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2401.10020.notes.md:13)、[Absolute Zero 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2505.03335.notes.md:14)。

B4 必须按实验分支判断：

- SEAL 的 **SQuAD** 使用 GPT-4.1 判分，**ARC** 使用任务正确性反馈；不能把整篇都归为依赖外部 judge。
- MASS 的主循环自行筛选，外部强模型不选择主实验 checkpoint；但任务规格部分由 GPT-5.5 生成，最终还有外部强模型评测，因此不满足 dossier 的严格全流程 B4。
- R-Zero 训练环没有强模型教师，但数学 benchmark 使用 GPT-4o 判答案。
- Self-Rewarding 不仅最终使用 GPT-4，训练中的 early stopping 还使用 Claude 2。[MASS 原文 pp.5–6、17](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2610.12176/2610.12176.pages.md:370)、[R-Zero 评测协议](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2508.05004.notes.md:21)、[Self-Rewarding 训练协议](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2401.10020.notes.md:21)。

**3. 是否本质上就是 SEAL / Search 加 recursion？**

**作为工程起点，是；作为论文贡献，这个表述已经不够。**

SEAL/Search 的关键结构是：

\[
\theta_t \xrightarrow{\text{生成数据与配置}}
\theta'_t\;\text{用于计算奖励},
\qquad
\theta_{t+1}=\text{外层 proposer 更新}.
\]

内层任务适配模型 \(\theta'_t\) 被丢弃。你要求它直接成为下一轮作者，确实不同。Search 在 §5.4、p.10 将持续 self-edit、进一步提高 self-edit 能力列为 future work；**提出过这个方向不等于已经完成实验**。[原文定位](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2601.14532/2601.14532.pages.md:625)

但现在已有更多递归先例：

- **Self-HPO**：当前 policy 修改自身 harness，RL 更新后接任下一轮 proposer；训练配方仍是固定 GRPO/BAR。实验为 Qwen3.5-35B-A3B、两轮，每轮六个 harness 候选、50 次 RL updates，使用执行 verifier。[§2–3、Appendix A](https://arxiv.org/html/2610.22791v1)
- **HASE**：同一 Qwen3-8B 在 GRPO 中生成任务解答和 harness edits，更新后继续编辑；可修改面不包含 LoRA/optimizer 配方。[§3、Algorithm 1、Table 1](https://arxiv.org/html/2607.03935v1)

因此，以下几项单独都不足以支撑论文：多跑几轮、换成 27B、多开放一个超参、让 JSON 配置变成训练脚本。

更有价值的问题是：

> **任务模型变强之后，是否也更会为固定目标模型设计有效更新；这种能力能否迁移到未见任务，以及何时会与任务专精发生冲突？**

**4. 最接近工作实际上怎么做**

下面区分内层适配与外层作者训练；这些配置是复现实验参考，不是 27B 的推荐最优值。

| 工作 | 模型、数据与流程 | LoRA / 更新配置 | 评测与关键限制 |
|---|---|---|---|
| **SEAL：知识注入** | Qwen2.5-7B；SQuAD；每轮 50 段×5 self-edits×3 seeds，再对优胜编辑做 ReST-EM；两轮 | 外层作者 SFT：r=64、α=128、lr=3e−4；内层适配另由人工搜索配置 | 200 段、974 问；GPT-4.1 判分；Table 2：32.7→47.0。内层适配模型不接棒。 |
| **SEAL：ARC** | Llama-3.2-1B-Instruct；11 个训练、8 个筛选后的评估任务；模型选增强、lr、epochs、loss 范围 | 内层 r=128、α=16；q/v/gate/down/up；外层 r=16 | Table 1：无 RL self-edit 20%，SEAL 72.5%；评估规模小且经过筛选。 |
| **Search over Self-Edit** | Qwen3-8B reasoning；SQuAD 50 段训练、200 段验证；自产模板＋补全；五轮外层训练 | 搜索上述六项超参；固定模板基线 r=32、α=64、lr=1e−3、10 epochs | Claude Haiku 判分；Table 2：Archive 峰值45.1%，低于初始人工 Rewrite 50.3%；最优 archive 种子始终未被超越。 |
| **MASS** | Qwen3.6-27B；自生成、自评研究任务轨迹；119 条轨迹、17.2M 监督 tokens；两轮 | r=64、α=128、dropout=.05、lr=3e−5；adapter 合并后下一轮重新建 LoRA；配方人工固定 | Table 1 用固定 L0 执行不同作者的 workflow，L1 workflow 胜率64%；部分研究基准上升，通用编码基准下降。 |
| **Self-Specialization** | MPT-30B 生物医学、LLaMA2-7B 金融；80/90 条人工种子→约5K自产样本；可迭代生成 | 4-bit QLoRA；r=8、α=16、all modules、lr=3e−4、batch32、3 epochs | Table 4：MPT-30B 0-shot F1 25.15→34.57→36.63；后续迭代基本饱和。 |
| **Absolute Zero** | 同一策略出题与解题；Qwen2.5 7B/Coder-7B 主实验；Python 构造答案并验证 | 人工固定 TRR++，lr=1e−6、500 steps；未报告 LoRA | 同权重自举的正面先例；更新配方不是模型设计的。 |
| **TT-SI** | Qwen2.5-1.5B-Instruct，另有7B；每个工具调用实例自产适配数据 | r=8、α=16、all linear、lr=1e−4、5 epochs | Table 1 平均54.66→60.14；Algorithm 1 每个实例之后恢复 θ₀，没有持续 handoff。 |

配置与表格出处：[SEAL pp.18–21](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2506.10943/2506.10943.pages.md:1098)、[Search pp.3–6、12–19](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2601.14532.notes.md:14)、[MASS pp.16–17](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2610.12176/2610.12176.pages.md:1128)、[Self-Specialization pp.4、7、17](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2310.00160.notes.md:12)、[Absolute Zero 配置](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2505.03335.notes.md:14)、[TT-SI 配置及 reset](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2510.07841.notes.md:13)。

其余候选的重要边界如下，不能因标题有 “self” 或 “recursive” 就视为完整覆盖：

| 候选 | 实际边界 |
|---|---|
| **Self-Developing** | 写模型合并算法；DPO 更新 factory，但合并结果不接任 factory，合并始终作用于固定 M0。§2.4、Algorithm 1。[笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2410.15639.notes.md:14) |
| **Self-MoE / Transformer²** | 前者自产领域数据训练专家，配方固定且无递归；后者选择或组合预训练 SVF 专家，SVF 不是 LoRA。各自 §3。[Self-MoE](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2406.12034.notes.md:12)、[Transformer²](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2501.06252.notes.md:12) |
| **Text-to-LoRA / Generative Adapter** | 直接生成 adapter 已有先例，但依靠另外训练的 generator。后者确有适配模型参与后续流式更新，不能说只有一步；却没有证明 generator 的编辑能力递归改善。Eqs.3–6 / §2.2。[T2L](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2506.06105.notes.md:14)、[GA](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2411.05877.notes.md:13) |
| **FT-Dojo / PostTrainBench / SIA / WHALE** | 自动训练工程与权重更新很相关，但关键 editor 是外部模型，目标模型不接替它。WHALE 未明确报告 LoRA，不能补写成 LoRA 系统。[FT-Dojo §3.2](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2603.01712.notes.md:14)、[PostTrainBench §5.2](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2603.08640.notes.md:12)、[SIA](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2605.27276.notes.md:13)、[WHALE](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2609.00196.notes.md:13) |
| **SOLAR** | 实验只完成 Level I；0.5B 结构化输出必要时由7B辅助；不能把 Level II/III 愿景算成已执行递归。Algorithm 1、Table 1。[笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2605.20189.notes.md:30) |
| **AIDE² / WeightLoop** | 前者主要改 harness，递归 ignition test 有限且不确定；后者由固定外部 LLM 写程序修改小预测器，不是 LLM 自改。§3.6 / Fig.1。[AIDE²](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2609.26457.notes.md:14)、[WeightLoop](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2611.07319.notes.md:13) |
| **两篇 recovery / self-distillation** | 使用历史参考模型或剪枝前模型帮助恢复；不是当前 checkpoint 自主设计更新并递归接棒。[SDFT](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2604.15794.notes.md:16)、[Self-Data Distillation](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2410.09982.notes.md:21) |

**5. 对约 27B 的可行性判断**

**工程上可行；持续改善配方设计能力，没有现成保证。**

- **规模不是主要未知量。** MASS 已做27B，Self-Specialization 已做30B。真正未知的是任务适配能否保留乃至改善 editor 能力，不能假设“比8B大”就会解决自举偏差。
- **自主配方可能比人工配方差。** Search 的高学习率候选曾严重损坏适配模型；后期模板和超参迅速同质化。它说明应先检验自主决策有没有价值，再扩大自由度。Table 2、§4.2、Table 5。[证据](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2601.14532.notes.md:27)
- **同 checkpoint 多角色存在具体风险。** R-Zero 的共享参数消融 Single-R-Zero，在 Table 6 中从首轮47.31降至第四轮43.89；双模型版本更晚达峰。但这没有证明27B必然崩塌。[证据](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2508.05004.notes.md:29)
- **Self-Rewarding 没有观察到整体 collapse。** 奖励判断准确率78.7→80.4→81.7，增量变小；部分普通 NLP 能力下降。只有两次 DPO handoff，不能写成已证明长期饱和。Table 3–4。[证据](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2401.10020.notes.md:32)
- **代码递归崩塌论文是风险证据，不是定论。** 最大模型7B、单 seed；很多配置第一轮已明显下降，缺少等强度非递归训练对照。compile/长度过滤也不等于功能正确性验证。Table 2–3、Tables 11–14。[证据](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2606.28438.notes.md:56)

计算预算也不能只算最终 LoRA：SEAL 的一次750候选内层评估约需2×H100、6小时；MASS 第一轮 SFT 约6×H100、19小时，**后者不包含搜索、轨迹生成和评测**。这些是各自协议的实测，不能直接线性外推成你的成本。[SEAL Appendix B.5](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2506.10943/2506.10943.pages.md:1319)、[MASS Appendix B.2](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2610.12176/2610.12176.pages.md:1157)。

直接生成权重应作为另一条研究路线单独定义。T2L 的实现需要专门训练 hypernetwork，不能据此推断普通27B LLM能可靠输出自己的 adapter 数值；也不能仅凭“同模型直接生成”尚未匹配到论文就提高当前方案的分数。

**6. 怎样让剩余 delta 足以承载论文**

建议把核心问题和实验固定为：

> **在固定提示、记忆、验证器及被编辑目标的条件下，检验任务 LoRA 造成的权重变化能否提升同一 LLM 在新任务上设计训练数据与更新配方的能力，并刻画这种迁移对连续自适配、遗忘和成本的影响。**

这是待验证的研究目标，不是目前已经成立的结果。

最重要的测量是，把不同代模型当作作者，却让它们编辑**相同的固定 recipient checkpoint**：

\[
E_t=\mathbb{E}\!\left[
Q\!\left(U(\bar{\theta};D_t,h_t)\right)-Q(\bar{\theta})
\right].
\]

其中作者是 \(\theta_t\)，recipient \(\bar{\theta}\)、任务分布、prompt/archive 和候选预算固定。这样才能区别“被编辑的模型更容易进步”和“作者更会设计更新”。

| 对照 | 回答的问题 |
|---|---|
| 更新后的作者 vs 冻结初始作者 | handoff 是否真正有贡献？ |
| 模型配方 vs 调优人工配方、同预算搜索 | 自主决策是否优于常规调参？ |
| 当前 checkpoint 连续更新 vs 固定基座累计数据重训 | 保留优化历史带来收益还是误差积累？ |
| 固定数据、交换不同代作者的 recipe | 提升是否来自更新方法，而非仅数据更好？ |
| 固定 recipe、交换不同代作者的数据 | 数据生成能力贡献多少？ |
| 固定 prompt/archive vs 演化的 prompt/archive | 改善来自权重还是外部记忆、策略文本？ |

实验还应满足三点：

1. **开发反馈与封存测试分离。** 固定外部评估不意味着可以反复用最终测试选更新；代码任务应检查功能正确性，而非仅可编译。
2. **报告完整多轮轨迹和多个训练 seeds。** 包含 task score、editor score、旧能力保持、无效配方率、数据多样性、拒绝更新及总成本。
3. **失败必须有机制解释。** 例如任务分数上升而 editor score 下降，再通过配方/数据交换识别遗忘或分布窄化；单纯再报告一次“递归会 collapse”不够。

若观察到稳定的跨任务 editor transfer，或识别出能被干预验证的 task/editor 冲突机制，论文就有实质内容。只有 warm-start 改动和一条上涨曲线，贡献仍薄。

**7. 怀疑型 reviewer 会引用什么，还缺哪些论文**

最有力的审稿意见会是：

- **Search + MetaRSI：**自主数据与 LoRA recipe 已经有人实现。
- **MASS + Self-HPO/HASE：**更新模型接任改进角色并非首次，27B也不是新设置。
- **Self-Specialization + Self-Rewarding：**自产数据、递归训练和双能力改善有长期先例。
- **T2L + Generative Adapter：**直接生成低秩更新同样不是新概念。

能抵御这些意见的，就是上面的 **weight-induced recipe-author transfer**，而不是改用“Self-LoRA”这个名称。

本轮新增的三篇已升级为全文证据。另有以下**待全文核查的 leads，不参与本轮 verdict**：

| Lead | 后续阅读目的 |
|---|---|
| [ThetaEvolve，2511.23473](https://arxiv.org/abs/2511.23473) | 进一步核查测试时学习与改进能力迁移的边界。 |
| [Phantom Gains，2608.20290](https://arxiv.org/abs/2608.20290) | 完善自改进测量的零假设与评估噪声控制。 |
| [Doc-to-LoRA，2602.15902](https://arxiv.org/abs/2602.15902) | 若走直接权重生成路线，补齐最近邻。 |
| [Macaron-V1，2608.09819](https://arxiv.org/abs/2608.09819) | 核查 continual learning、Mixture-of-LoRA 与递归组合。 |
| [AI4AI-Bench，2608.20318](https://arxiv.org/abs/2608.20318) | 若开放训练代码生成，寻找配方设计评估协议。 |

Dossier 原列的 FST、Drag-and-Drop LLMs、STaR / ReST^EM / SPIN 仍按 lead/background 处理，未据摘要给它们追加覆盖主张。

📋 **本次操作回顾**：完成 Topic B 证据审查、补充三个关键全文近邻和独立复核；未修改文件。  
📊 **当前状态**：CAUTION，5/10；值得进入有明确对照的 pilot。环境只读，报告与 review trace 未保存到本地。  
💡 **下一步建议**：先检验“不同代作者对固定 recipient 的更新质量”，把这项结果作为是否扩展到完整27B递归实验的依据。