**结论：PROCEED，6/10。这个方向通过新颖性初筛，优先推进 (a) self-compression。** 最值得研究的差异是：**模型自主压缩当前自身权重，并由压缩后的 checkpoint 接任下一轮编辑者**。相比之下，(b) 如果只是“自己生成数据／配置，再训练 LoRA 应用到自己”，已有直接先例，不能作为主要新意。

我已阅读 dossier 指向的全部 **14 份全文笔记**，核对关键方法与实验原文，并补查截至 **2026-11-13** 的文献。另进行了 fresh-agent 复核；其评分为 **5/10，同样判定 PROCEED**。以下重叠判断依据已读笔记及原文；新发现和 ABSTRACT-ONLY 文献只列为线索。

| Claim | 最接近的工作及具体证据 | 判定与剩余差异 |
|---|---|---|
| **C1：自主压缩自身权重，提高推理效率** | **Self-Pruner**：GPT-4o 生成剪枝比例，目标为 LLaMA/Vicuna；Alg.1，p6；Table 7，p9 有真实吞吐测量。[笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2502.14413.notes.md:12)。**Adaptive Pruning by Agents**：Gemini 决定 Qwen3 的逐层稀疏率；§3.3、Alg.1，pp6–8。[笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2601.09694.notes.md:12)。**ProfilingAgent**：GPT-4o 为视觉模型选择剪枝、动态量化；Fig.4、Alg.1、Tables 3–4，pp5–9。[笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2509.05584.notes.md:12)。 | **agent 驱动压缩已有；当前编辑者压缩自身、压缩后继续担任编辑者，未被这些工作覆盖。** 差异在 checkpoint 关系及其后果。 |
| **C2：自产数据／超参／代码／权重，形成自己的任务 LoRA** | **SEAL**：同一模型生成 self-edit，再更新自身；ARC 中选择增强和优化超参，并用 LoRA 适配。§3.1、Alg.1，p4；§4.1，p6；Table 1，p7。[笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2506.10943.notes.md:13)。 | **按 dossier 的“或”定义，已实质覆盖。** 不能因为 SEAL 不写训练代码、不直接输出权重，就否认其覆盖数据／超参路线。 |
| **C3：更新后的模型接任编辑者，形成共同改善** | **MASS**：更新后的同一 checkpoint 接任执行者、workflow 优化者和评估者；Alg.1，p4；第二轮训练，p17。Table 1，p7 固定执行器后，更新模型生成的 workflow 获胜率为 0.64。[笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2610.12176.notes.md:13)。 | **checkpoint 接棒和部分角色能力共同改善已有。** 尚未覆盖的是：自主压缩改变编辑者本身后，其**权重编辑能力**如何变化。不能泛称“首次递归更新权重”。 |
| **C4：权重操作作为一等动作** | **Self-Developing** 生成直接操作任务向量的 merging 程序；§2，pp3–4；Alg.1，p21。[笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2410.15639.notes.md:13)。**SIA** 在 harness 更新和 LoRA 权重更新之间选择，并选择 RL 算法；Fig.2、§5，pp6–8。[笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2605.27276.notes.md:13)。 | **C4 本身已有。** 提供更多算子可以构成方法设计，但“把权重加入动作空间”不能单独主张新颖性。 |

这里需要修正笔记中几处容易放大差异的解读。

**首先，输出 JSON、剪枝比例，再调用固定工具，仍然属于权重级动作。** Self-Pruner 和 ProfilingAgent 不需要亲自生成每个权重数值，才算自主选择压缩。它们与 C1 的决定性区别是编辑者和压缩对象不同。也不能要求一篇论文同时实现 pruning、quantization、LoRA、merging，才承认它覆盖 C4。

**其次，SEAL 不能被概括为“没有连续自编辑”。** 主训练 Alg.1 确实将内层适配后的 \(\theta'_t\) 用于计算奖励，下一轮编辑策略来自外层更新；但 §5、Fig.6，pp8–9 另有连续 self-edit 实验。可守的区别是：**该实验测量遗忘，没有验证先前编辑使模型更擅长后续编辑**。[SEAL 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2506.10943.notes.md:19)

**第三，已有方法的实验不足不会撤销其先行性。** 笔记中有关基线公平性、数据泄漏或替代解释的【推断】【猜测】，不能直接当作事实，更不能用来宣布已有机制“不算做过”。

对 **(b) 是否基本就是 SEAL**，需要分三种实现回答：

- **自产数据／选择部分训练配置 → 自身 LoRA：基本是。** 更早的 **Self-Specialization** 已用自产数据训练自身 QLoRA，并让适配后的模型继续生成下一轮数据；§3.5，p4；Table 4，p7。**Self-MoE** 已扩展到多个自身 LoRA 专家及组合；§3.1–3.2，pp4–5；Table 1，p6。[Self-Specialization 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2310.00160.notes.md:12)、[Self-MoE 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2406.12034.notes.md:12)
- **自己写训练代码：超出 SEAL 的具体实现，但单凭增加代码生成，差异仍薄。** PostTrainBench 已展示 agent 自主找数据、写训练代码、选择 LoRA／全参训练；其对象是另一个模型。必须说明“训练自身并接棒”带来什么可验证的新机制或发现。依据 §2，pp2–3；§5.2，pp7–8。[PostTrainBench 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2603.08640.notes.md:12)
- **直接生成 LoRA 权重：不是 SEAL，但已有强先例。** Text-to-LoRA 的 hypernetwork 从任务描述生成 LoRA；§3，pp3–4；Table 2，p6。Generative Adapter 使用目标 LM 的隐藏状态生成低秩增量，且已适配模型参与下一段上下文的处理；§2，pp4–5。不能仅凭它有新增生成器 \(G\)，就排除所有合理系统边界下的“self”。仍未验证的关键是编辑能力跨轮改善。[Text-to-LoRA 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2506.06105.notes.md:14)、[Generative Adapter 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2411.05877.notes.md:13)

此外，“根据任务选择自身权重调制”也不能作为独有差异：Transformer² 已有自选预训练 SVF 专家的两遍推理流程；区别是它没有生成新专家或验证递归编辑能力。见适配策略 p6、Table 2，p8。[Transformer² 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2501.06252.notes.md:12)

**组合层面，最接近的参照是两组工作，而不是一篇已经完整覆盖的论文。**

对 self-compression，是 **Self-Pruner／Adaptive Pruning + MASS**：前者提供压缩决策，后者提供更新 checkpoint 接任优化者的结构。值得研究的未知是：**压缩同时改变了被优化对象与未来优化者；推理变便宜，但编辑判断可能变差，这两种效应如何共同决定长期收益？** 这可以产生独立贡献，并不要求先发明全新的量化算法。

对 self-LoRA，是 **SEAL + Self-Specialization + MASS**。只增加迭代次数，很难形成强差异。Self-Developing 也不能被忽略：它的 factory 会改善，但合并始终作用于固定 \(M_0\)，下一轮 factory 来自 DPO，而非合并结果；§2.4，p4。AIDE² 的 ignition test 则已经提出检验“改进后的系统是否更会改进系统”，但它操作 harness，且该实验结论不确定；§3.6、Fig.7，pp10–11。[Self-Developing 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2410.15639.notes.md:13)、[AIDE² 笔记](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2609.26457.notes.md:13)

审稿人最可能用前三组工作分别质疑“自主压缩”“自身 LoRA”“递归接棒”。我认为能守住的一句话 delta 是：

> **研究由当前 LLM checkpoint 自主选择并执行自身权重压缩、再由压缩后的 checkpoint 接任下一轮权重编辑者的闭环，并在匹配总预算下检验真实推理效率、任务能力与后续编辑能力的跨轮权衡。**

这是拟研究贡献，不能提前写成已经证明的“共同提升”。它也无需依赖“首次 RSI”“首次自生成 LoRA”或“首次权重级动作”等站不住的表述。

让这个 delta 成为论文贡献，最有辨别力的 pilot 应包括：

1. **接棒对照**：更新后的编辑者，与同一初始 checkpoint 的冻结编辑者比较；统一工具、反馈、历史和总预算。
2. **单独测编辑能力**：让不同轮次编辑者处理同一组留出的目标 checkpoint，避免把任务分数提高直接解释成编辑能力提高。
3. **测真实成本**：同时记录 latency／throughput、能力保持以及搜索和恢复训练成本。稀疏率下降不等于实际加速。

这些实验即使发现递归接棒存在稳定的退化边界，也可能形成有价值的发现；不必预先承诺每轮都共同进步。

补查发现以下**待全文核对线索**。它们目前没有用于降低评分或否决方向：

| 线索 | 精读时最需要核对的问题 |
|---|---|
| [CompressionBench，2611.06858](https://arxiv.org/abs/2611.06858)，11 月 4 日首发、11 月 9 日更新 | 是否包含自身压缩、量化，以及压缩后编辑者接棒；这是最高优先级。 |
| [Search over Self-Edit Strategies for LLM Adaptation，2601.14532](https://arxiv.org/abs/2601.14532) | 自主选择数据与超参，是否已经覆盖拟议 self-LoRA 的具体增量。 |
| [JoMoHa，2611.11499](https://arxiv.org/abs/2611.11499)；dossier 已列的 [WHALE](https://arxiv.org/abs/2609.00196)、[Learning, Fast and Slow](https://arxiv.org/abs/2605.12484) | 更新者、执行者和权重更新策略之间的继承关系。 |
| [Self-calibration for Language Model Quantization and Pruning，2410.17170](https://arxiv.org/abs/2410.17170) | “自己生成校准数据”与“自己决定压缩策略”的边界。 |
| [Can Compressed LLMs Truly Act?／ACBench，2505.19433](https://proceedings.mlr.press/v267/dong25k.html)；[Anosognosia in LLMs，2610.06174](https://arxiv.org/abs/2610.06174) | 压缩后的 agent 能力与自监测，应采用什么评价和已有发现。 |
| [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)（作者研究札记）；[WeightLoop，2611.07319](https://arxiv.org/abs/2611.07319) | 直接权重操作与固定／更新编辑者的关系。 |
| [A Modern Self-Referential Weight Matrix That Learns to Modify Itself](https://proceedings.mlr.press/v162/irie22b.html)；[Metalearning Continual Learning Algorithms](https://openreview.net/forum?id=IaUh7CSD3k) | 更早的 self-referential 权重更新机制，避免对“自己修改自己”作过宽首次声明。 |
| [AdaRSI，2610.20474](https://arxiv.org/abs/2610.20474) | 自主决定何时更新权重的邻近机制。 |

因此，评分和建议具体是：

- **整体：PROCEED，6/10。** 有明确邻居，也有值得 pilot 的可辨认差异；尚无固定方法和结果，因此不是成熟论文的新颖性评分。
- **(a) self-compression：PROCEED，优先推进。** 已读证据没有封死“自身压缩后接任编辑者”的问题。
- **(b) self-LoRA：PROCEED WITH CAUTION。** 具体薄弱处是数据／超参路线已被 SEAL 等覆盖，直接权重路线又与 Generative Adapter 接近。需要确定额外机制或发现；仅把已有自产数据 LoRA 流程重新命名为 RSI，不构成新贡献。
- **不建议 ABANDON 整体方向。** 当前没有一篇已核对论文可具名为完整实现上述 self-compression 闭环及其能力评估的先例。

📋 已完成 14 份笔记审查、关键原文核对、补充检索及独立复核。  
📊 未主动修改文件；目录不是 Git 仓库，只读权限使报告与 review trace 无法落盘。  
💡 下一步优先精读 CompressionBench，再开展冻结编辑者与递归接棒编辑者的同预算 pilot。