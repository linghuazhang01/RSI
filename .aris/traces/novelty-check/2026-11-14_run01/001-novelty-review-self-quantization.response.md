**判决：PROCEED，7/10。Topic A 通过本轮 novelty check。** 在已核实的文献中，没有发现同时完成“同一 checkpoint 自主量化自身、量化产物接任下一轮 editor、分析 editor 退化如何影响后续压缩”的工作。分数针对这个完整研究问题；它不代表实验成功概率。

本轮覆盖 dossier 中的 22 份全文阅读笔记，回查关键方法、表格和原文，并补充检索至 **2026-11-14**。另补读了两篇遗漏的近邻：**AutoQS** 和 **APQF**。独立上下文的同模型家族 reviewer 也给出 **PROCEED，7/10**；这是早期查新结论，不能证明未检出的工作不存在。

可以保留的一句话贡献是：

> 在没有更强外部模型参与决策的闭环中，让同一 checkpoint 自主量化自身，并由被接受的量化模型接任下一轮压缩者，通过匹配目标、预算与反馈的实验，测量 editor 退化对后续压缩质量、成本和停止边界的影响。

**A1–A4 的新颖性并不均匀，主要贡献集中在 A1+A2 的组合及其研究发现。**

| Claim | 最近先例及全文证据 | 已覆盖的部分 | 尚存的 delta |
|---|---|---|---|
| **A1 自主量化自身** | [CompressionBench notes](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2611.06858.notes.md)，§2–3，pp.3–6；[Self-calibration notes](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2410.17170.notes.md)，§3–4，pp.3–5 | 外部 agent 自主选择、实现和验证压缩；目标模型为自身压缩生成校准数据 | **决策者就是被压缩 checkpoint**。仅替换 editor 身份的单轮演示较薄，递归中的影响更有研究价值 |
| **A2 压缩产物接任 editor** | CompressionBench p.12；[MASS notes](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2610.12176.notes.md)，Algorithm 1 p.4、训练流程 p.17 | MASS 已让更新后的模型担任下一代执行者、优化者和评估者；CompressionBench 已提出后续压缩能力问题 | **量化改变 editor 能力，而 editor 又决定下一次量化**的反馈关系及因果分析尚未被这些实验覆盖 |
| **A3 保持能力并获得效率** | CompressionBench §2.4、Table 1；[ACBench notes](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2505.19433.notes.md)，评测设置及 Table 10 p.33 | 外部验证、存储约束、能力评测、压缩后的 agentic 测试和速度测量均有先例 | 将这些要求落实为递归接任的验收条件有意义，但主要是**必要验证协议** |
| **A4 LLM 分配优于经典分配** | [GAMMA notes](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-quantization/papers/2605.18475.notes.md)，§3、Tables 2、5–7；下述 AutoQS | 自动 mixed-precision 分配、LLM 量化配置搜索及随机 proposer 对照已有 | self-editor 能否在相同资源下达到更好的质量或更低搜索成本，仍是**待验证假设** |

组合层面不能主张“首次 agent-driven compression”或“首次更新后的模型接任优化者”。可以主张研究**压缩对象与压缩决策者耦合后的行为和边界**。

CompressionBench 的差异确实能用一句话核验。其 p.12 原文明确写道：

> “Extending this step toward RSI requires testing whether a compressed model retains the ability to carry out subsequent compression tasks.”

这句话共 22 个词，已在[本地全文 p.12](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2611.06858/2611.06858.pages.md:820)核实。**这是已经公开提出、尚未在该文执行的方向。竞速风险明确，具体团队的后续进度未知。** 按 dossier 的判决规则，公开路线图构成竞争信息，不构成否决。

最接近工作的实际做法如下，可以直接用于实验设计：

| 工作 | 方法、模型与数据 | 评价及可借鉴之处 |
|---|---|---|
| **CompressionBench** | 六种外部 coding agent 压缩 Nemotron/Qwen，包括 Qwen3.8-27B；一张 H200、8 小时；自主选算法并交付权重与推理代码。提供 FineWeb、UltraChat、WikiText 等数据 | 搜索用固定 BF16 生成续写的 likelihood；独立验证存储及 GSM8K/HumanEval/TriviaQA 各 50 题，平均准确率至少为 BF16 的 80%。测 decode TPS，但速度不进入验收。见 §2–3、Appendix A |
| **[Self-Pruner](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2502.14413.notes.md)** | GPT-4o 生成逐层剪枝率，并执行进化搜索的初始化、交叉、变异；Wanda-sp 固定执行。目标为 LLaMA/Vicuna 7B–70B；WikiText-2 PPL 作 fitness | 评测七项常识任务及 vLLM/A100 吞吐；7B 的 50% 剪枝获得 1.82× 吞吐。没有目标模型接任，也缺少同预算普通 GA 对照。Algorithm 1 p.6、Tables 4–7 p.9 |
| **[Adaptive Pruning by Agents](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2601.09694.notes.md)** | Gemini-3-Flash 读取层敏感度、梯度、当前 PPL，向 Qwen3-4B/8B 输出额外稀疏率；Wanda 执行，单轮 PPL 恶化超过 15% 时回滚；C4 校准 | WikiText-2、MMLU、FreebaseQA；没有速度测量。可借鉴反馈、停止和回滚机制；其非结构化方法对比 N:M magnitude 的实验不能隔离 agent 收益。pp.5–10、Table 4 p.15 |
| **Self-calibration** | Gemma/Phi/OPT/Mistral/Llama 2B–8B 自采样 128×2048 token；人类固定 AWQ/GPTQ W4-g128 或剪枝方法；五套校准集 | 十项零样本任务；没有 agent 行为或速度测试。适合作为“固定原模型自产校准数据”的对照，不足以构成 A1 |
| **GAMMA** | 预先生成 QTIP/AQLM 的 2/3/4-bit 候选；冻结候选权重，学习模块级偏好，再用 ILP 满足预算；RedPajama 1024×2048；8B–32B | 同 QTIP 后端下，Qwen3-8B 的 3-bit 平均分为 76.33，GAMMA 为 81.16；更低 bit 未自动更快。必须计入候选预量化成本。Table 2 PDF p.9、Tables 5–7 PDF p.13 |
| **新增 [AutoQS](https://arxiv.org/html/2611.11975v1)** | Qwen-Plus／DeepSeek proposer 对其他 7B 模型搜索固定 bit 下的 group size；有反馈和随机 proposer 对照 | 没有自身接任或 agentic/速度评价。§3–4、Tables 2/8、Appendix A–E |
| **新增 [APQF](https://arxiv.org/html/2608.05499v1)** | 外部 LLM 对 CNN/ViT 选择剪枝率、bit-width 和恢复训练；ImageNet/CIFAR-10；QAT、KD、固定 evaluator | 主要报告准确率和 BOPs；不是 LLM 自量化。§3–4、Tables 1–4；§6 将 LLM 压缩列为未来方向 |

几个容易被审稿人混用的“self/recursive”先例也需要准确区分：

- **[Self-Developing](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2410.15639.notes.md)** 的 factory 写 merging 算法，再由 DPO 更新 factory；合并结果不接任 factory，算法始终作用于固定 \(M_0\)。见 §2.4 p.4。
- **[AIDE²](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2609.26457.notes.md)** 的主要对象是 harness；真正更换外层编辑者的 ignition test 在 p.11、Fig.7，结果不确定。
- **[WeightLoop](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2611.07319.notes.md)** 只调用外部 LLM 一次来编写固定编辑器，随后修改小型生物预测模型；LLM 和编辑器代码都不更新。见 §2，pp.1–3。
- **MASS** 是重要结构近邻，但其 FP8 是人为固定的部署步骤，不是模型自主选择的压缩动作。不能把这一点写成“完全没有递归先例”。

**27B 自任 compressor 有合理可行性，但现有证据还没有直接证明它能做好。** CompressionBench 的 27B 是 target；Appendix D 中名为 Qwen 的 proposer 是 Qwen3.8-Flash，不能误认成 27B 自己。MASS 的 Qwen3.6-27B coding agent 支持工具执行可行性，但不能替代压缩任务 pilot。

量化后一般 agent 能力也有正面证据：ACBench 的 Qwen2.5-32B 在 WorfBench 上为 BF16 0.72、AWQ 0.71、GPTQ 0.69；但 32B 只覆盖这一工作流任务，不能外推为“能够自主运行压缩实验”（Table 10，p.33）。

**建议把 16 GiB 用于流程验证，12→10 GiB 用于主要研究，8 GiB 用于压力测试。** 这是实验建议，不是已证实的最优区间。按约 270 亿参数换算：

| 权重存储预算 | 总字节折算的 effective bits/parameter |
|---|---:|
| 16 GiB | 5.09 |
| 12 GiB | 3.82 |
| 10 GiB | 3.18 |
| 8 GiB | 2.55 |

这是总预算的折算值，必须容纳 scales、zeros、未量化 embedding/head 等；不能把 16 GiB 理解为 16-bit。

CompressionBench 的直接 27B pilot 支持这个区间有辨识度：W4-g128 实际包为 **15.00 GiB，C4 PPL 增加 3.19%**；混合 W3/W4 为 **11.98 GiB，增加 10.53%**（Table A4，p.19）。该文没有测 10 GiB；pilot 与主表的 runtime、评价背景不同，不能跨表计算速度提升。

“4-bit 安全、3-bit 崩塌”应当从笔记建议中删去或收紧：

- [Quantization Hurts Reasoning?](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2504.04823.notes.md) 的 R1-Distill-Qwen-32B 在 W4 下平均下降 AWQ 0.4、GPTQ 1.9 个百分点，W3 下下降 3.9、5.1 个百分点。这支持把 3–4 bit 作为探索区间，但不是通用阈值（Table 1，p.5）。
- [Flat Score, Amplified Failures](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2607.27275.notes.md) 直接测了 27B：INT4 下 Qwen3.5-27B telecom 工具错误率从 **0.26%→1.21%**；Qwen3.6-27B 为 **0.38%→0.47%**，没有显著放大。最终成功率却都未显著下降（Table 1，p.4）。模型与配方差异必须实测。
- [Which Decisions Low-Bit Quantization Breaks](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2608.06564.notes.md) 的 3-bit whether-to-call 翻转率中位数为 RTN 47%、GPTQ 21%、GGUF 10%；这也否定单一 bit 阈值。其首 token margin 结果不能直接代替完整 rollout（Table 1 p.5、Appendix A）。

工程上还要预留构建资源：27B 的 BF16 权重本身约 **50.3 GiB**，量化阶段另需校准与工作内存。最终产物能装入 10–16 GiB，不表示整个自压缩过程只需同样的显存。

**评价需要同时测模型能力、压缩者能力与实际执行成本。** 推荐如下最小组合，阈值应在搜索前固定：

| 维度 | 必需测量 |
|---|---|
| 一般能力 | PPL/KL 作便宜筛选；知识、指令遵循、难数学和代码生成作能力门槛。不能只用 MMLU 或 GSM8K |
| Agentic 能力 | BFCL 实际生成，以及固定多轮工具任务；报告成功率、工具名/参数错误、遗漏调用、重试、预算耗尽 |
| **Compressor 能力** | 合法计划率、可运行产物率、有效候选数、报错修复率、单位搜索成本获得的质量改善 |
| 存储与内存 | 实际 package bytes、resident model storage、加载及运行峰值显存；单列 KV cache/workspace |
| 速度与总成本 | 同硬件和 runtime 下的 TTFT、decode TPOT/TPS、完整任务耗时、总 token、GPU-hours；统计失败候选成本 |

这些测量不能互相替代。例如 CompressionBench 主表中，27B/12 GiB 的 Sonnet 产物 PPL 接近 BF16，但 TPS 是 **16.7 对 61.4**（Table 1，p.6）。[UniComp](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2602.09130.notes.md) 的效率测试则使用输入 1024、输出 16 的负载，不能直接代表长输出 agent 的性能（Table 3、Appendix A.3）。

固定外部 evaluator 的方向正确，但**接受门与最终封存测试应分开**，能力底线相对于第零轮定义，避免每轮允许一点退化后累计失控。主实验优先使用确定性答案、执行测试和固定环境；如果借助更强 LLM judge，必须如实收紧“没有更强模型参与整个循环”的表述。

**最关键的实验不是单独跑出 16→12→10 GiB 轨迹，而是拆开三种原因：预算变紧、权重重复量化、editor 退化。**

建议采用以下控制矩阵：

| Editor | 每轮从原始高精度权重构建 | 对上一轮量化权重继续压缩 |
|---|---|---|
| 冻结的同源 BF16 editor | 非递归搜索参照 | 测重复量化损伤 |
| 当前量化 editor | 测 editor 退化，减弱累计权重误差的干扰 | 完整递归自量化主实验 |

再在每个固定 target checkpoint 上，让 BF16 editor 与量化 editor 接收**相同预算、历史、工具和校准数据**，配对提出方案并重放执行。冻结 BF16 editor 只属于对照组，主循环不能向它求助。

由此才能区分：

- 当前 editor 失败，但固定 editor 在同一 target、同一预算下成功：支持存在 editor 瓶颈。
- 两者都失败，而从原始权重重建成功：支持累计量化损伤。
- 所有方法都失败：只能说明在当前候选空间、预算与搜索成本下未找到可行解，不能宣称证明了根本压缩极限。

自生成校准数据应单独做消融：固定原模型生成数据与当前模型重新生成数据对比。否则数据分布变化会成为第四个混杂因素。也不应预设一定出现 collapse；稳定接任或明确的停止边界同样可以成为结果。

A4 的比较必须包含 **uniform PTQ、简单敏感度分配、随机/进化搜索、GAMMA**。要同时控制实际 bytes、量化后端、候选空间、校准数据和评估预算，并计入 agent 推理、候选预计算、失败执行的总成本。若完整系统允许选择不同 quantizer，应另设固定后端的分配实验，才能归因于 LLM 决策。

另外有三处证据需要在后续方案里收紧：

- “推理校准带来 +9.8 分”来自 **1.5B** 实验，不能外推到 27B；见 Quantization Hurts Reasoning? Table 2 p.9。
- [The Structure of Quantization Damage](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2609.01587.notes.md) 的全局细粒度优势来自 ≤8B、per-row RTN 起点和整层保护对照；不能据此否定所有 mixed precision。Table 1 的差距是恢复比例，不是原始准确率百分点。
- [Self-Distillation recovery](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2604.15794.notes.md) 与 [Self-Data Distillation](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2410.09982.notes.md) 支持特定任务恢复，但没有建立 27B 自主递归压缩能力。恢复训练会引入额外变量，宜放在基本闭环之后。

审稿人最强的质疑会是：**“这是 CompressionBench/AutoQS 的压缩搜索，加上 MASS 式接任。”** 这项质疑对单轮配置搜索成立；要让完整论文成立，需要拿出上述匹配干预，回答“压缩如何改变继续压缩的能力”，并给出实际成本与能力边界。若 A4 没赢，应删除“更优分配器”的结论；只要 A2 的发现扎实，研究问题仍可能成立。

需要补入 dossier 的遗漏文献是已全文核对的 **AutoQS、APQF**。以下仍仅作为 **leads，不参与本轮否决或重叠判决**：

- [AMQ，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1799/)：优先补读的 mixed-precision 搜索基线。
- [Compute Where it Counts: Self Optimizing Language Models，ICML 2026](https://proceedings.mlr.press/v306/akhauri26a.html)：需要核对自主效率控制的边界。
- [Safety-Guided Self Compression](https://arxiv.org/abs/2505.00350)：需要澄清标题中的 self 与本方案的关系。
- dossier 原有的 Pruner-Zero、EvoPress、Q-Strata、MixQuant 继续保持 ABSTRACT-ONLY，不能用来认定核心贡献已被完成。

📋 本次操作回顾：完成文献证据核对、近期补检与独立复核；未修改文件。  
📊 当前状态：**完整 Topic A：PROCEED，7/10**；仅做 A1 单轮演示则贡献偏薄。当前环境只读，报告、trace 与 Obsidian 未落盘。  
💡 下一步建议：先做 **BF16 自任 compressor 的单轮 pilot，再做一次真实 hand-off 和匹配 editor 对照**；优先验证核心反馈关系。