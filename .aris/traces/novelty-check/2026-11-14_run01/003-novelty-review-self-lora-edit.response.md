**结论：PROCEED，6.5/10。B′ 通过这轮早期查新，值得进入 8B pilot。** 相比此前的 5/10、PROCEED WITH CAUTION，我会上调判断。已核验文献没有完整实现“自身已训练 LoRA 的代码编辑 → 编辑后的模型接任 → 固定 recipient 上的新任务编辑能力测量”这一组合。

这个判断针对 dossier 定义的完整方向。**C2 的等预算优势、C3 的递归编辑能力提升，目前仍是待验证假设。** 单轮调几个合并系数、观察任务分数上涨，还不足以支撑完整贡献。

我读完了 dossier、设计文件及其指向的 26 份笔记，回查了关键方法、算法和实验原文，并补查截至 2026-11-14 的相关工作。新增三篇完成全文文本核验；未完成全文的候选单独列为 leads，没有用于改变 verdict。

**1. C1–C4 的最近先例与剩余差异**

| Claim | 已有工作的实际覆盖 | B′ 可以保留的差异 |
|---|---|---|
| **C1：同一 checkpoint 编码编辑自身 LoRA** | **Self-Developing 最接近。** 第一代 factory 明确由 \(M_0\) 克隆，因此已经存在“同权重模型写合并代码改善种子”的交集。随后 factory 经 DPO 更新，生成算法始终作用于固定 \(M_0\)，素材是外部专家的 task vectors。见 §2，pp.3–4、Alg.1。 | 编辑对象是自身任务训练得到的 adapter；接受后的编辑产物直接成为后继 editor。“同一 checkpoint”单独不是充分差异。[原文](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2410.15639/2410.15639.pages.md:169) |
| **C2：等评估预算胜过基线** | LoraHub、Sakana、EvoGM 已覆盖无梯度合并搜索；PARA、TaDA 已覆盖谱截断、诊断门控等编辑。Self-Developing 已与人工合并规则比较。 | **相同动作空间、信息和完整预算下，LLM proposer 是否有稳定优势**，仍可成为经验贡献。算子本身不能当作新贡献，具体预算问题见下文。 |
| **C3：接任与 editor-skill 测量** | **MASS** 已让后代接任 workflow optimizer，并在固定 \(L_0\) executor 上比较两代 workflow，Table 1 报告胜率 0.64。**MetaRSI** 已在同一个 \(S_1\)、同预算比较原始与 meta 改写后的 improver，Fig.11(c) 为 +4.9 对 +7.2。 | 新意应落在**权重编辑程序的能力，以及 post-hoc 编辑对这种能力的贡献**。不能泛称首次“固定 recipient 测 improver”或首次“未专门训练 optimizer 却有所提升”。[MASS，p.7](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2610.12176/2610.12176.pages.md:487)、[MetaRSI，p.24](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/src_2609.06396/2609.06396.pages.md:1576) |
| **C4：持续学习、遗忘控制、固定外部门** | SEAL §5/Fig.6 已测连续 self-edit 与 retention。新增全文核验的 SLoRA、Audit Before You Merge 更直接覆盖训练后 LoRA 过滤、连续合并和相关评测协议。 | C4 主要提供研究场景与实验约束；独立新颖性有限。贡献需要体现在编辑带来的稳定性—可塑性改善，以及后继 editor 的表现。[SEAL，pp.8–9](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2506.10943/2506.10943.pages.md:587) |

**组合层面最有力的差异，是 C1 与 C3 的连接。** Self-Developing 更新的是算法 factory，MASS 改进的是 workflow optimizer，WSP 主要由固定外部模型改另一模型；尚未在已全文核验的工作中找到 B′ 所定义的完整链条。

**2. 从“自选 LoRA recipe”转向“训练后权重编辑”，确实改变了判断**

MetaRSI 的 Model-RSI 在有界空间选择 adapter 配置、学习率、训练安排等，然后由固定后端执行训练。其 Eq.18–19 明确了这一机制。B′ 则在 raw LoRA 已经训练完成后，让模型决定具体的参数变换，编辑阶段不再执行梯度训练。这是实质性的动作空间变化。[MetaRSI，p.14](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/src_2609.06396/2609.06396.pages.md:914)

因此，最直接的近邻从 MetaRSI／Search over Self-Edit，转向了 **Self-Developing＋WSP＋训练后 adapter 操作**。新颖性提高，但不能据此忽略这些更近的先例。

另外，**“从固定 \(\theta_0\) 重训”不能用来证明 MetaRSI 没有递归或接任**：它明确有后继系统和多 term 更新，只是没有隔离 B′ 所关心的“editor 权重变化带来的权重编辑能力”。

**3. WSP 是直接竞争方向，属于 race**

| 工作 | 已完成的内容 | 与 B′ 的关键边界 |
|---|---|---|
| **Sparks of WSP blog** | Astra 写代码修改其他模型的权重；“修改自身权重用于 RSI”已在展望中提出。 | 自身权重编辑的概念不能宣称首创；尚无自身 LoRA 编辑与后继 editor 实验。[Exp.8／展望](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/src_wsp_blog/wsp_blog.txt:350) |
| **2611.09382，WSP-Scope** | 外部 Astra 诊断并编辑 Qwen；已允许 rank-one、缩放、稀疏编辑及组合。 | 每次搜索从原始 checkpoint 开始，无被编辑模型接任。因此“低秩编辑”“组合算子”本身也不是差异。[§2–3，pp.3–4](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/src_2611.09382/2611.09382.pages.md:146) |
| **2611.09376，Robot WSP** | 外部 Astra 编辑机器人策略，主成功来自 token embedding 行的调整。 | 编辑者与 recipient 不同，没有 editor 接任或 \(E_t\)。[§2，p.2](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/src_2611.09376/2611.09376.pages.md:89) |
| **WeightLoop** | LLM 一次生成可复用编辑程序，之后随实验反馈修改预测器。 | LLM 和程序始终固定；补丁从 \(\theta_0\) 重建，不构成后继 editor 的能力递归。[§2，p.2](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/src_2611.07319/2611.07319.pages.md:116) |

2611.04268 虽然标题包含 recursive loop，但全文是猜想与伪代码，没有实验；设想中也是固定 LLM 反复编辑 UNet。[Listing 2／结论](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/src_2611.04268/2611.04268.pages.md:296)

建议采用这一句定位：

> **在无更强外部模型、无专门 editor 训练的主实验中，检验开放 LLM 对自身已装载 LoRA 编写训练后编辑程序、由编辑后的权重接任，是否能在固定 recipient、未见任务和相同预算下产生超出普通任务训练的编辑能力提升。**

这句话描述的是待检验贡献；现阶段不能改写成已经证明的结果。

**4. 可行性证据：有编辑空间，但 LLM 优势与递归收益均未确定**

关于 **raw LoRA 之后是否还有提升空间**：

- **PARA** 提供直接的单 adapter 证据：p.6 报告部分压缩版本略胜父 adapter，并称其他图像数据集有相似行为。不过，主表的 LoRA 行没有清楚标为同一个高 rank 父 adapter，不能把主表差值直接解释为纯粹的事后编辑增益，也不能据此预测 8B／27B LLM 的收益。[p.6／Fig.4、8](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/src_2604.27796/2604.27796.pages.md:410)
- **TaDA** 支持多 adapter 干扰处理：语言任务报告均值 0.452，对 DARE-TIES 为 0.416；视觉消融中去掉 probe gating 从 85.9 降至 84.7。但这是两个 task/domain adapters 的合并，基线系数固定为 0.5/0.5，不能推出它胜过充分调优的搜索。[方法与 Tables 3、7](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/2606.05016.notes.md)
- **Sakana** 发现删掉 MetaMath-13B 第 30 层可使 MGSM-JA 从 8% 到 10%。这证明某些训练后结构操作可以有效，但操作对象是完整 transformer 层，不能直接当作 LoRA 层删除的证据。[p.13／Fig.5](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/src_2403.13187/2403.13187.pages.md:816)

关于 **LLM 是否优于同预算非 LLM 搜索**，现有证据不足以预设答案：

- **AutoQS：73.8521 对 73.4465** 是单个搜索种子的最佳 Macro3，差约 0.41。设计文件把 **0.94 写成“方差”是错误的**：它是两个搜索种子的 Macro7 分差。指标也不同，不能直接拿来判断上述 0.41 是否显著。[Table 2，p.8；Table 8，p.16](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-quantization/papers/src_2611.11975/2611.11975.pages.md:475)
- **Robot WSP：** 原模型 86.3%，编辑 C 为 92.0%，单次匹配范数的随机方向为 85.2%，editor-informed 字典控制为 90.5%。C 与字典直接比较 \(p=0.0614\)。随机方向控制不是随机搜索；字典复用了 editor 找到的对应关系，也没有计入发现这些对应关系的成本。[Table 3／§3.4，p.4](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/src_2611.09376/2611.09376.pages.md:207)
- **WSP-Scope：** Math 三条轨迹的增益为 +29.45、+5.00、+27.30，说明路径敏感；但其中一次中途改变了搜索目标，不能把三者当作严格同条件的 seed 方差。9B Math 的 dev 增益没有迁移到 held-out，也支持独立验收的必要性。[§3.4／附录 A、E](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/2611.09382.notes.md)

非 LLM 基线还需要忠实实现。LoraHub 的公式是分别混合两个 LoRA 因子后相乘，不等于直接混合 \(\Delta W\)；Sakana 的搜索与静态基线没有等预算；EvoGM 的 Qwen3-8B 成本表为 1,280 次候选评估，对 Model Swarm 的 900 次。它们提供重要算法基线，但原论文的排名不能直接移植为 B′ 的等预算结论。[LoraHub](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/2307.13269.notes.md)、[Sakana](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/2403.13187.notes.md)、[EvoGM](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/2605.29295.notes.md)

关于 **递归风险**：

R-Zero 的共享参数消融确实较早退化，但其机制是没有外部真值的自出题与伪标签训练；代码递归崩塌论文主要采用编译、长度等弱门。这些不能证明 B′ 的功能性外部验收必然失败。更直接的风险是：**目标任务提升，同时代码能力、旧任务或 editor 能力下降**。MASS 就同时报告了部分研究任务提升与编码 benchmark 下降。[R-Zero，Table 6](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/papers/2508.05004.notes.md)、[递归代码自训练](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2606.28438.notes.md)、[MASS，Table 7](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2610.12176.notes.md)

Anosognosia 和 WSP reading critique 约束的是未经工具辅助的自感知／读权重能力，不能外推否定明确提供谱、消融和 dev 反馈的编辑系统。[Anosognosia](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/rsi-model-edit/papers/2610.06174.notes.md)、[reading critique](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora-edit/papers/2611.08070.notes.md)

**5. 怀疑者最有力的攻击，以及让差异成立的最小实验**

最有力的质疑会是：

> “Self-Developing 已让 LLM 写合并代码，PARA／TaDA／SLoRA 已提供编辑操作，MASS 已研究后代 optimizer 的提升；这里是否只是把现成组件接起来？”

能够回应它的是**等预算优势与编辑造成的后继能力变化**。建议先用一个 8B 模型、三轮任务序列、至少三个训练种子做以下实验。

首先修正 checkpoint 身份。设计 §2.2 先指定 editor=\(M_t\)，再训练 \(A_{\rm raw}\)，没有明确让 editor 装载它。要落实 dossier 的 C1，应写成：

\[
R_t=M_t\oplus A_t^{\rm raw},\qquad
p_t\sim R_t(\cdot\mid \operatorname{diagnostics}(R_t)),\qquad
M_{t+1}=\operatorname{Apply}(p_t,R_t).
\]

即**先装载 raw LoRA，由这一权重状态生成编辑程序**。原设计对应位置在[第 49 行](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/SELF_LORA_EDIT_DESIGN.md:49)。

然后，在每个共同父 checkpoint 上保留四个分支：

| 分支 | 用途 |
|---|---|
| Parent | 训练前的 editor |
| Raw | 仅加入普通训练得到的 LoRA |
| LLM-edited | 从同一个 Raw 执行 LLM 编辑 |
| Non-LLM-edited | 从同一个 Raw 执行最佳非 LLM 编辑 |

对它们使用相同初始 prompt、历史条件、工具接口、固定 recipient 和 raw-adapter 测试库；任务对所有代均保持未见，反馈不回流主循环。分别报告：

\[
\Delta E_{\rm train}=E(\mathrm{Raw})-E(\mathrm{Parent}),
\qquad
\Delta E_{\rm edit}=E(\mathrm{LLM\!-\!edited})-E(\mathrm{Raw}).
\]

**固定 recipient 的 \(E_t\) 本身只控制了 recipient 变化，没有控制普通任务训练或上下文历史。** 若非 LLM-edited 分支同样提高 \(E\)，则还不能把收益独占归因于 LLM self-edit。

单轮比较至少包含 no-edit、最佳单 checkpoint、标量插值、PARA、TaDA／SLoRA、random、CMA-ES；主实验再补齐 LoraHub-style 和 EvoGM。共享动作空间，计入逐层消融、诊断前向、候选筛选的预算，并报告生成 token、GPU 时间。仅把候选数都设成 32，并不自动公平。

最后加入 **current-editor 接任与 frozen-editor** 的循环对照。使完整主张成立的结果应包括：

- 在相同预算下改善 target／retention 的取舍；
- \(\Delta E_{\rm edit}\) 在未见任务上稳定为正；
- 接任对后续编辑的实际收益超过 frozen editor。

若只有目标任务分数上涨，就应收窄为训练后 adapter 编辑结果。

还需修正两处协议：P0 不能查看最终 sealed test 后决定方向，应使用可弃用的 pilot holdout；dev2 若反复提供接受反馈，也会成为自适应选择的一部分，需预定使用次数或采用新分片。编辑历史 adapters 时则应替换相应栈成员，避免将已包含旧 adapter 的合并结果再次追加。[设计 P0](/Users/linghuazhang/Desktop/Project/RSIModelEdit/novelty/self-lora/SELF_LORA_EDIT_DESIGN.md:137)

**6. dossier 缺少的工作**

以下三篇已完成全文文本核验，可以加入正式证据表：

| 新增工作 | 应加入的原因 | 未覆盖的部分 |
|---|---|---|
| **SLoRA: Balancing Plasticity and Forgetting…，ACL 2026** | §4.4／Alg.1 明确包含任务 raw LoRA 训练后进行子空间去噪，再继续任务序列；是 C4 和 headroom 的直接近邻。 | 固定算法，无 LLM 编辑程序、editor 接任或技能测量。[全文，pp.5、17](https://aclanthology.org/2026.acl-long.247.pdf) |
| **Audit Before You Merge，2611.10684** | 连续、无梯度 LoRA 合并；§3／App.E 有逐 block 消融及 probe／selection／test 分离；App.M 有对历史子空间的正交投影。 | 无 LLM editor。三份数据划分适用于 block-probing 协议，不能归给它所有实验。[全文](https://arxiv.org/html/2611.10684v1) |
| **ReCoLoRA，2607.07719** | §3.5 用固定 SVD 规则对当前有效权重做 recursive consolidation，再训练下一任务。 | 其递归指权重巩固；不包含代码编辑者接任，也不能直接充当纯 post-hoc 编辑收益的证据。[全文](https://arxiv.org/html/2607.07719v1) |

以下仍为 **leads，不影响本轮 verdict**：

- [Skill Weaving／SkillWeave](https://arxiv.org/html/2605.22205v1)、[Superficial Self-Improved Reasoners Benefit from Model Merging／IMM](https://aclanthology.org/2025.emnlp-main.301.pdf)：已读方法、主要结果和部分附录，尚未完成全文。
- [AC/DC：Discovering Novel LLM Experts via Task-Capability Coevolution](https://arxiv.org/html/2604.14969v1)：已读正文及多数附录，仍有未覆盖附录与抽取缺口。
- [LoRA on a Budget](https://arxiv.org/abs/2610.36386)、[Not All Ranks Are Equal](https://arxiv.org/abs/2609.22237)：摘要级线索，值得补查函数感知压缩与 rank 分配基线。
- dossier 原有的 ABSTRACT-ONLY 工作仍保持该状态，没有作为否定新颖性的依据。

**7. 分数与判断的适用范围**

我的最终判断是 **PROCEED，6.5/10**，评价的是 dossier 的完整研究问题。它存在可核查的组合差异，竞争密集并不构成否决理由。

同模型家族的独立复核对**当前设计文本**给出 **CAUTION、6/10**，具体异议是 checkpoint 身份、Level 1 的参数选择性质，以及 raw-training confound。我采纳这些修正，但将它们视为可以在 pilot 前解决的设计缺口。若实际工作最终只包含 Level 1 单轮搜索和任务分数提升，CAUTION 会更合适。

📋 本次操作：完成文献审查、原文核验与独立复核；未修改文件。  
📊 当前状态：尚无 pilot 结果；当前目录不是 Git 仓库。只读权限使报告、review trace 和知识库记录无法落盘。  
💡 下一步：先明确 raw LoRA 装载后的 editor 身份，并把四分支配对对照写入 P0–P2，再启动 8B pilot。