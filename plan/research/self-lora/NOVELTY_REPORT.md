# Novelty Check Report — Topic B：模型自己给自己做 LoRA（递归）

日期：2026-11-14
- 审查：Claude 全文精读，加上 Codex gpt-6-astra xhigh 独立复核（session `01a1d7cc-a4a0-7b42-bd71-5c6fe5a7ca88`，trace 在 `.aris/traces/novelty-check/2026-11-14_run01/002-*`）。
- Codex 补充了 3 篇：MetaRSI、Self-HPO、HASE。这三篇已由我们自己全文精读、写好笔记，并抽查过关键数字，之后才纳入判断。

## Proposed Method
一个开源模型（计划用约 27B）不借助任何更大的外部模型，自己完成整套适配：
- 自己生成或筛选训练数据，自己决定 LoRA 的 rank、目标模块和超参，必要时自己写训练代码；另一条路线是直接生成 adapter 权重。
- 把 adapter 装到自己身上，提升下游任务表现。
- 微调后的模型当下一轮的编辑者，形成递归。

## Core Claims
- **B1 自定更新配方**：同一个模型既生成数据，又决定怎么更新自己（超参、LoRA 配置、训练代码），或直接生成 adapter。
- **B2 用在自己身上**：adapter 装回同一个模型，提升下游任务。
- **B3 递归接任**：微调后的模型当下一轮编辑者，测量"编辑能力"和"任务能力"是否一起提升，或刻画失败的方式。
- **B4 没有更强的外部模型**：代码执行器、固定 benchmark 这类非学习的验证器可以使用。

## Reading Coverage
笔记路径缩写：R 指 `novelty/rsi-model-edit/papers/`，L 指 `novelty/self-lora/papers/`。

- **全文精读 26 篇**：
  - 自编辑：SEAL 2506.10943、Search over Self-Edit 2601.14532、Self-Developing 2410.15639
  - 自产数据、LoRA 装回自身：Self-Specialization 2310.00160、Self-MoE 2406.12034、Transformer² 2501.06252
  - 直接生成 adapter 权重：Text-to-LoRA 2506.06105、Generative Adapter 2411.05877
  - 外部 agent 训练目标模型：SIA 2605.27276、PostTrainBench 2603.08640、FT-Dojo 2603.01712(L)、WHALE 2609.00196(L)、SOLAR 2605.20189(L)
  - 递归接任：MASS 2610.12176、AIDE² 2609.26457、WeightLoop 2611.07319、**MetaRSI 2609.06396(L)**、Self-HPO 2610.22791(L)、HASE 2607.03935(L)
  - 自博弈、自奖励：Absolute Zero 2505.03335(L)、R-Zero 2508.05004(L)、Self-Rewarding 2401.10020(L)
  - 测试时自训练：TT-SI 2510.07841(L)
  - 递归崩塌与恢复：When AI Reviews Its Own Code 2606.28438、SDFT recovery 2604.15794、Self-Data Distillation 2410.09982
- **仅读摘要，作为线索或背景，不参与判断**：
  - 原有：FST 2605.12484、Drag-and-Drop LLMs 2506.16406、STaR / ReST^EM / SPIN。
  - Codex 新列的线索：ThetaEvolve 2511.23473、Phantom Gains 2608.20290、Doc-to-LoRA 2602.15902、Macaron-V1 2608.09819、AI4AI-Bench 2608.20318。
- **存在性核验**：`verify_papers.py` 对 23 篇全部 PASS（`verified.json`）。MetaRSI、Self-HPO、HASE 三篇是从 arXiv 下载后全文读过的。

## Per-Paper Analysis（最相关的 8 篇详写，其余见笔记）

**1. MetaRSI / RSI²（2609.06396 v2，2026-09）：整体上最近的工作**
- **方法**
  - 只用一个模型 Qwen3.5-35B-A3B（MoE），由它担任所有角色：出题、盲解认证、诊断与提议、调度和 meta 改写（§5.1，p.21）。
  - Model-RSI 分支里，模型在一个**有界的 recipe 空间**中提议 LoRA 配置，包括目标模块、层、rank、scaling、lr、schedule、batch、序列长度和早停（Eq.18，p.14）。
  - 训练代码、后端、数据划分和评估器都是受保护的，模型不能改（Table 10，p.45）。
  - 每个候选都**从固定基座 θ0 用累积数据重新训练**（Eq.19，p.14）。
- **证据**
  - 单轮 4 个基准平均 +10.9，其中只跑 Model-RSI（LoRA）是 +3.9（Table 3，p.22）。
  - 连续 5 轮，有 meta 更新累计 +26.2，无 meta 更新 +20.7，单轮增益逐轮衰减（Fig.11、Table 4，p.24–25）。
- **问题**
  - 没有报告 LoRA 的 rank、lr 和数据量的具体数值。
  - 只给 5 个 seed 的均值，没有标准差。
  - 下一轮的提议者和作者是 θ0 还是新权重，论文没有逐个角色说清。
  - "改进者变强"（Fig.11c）比较的是文本层的调度和策略改写，**不是权重**；论文没有实验让不同代的权重去编辑同一个固定的被编辑模型。
  - 作者自己承认：闭环只能放大已有能力，只覆盖可自动验证的领域。
- **与 claim 的关系**
  - B1 大部分覆盖：同一模型既生成数据，又在有界空间里提议 LoRA recipe；但不写训练代码，也不生成权重。
  - B2 覆盖。
  - B3 部分覆盖，未证实：每轮从 θ0 重训，权重层面的作者能力提升没有被隔离。
  - B4 覆盖。

**2. SEAL（2506.10943，NeurIPS 2025）**
- **方法**：同一个模型生成 self-edit（训练数据，ARC 任务里还包括增强和超参选择），经固定的 LoRA SFT 得到 θ′，用 θ′ 的表现做奖励，外层用 ReST^EM 更新生成策略（Alg.1，p.3–4）。
- **证据**
  - ARC：Llama-3.2-1B 达到 72.5%，不用 RL 的自编辑是 20%（Table 1，p.7）。
  - SQuAD 知识注入：Qwen2.5-7B 从 32.7 到 47.0（Table 2，p.8）。
- **问题**
  - 内层适配出的 θ′ 用完就丢，不接任编辑者。
  - SQuAD 用 GPT-4.1 判分。
  - ARC 只有 8 个经过筛选的评估任务。
  - 知识注入时内层 LoRA 的超参是人工网格搜出来的。
- **与 claim 的关系**：B1 中"自生成数据"这一半覆盖；B2 覆盖；B3 结构不同；B4 在 ARC 分支成立，SQuAD 分支不成立。

**3. Search over Self-Edit Strategies（2601.14532）**
- **方法**：Qwen3-8B 自己写数据模板，并选择 rank、alpha、dropout、lr、epochs、grad_accum，外层跑 5 轮。
- **证据**：带 archive 的版本峰值 45.1%，**低于**人写的 Rewrite 基线 50.3%；模板多样性会坍缩（Table 2）。
- **问题**：高学习率候选会严重损坏模型；仍然是"θ′ 丢弃"的结构；用 Claude Haiku 判分。
- **与 claim 的关系**：B1 中"自选 LoRA 超参"已有先例，而且结果是负面的；B3 未覆盖，被作者列为 future work（§5.4，p.10）。

**4. MASS（2610.12176）**
- **方法**：Qwen3.6-27B 自己生成并自评研究任务轨迹，用 LoRA SFT 合并回自身（r=64，α=128，lr 3e-5，配方人工固定）。下一代担任执行者、workflow 优化者和评估者，共 2 轮。
- **证据**：用固定的 L0 执行时，L1 写的 workflow 胜率 0.64（Table 1，p.7）；test 集上 L1 对 L0 的胜率是 53.9%（Fig.3b）。
- **问题**：任务规格部分由 GPT-5.5 生成，最终评测用外部强模型；递归改进的是 workflow 能力，不是"设计权重更新"的能力。
- **与 claim 的关系**：B2 覆盖；**B3 的"27B 接任"结构已有先例**；B1 未覆盖（配方人工固定）；B4 不完全满足。

**5. Self-HPO（2610.22791）与 HASE（2607.03935）**
- **方法**
  - Self-HPO：Qwen3.5-35B-A3B 冻结后写 harness，再从当前 policy 继续做 GRPO（每轮 50 次更新），共 2 轮。
  - HASE：一个 Qwen3-8B 在 GRPO 中既解题又编辑白名单内的 harness，权重持续更新。
- **证据**
  - Self-HPO 把平均 resolve rate 从 57.8 提到 64.8（Table 1，p.5）。
  - HASE 在 S2D 上得 86.98，与使用 GPT-OSS-120B 的 Meta-Harness 持平（Table 2）。
- **问题**
  - 两篇改的都是 harness，不是训练配方；都是全参 RL，不是 LoRA。
  - Self-HPO 的 Table 3 中，P1 和 P0 作者的差距在标准差以内，编辑能力没有被隔离。
- **与 claim 的关系**：B3 的结构（更新后的模型继续当编辑者）已有先例；B1 未覆盖。

**6. Self-Rewarding LMs（2401.10020）**
- **方法**：Llama 2 70B 生成回答并给自己打分，迭代做 DPO，共 3 轮。
- **证据**：AlpacaEval 2.0 胜率从 9.94% 到 15.38% 再到 20.44%；判分准确率 65.1 → 78.7 → 80.4 → 81.7%（Table 1、Table 4）。
- **问题**：prompt 来自外部的 Llama 2-Chat；用 Claude 2 选 checkpoint；部分 NLP 基准逐轮下降；不是 LoRA。
- **与 claim 的关系**：B3 中"任务能力和判断能力一起提升"的最早先例；B1、B4 不满足。

**7. R-Zero（2508.05004）与 Absolute Zero（2505.03335）**
- **方法**
  - R-Zero：出题者和解题者从同一基座分开训练。
  - Absolute Zero：同一个 checkpoint 既出题又解题，用 Python 执行器验证。
  - 两篇的训练配方都由人固定，都是全参 RL。
- **证据**
  - R-Zero：4B 在第 3 轮达峰后下降；伪标签准确率 79% → 69% → 63%；共享参数的版本在第 1 轮达峰 47.31，之后跌到 43.89（Fig.3、Table 5、Table 6）。
  - Absolute Zero：7B-Coder 平均分 40.2 → 50.4（Table 1）。
- **与 claim 的关系**：给出了"同一权重自举"的正面证据和崩塌证据；B1 中的"决定怎么更新"都不覆盖。

**8. Self-Specialization（2310.00160）**
- **方法**：MPT-30B 自己生成约 5K 条领域数据，用 QLoRA 训练后装回自身。配置 r=8、α=16、lr 3e-4、3 epochs，均由人固定。
- **证据**：0-shot F1 从 25.15 到 34.57，再到 36.63；后续迭代基本饱和（Table 4）。
- **与 claim 的关系**：B2 覆盖；B3 弱覆盖；B1 只覆盖数据这一半。

**其余 18 篇**都不单独覆盖 B1+B3，具体结论见各自笔记：
- 直接生成 adapter（T2L、Generative Adapter）需要单独训练的 hypernetwork。
- FT-Dojo、PostTrainBench、SIA、WHALE、SOLAR 的编辑者是外部大模型（GPT-5.2、Claude 等）。SOLAR 还要 7B 模型帮忙格式化输出，而且只完成了 Level I。
- TT-SI 每个实例训一个临时 adapter，用完重置回 θ0，配方固定。
- 递归自训练崩塌那篇说明，用模型自评当筛选门会失效。

## Core Claims
1. **B1 自定配方**
   - 最近的工作：MetaRSI（同一模型在有界空间里提议 LoRA recipe）、Search over Self-Edit（同一模型选 6 个超参）。
   - 还没人做的：写训练代码，或在 recipe 空间之外改训练方法；自选超参本身已不新，而且 Search 显示它会输给人工配方。
2. **B2 用在自己身上**
   - 最近的工作：SEAL、Self-Specialization、MASS、MetaRSI。
   - 几乎不承担新意。
3. **B3 递归接任**
   - 最近的工作：MASS（27B，接任 workflow 角色）、Self-HPO 和 HASE（接任 harness 角色）、Self-Rewarding（接任评判角色）、MetaRSI（5 轮，但从 θ0 重训，改进者变强来自文本层）。
   - **还没人做的：权重变化本身是否让模型更会设计权重更新。** 具体做法是让不同代的模型当作者，编辑同一个固定的被编辑模型来隔离测量；以及从当前 θ_t 连续更新，而不是从 θ0 重训。
4. **B4 没有更强的外部模型**
   - 最近的工作：MetaRSI、Absolute Zero、HASE 的训练闭环。
   - 这是有价值的实验约束，但不能单独作为创新点。

## Closest Prior Work
| Paper | Year | Venue | Read | Overlap | Key Difference |
|---|---|---|---|---|---|
| MetaRSI 2609.06396 | 2026 | arXiv | FULL | 同一模型自产数据，提议 LoRA recipe，无外部模型，5 轮 | recipe 空间有界；从 θ0 重训；没有隔离"权重让作者变强" |
| SEAL 2506.10943 | 2025 | NeurIPS | FULL | 同一模型生成 self-edit 数据和超参，LoRA 更新自己 | θ′ 丢弃，不接任；SQuAD 用 GPT-4.1 判分 |
| Search over Self-Edit 2601.14532 | 2026 | arXiv | FULL | 自选 LoRA 6 个超参和数据模板 | 不如人写模板（45.1 vs 50.3）；无接任 |
| MASS 2610.12176 | 2026 | arXiv | FULL | 27B 自产数据，LoRA 合并回自身，下一代接任 | 配方固定；接任的是 workflow 角色；有外部强模型 |
| Self-HPO 2610.22791 | 2026 | arXiv | FULL | 更新后的 policy 接任 harness 编辑者 | 改 harness 不改配方；全参 RL |
| HASE 2607.03935 | 2026 | arXiv | FULL | 单模型持续更新并编辑 harness | 同上；8B |
| Self-Rewarding 2401.10020 | 2024 | ICML | FULL | 任务能力和判断能力一起提升 | 判分不是设计更新；有外部模型参与 |
| Self-Specialization 2310.00160 | 2024 | NAACL Findings | FULL | 30B 自产数据，QLoRA 装回自身 | 配方固定，迭代后饱和 |

## Overall Novelty Assessment
- **Score：5/10**
- **Recommendation：PROCEED WITH CAUTION。** Claude 与 Codex 独立判断一致，Codex 的独立子审查也是 CAUTION 5/10。
- **为什么判 CAUTION**：MetaRSI 已经做到"同一个模型自产数据、在有界空间里提议 LoRA recipe、不用外部模型、跑 5 轮"（§4.2.3、§5.1、Table 3–4）。所以"模型自己给自己做 LoRA，并递归循环"这个宽泛的说法**已经有人做了**。按规则这还不到 ABANDON：MetaRSI 没有从当前 θ_t 连续更新，也没有隔离"权重变化让作者变强"。
- **怎样才能撑起一篇论文**：把核心问题收窄为下面这个，任何一种结果都有价值：

  > 任务 LoRA 带来的权重变化，是否提升了同一模型在**新任务**上设计训练数据和更新配方的能力？这种迁移在什么条件下与任务专精冲突？

  - 观察到稳定的跨任务"作者能力迁移"，是正面结果。
  - 找到能被干预验证的冲突机制，比如任务分数上升但编辑分数下降，可以用配方交换和数据交换定位原因，也是有效结果。
  - 只给出一条递归上升曲线，贡献会很薄。
- **风险**：审稿人会引用 MetaRSI 和 Search 说"自主数据加 LoRA recipe 已有人做"；引用 MASS、Self-HPO、HASE 说"更新后接任不是首次，27B 也不新"；引用 T2L 和 Generative Adapter 说"直接生成低秩更新不新"。

## Suggested Positioning
> 在固定提示、记忆、验证器和被编辑目标的条件下，检验任务 LoRA 造成的权重变化能否提升同一 LLM 在新任务上设计训练数据与更新配方的能力，并刻画这种迁移对连续自适配、遗忘和成本的影响。

**核心测量**（Codex 建议，我同意）：让第 t 代模型当作者，编辑一个**固定**的被编辑模型 θ̄，测编辑后的提升：E_t = 期望[ Q(用 θ_t 设计的数据和配方去更新 θ̄) − Q(θ̄) ]。这样才能分清是"被编辑的模型更容易进步"，还是"作者更会设计更新"。

| 对照 | 回答的问题 |
|---|---|
| 更新后的作者 vs 冻结的初始作者 | 接任是否真的有贡献 |
| 模型自定配方 vs 调好的人工配方 / 同预算搜索 | 自主决策是否胜过普通调参（Search 的结果是负面的） |
| 从当前 θ_t 连续更新 vs 从 θ0 累积数据重训（MetaRSI 的做法） | 保留更新历史是带来收益，还是累积误差 |
| 数据固定，换不同代作者的配方 | 提升是否来自更新方法 |
| 配方固定，换不同代作者的数据 | 数据生成能力贡献多少 |
| 提示和 archive 固定 vs 允许演化 | 改善来自权重，还是外部记忆和策略文本 |

**其他要求**
- 用于开发反馈的评测和最终封存的测试要分开。
- 报告多个训练 seed 的完整多轮轨迹。
- 失败要有机制解释。
- 直接生成权重是另一条路线：T2L 需要专门训练 hypernetwork，不能据此推断普通 27B 模型能输出可靠的 adapter 数值，所以单独立项。

## 之前类似工作是怎么做的（方法 / 模型 / 数据集 / LoRA 配置 / 评测）
| 工作 | 模型 | 谁做什么 | LoRA / 更新 | 数据集与评测 |
|---|---|---|---|---|
| MetaRSI | Qwen3.5-35B-A3B | 同一模型出题、认证、提议 recipe、调度 | 有界 recipe 空间，从 θ0 重训，具体数值未报告 | 4 个可自动验证的基准（见笔记），5 seed 均值，5 轮 |
| SEAL | Llama-3.2-1B（ARC）、Qwen2.5-7B（SQuAD） | 同一模型生成数据（ARC 还选增强和超参） | ARC 内层 r=128；SQuAD 外层 r=64、α=128 | ARC 8 个任务；SQuAD 200 篇、974 题，GPT-4.1 判分 |
| Search over Self-Edit | Qwen3-8B | 自写模板，自选 6 个超参 | 基线 r=32、α=64、lr 1e-3 | SQuAD 无原文 QA，Claude Haiku 判分 |
| MASS | Qwen3.6-27B | 自产并自评轨迹 | r=64、α=128、lr 3e-5，合并后再量化为 FP8 | 自建研究任务 + 编码基准；6×H100 约 19 小时/轮 |
| Self-Specialization | MPT-30B、LLaMA2-7B | 自产约 5K 条数据 | QLoRA r=8、α=16、lr 3e-4、3 epochs | 生物医学 / 金融 NLP，0-shot F1 |
| TT-SI | Qwen2.5-1.5B / 7B | 每个实例自产 1 条数据 | r=8、α=16、lr 1e-4、5 epochs，用完重置 | 4 个工具调用基准，54.66 → 60.14 |
| Absolute Zero | Qwen2.5-7B / Coder | 同一模型出题解题，Python 验证 | 全参 RL，lr 1e-6、500 步 | HumanEval+、MBPP+、LCB、AIME、MATH500 等 |
| R-Zero | Qwen3 0.6B–8B | Challenger / Solver 分开训练 | 全参 GRPO，3–4 轮 | AMC、AIME、MATH-500、GSM8K、MMLU-Pro、SuperGPQA |
