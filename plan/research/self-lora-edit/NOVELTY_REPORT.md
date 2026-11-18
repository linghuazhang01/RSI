# Novelty Check Report：Topic B′，模型写代码编辑自己的 LoRA（递归）

- 日期：2026-11-14
- 审查方式：Claude 全文精读，Codex gpt-6-astra xhigh 独立复核。Codex session 为 `01a1d863-0ce9-7a13-b741-1de0e4c9c1bf`，trace 在 `.aris/traces/novelty-check/2026-11-14_run01/003-*`。
- Codex 另外补了 3 篇（SLoRA、Audit Before You Merge、ReCoLoRA）。这 3 篇由我们自己全文精读、写好笔记、抽查过数字后，才纳入判断。
- 方案文件：`novelty/self-lora/SELF_LORA_EDIT_DESIGN.md`，已按审查意见修正。

## Proposed Method
一个开源模型（pilot 用 8B，主实验约 27B）表示为"冻结基座 + adapter 栈"，每一轮按以下步骤进行：
1. 先用梯度训出当前任务的 raw LoRA。
2. 装上 raw LoRA 后，同一个模型调用诊断工具，写代码在权重空间编辑自己的 adapter，不再训练。可用操作包括逐层缩放、rank 截断、丢层、与历史 adapter 或训练中途 checkpoint 插值合并、TIES/DARE、正交化。
3. 固定的外部评测验收候选：dev 打分、dev2 验收、test 封存。
4. 通过验收的模型当下一轮编辑者。

全程不用任何更大的外部模型。

## Core Claims
- **C1 自编辑**：同一个 checkpoint 事后用代码编辑它**自己**训好的 LoRA，不需要再训练。
- **C2 胜过基线**：在相同评估预算下，胜过不编辑、人工合并规则、解析式截断，以及非 LLM 搜索。
- **C3 递归，并测量编辑能力**：用"固定的被编辑模型 + 新任务"测编辑能力 E_t。
- **C4 设定**：连续多任务学习，靠 adapter 空间的操作控制遗忘；没有更强的外部模型；外部评测固定，dev / dev2 / test 分开。

## Reading Coverage
笔记路径缩写：E = `novelty/self-lora-edit/papers/`，R = `novelty/rsi-model-edit/papers/`，L = `novelty/self-lora/papers/`，Q = `novelty/self-quantization/papers/`。

- **全文精读 29 篇**
  - **权重空间编程（WSP）一线（E）**：Sparks of WSP 博客、2611.09382 WSP-Scope、2611.04268、2611.08070、2611.09376；另有 WeightLoop 2611.07319（R）。
  - **LLM 写合并代码**：Self-Developing 2410.15639（R）。
  - **非 LLM 的合并 / 编辑（E）**：LoraHub 2307.13269、Sakana 2403.13187、EvoGM 2605.29295、TaDA 2606.05016、PARA 2604.27796。
  - **持续学习中的 LoRA 后处理（E，Codex 补充）**：SLoRA（ACL 2026）、Audit Before You Merge 2611.10684、ReCoLoRA 2607.07719。
  - **自定配方 / 接任**：MetaRSI 2609.06396（L）、Search over Self-Edit 2601.14532（R）、SEAL 2506.10943（R）、MASS 2610.12176（R）、Self-HPO 2610.22791（L）、HASE 2607.03935（L）。
  - **直接生成或选择 adapter**：Transformer² 2501.06252、Text-to-LoRA 2506.06105、Generative Adapter 2411.05877（均为 R）。
  - **LLM 与随机搜索的同预算对比**：AutoQS 2611.11975（Q）。
  - **风险**：R-Zero 2508.05004（L）、When AI Reviews Its Own Code 2606.28438（R）、SDFT recovery 2604.15794（R）、Anosognosia 2610.06174（R）。
- **仅读摘要（作为线索或背景，不参与判断）**：
  - 2611.11604（WSP 与模型检验）、TIES、DARE、O-LoRA、2611.17191；
  - Codex 列出的线索：SkillWeave 2605.22205、IMM（EMNLP 2025）、AC/DC 2604.14969、LoRA on a Budget 2610.36386、Not All Ranks Are Equal 2609.22237。
- **存在性核验**：全部 arXiv 论文都是从 arXiv 下载 PDF 后全文读完的，按规则视为已核验。SLoRA 来自 ACL Anthology 官方 PDF；WSP 博客来自作者主页。`verify_papers.py` 的批量核验见文末。

## Per-Paper Analysis（最相关的 9 篇详写，其余见笔记）

**1. Self-Developing（2410.15639）：C1 方向上最近的先例**
- **方法**：第一代 factory 是种子模型 M0 的克隆，它写 Python 合并函数，把 3 个外部 7B 模型的任务向量合并到 M0 上。factory 用 DPO 更新，共 3 轮（§2，p.3–4，Alg.1）。
- **证据**：GSM8K 从 70.1 提到 76.1，MATH 从 0.5 提到 8.5（Table 1，p.6）。
- **问题**：
  - 合并对象是外部现成模型，不是模型自己训的 adapter。
  - 合并结果不接任下一轮，算法始终作用于固定的 M0。
  - 每轮逐轮收益很小。
- **与 claim 的关系**："同权重模型写合并代码改善种子"这个交集已经有了，所以只说"同一个 checkpoint"不足以构成差异。C1 中"编辑自己训出的 adapter"和 C3 都没有覆盖。

**2. Sparks of Weight-Space Programming（MIT CSAIL 博客，2026-11-02）：WSP 一线的起点**
- **方法**：GPT-6 Astra 读写别的模型的基座权重。小网络直接写数值，Qwen3.5-0.8B 这类模型则在 Python kernel 里写代码改张量，禁止梯度训练。
- **证据**：从 Qwen3.5-0.8B 第 15、19 层 MLP 各 64 列中减掉一个"思考循环"方向，共改 131,072 个权重（0.016%）。用 50 次前向后，MATH-500 从 7.9% 升到 48.1%，GSM8K 从 13.8% 升到 67.1%（Exp.8）。
- **问题**：
  - 收益主要来自修好截断：撞 token 上限的比例从 93% 降到 33%。
  - 同样的做法在 Gemma 3 270M 上失败（6.8 → 7.0–7.7）。
  - 对照组没有对齐预算，作者在 FAQ 里自己承认了这一点。
  - "LLM 改自己的权重用于 RSI"只出现在展望里。
- **与 claim 的关系**：C1–C4 都没有覆盖。但"自身权重编辑"这个概念已经被公开提出，不能说是首创。

**3. Exploring WSP（2611.09382，2026-11-06）：流程上最像 B′**
- **方法**：GPT-6 Astra 用诊断工具（logprob、隐状态、激活）和代码编辑 Qwen3.5-0.8B/9B，做 rank-1 更新、缩放、置零 head 这类操作。每次搜索都从原始权重开始，编辑者也可以选择保留原权重。
- **证据**：0.8B 在 held-out 上 Math +29.45；9B Math 在 dev 上 +29.69，held-out 上 −0.35（Table 2，p.5）。三次 Math 搜索分别得 +29.45、+5.00、+27.30，结果依赖搜索路径（图 4、5）。
- **问题**：没有梯度、合并规则或随机 / 进化搜索作对照；dev 是自适应选出来的。
- **与 claim 的关系**：C1 和 C3 不覆盖；C2 只证明了比不编辑好。它反过来证明了 B′ 必须设 dev2 和封存的 test。

**4. 2611.09376（编辑机器人策略的权重）**
- **方法**：GPT-6 Astra 编辑 π0.5 的 token embedding 行，把同义词行插值到锚词。
- **证据**：改写指令的 held-out 成功率从 86.3% 升到 92.0%；同范数的随机方向只有 85.2%；按编辑者找到的词典直接替换达到 90.5%，和 LLM 编辑比 p=0.061，不显著（Table 1、3，p.3–4）。
- **问题**：只在 embedding 行这种有符号锚点的参数上成功，16 个会话里非语言类大多回退了。
- **与 claim 的关系**：C1–C3 不覆盖。它说明"LLM 编辑权重"的收益可能被简单规则解释掉。

**5. PARA（2604.27796）：解析式截断基线**
- **方法**：在全模型奇异值池上用一个全局阈值截断训好的 LoRA，自然得到非均匀 rank，必要时整层丢掉。不需要数据。
- **证据**：作者称压缩后的版本略好于父 adapter（图 3/4，p.6）。主表中视觉任务 PARA 88.83，LoRA 84.30（Table 1）。
- **问题**：主表的 LoRA 行像是原生 rank 4 而不是父 adapter，增益可能主要来自"用高 rank 训练"；只有单种子。
- **与 claim 的关系**：不覆盖 C1–C4，但说明单纯 SVD 截断就可能带来增益，LLM 编辑必须胜过它。

**6. TaDA（2606.05016）：人工诊断加编辑规则的强基线**
- **方法**：用 32 条探针算逐层激活范数比，做 sigmoid 门控，再按 SVD 分量的重叠度过滤后合并两个 LoRA。全部是固定的手工规则。
- **证据**：语言任务平均 0.452，DARE-TIES 0.416（Table 3，p.9）；探针门控贡献约 1.2pp，SVD 过滤约 0.5pp（Table 7）。
- **问题**：基线的合并系数固定为 0.5/0.5，没有搜索；疑似在测试集上调过参。
- **与 claim 的关系**：诊断和算子与 B′ 高度重合，相当于"人类专家写死的 B′"。必须作为基线。

**7. SLoRA（ACL 2026）：C4 方向上最近的工作**
- **方法**：持续学习中，每个任务训完 raw LoRA 后，按它和基座子空间的 Grassmann 相似度用闭式规则选截断 rank，做 SVD 去噪，不用旧数据（式 3–5，p.4）。
- **证据**：TRACE 8 任务上，Llama3.1-8B 从 52.81 升到 60.19，Qwen2.5-32B 从 56.51 升到 68.61；遗忘指标 AFR 从 9.31 降到 4.78（Table 1、2、5）。
- **问题**：
  - 15 任务长序列上不如 SD-LoRA（72.15 vs 74.37）。
  - 没有 dev/test 划分。
  - 正文给的提升百分比和表里的数字对不上。
- **与 claim 的关系**：C4 中"持续学习里对 raw LoRA 做免训练的事后处理来降遗忘"**已经被占了**。C1–C3 不覆盖，没有 LLM。

**8. Audit Before You Merge（2611.10684）**
- **方法**：依次合并 HuggingFace 上现成的 LoRA，比较 naive / TA / TIES / DARE / MagMax 和系数坐标搜索；用逐 block 消融做诊断，用正交投影合并。probe / α 选择 / test 三方分开。
- **证据**：
  - Llama-3.1-8B 上 GSM8K：不合并 0.707，naive 0.603，TA 0.705，TIES 0.712（Table 1，p.4）。
  - 负系数搜索把 probe 从 0.405 提到 0.490，但在完整测试集上只有 0.672（App.C）。
- **问题**：没有一个 adapter 单独使用时能提升 GSM8K，提升空间本来就小。
- **与 claim 的关系**：C1–C3 不覆盖。它说明人工规则已经很强，而在小 probe 上搜索容易过拟合。

**9. MASS（2610.12176）与 MetaRSI（2609.06396）：C3 方向上最近的工作**
- **MASS**：L1 写的 workflow 在固定的 L0 执行者上胜率 0.64（Table 1，p.7）。
- **MetaRSI**：在同一个系统 S1、同样预算下，比较原始改进者和 meta 改写后的改进者，提升分别是 +4.9 和 +7.2（Fig.11c，p.24）。
- **问题**：两者测的都是 workflow 或文本层的改进能力，不是"权重编辑程序"的能力。
- **与 claim 的关系**：不能泛泛地说"首次用固定被编辑对象测量改进者能力"。新意要落在"权重编辑能力，以及编辑本身对这种能力的贡献"上。

**其余 20 篇**的结论见各自笔记，这里只列与判断相关的要点：
- **LoraHub、Sakana、EvoGM 是必须复现的非 LLM 搜索基线。**
  - LoraHub 的公式是 (ΣwA)(ΣwB)，不等于直接对 ΔW 加权。
  - Sakana 发现逐层配置在数据少时没有收益。
  - EvoGM 的 Single Best（0.405）高于 EvoGM 本身（0.380）。
- **WeightLoop、2611.04268、2611.08070**：编辑者都是外部固定模型，或者没有实验。
- **AutoQS**：单种子下 LLM 73.85、随机 73.45；两个搜索种子的 Macro7 相差 0.94，这是分差，不是方差。
- **R-Zero、递归代码自训练**：属于递归风险。但 R-Zero 的机制是没有真值的伪标签，代码那篇用的是弱门，都不能证明 B′ 用功能性外部验收时必然失败。

## Core Claims
1. **C1 自编辑**
   - 最近的先例：Self-Developing（同权重模型写合并代码，作用于固定的 M0、合并外部模型），以及 WSP 一线（外部模型写代码改别的模型）。
   - 还没人做的：编辑对象是模型**自己训出来的** adapter，而且编辑产物直接接任后继编辑者。
2. **C2 胜过基线**
   - 最近的先例：LoraHub、Sakana、EvoGM（非 LLM 搜索），PARA、TaDA、SLoRA（固定规则）。
   - 还没人做的：在**相同动作空间、相同信息、相同完整预算**下，LLM 是否有稳定优势。这只能作为经验贡献，算子本身不算新。
3. **C3 递归与编辑能力**
   - 最近的先例：MASS、MetaRSI（固定被编辑对象测改进者能力，但测的是 workflow 或文本层）。
   - 还没人做的：**权重编辑程序的能力**，以及事后编辑本身（区别于普通任务训练）对后继编辑能力的贡献。
4. **C4 设定**
   - 最近的先例：SLoRA、Audit、ReCoLoRA、SEAL §5。
   - 还没人做的：这一条主要是研究场景和实验约束，单独的新颖性有限。

## Closest Prior Work
| Paper | Year | Venue | Read | Overlap | Key Difference |
|---|---|---|---|---|---|
| Self-Developing 2410.15639 | 2024 | arXiv | FULL | 同权重模型写合并代码，无外部模型 | 合并外部模型、作用于固定 M0、不接任 |
| WSP-Scope 2611.09382 | 2026 | arXiv | FULL [UNVERIFIED-by-helper] | 带诊断工具写代码编辑权重，dev 选优，可回退 | 外部 GPT-6 Astra 改别的模型；不是 adapter；不递归 |
| Sparks of WSP（博客） | 2026 | blog | FULL | LLM 写代码改权重，展望中提到自改 | 外部模型；单次会话；无预算对齐 |
| SLoRA | 2026 | ACL | FULL | 持续学习中对 raw LoRA 做事后去噪降遗忘 | 固定闭式规则，无 LLM 编辑者 |
| TaDA 2606.05016 | 2026 | arXiv | FULL | 逐层诊断 + SVD 过滤 + 缩放合并 LoRA | 手工规则；两个 adapter |
| PARA 2604.27796 | 2026 | arXiv | FULL | 训后 rank 截断、丢层 | 解析阈值规则 |
| EvoGM 2605.29295 | 2026 | ICML | FULL | 搜索 LoRA 专家的合并系数，多轮 | 学习出的 MLP 生成器；每轮重置；只有全局标量 |
| MASS 2610.12176 | 2026 | arXiv | FULL | 27B 后代接任并在固定执行者上测能力 | 测的是 workflow 能力，不是权重编辑 |
| MetaRSI 2609.06396 | 2026 | arXiv | FULL | 同预算比较原始和改进后的改进者 | 文本层 meta 改写；改训练配方，不做事后编辑 |

## Overall Novelty Assessment
- **Score：6.5/10。** Codex 判 6.5；Codex 的独立子审查针对设计**修正前**的文本判 CAUTION 6/10，具体异议有三条：编辑者身份、Level 1 只是参数选择、普通训练带来的混杂。我判 6.5，这三条已在设计文档中修正。
- **Recommendation：PROCEED。** 比上一版"自选 LoRA 配置"（CAUTION 5/10）高，原因是动作空间发生了实质变化：MetaRSI 是在训练**之前**选配置，B′ 是在训练**之后**由模型决定具体的参数变换。
- **Key differentiator**：模型对**自己训出的 adapter** 写事后编辑程序，编辑后的权重接任编辑者；再用固定的被编辑模型、新任务和相同预算，检验编辑本身是否带来了超出普通任务训练的编辑能力提升。
- **Risk**：
  1. **WSP 一线是直接竞速**。博客已在展望里提出"模型改自己的权重"，三周内已有 5 篇跟进。这是竞速，不构成否决，需要尽快出 pilot。
  2. **最强的审稿质疑会是"这只是把现成组件接起来"**：Self-Developing 写合并代码，PARA / TaDA / SLoRA 提供算子，MASS 研究后代的改进能力。能回应这一点的只有两件事：等预算下的优势，以及编辑造成的后继能力变化。
  3. **LLM 可能不比简单规则或搜索强**。AutoQS、2611.09376 和 Audit 都出现过这种情况。
  4. **如果最终只做到 Level 1 单轮搜索加任务分数提升**，贡献会退回 CAUTION 水平。

## Suggested Positioning
> 在无更强外部模型、也不专门训练编辑能力的主实验中，检验开放 LLM 对自身已装载的 LoRA 编写训练后编辑程序、并由编辑后的权重接任，是否能在固定被编辑对象、未见任务和相同预算下，带来超出普通任务训练的编辑能力提升。

这句话描述的是待检验的贡献，不是已经证明的结果。

**让差异站得住的最小实验**：8B 模型、3 轮任务序列、至少 3 个训练种子。
- **四个分支**：在每个父 checkpoint 上保留 Parent、Raw、LLM-edited、Non-LLM-edited，分别报告：
  - ΔE_train = E(Raw) − E(Parent)
  - ΔE_edit = E(LLM-edited) − E(Raw)
- **单轮基线**：不编辑、最好的单个 checkpoint、标量插值、PARA、TaDA、SLoRA、随机搜索、CMA-ES；主实验再加 LoraHub（按原公式实现）和 EvoGM。
- **预算对齐**：动作空间共享；逐层消融、诊断前向和候选筛选都计入预算；同时报告生成的 token 数和 GPU 时间。
- **递归对照**：当前编辑者接任 vs 冻结的编辑者。
- **完整主张成立的条件**：
  1. 相同预算下改善了目标任务与遗忘之间的取舍；
  2. ΔE_edit 在未见任务上稳定为正；
  3. 接任带来的收益超过冻结的编辑者。

  如果只有目标任务分数上涨，就把论文收窄为"训练后 adapter 编辑"。

## 之前类似工作是怎么做的
| 工作 | 谁编辑谁 | 操作 | 数据 / 评测 | 预算与对照 |
|---|---|---|---|---|
| Self-Developing | M0 的克隆写代码，合并 3 个外部 7B 模型到 M0 | 合并函数（任务向量） | GSM8K / MATH，dev 100 / 600 题 | 每轮 3000 个候选，对照 TA / TIES 网格 |
| WSP-Scope | GPT-6 Astra → Qwen3.5-0.8B / 9B | rank-1、缩放、置零 head | 7 个基准，dev + held-out | 50 次前向 + 300 次分析，只对照不编辑 |
| Sparks of WSP | GPT-6 Astra → 小网络 / Qwen3.5-0.8B | 写数值 / 写代码改张量 | MATH-500、GSM8K、IFEval | 10–50 次前向，对照不编辑和 PPO |
| LoraHub | CMA-ES → 20 个 FLAN LoRA | 每个模块一个全局系数 | BBH 27 个任务，5-shot | 40 次迭代，1 分钟 |
| Sakana | CMA-ES → 7B / 13B 全模型 | DARE-TIES 系数 + 层路径 | MGSM-JA、JP-LMEH | 1000 次 trial |
| EvoGM | MLP 生成器 → 10 个 Qwen2.5-1.5B LoRA | 全局 λ | Tulu 多任务，验证 200 / 测试 1000 | 1280 次评估，对照 TA / TIES / DARE / CMA / PSO |
| TaDA | 手工规则 → 2 个 LoRA | 探针门控 + SVD 过滤 | 6 个科学 QA、ViT | 一次约 31 分钟 |
| PARA | 解析规则 → 单个 LoRA | 全局奇异值阈值截断 | 视觉 / NLU / 常识 / 数学 | 不需要数据 |
| SLoRA | 闭式规则 → 每个任务的 raw LoRA | Grassmann 选 rank + SVD 去噪 | TRACE 8 任务，7B–32B | 无搜索，测遗忘 AFR |

## 附：存在性核验状态
- `verify_papers.py` 跑了两次，arXiv API 都超时：25 个 ID 中 9 个 verified，16 个 verify_pending，0 个被判为不存在（`verified.json`、`verified_retry.json`）。
- 这 16 个里有 12 个在今天另外两个查新目录的 verified.json 中已经 PASS。
- 剩下 4 篇是 WSP 跟进论文：2611.09382、2611.04268、2611.08070、2611.09376。它们的 PDF 都是从 arxiv.org/pdf/<id> 直接下载、全文读完的。按 citation-discipline 规则，这算存在性已核验。不过它们没有经过 helper 的二次确认，所以在表中标为 [UNVERIFIED-by-helper]。
