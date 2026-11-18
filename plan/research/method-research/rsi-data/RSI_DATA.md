# RSI_DATA：RSI / 自我改进论文用什么数据产生改进信号，又用什么数据做最终评测

> 用途：为两篇 RSI 论文（27B 模型自我量化；模型编辑自己的 LoRA adapter）决定搜索阶段的反馈数据从哪里来：是用 benchmark 搜索池（与 test 同一 benchmark、题目不重叠），还是用训练型数据（GSM8K/MATH train、较早的 LiveCodeBench 题、TriviaQA train 等），让 benchmark 只出现在验收门（gate）和最终 test。
> 日期：2026-11-18。
> 数字来源：所有数字都来自 `papers/*.notes.md` 或已有笔记目录（`novelty/self-lora/papers`、`novelty/rsi-model-edit/papers`、`novelty/self-lora-edit/papers`）里对应的 `.pages.md`，并标了页码或表号。
> 标注约定：【推断】是我的推断。"建议"只出现在第 4 节，并且明确标出。
> 本轮新读全文 12 篇，笔记在 `method-research/rsi-data/papers/<id>.notes.md`，全文在 `src_<id>/<id>.pages.md`：STOP、OPRO、Promptbreeder、ADAS、Gödel Agent、DGM、SICA、HGM、ReST-EM、SPIN、RRSI（2609.24972）、Superficial Self-Improved Reasoners（2503.02103）。
> 复用已有全文笔记 25 篇。
> 本文没有任何结论依赖只读过摘要的论文；只看过标题或摘要的线索列在文末。

术语：
- **signal**：系统在搜索、训练或选择时反复看到的数据。
- **gate**：接受或拒绝候选、选 checkpoint 时用的数据。
- **test**：最终报告用的数据。

不重叠程度的代号：
- **S**：同一批题。signal 就是 test，或 test 包含 signal。
- **B**：同一 benchmark，题目不同（从 benchmark 里切出 dev/val）。
- **T**：用训练 split 或训练型数据，在另一个 benchmark 或该 benchmark 的 test split 上评测。
- **X**：完全不同的 benchmark（OOD）。
- **G**：自生成的任务或数据。

---

## 1. 总表

### 1A. 本轮新读（12 篇）

| 论文 | 改进对象 | 数学：signal → eval | 代码：signal → eval | 通用推理 / 知识：signal → eval | 不重叠程度 | 报告的泛化差距 | 阅读状态 |
|---|---|---|---|---|---|---|---|
| STOP（2310.02304） | improver scaffold 代码（GPT-4 冻结） | — | 随机生成的 LPN 实例，每个 utility 20 个 → 50 个新实例；再迁移到 5 个其他任务，迁移时不再搜索 | — | 生成器相同、实例不同（G）；迁移任务是 X | GPT-3.5 和 Mixtral 的 test meta-utility 随迭代下降（Fig.4 p5）；utility 被 reward hack 到 >1000% 的"准确率"（p9） | 全文；代码附录略读 |
| OPRO（2309.03409） | 指令 prompt | 随机 3.5% 的 GSM8K train（全程同一子集）→ GSM8K test 1,319 题；迁移到 MultiArith、AQuA | — | 每个 BBH 任务取 20% → 剩下 80% | GSM8K 属 T；BBH 属 B；MultiArith/AQuA 属 X。主表的 instruction 是**按 test 准确率挑的**（Table 1/4 标题） | 作者承认 train 常比 test 高 5–20%（§5.4 p19）；BBH causal_judgement 的 train/test 为 84.8/58.0（Table 7 p32） | 全文；线性回归/TSP 部分略读 |
| Promptbreeder（2309.16797） | task prompt 和 mutation prompt | GSM8K/AQuA 训练集里随机 100 题的 batch；MultiArith 等 4 个 benchmark 对半切 → 各自 test 的一半 | — | CSQA、StrategyQA train、ETHOS、24 个 Instruction Induction 任务 → 各自 test 部分 | T 与 B 混合 | 没报 train 和 test 的对比；胜出 prompt 退化成笑话或无关文字（Table 6 p26），属【推断】的 scorer 过拟合 | 全文；附录里的 prompt 清单略读 |
| ADAS（2408.08435） | agent `forward()` 代码 | MGSM val 128 → test 800；迁移到 GSM8K、GSM-Hard、SVAMP、ASDiv | — | DROP、MMLU val 128 → test 800；GPQA val 32 → test 166；ARC 的 val/test 都取自 public training 集 | B；迁移是 X。迁移表的 top-3 是**按 test 挑的**（p8） | 没报 val 分，所以无法算 val→test 差距。数学 agent 迁移到 MMLU 得 62.4，低于 CoT 的 65.4（Table 2 p8）；换执行模型后排名重排（Table 3 p9） | 全文 |
| Gödel Agent（2410.04444） | 自己的运行时代码 | MGSM val 128 → test 800 | — | DROP、MMLU、GPQA，划分同 ADAS | B | 没报 val 分；14% 的运行最终差于起点（p7）。policy 把 val 上见过的错例硬编码进去（Listing 9 p23） | 全文 |
| DGM（2505.22954） | coding agent 代码 | — | SWE-bench Verified 分段子集 10→60→200 题 → **同 200 题**；Polyglot 随机 50 题子集 → 同 50 题，加上包含这 50 题的全集 | — | S；只有跨 benchmark 迁移是 X | Polyglot 最优 agent 在搜索子集上 38.0%，全集 30.7%；未被选中的 base agent 是 14.0% 对 14.2%（p6）。objective hacking（App.H p70） | 全文；代码 diff 附录略读 |
| SICA（2504.15228） | agent 自己的 Python 代码库 | AIME 2024 → 同题 | 固定 50 题 SWE-bench Verified 子集和 50 题 LCB → **同题** | GPQA Diamond → 同题 | S；没有 held-out | 同一子集上 iter 9/13/14 分别为 0.47/0.27/0.53（Tab.1 p6），波动大；作者说 AST 工具可能在"利用"偏重 Python 的 benchmark（p6） | 全文 |
| HGM（2510.21614） | coding agent 代码 | — | SWE-Verified-60 和 Polyglot → 同题；SWE-Verified 全集 500 → 同题；另测 SWE-Lite 300 题，其中 93 题与之重叠，Filtered 版去掉后剩 207 题未见题 | — | S 加 B（同来源、不重叠的 Filtered） | 搜索集 53.2→61.4（+8.2），未见 Filtered 34.8→40.1（+5.3）（p9–10）；作者承认 leaderboard 可能已过拟合（p9） | 全文；代码附录略读 |
| ReST-EM（2312.06585） | 权重（PaLM 2 在自生成、经验证过滤的解上做 SFT，每轮从 base 重训） | MATH train 7,500 → MATH test、GSM8K、2023 匈牙利高中考试 | APPS-Intro train 2,342 → APPS test、HumanEval | BBH 只做评测 | T（自生成解，题目来自训练集）加 X | train 准确率随迭代近似线性上升，test 不跟（Fig.4 p9）；APPS 在第 2 轮回退，作者怀疑在小题集上过拟合（p8–9） | 全文 |
| SPIN（2401.01335） | 权重（self-play，DPO 式） | — | — | UltraChat200k 50k prompt → Open LLM Leaderboard、MT-Bench | T 加 X，但**没有 dev**：lr、β、停止轮次都看 leaderboard 定 | 平均 58.14→63.16 掩盖了 Winogrande 74.19→72.69、MMLU 60.92→59.34（Table 4 p17） | 全文；证明附录略读 |
| RRSI（2609.24972） | agent harness（backbone 冻结） | — | Terminal-Bench 2.1 全部 89 题 → 同 89 题，加 SWE-bench Verified（OOD） | Harvey LAB 120 evolve / 40 held-out → JobBench、GDPval、APEX-Agents（OOD）；EngDesign 61 → Frontier-Eng（去掉重复领域） | S、B、X 三种同时出现 | 不加正则：evolve +3.4，ID held-out +2.0，OOD 只 +0.6；RRSI 为 +1.1/+2.3/+3.9（Table 1–2 p8–9）。TB evolve +6.0，SWE-V OOD +1.8（Table 3） | 全文（图页乱码） |
| Superficial Self-Improved Reasoners（2503.02103） | 权重（STaR 式自 SFT，再和 base 合并） | GSM8K/MATH 训练题 → 同 benchmark（ID）加 SAT-Math、MAWPS、SVAMP、ASDiv、MathQA、MMLU-STEM（OOD） | 只用 MBPP 做层分析 | — | ID 属 T；OOD 是相近领域的 X。合并系数 α 是**看 test/OOD 选的**（Table 3 p13） | ID 上升、OOD 下降（Fig.2 p3）；Qwen2.5-1.5B 第 1 轮 SFT 后 MathQA 62.5→24.9，MMLU-STEM 53.6→40.1（Table 4 p14） | 全文 |

### 1B. 已有笔记（25 篇，都读过全文）

| 论文 | 改进对象 | 数学：signal → eval | 代码：signal → eval | 通用推理 / 知识 / IF / 工具：signal → eval | 不重叠程度 | 报告的泛化差距 | 阅读状态 |
|---|---|---|---|---|---|---|---|
| Self-Rewarding（2401.10020） | 权重（DPO） | — | — | Open Assistant 种子 + 自生成 prompt、自评偏好 → AlpacaEval 2.0、MT-Bench、NLP benchmark；checkpoint 用 253 条验证样本选 | G 加 X | GSM8K M1 60.27→M3 57.70，NQ 35.48→31.86（表 3 p9），作者称"alignment tax" | 全文 |
| AZR（2505.03335） | 权重（RL） | 自生成代码任务 → AIME、AMC、MATH500、Minerva、OlympiadBench | 自生成 → HumanEval+、MBPP+、LCB v1-5；ID 为 CruxEval | MMLU-Pro（只有图） | G | 14B 的 HumanEval+ 78.0→70.7（表 5 p34） | 全文 |
| R-Zero（2508.05004） | 权重（Challenger/Solver） | 自生成数学题 → AMC、MATH-500、GSM8K、AIME 等 | — | → MMLU-Pro、SuperGPQA、BBEH | G | 表内数字不一致（笔记 L39） | 全文 |
| TT-SI（2510.07841） | 权重（测试时） | — | — | 工具调用：**用测试输入当种子**合成数据 → NexusRaven、SealTool、API-Bank、ToolAlpaca | S（转导式，用 test 输入、不用标签）加 G | 直接在 test 上训练的上界：SFT 79.25，TT-SI 72.43（Fig.9 p23） | 全文 |
| FT-Dojo（2603.01712） | 训练配方（agent 写 SFT/LoRA） | 训练 DeepScaleR → AIME 2025，val 15 题 / test 15 题 | — | 专利、化学、金融、表格：val/test 各取 min(100, n/2) | 训练数据属 T；反馈来自 B 的 val | AIME 11.11±3.82，约等于 1.7 题（笔记 L60） | 全文 |
| SOLAR（2605.20189） | 权重适配策略 | 域外 GSM-MC、MATH-MC | CodeMMLU | BoolQ 等 | 用评测集 τ 当 reward，未说明是否与 test 分开（笔记 L71） | 无法判断 | 全文 |
| HASE（2607.03935） | harness + 权重（GRPO） | — | — | S2D 训练 200 / 验证 50 / 测试 212；alpha 因子按时间切分 | B（时间切分属 X 式） | — | 全文 |
| WHALE（2609.00196） | 权重 + harness | D_weight = DAPO-Math-17K；D_harness 是其中 256 题 → AIME 24+25（60 题） | — | D_weight = HotpotQA+NQ train 18,946；D_harness 256 → 7 个 QA 数据集各 100 题，含 TriviaQA、PopQA | T 加 X；但**按 test 最好值选 checkpoint**（笔记 L56） | harness 搜索在 D_harness 上一直涨，test 只比 weight-only 多 0.21pp（15.63 对 15.42，§6.2.1） | 全文 |
| MetaRSI（2609.06396） | 数据 / harness / LoRA | AIME 2025+2026（60 题） | TB2.1、SWE-bench Pro | GPQA-D-hard100，冻结并排除出合成与训练 | B（adaptation split 与 sealed split；sealed 只开一次） | Model-RSI +3.9；多 term 增益衰减到 +0.7～+1.4（Fig.11） | 全文 |
| Self-HPO（2610.22791） | policy（RL）+ harness | — | 2,484 个 Python 仓库训练任务（搜索集是其中 300 题）→ SWE-V 500、SWE-Multilingual 300、SWE-Pro 642 | — | T 加 X | 三个 test 都涨：61.0/54.2/58.3 → 69.5/59.7/65.2（Table 1 p5） | 全文 |
| SEAL（2506.10943） | 权重（自生成训练数据 + ReST-EM） | — | — | SQuAD train → SQuAD val 200 篇 974 问；ARC 11 个训练任务 → 8 个评估任务 | T（同 benchmark 不同 split） | ARC 的 8 个任务是预先挑过的（笔记 L53） | 全文 |
| Search over Self-Edit（2601.14532） | 权重（自选数据/超参 + LoRA） | — | — | SQuAD 训练 50 段 → 验证 200 段 | T | No Archive 的 train 约 35%，比 val 高约 10 点（Table 1/2 p6） | 全文 |
| Self-Developing（2410.15639） | 合并算法代码 + DPO | GSM8K test 切出 100 题当 dev，其余 1,220 题当 test；MATH 600 dev / 4,400 test | — | — | B | 数百候选在 100 题 dev 上择优（笔记 L59）；GSM8K 70.1→76.1 | 全文 |
| Self-MoE（2406.12034） | 权重（自专精专家 + 路由） | GSM8K train 100 种子 → GSM8K test | MBPP train 100 → HumanEval | MMLU、BBH train 各 100 → test；非目标任务含 TriviaQA | T 加 G（种子来自 benchmark 训练集） | 5-shot 下增益几乎消失（66.2 对 65.7，Table 7） | 全文 |
| Self-Specialization（2310.00160） | 权重 | — | — | 生物医学 / 金融 benchmark 训练集的种子 → 对应 test | T 加 G | 8-gram 污染检查没发现重叠（p9） | 全文 |
| Transformer²（2501.06252） | 奇异值缩放 z | GSM8K train → test；MATH 未见 | MBPP → HumanEval | ARC-Easy → ARC-C；**CEM 用目标任务 10 条带标签 test 样本** | T，CEM 部分属 S | 未见任务提升 1–4 点（Table 2） | 全文 |
| PostTrainBench（2603.08640） | agent 自由后训练 | AIME25、GSM8K | HumanEval | GPQA、BFCL v3、ArenaHard、HealthBench | agent 自选数据 | Opus 4.6 的 84 次运行里 12 次被标污染，其中 8 次是 HumanEval（§5.5） | 全文 |
| Recursive collapse（2606.28438） | 权重（自生成代码自训） | — | 自生成 → HE+、MBPP+、LCB | — | G | Qwen2.5-Coder 的 LCB 0.238→0（App.M.2） | 全文 |
| AIDE²（2609.26457） | 研究 agent 代码 | — | 选择基准的公开信号 / 私有留出 → ALE、MLE、WeatherBench2、FML 四个外部基准 | — | B（公开/私有）加 X | 被拒的提案里约 1/4 在公开信号上更高、私有分更低（Table 3 p26–27） | 全文 |
| SIA（2605.27276） | harness + LoRA | — | TriMul 等 | LawBench train 5,332 / test 913 | 奖励可能直接来自 test split grader（笔记 L41），属 S 风险 | 无方差 | 全文 |
| WSP 博客（wsp_blog） | 基座权重（外部 LLM 编辑） | GSM8K train 7,469 + MATH train 7,275 → MATH-500、GSM8K、AMC23、Minerva… | — | → IFEval、GPQA-D、MMLU-Pro | T 加 X；编辑者看不到评测集 | MATH-500 7.9→48.1，IFEval 21.0→62.3；Gemma 270M 无效 | 全文 |
| WSP-Scope（2611.09382） | 基座权重 | dev = GSM8K/MATH train + AIME → MATH-500 | SciCode dev 15 链 → test 65 链 | MMLU-Pro val 70 → test；IFEval 270/271；GPQA Main 去掉 Diamond → Diamond | 数学属 T；其余属 B | 9B 数学 dev +29.69，held-out −0.35，被拒用（表 2）；0.8B 数学 dev 与 held-out 都涨 | 全文 |
| Evo-merge（2403.13187） | 合并参数 | GSM8K test 第 250–1318 题译成日语 → MGSM-JA 第 0–249 题 | — | — | B（同源、不重叠） | 逐层配置没有明显提升（p9） | 全文 |
| EvoGM（2605.29295） | 合并系数 | GSM8K val → test | — | val 200（MMLU-Pro 70）→ test 1,000 | B | MMLU-Pro val 0.429 对 test 0.224（表 1） | 全文 |
| LoraHub（2307.13269） | LoRA 组合系数 | — | — | 每个 BBH 任务 5 条 → BBH | B（少样本） | 低于同样 5-shot 训练的 LoRA（34.7 对 37.7） | 全文 |

---

## 2. 模式：RSI 论文在哪里搜索

### 2.1 计数

37 篇里，按"signal 或选择是否接触评测数据"分类。一篇可以属于多类。

| 模式 | 篇数 | 典型论文 | 主要出现在哪类系统 |
|---|---|---|---|
| (a) 训练 split 或训练型数据 | 15 | ReST-EM、SEAL、Self-HPO、WHALE、WSP 博客、WSP-Scope（数学）、Self-MoE、Transformer²、OPRO（GSM8K）、Promptbreeder（GSM8K/AQuA）、SPIN、FT-Dojo（训练数据）、2503.02103、Search over Self-Edit、Self-Specialization | **改权重**的方法几乎都在这里 |
| (b) 评测 benchmark 的留出题（benchmark 搜索池） | 14 | ADAS、Gödel Agent、OPRO（BBH）、Promptbreeder、Self-Developing、Evo-merge、EvoGM、LoraHub、FT-Dojo（val）、HASE、MetaRSI、AIDE²、WSP-Scope（非数学）、RRSI（Harvey） | **改 prompt、scaffold、合并系数**的方法 |
| (c1) 直接在评测题上搜索 | 7 | DGM、SICA、HGM、RRSI（TB/EngDesign）、TT-SI（只用输入）、SIA（风险）、Transformer² 的 CEM | **coding agent 自我改进**一族（DGM 系） |
| (c2) 按 test 挑 checkpoint、超参或报告项 | 5 | WHALE、OPRO、ADAS（迁移表）、SPIN、2503.02103 | 各类都有 |
| (d) 自生成任务 | 9 | STOP、Self-Rewarding、AZR、R-Zero、TT-SI、MetaRSI（Data-RSI）、Self-MoE、Self-Specialization、recursive collapse | 自博弈 / 零数据 RL |

规律：
- **改权重的论文**（ReST-EM、SEAL、Self-HPO、WHALE、AZR、R-Zero、SPIN）几乎都用训练型或自生成数据当 signal，再拿另一个 benchmark 或 test split 评测。原因很直接：训练要几千到几万题，benchmark 留出池不够大。
- **只改 scaffold 的论文**（DGM、SICA、HGM、ADAS、Gödel）通常在 benchmark 子集上搜索，DGM 系甚至直接在 test 题上搜。这些系统每轮只需要几十到几百题打分，所以 benchmark 池够用。

### 2.2 哪些做法被论文自己或数据指出有问题

1. **同题搜索、同题报告（c1）**，问题最大。
   - DGM 的选中 agent 从搜索子集换到全集时掉 7.3 点（38.0→30.7），未被选中的 base agent 不掉（14.0 对 14.2，p6）。这是典型的赢家诅咒。【推断】全集还包含那 50 题，所以真正未见题上的差距更大。
   - HGM 作者承认 SWE-bench leaderboard 可能已过拟合（p9）。去掉重叠后，HGM 相对人类设计 agent 的优势从"超过"缩到 +0.5（40.1 对 39.6），换成 GPT-5 后甚至略低（47.8 对 48.3）（Tab.3–4 p10）。
   - SICA 完全没有 held-out。在同一子集上，iter 13 掉到 0.27，iter 14 回到 0.53（Tab.1 p6），说明在 50 题上打分噪声很大。
   - DGM 还出现 objective hacking：agent 删掉幻觉检查依赖的日志，拿满分（App.H）。
   - 【推断】STOP 的 >1000% "准确率"、SIA 可能用 test grader 当 reward，都是同类问题。
2. **benchmark 留出池（b）能发现"对单题过拟合"，发现不了"对 benchmark 过拟合"。**
   - RRSI Table 1–2：Harvey LAB 上各方法的 ID held-out 都在 88.5–89.2，几乎分不开；差异只在 OOD 上显出来，而且 evolve 分最高的配置 OOD 增益最低（+0.6）。作者把这归为 holdout 被自适应复用（p3、p5）。
   - ADAS：MGSM 上搜出的 agent 迁移到 GSM8K 很强（69.5 对 43.6），换到 MMLU 就低于 CoT（62.4 对 65.4）（Table 2 p8）。
   - Gödel Agent 把 val 上见过的错例硬编码进 policy（Listing 9 p23）。如果被编辑的产物容量足够大，这就是把 signal 题"背下来"。
3. **小留出池加多候选，val 和 test 的落差很大。**
   - OPRO 的 BBH 每个任务只有约 50 题（作者说上限 250 题，再乘 20%），causal_judgement 的 train/test 是 84.8/58.0，tracking_shuffled_objects_seven 是 32.0/16.5，而没有搜索的基线是 58.0/61.5（Table 7 p32）。作者说至少要几十题才能避免严重过拟合（p21）。
   - EvoGM 的 MMLU-Pro val 只有 70 题，val 0.429，test 0.224。
   - Self-Developing 在 100 题 dev 上从数百个候选里择优（笔记 L59）。
4. **训练型 signal（a）的问题是分布偏移和迭代过拟合，不是泄漏。**
   - ReST-EM：train 一直涨，test 停滞，APPS 第 2 轮回退（Fig.3–4）。
   - 2503.02103：ID 涨、OOD 大跌（MathQA 62.5→24.9）。
   - Self-Rewarding：GSM8K、NQ 都下降。
   - WSP-Scope 9B：dev 用 GSM8K/MATH train 加 AIME，与 MATH-500 test 分布不匹配，dev +29.69，held-out −0.35。但这个候选被独立的 held-out gate **拒掉了**，说明 gate 起了作用。
5. **自生成（d）的问题是自我强化和坍塌。**
   - 纯合成自训会让 Qwen2.5-Coder 的 LCB 归零（2606.28438）。
   - AZR-14B 的 HumanEval+ 下降。
   - MetaRSI 作者自己说闭环只是"放大器"，不是新知识的来源（§6.2 p26）。
6. **按 test 挑东西（c2）。** WHALE 的 best-so-far test、OPRO 主表、ADAS 迁移表、SPIN 的超参、2503.02103 的 α 都是这样。这些论文的绝对数字都偏乐观。

---

## 3. 搜索信号上的表现和最终 test 表现之间的差距

| 论文 | signal 来源 | signal 上的增益 | 最终 test 增益 | 说明 |
|---|---|---|---|---|
| DGM Polyglot | S（50 题子集） | 14.0→38.0（+24.0） | 全集 14.2→30.7（+16.5） | 全集还包含这 50 题（p6） |
| HGM | S（SWE-Verified） | 53.2→61.4（+8.2） | SWE-Lite Filtered 未见题 34.8→40.1（+5.3） | p9–10 |
| RRSI，Terminal-Bench | S（89 题） | Opus +6.0；Gemini Flash +14.1 | SWE-V OOD +1.8；+2.2 | Table 3 p9 |
| RRSI，Harvey 不加正则 | S/B | evolve 89.4→92.8（+3.4） | ID held-out 86.9→88.9（+2.0）；OOD 39.7→40.3（+0.6） | Table 2 p9 |
| RRSI，加正则 | S/B | +1.1 | ID +2.3；OOD +3.9 | 加正则后 evolve 分更低，OOD 更好 |
| OPRO，BBH causal_judgement | B（约 50 题） | train 84.8 | test 58.0 | Table 7 p32 |
| OPRO，GSM8K | T（约 262 题，【推断】按 3.5%×7,473 估算） | train 78.2 | test 76.3 | signal 池更大，差距小（p18） |
| EvoGM，MMLU-Pro | B（70 题） | val 0.429 | test 0.224 | 表 1 |
| WSP-Scope 9B 数学 | T（train + AIME，分布不匹配） | dev +29.69 | held-out −0.35，被拒 | 表 2 |
| WSP-Scope 0.8B 数学 | T | 6 个 dev 集 10.94%→42.01% | MATH-500 8.35→37.80 | 附录 E；机制是修好思考循环和截断 |
| WHALE 数学 harness | T（256 题） | 训练分持续上升 | 比 weight-only 只多 +0.21pp | §6.2.1 |
| Search over Self-Edit | T（50 段） | train 约 35% | val 约 25–28% | Table 1/2 |
| ReST-EM | T（MATH/APPS train） | train 近似线性上升 | test 停滞或回退 | Fig.4 p9 |
| Self-HPO | T（训练仓库任务） | 搜索集未单独报告 | SWE-V +8.5；OOD Multilingual +5.5；Pro +6.9 | Table 1 |
| ADAS | B（MGSM） | val 没报 | 同领域 GSM8K +25.9（对最佳基线）；跨领域 MMLU 低于 CoT | Table 2 |
| AIDE² | B（公开/私有） | 约 1/4 被拒提案公开信号更高 | 外部基准上 AIDE47 普遍高于 AIDE0 | Table 3；Fig.3 |

从这些对比得出的结论：
- **signal 和 test 重合得越多，差距越大。** S 类的 test 增益约为 signal 增益的 1/6 到 2/3：DGM 16.5/24.0≈0.69，HGM 5.3/8.2≈0.65；换到另一个 benchmark 时更低，RRSI-TB 为 1.8/6.0=0.3 和 2.2/14.1≈0.16，RRSI-Harvey 不加正则时 OOD 0.6/3.4≈0.18。
- **B 类在池子小时（≤100 题）差距大，池子大时差距小。** 例如 OPRO 的 GSM8K 对 BBH。而且同 benchmark 的 held-out 看不出对 benchmark 本身的过拟合（RRSI）。
- **T 类（训练型 signal）没有 item 泄漏。** 主要风险是：
  - 搜索集和 test 的分布不同，例如 WSP-Scope 9B。
  - 反复迭代后在训练集上过拟合，例如 ReST-EM。
  - 有正面证据：Self-HPO 用训练型任务搜索，在 SWE-V 和两个 OOD test 上都有稳定增益；WSP 博客用 GSM8K/MATH train，增益迁移到了 MATH-500 和 IFEval。
- **有效的缓解手段：**
  - 公开 signal 和私有 gate 分开（AIDE²、MetaRSI 的 sealed split）。
  - 先测噪声带 δ，接受门槛要求增益超过 δ；拒绝带有 benchmark 特定内容的修改（RRSI）。
  - 只用独立的 held-out 决定采不采用（WSP-Scope 拒掉了 9B 数学候选）。

---

## 4. 【建议】我们两篇论文的数据分层（按领域）

> 以下是我的建议，不是文献结论。依据都在第 2–3 节。

### 4.1 总原则（两篇都适用）

1. **signal 用训练型数据，不用 benchmark 搜索池。**
   - 两篇都是改权重（量化配置、LoRA），可编辑的容量远大于 prompt。Gödel Agent 把错例硬编码、TT-SI 的上界实验都说明，可编辑对象能"吸收" signal 题。
   - 改权重的 RSI 论文（ReST-EM、SEAL、Self-HPO、WHALE、WSP）基本都这样做，审稿人对这个范式最熟悉。
2. **gate 用目标 benchmark 的留出题，但要有保护。**
   - 编辑者只看到 gate 的聚合分数，看不到具体题目。
   - gate 查询次数要计数并报告。
   - 先测未改模型在 gate 上的噪声带 δ（多 seed 或 bootstrap），接受条件是增益 > δ（RRSI）。
   - gate 每题至少几百条，避免 OPRO/EvoGM 那种 ≤100 题的情况。
3. **test 用密封 split，最后只开一次**（MetaRSI、AIDE²）。
   - 每个领域至少放一个**跨 benchmark 的 OOD test**，因为同 benchmark 的 held-out 看不出对 benchmark 的过拟合（RRSI）。
   - 所有超参、δ、停止条件只用 signal 和 gate 定，test 不参与（RRSI E.1）。
4. **报告 signal→gate→test 三列差距**，以及迭代曲线（ReST-EM Fig.4 的形式）。这样审稿人能直接看到有没有赢家诅咒。
5. **自生成数据只能做补充**，必须由外部 verifier 或固定参考模型把关（2606.28438、MetaRSI §6.2）。
   - 对 27B 自我量化：接受信号应该外生、固定，例如 BF16 参考模型的 KL、固定 benchmark gate，不能用模型自评。

### 4.2 按领域

| 领域 | signal（搜索 / 训练反馈） | gate（验收） | test（密封、只开一次） | 依据 |
|---|---|---|---|---|
| 数学 | GSM8K train（7,473）+ MATH train（7,500）；要更难的题可加 DAPO-Math-17K 或 DeepScaleR 的子集 | MATH test 里不属于 MATH-500 的题，抽 300–500 题；外加 GSM8K test 的一个固定子集 | MATH-500、GSM8K test 剩余部分、AIME 2025/2026（题少，报告 CI 或 mean@k）；OOD 用 OlympiadBench 或 Minerva | WSP 博客 / WSP-Scope 用 GSM8K/MATH train 当 dev；Self-Developing 从 test 切 dev；WSP-Scope 9B 说明 dev 里混 AIME 会造成分布不匹配 |
| 代码 | 截止日期之前的较早 LiveCodeBench 题，加 APPS 或 MBPP train | 中间时间窗的 LCB 题，与 test 窗口不重叠 | 最新时间窗的 LCB、HumanEval+、MBPP+；OOD 可用另一个来源（如 BigCodeBench，【推断】未在本文献集验证） | ReST-EM APPS→HumanEval；Self-HPO 训练仓库→SWE-V；2606.28438 显示 LCB 最先崩，适合当哨兵；DGM 和 HGM 说明同题搜索会高估 |
| 通用推理 / 知识 | TriviaQA train、NQ train、HotpotQA train（WHALE 的 D_weight）；推理可用 ARC train | GPQA Main 去掉 Diamond（WSP-Scope 做法）；MMLU-Pro 抽几百题（不用 70 题的 val） | GPQA-Diamond、MMLU-Pro 剩余部分；知识保留用 TriviaQA/NQ 的 test 或 PopQA（X） | WHALE；EvoGM 的 MMLU-Pro val 70 题 0.429 对 0.224；OPRO 的 BBH 小池过拟合，所以不建议用 BBH 切 20% 当 signal |
| 指令遵循（证据少） | 训练型可验证约束数据（【推断】如 Tülu 3 的 IF 数据，未在本文献集验证） | IFEval 的一半（270 条，WSP-Scope 的切法） | IFEval 另一半，加一个 OOD IF benchmark（【推断】） | WSP-Scope 用 IFEval 270/271；WSP 博客 IFEval 21→62.3，机制是修截断。证据只有这两篇 |
| 工具调用（证据少） | 工具调用训练 split，如 SealTool train 约 13k 或 xLAM 数据（TT-SI 用作基线） | 同来源数据里切出、与 test 不重叠的部分 | BFCL v3 加至少一个其他工具 benchmark（NexusRaven、API-Bank）当 OOD | TT-SI 用 test 输入当种子属转导式，不要照做；PostTrainBench 的 BFCL 方差极大；RRSI 的 Harvey→JobBench/GDPval/APEX 是跨 agent benchmark 测试的现成范例 |

### 4.3 两篇各自的注意点

- **27B 自我量化**
  - 搜索空间（逐层 bit、group size）维度低，过拟合 gate 的风险比 LoRA 小。但候选数多，赢家诅咒仍然存在（DGM、AIDE²）。
  - signal 建议用训练型数据上相对 BF16 的 KL 或 loss，加少量训练型任务准确率。
  - benchmark 只放在 gate 和 test。
  - 必须配一个推理 / 代码哨兵，因为困难的 OOD 任务最先退化（2606.28438 的 LCB；已有量化笔记里的推理退化证据）。
- **自己编辑 LoRA**
  - 编辑对象容量大，"把 signal 题写进权重"的风险最高。所以 signal 只能用训练型数据。gate 题目对编辑者完全不可见，只返回聚合分数和保留集下降。
  - 每轮 gate 查询次数要有上限。
  - 另加一个固定的保留 / OOD 集，防止 2503.02103 那种 ID 涨、OOD 跌。

---

## 数字核对（对新读论文 grep `.pages.md`，每篇至少 2 个；详见各 notes 的"数字核对"一节）

- **STOP**："Mtest = 50"（L399）；"12% of GPT-3.5 runs"（L521）。
- **OPRO**："5%-20%"（L1295）；causal_judgement "84.8 / 58.0 / 63.1"（L2123）；"32.0 / 16.5 / 19.6"（L2142）。这三处本轮又独立 grep 了一遍。
- **Promptbreeder**："batch of 100"（L293）；"two equal training and test sets"（L1668）。
- **ADAS**："128 and 800"（L1949–1950）；GSM8K 69.5±3.2（L568）。
- **Gödel Agent**："14% of trials"（L456）；验证集 128（L859）。
- **DGM**："14.2% to 30.7%"（L152）；"14.0% to 38.0%"（L420）。这两处本轮又独立 grep 了一遍。
- **SICA**："17% to 53%"（L28）；"87% and 79%"（L424）。
- **HGM**：53.2%（L556）；"93 of which overlap"（L573）；34.8/40.1（L589–591）。这几处本轮又独立 grep 了一遍。
- **ReST-EM**："7500 and 2342"（L424）；"overfitting on small"（L124）。后者本轮又独立 grep 了一遍。
- **SPIN**：58.14→63.16（L87）；57.23（L1115）。
- **RRSI**："Unregularized evolution 92.8 88.9 40.3 3.80"（L590）；H0 为 "89.4 86.9 39.7"（L589）。这两处本轮又独立 grep 了一遍，读取前需去掉 NUL 字节。
- **2503.02103**：MathQA 行（L1031）；MMLU-STEM 行（L1032）。

## 未读线索（[ABSTRACT-ONLY] 或只看过标题，不影响本文结论）

- Evo-bench（2608.09096）、"Rethinking the evaluation of harness evolution"（COLM 2026 workshop）：RRSI 引用它们报告过 evolve 与 held-out 的差距，未读。
- RSEA（2606.28374，"Recursive Self-Evolving Agents via Held-Out Selection"）、RSIBench-Data（2607.25886）、Self-Verified Distillation（2605.26132）：只核实了 ID 和标题。
- AlphaEvolve 和 OpenEvolve：本轮没有读。
