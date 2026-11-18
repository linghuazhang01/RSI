# CODE_RSI_AGENTS：代码领域里改 agent / scaffold / 程序 / 训练配方的自我改进论文，搜索或训练用什么数据，评测用什么

> 目的：为我们选**代码训练集**（既当执行判分的反馈信号，也当梯度训练数据）。问题：代码领域的自我改进 / RSI 工作在什么数据上搜索或训练，在什么上评测；评测是否包含我们打算留作 held-out 的 HumanEval(+) / MBPP(+) / LiveCodeBench。
> 日期：2026-11-19。范围：改代码、prompt、工具、环境、自身 agent 或训练配方的系统（可能同时改权重）。纯权重自训练（ReST-EM、CURE、AZR、ZeroCoder、RLAR、RL Tango 等）由另一份 `CODE_RSI_WEIGHTS.md` 覆盖，这里不重复。
> 来源：本轮新读全文 13 篇（笔记在 `papers/<id>.notes.md`，全文在 `papers/src_<id>/`）；复用已有全文笔记 8 篇（`../rsi-data/papers/`、`../../self-lora/papers/`、`../../rsi-model-edit/papers/`）。每个数字都标了页码或表号；原文没写的标 not stated。【推断】是我的推断，【建议】是我的建议。
> 没有只读摘要的论文进入表格或结论。

代号（同 `../rsi-data/RSI_DATA.md`）：
- **S**：signal 与 test 是同一批题（或 test 包含 signal）。
- **B**：同一 benchmark，题目不同。
- **T**：训练 split 或训练型数据，评测在 test split 或别的 benchmark。
- **X**：完全不同的 benchmark。
- **G**：自生成任务/测试，或测试时逐题自我改进（没有独立搜索集）。

---

## 1. 总表（21 篇，代码领域）

### 1A. 本轮新读（13 篇）

| 论文 | 改进对象 | 搜索/训练数据（名称、规模、子集） | 不相交 | 评测集 | 评测是否含 HE(+)/MBPP(+)/LCB | 出处 |
|---|---|---|---|---|---|---|
| **EvoTrainer**（2606.03108） | trainer agent（Sonnet 4.6）改 4B/9B 策略模型的 RL 训练配方和诊断 harness；人类批准启动与晋升 | SWE：**swe-rebench-v6 train 8,622 实例**；Coding：**TACO-verified 11,897 题**；Math：BigMath-Hard 6,429 | 训练 T；但版本取舍用的 "validation" 数值与报告的测试数值一致，【推断】选择层面是 S | swe-rebench-v6 test 中 77 个 held-out Python 实例；**LCB-v6 中 175 道较新的 AtCoder ABC 题**（排除较早题，日期窗口 not stated）；AIME24/25、CNMO24 | **LCB：是**（v6，175 题）；HE/MBPP：否 | p6 §4.1；Table 3 p7；D.1–D.2 p14；Table 8 p14；E.1 p16 |
| **EnvHarness**（2608.19880） | 同骨干的 EnvRigger 改造静态环境（状态/规则/观测），再抽 skill 给冻结 agent | **SWE-bench Lite 100 题**作训练种子；**SWE-smith 只是基线**（同种子、同模型生成任务），不是训练数据 | B | SWE-bench Verified 中不在 Lite 里的 407 题（另有 ALFWorld、WebArena、OfficeQA、SpreadsheetBench） | 否 | p8 §4.1；Table 3 p9；Table 7 p25；E.2 p25 |
| Live-SWE-agent（2511.13646） | 解题过程中在线自建工具（改自己的 scaffold） | 无；逐题测试时 | G | SWE-bench Verified 500；SWE-Bench Pro 731；Verified-60 | 否 | p5；Table 1–2 p6；Table 3 p7 |
| SE-Agent（2508.02085） | 逐题轨迹池的修订/重组/精炼（规则 + LLM 评审打分） | 无；逐题测试时 | G | SWE-bench Verified 500 | 否 | 式 8–9 p6；Table 1 p7 |
| AlphaEvolve（2506.13131） | 进化改写程序（算法、启发式、内核分块） | 每个问题自带评估器；内核：真实输入形状**对半切**训练/评估；调度：历史负载训练、较新负载测试 | 数学构造 S；内核 B；调度时间切分 | 50+ 数学问题、Borg 调度、TPU 内核 | 否 | p7；p10；p14；p16 |
| Reflexion（2303.11366） | 测试时文字反思记忆 | 对每道测试题**自生成最多 6 个单测**，不看真实测试 | G | HumanEval-PY/RS、MBPP-PY/RS、LeetcodeHardGym 40 题 | **HE、MBPP 原版：是**；LCB：否 | p7 §4.3；Table 1 p7；Table 2 p8 |
| Self-Debugging（2304.05128） | 测试时自解释 + 执行反馈迭代修复 | MBPP test 500 题，题面带第 1 个单测，另 2 个隐藏 | G | MBPP 500、TransCoder 560、Spider dev | **MBPP 原版：是**；HE、LCB：否 | p5–6；MBPP 表 p9 |
| AFlow（2410.10762） | MCTS 改写代码表示的 workflow | 每个数据集**随机 20% 作 validation**，再只留高方差题；HumanEval、MBPP 用全集再切 | B | 同数据集剩余 80% | **HE、MBPP 原版：是**（搜索与测试同源）；LCB：否 | p6；p7 §5.1；Table 1 p8；p18 |
| RSI Arena（2610.33624） | 10 个前沿 agent 各自 post-train 同一基座（Nemotron 3.5 30B-A3B），自选数据、自写训练代码 | agent 自选公共数据（Tülu 3、Nemotron post-training 与 code 数据等）；规则禁止用 benchmark 测试数据；代码数据具体来源未汇总，not stated | T/X；最终评测对 agent 不可见 | 20 个 benchmark，代码为 **LCB v6 175 题** 和 SciCode | **LCB：是**（v6，175 题）；HE 只出现在部分 agent 的内部自检 | p3；Table 1 p5；p27；Table 6 p28；附录 I p32–33 |
| NeoHorse-1（2609.08183） | 路由 harness 的部署轨迹 → SFT 课程 + 固定 teacher 的 OPD；按评测缺陷调下一轮配比 | 私有部署轨迹（规模只以监督 token 数给出）；对照用 Toucan | 训练 G/X；数据消融的 dev 套件就是报告的 LCB/HE 等，S 风险 | HumanEval、LCB v6、BFCL v4、τ2、IFEval 等 10 个 | **HE 原版、LCB v6：是**（日期窗口 not stated） | p8 §3.5；Table 1 p14；Table 2 p15；Table 3 p17 |
| Ouroboros（2608.08311） | 评审过的 commit 改自己的 harness 源码 | 日常任务和人类交互（161 天部署） | X（冻结种子跑评测；种子开发期是否看过 benchmark 失败 not stated） | Terminal-Bench 2.1、SWE-bench Pro、OSWorld-Verified、CL-Bench、GAIA | 否 | p1；p3；p4；Table 2 p6 |
| SpecForge（2610.15708） | 改行为规格文档，再据此生成代码 | DevEval+ 20 模块 247 实体；搜索只用 LLM 生成的测试（期望输出来自参考实现） | S（同一批模块）+ G 信号；最终用隐藏真实测试 | DevEval+；跨语言翻译、新功能实现 | 否 | p6 §4.1；Table 1 p7；Table 2 p27；附录 H p33 |
| ShinkaEvolve（2509.19349） | AlphaEvolve 式程序进化 | ALE-Bench LITE 10 题，适应度 = **公开测试用例**分数 | B（公开测试 vs 私有测试） | 同题私有测试集 | 否 | p8 §4.3；附录 B.3 p23 |

### 1B. 复用已有笔记（8 篇，全文已读）

| 论文 | 改进对象 | 搜索/训练数据 | 不相交 | 评测集 | 评测是否含 HE(+)/MBPP(+)/LCB | 出处 |
|---|---|---|---|---|---|---|
| STOP（2310.02304） | improver scaffold 代码 | 随机生成的 LPN 实例，每个 utility 20 个 | G（同生成器、新实例） | 50 个新 LPN 实例 + 5 个迁移任务 | 否 | p6；Fig.4 p5；App.K p45 |
| DGM（2505.22954） | coding agent 代码库 | **SWE-bench Verified 子集 10→60→200**；**Polyglot 随机 50 题**；诊断 prompt 里放入官方私有测试 patch | S | 同 200 题；Polyglot 50 题与全集 | 否 | p6；Fig.2 p7；App.C.3 p29–30 |
| SICA（2504.15228） | agent 自己的代码库 | **SWE-bench Verified 固定 50 题** + **LCB 随机 50 题** + 两个合成 benchmark | S | 同 50+50 题 | **LCB：是**（同一 50 题，版本/窗口 not stated） | p5–6；Table 1 p6 |
| HGM（2510.21614） | coding agent 代码库 | SWE-Verified-60、Polyglot、SWE-Verified 全集 500 | S；SWE-Lite Filtered 207 题为 B | 同题；SWE-Lite 300（去重后 207） | 否 | p7；p9–10；Table 2–4 |
| RRSI（2609.24972） | agent harness | **Terminal-Bench 2.1 全部 89 题** | S；SWE-bench Verified 为 X | TB2.1 同题；SWE-bench Verified（OOD） | 否 | p6；Table 3 p9 |
| Self-HPO（2610.22791） | harness（policy 自己写）+ 全参 RL | **2,484 个 Python 仓库任务（SWE-rebench、SWE-rebench V2、Scale-SWE）**，搜索集为其中随机 300；proposer 能看参考修复 | T | SWE-bench Verified 500、Multilingual 300、Pro 642 | 否 | p4；Table 1 p5；App.A p16；App.D p19 |
| MetaRSI（2609.06396） | 数据 / harness / LoRA 配方 | TB2.1 与 SWE-bench Pro 的 adaptation split（大小 not stated） | B（sealed split 只开一次） | TB2.1 89、SWE-bench Pro 731 的 sealed split；GPQA、AIME | 否 | p21；Table 3 p22 |
| PostTrainBench（2603.08640） | 前沿 agent 自主 post-train 小模型 | agent 自选，如示例轨迹里用 Magicoder OSS-Instruct 并自写去污染过滤；**agent 手里的 evaluate.py 直接在 HumanEval 上打分**（"humaneval (150 samples)"） | 训练 T；内部选择对着目标 benchmark，S 风险 | HumanEval（另有 AIME25、GSM8K、GPQA、BFCL、ArenaHard、HealthBench） | **HE 原版：是** | p3；Fig.3 p4；§5.5 p8–9 |

不进表的：ADAS、Gödel Agent、FT-Dojo、AIDE²（已有笔记，不含代码 benchmark 或只有 ML 研究任务）；Evo-Bench（2608.09096，本轮读了 §1–3，全是 Search/Office/General agent 任务，见 `papers/2608.09096.notes.md`）。

### 1C. 两个指定论文的核实结果

- **EvoTrainer 存在（2606.03108）**。全文确认：SWE 训练用 swe-rebench-v6 train split 的 **8,622 个实例**，评测用对应 test split 的 **77 个 held-out Python 实例**（p6）。另外 Coding 域训练 **TACO-verified 11,897 题**，评测 **LCB-v6 的 175 道 AtCoder ABC 题**（p6）。注意：77 题评测集也是版本选择用的 "validation"（附录数值与 Table 3 一致，【推断】）。
- **EnvHarness 存在（2608.19880）**。全文**不支持**"它用 SWE-smith 训练"：SWE 训练种子是 **SWE-bench Lite 100 题**，评测是 **Verified 中不在 Lite 的 407 题**；SWE-smith 是**同种子、同模型、同环境数量的对照基线**（p8；Table 7 p25；E.2 p25）。结果 52.58 对 SWE-smith 50.12（Table 3 p9）。

---

## 2. 搜索/训练数据出现频次（21 篇）

一篇可以计入多类。

| 数据（作为搜索或训练信号） | 篇数 | 论文 |
|---|---|---|
| 测试时逐题、自生成测试或工具（没有独立训练集） | 6 | Reflexion、Self-Debugging、Live-SWE-agent、SE-Agent、SpecForge（DevEval+）、STOP（LPN 生成器） |
| SWE-bench Verified 子集（就是评测题本身） | 3 | DGM、SICA、HGM |
| SWE-rebench 系列（train split / 任务池） | 2 | Self-HPO（SWE-rebench + V2 + Scale-SWE，共 2,484）、EvoTrainer（swe-rebench-v6 train 8,622） |
| Polyglot（随机子集） | 2 | DGM（50）、HGM |
| Terminal-Bench 2.1 | 2 | RRSI（全部 89）、MetaRSI（adaptation split） |
| agent 自选公共训练数据 | 2 | PostTrainBench、RSI Arena |
| 部署 / 人类交互轨迹 | 2 | NeoHorse-1、Ouroboros |
| 单问题评估器（优化问题本身） | 2 | AlphaEvolve、ShinkaEvolve（ALE-Bench LITE 公开测试） |
| LiveCodeBench（作为搜索集） | 1 | SICA（随机 50 题，同题报告） |
| TACO-verified | 1 | EvoTrainer（11,897） |
| HumanEval / MBPP 的 20% validation 切分 | 1 | AFlow |
| SWE-bench Lite（作训练种子） | 1 | EnvHarness（100 题） |
| SWE-bench Pro（adaptation split） | 1 | MetaRSI |
| SWE-smith | 0（作训练）；1（作基线） | EnvHarness 的对照 |
| SWE-Gym、R2E-Gym、APPS、CodeContests | 0 | 本集合里没有一篇把它们当搜索/训练数据 |

规律：
- **只改 scaffold 的 coding agent**（DGM、SICA、HGM、RRSI）几乎都直接在评测 benchmark 的题上搜索。
- **同时改权重的**（Self-HPO、EvoTrainer）用大规模训练任务池（SWE-rebench、TACO），因为 RL 需要几千题。
- **测试时方法**（Reflexion、Self-Debugging、Live-SWE-agent、SE-Agent）没有训练集，对"选训练集"没有直接帮助。

---

## 3. 评测是否含 HE(+) / MBPP(+) / LCB

### 3.1 按论文
- **含 LCB（4 篇）**：SICA（随机 50 题，且是搜索题本身，版本 not stated）、EvoTrainer（v6，175 道较新 AtCoder ABC 题）、RSI Arena（v6，175 题）、NeoHorse-1（v6，规模与窗口 not stated）。
- **含 HumanEval 原版（4 篇）**：Reflexion、AFlow、PostTrainBench、NeoHorse-1。
- **含 MBPP 原版（3 篇）**：Reflexion、Self-Debugging（test 500）、AFlow。
- **含 HumanEval+ 或 MBPP+（EvalPlus）：0 篇。** 本集合里没有一篇 agent / scaffold / 配方类自我改进论文用 HE+ 或 MBPP+ 评测。用 HE+/MBPP+ 的 RSI 论文都在权重自训练一侧（例如 `RSI_DATA.md` 中的 AZR、2606.28438），归 `CODE_RSI_WEIGHTS.md`。
- 合计 8 篇至少含其一；其余 13 篇（STOP、DGM、HGM、RRSI、Self-HPO、MetaRSI、EnvHarness、Live-SWE-agent、SE-Agent、AlphaEvolve、Ouroboros、SpecForge、ShinkaEvolve）**都不含**这三者。代码评测以 SWE-bench 系、Terminal-Bench、Polyglot 为主。

### 3.2 按训练/搜索数据：哪些 RSI 论文用了它，并且评测含 HE+/MBPP+/LCB

| 训练/搜索数据 | 用了它的论文 | 其中评测含 HE(+)/MBPP(+)/LCB 的 |
|---|---|---|
| swe-rebench（train split / 任务池） | Self-HPO、EvoTrainer | **EvoTrainer 评测含 LCB-v6**，但 LCB 对应的训练集是 TACO-verified，swe-rebench 只对应 SWE 评测。Self-HPO 不含 |
| TACO-verified | EvoTrainer | **EvoTrainer：TACO-verified 11,897 → LCB-v6 175 题（排除较早题），Avg@8 46.71→51.29，与人工 RL 基线 50.71 差异不显著**（Table 3 p7；p6） |
| SWE-smith | 无（EnvHarness 只作基线） | — |
| SWE-Gym | 无 | — |
| R2E-Gym | 无 | — |
| APPS | 无（本集合内）。权重侧先例：ReST-EM 用 APPS-Intro train 2,342 → APPS test + HumanEval（见 `RSI_DATA.md`） | — |
| CodeContests | 无 | — |
| LCB 较早题作训练 | 无。SICA 是在 LCB 50 题上**搜索并在同题报告**（S） | SICA（LCB，同题） |
| Polyglot | DGM、HGM | 无 |
| SWE-bench Verified 子集 | DGM、SICA、HGM | SICA（LCB，同题） |
| SWE-bench Lite | EnvHarness | 无 |
| Terminal-Bench 2.1 | RRSI、MetaRSI | 无 |
| HumanEval/MBPP 20% val | AFlow | **AFlow：HE、MBPP 原版的另外 80%**（B） |
| agent 自选公共数据 | PostTrainBench、RSI Arena | **PostTrainBench：HE 原版**（内部 evaluate.py 直接打 HumanEval）；**RSI Arena：LCB v6 175** |
| 部署轨迹 | NeoHorse-1、Ouroboros | **NeoHorse-1：HE 原版 + LCB v6**（dev 套件与测试同为这些 benchmark） |
| 测试题上自生成测试 | Reflexion、Self-Debugging | **Reflexion：HE、MBPP；Self-Debugging：MBPP** |

结论：在本集合里，"在独立代码题库上训练 → 在 LCB 上 held-out 评测"的直接先例**只有 EvoTrainer（TACO-verified → LCB-v6）**，以及选择数据不受控的 RSI Arena（agent 自选 → LCB v6）。没有任何一篇以 HE+/MBPP+ 为评测。

---

## 4. 值得注意的发现

1. **在测试题上搜索会虚高（再次确认）。**
   - DGM：Polyglot 搜索子集 38.0，全集 30.7；未选中的 base 14.0 对 14.2（`RSI_DATA.md`，p6）。
   - HGM：搜索集 +8.2，未见的 SWE-Lite Filtered +5.3（p9–10）。
   - RRSI：TB2.1 evolve +6.0，SWE-V OOD +1.8（Table 3 p9）。
   - SICA：同一 50 题上第 9/13/14 代为 0.47/0.27/0.53（Table 1 p6），LCB 只从 0.65 到 0.71。
2. **"held-out" 有时也是选择集。** EvoTrainer 的 77 题 SWE 测试集与 LCB 175 题，数值上就是版本取舍用的 validation（D.1 p14、Table 8 p14 与 Table 3 p7 一致），【推断】其增益含选择偏差；单 seed（p9）。NeoHorse-1 的数据消融 dev 套件就是报告的 LCB/HE（p17）。PostTrainBench 的 agent 直接在 HumanEval 上自评（Fig.3 p4）。
3. **执行环境泄漏答案。**
   - EvoTrainer：模型通过 git 历史看到参考 patch，validation 48.80，清理后 31.04（p14 D.2）。
   - DGM 诊断 prompt 含私有测试 patch（App.C.3）；Self-HPO 的 proposer 能读参考修复（App.D p19）。
   - PostTrainBench：Opus 4.6 的 84 次运行里 12 次被标污染，其中 8 次是 HumanEval（§5.5 p8–9）。
   - 【推断】HumanEval 是最容易被"找到"的代码 benchmark，用它做 held-out 必须做 n-gram 去污染。RSI Arena 的 8-gram 检查是可直接借鉴的流程（附录 I p32）。
4. **自生成测试作为信号：要有执行接地。**
   - Reflexion 在 MBPP-PY 上反而下降（80.1→77.1），自生成测试假阳性率 16.3%，HumanEval 只有 1.4%（Table 1 p7，p8）。
   - SpecForge：期望输出来自参考实现时，LLM 测试信号 93.2 接近真实测试 94.8；换成 LLM 评审只有 73.6（Table 2 p27）。
   - 【推断】对我们的执行判分反馈：测试用例的期望输出必须来自参考解或官方测试，不能来自模型自判。
5. **HumanEval 已接近饱和。** NeoHorse-1-9B 98.17、Gemma-4-12B 100.00（Table 2 p15）；Reflexion 91.0（2023）。9B 上 LCB v6 完全没动（65.14→65.14），HE 涨 5.5（Table 2 p15）。【推断】HE 原版区分度低，LCB（较新窗口）更能反映增益。
6. **SWE 类训练池现成可用。** swe-rebench-v6 train（8,622，EvoTrainer）、SWE-rebench + V2 + Scale-SWE（2,484，Self-HPO）都给出了 held-out 增益：Self-HPO SWE-V 61.0→69.5、Multilingual 54.2→59.7、Pro 58.3→65.2（Table 1 p5）；EvoTrainer SWE-9B 30.19→38.16（Table 3 p7）。但这类任务每条需 Docker 环境，成本高；且它们的评测不是 HE/MBPP/LCB。

---

## 5. 【建议】对我们选代码训练集的含义（不是文献结论）

- **函数级/竞赛级代码训练池**：本集合唯一的直接先例是 **TACO-verified（11,897 题）→ LCB-v6 较新题**（EvoTrainer）。APPS 的先例在权重侧（ReST-EM）。所以 TACO-verified 或 APPS 都可以作为执行判分的反馈 + 梯度数据，**评测留给 HE+/MBPP+ 与发布日期晚于训练数据截止的 LCB 窗口**。
  - 必须做题目级去重：EvoTrainer 是靠"排除较早发布的 LCB 题"处理污染的（p6）。【推断】TACO 与 LCB 可能共享 AtCoder/Codeforces 来源，需要按题源与日期去重；本文献集没有验证这一点。
  - HumanEval 原版易被污染、已饱和，建议只作哨兵，不作主指标。
- **选择与报告要分开**：EvoTrainer、NeoHorse-1、PostTrainBench 都出现了"测试集即选择集"的问题。gate 应从训练池里切出（或用 LCB 中间窗口），HE+/MBPP+/最新 LCB 只开一次。
- **SWE 训练池**（swe-rebench、SWE-smith、SWE-Gym、R2E-Gym）只有在我们要评 SWE-bench 时才有先例；本集合里没有论文用它们然后去评 HE+/MBPP+/LCB。

---

## 6. 未读线索（不影响结论）

- SWE-Gym（2412.21139）、R2E-Gym、SWE-smith、swe-rebench 原论文：是数据来源，本轮没读全文；EnvHarness 引用了 SWE-Gym 和 SWE-smith，但没有用它们训练。
- Self-Refine、LATS、MaAS、EvoMAC、TextGrad、OpenEvolve：可能含 HumanEval/MBPP 评测，本轮没有读。
- NeoHorse-1 引用了 Nemotron-SFT-SWE-v3（参考文献 [45]），正文是否用它训练没有确认。
