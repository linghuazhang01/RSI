# CODE_RSI_WEIGHTS：代码领域"更新权重"的自我改进论文用什么训练 / 信号数据、在什么上评测

> 用途：为两篇 RSI 论文（自我量化；模型写自己的 LoRA）选一个**代码训练集**，它既当执行打分的反馈信号，也当梯度训练数据。本文只覆盖**更新模型权重**的方法（自生成数据、自验证、自博弈）；agent / scaffold 自改进（DGM、SICA 等）由另一份文件覆盖，这里不重复。
> 日期：2026-11-19。
> 数字来源：每个数字都来自对应论文全文（`papers/src_<id>/<id>.pages.md`）或已有全文笔记，并标页码 / 表号。论文没写的标"未说明"。【推断】是我的推断；"建议"只出现在第 6 节。
> 本轮新读全文 19 篇（笔记在 `papers/<id>.notes.md`）。复用已有全文笔记 4 篇：ReST-EM（`../rsi-data/papers/2312.06585.notes.md`）、AZR（`../../self-lora/papers/2505.03335.notes.md`）、递归自训练崩塌（`../../rsi-model-edit/papers/2606.28438.notes.md`）、Self-MoE（`../../rsi-model-edit/papers/2406.12034.notes.md`）。没有任何结论只依赖摘要。

缩写：HE = HumanEval，MBPP+ / HE+ = EvalPlus 版本，LCB = LiveCodeBench，CC = CodeContests，CF = CodeForces。
"自改进程度"：**强** = 同一模型自己生成数据 / 测试 / 题目并自我验证；**中** = 自己生成解，但奖励依赖人工测试或 golden 解；**弱 / 否** = 数据或奖励来自外部更强模型（只作参照，不作 RSI 证据）。

---

## 1. 总表

| # | 论文 | 改进对象 | 自改进程度 | 训练 / 信号数据（名称、规模、划分） | 每轮怎么用 | 评测集 | 评测含 HE(+)/MBPP(+)/LCB？ | 与评测不相交？ | 出处 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ReST-EM（2312.06585） | 权重（PaLM 2，自生成解上 SFT，每轮从基座重训） | 中 | **APPS-Introductory train 2,342 题**（每题 64 样本，至多留 10 个正确解） | E-step 采样 + 测试过滤 → M-step SFT；APPS 2 轮 | APPS-Intro test、HumanEval | HE（非 plus） | APPS train→test 同 benchmark 不同 split；HE 跨 benchmark | p7–9，Fig.3–4 |
| 2 | CodeRL（2207.01780） | 权重（CodeT5-770M，actor-critic） | 中 | **APPS train 5,000 题**；附录另用 MBPP train 374 题 | Lce 预热 10 epoch 后加 RL；单阶段 | APPS test 5,000；MBPP test 500（零样本） | MBPP（非 plus） | 官方划分；作者测了 APPS train 与 MBPP 代码行重叠（12.6% 的 MBPP 程序过半行重复） | p11–16，Table 1/6，Fig.7 |
| 3 | LEDEX（2405.18649） | 权重（SFT + PPO，自调试 / 解释） | 中（§4.3.2 有纯自举版本） | **MBPP train 374、APPS train 5,000、CC 6,627**；RL 只用 MBPP+APPS train | 被训模型每题采 20 解找错解 → 修复须重新过测试 → SFT 2 epoch → PPO | MBPP、HE、**MBPP+、HE+** | **HE+、MBPP+** | MBPP 按 split 不相交；未做去污染 | p3、p5–6，Table 1–2、Table 8 |
| 4 | ZeroCoder（2604.07864） | 权重（GRPO，同一模型交替当 Coder/Tester） | 强（无标签） | **APPS train 5,000 题**，预过滤后约 1k 题（正文与 Table 4 数字对不上） | 在线 RL 150 步，每题 8 代码 + 8 测试 rollout | **LCB（2024-10 至 2025-02）**、MBPP、APPS test 300、CF 300 | **MBPP（非 plus）、LCB** | APPS 同 benchmark 不同 split；CF 子集声明与 APPS 不重叠 | p5–6，Table 1 p7，Table 4 p10 |
| 5 | CURE（2506.03136） | 权重（RL，同一模型写代码 + 单测） | 中（coder 奖励用 GT 测试） | **CC 难度≤2，随机切 4.5k train / 200 eval** | 在线 RL，16+16 rollouts，7B/14B 350 步 | LiveBench、**MBPP test**、**LCB v2（511 题）**、CC 200、CF 500 | **MBPP（非 plus）、LCB v2** | CC 自切同分布；CF 声明与 CC 不重叠 | p8，Table 1 |
| 6 | ACE（2605.16299） | 权重（LoRA，solver SFT + adversary KTO，各一个 adapter） | 中（solver 用 GT 测试 + 自生成对抗测试） | **CC 约 4.5k 题** | 5 轮自演化；每题 16 程序 + 16 测试输入 | MBPP（split 未说明）、**LCB v2**、CC held-out | **MBPP、LCB v2** | CC 同分布；MBPP/LCB 跨源 | p6–7，Table 1 |
| 7 | ReVeal（2506.11442） | 权重（多轮 RL，TAPO） | 中（验证奖励用 golden 解） | **TACO 26,443 → 过滤后 11,151 train / 509 test** | RL 最多 3 轮生成-验证 | **LCB V6（2025-02 至 05）**、CC；附录 **LCB V5** | **LCB v6/v5** | 时间上与 LCB 不相交；TACO 与 CC 重叠未讨论 | p5–7，Table 1，Table 7–8 |
| 8 | DuST（2605.11299） | 权重（GRPO 训练模型给自己的候选排序） | 强（自采样 + 执行标签） | **rStarCoder seed 子集约 10K 竞赛题** → 6.9K 题 / 37K 组 | 数据一次性构造；GRPO 3 epoch | **LCB v6（2025-02 至 05，131 题）**、**LCB v5（2024-08 至 2025-02）** | **LCB v6/v5** | 靠 LCB 时间窗；未单独去重 | p6、p14 |
| 9 | Sol-Ver（2502.14948） | 权重（Llama 3.1 8B，SFT + DPO） | 强 | **自生成 103,280 题**（以 OSS-Instruct 片段为灵感；题面模板取自 MBPP/APPS/CC train） | 3 轮，每轮自生成解与测试 → SFT → DPO | MBPP、LCB（版本未说明） | **MBPP（非 plus）、LCB** | 题目自生成，但借 MBPP train 模板，未去污染 | p4–6，Table 1 |
| 10 | SelfCodeAlign（2410.24198） | 权重（SFT） | 强 | **The Stack V1 种子 250k 函数 → 自生成 74k 指令-回答对**；子串去污染 | 单轮：自生成指令 → 自写回答 + 测试 → 执行过滤 | **HE+、MBPP+**、**LCB（2023-05 至 2024-02，按起始日期分段）**、EvoEval、ClassEval、DS-1000 等 | **HE+、MBPP+、LCB** | 是（显式去污染 p18；LCB 日期后移分数几乎不变） | p3–8，Table 1–2、Table 8 |
| 11 | InverseCoder（2407.05700） | 权重（SFT） | 强（自生成指令、自评选择，无执行） | **evol-codealpaca-v1 111,183 条**（已去 HE/MBPP/MultiPL-E/DS-1000 污染） | 单轮；第二轮无显著提升 | **HE(+)、MBPP(+)**、MultiPL-E、DS-1000 | **HE+、MBPP+** | 是（显式去污染 p4） | p4–7，Table 3 |
| 12 | CodeDPO（2410.05605） | 权重（DPO） | 弱–中（DeepSeekCoder-v2 生成，代码×测试互验） | OSS-Instruct 式种子 → **93k 正确性 + 21k 效率偏好样本** | 一次性构造 → DPO 10 epoch | **HE(+)、MBPP(+)**、DS-1000、**LCB（版本未说明）** | **HE+、MBPP+、LCB** | 跨源；未去污染 | p3–7，Table 1–4 |
| 13 | SCoRe（2409.12917） | 权重（Gemini 1.0 Pro，两阶段多轮 RL，自纠正） | 中（自生成轨迹，二值测试奖励） | **"MBPP"**（split 与题数未说明） | 在线 RL 1,500 步；按训练奖励选 checkpoint | HE（两次尝试）、MBPP-R | HE（非 plus） | MBPP→HE 跨 benchmark | p11–12，Table 3，Table 5 |
| 14 | Self-MoE（2406.12034） | 权重（自专精 LoRA 专家 + 路由） | 强（自生成数据，种子人工取） | **MBPP train 100 条种子**（改成 HE 格式） | 自生成指令数据 → 训 LoRA 专家 | HE；MBPP 作非目标任务 | HE、MBPP（非 plus） | MBPP train→HE 跨 benchmark | 笔记：附录 A p18，Table 1 p6，Table 3 p8 |
| 15 | SAGE（2603.15255） | 权重（**LoRA r128**，四角色联合 RL，200 步） | 强（Challenger 出题、外部验证器判） | **500 条种子：HumanEval 87、MBPP 109、MATH 156、GSM8K 148** | 迭代扩题库 → 解 → 联合更新 | **HE+、MBPP+（ID）**、**LCB v1–v5（2023-05 至 2025-02，OOD）**、数学 6 项 | **HE+、MBPP+、LCB** | **否**：HE 没有训练集，87 条 HE 种子必与 HE+ 重叠【推断】 | p6–7，Table 1/3，附录 B Table 5 p13 |
| 16 | AZR（2505.03335） | 权重（RL，同一模型出题 + 解题） | 强（零外部数据，Python 执行器） | **无外部数据**（种子只有恒等函数） | 在线自博弈 500 步 | **HE+、MBPP+、LCB v1-5（2023-05 至 2025-02）**、CruxEval、数学 | **HE+、MBPP+、LCB** | 是（无训练题） | 笔记：表 1 p9，表 5 p34，附录 D p51 |
| 17 | 递归自训练崩塌（2606.28438） | 权重（自生成代码回流 SFT，5–20 轮） | 强（含自门控） | 固定 prompt 池（来源与规模未说明），每轮 5,000 条自生成样本 | 每轮 3,000 步；有/无外生门、自门对比 | **HE+、MBPP+**（主）、HE、MBPP、**LCB**（版本未说明） | **HE+、MBPP+、LCB** | 是（无训练题） | 笔记：§3.1 p7，Table 2 p8，附录 M.2 p29 |
| 18 | FTTT / OpTune（2502.15771） | 权重（测试时逐题 LoRA 更新） | 中（二值验证 + 自反思） | **评测题本身**（贪心失败子集）；OpTune 优化器在 MBPP train 上训练 | 预算 32 次尝试，每次失败后更新 | MBPP、HE（失败子集） | HE、MBPP（非 plus） | **否（同题，测试时用评测题的验证反馈）** | p2–6，Table 2，p12 |
| 19 | CodeIt（2402.04858） | 权重（CodeT5+ 系，hindsight relabeling + 优先回放） | 强 | ARC train 400 + 19,200 变异程序；**搜索集 = ARC eval 400 题（只用 demonstration）** | 每 meta-iteration 每题采 24 程序；回放训练 1 epoch | ARC eval（pass@3） | 否 | **否（同题不同样例）** | p4–5，Table 1，附录 B.2 |
| 20 | SSR（2512.18552） | 权重（32B CWM-sft，同一模型注入 / 修复 bug） | 强（无 issue、无测试） | **CWM 的仓库 Docker 镜像**（数量未说明） | 在线 RL 150 步，约 2.5B token | SWE-bench Verified 500、SWE-Bench Pro 731 | 否 | 仓库重叠未说明 | p3、p6–7，p15 |
| 21 | AceCoder（2502.01718） | 权重（RM + Reinforce++） | **否**（GPT-4o-mini 出测试，Qwen-32B 过滤） | **AceCode-87K**（Evol/OSS-Instruct/Stack 种子；87,149 题、1.38M 测试）；RL 取难的 25% | 1 episode | **HE(+)、MBPP(+)**、BigCodeBench、**LCB V4** | **HE+、MBPP+、LCB V4** | 跨源；未去污染 | p3–6，Table 1、Table 4 |
| 22 | RLAR（2603.00724） | 策略权重（GRPO 80 步）；"自进化"的是 GPT-5 驱动的奖励工具库 | **否** | **LeetCode-Dataset train 2,585 / test 223** | 多任务混合 GRPO | LeetCode test 223、**MBPP 全部 974 题（OOD）** | MBPP（非 plus，且用全集） | LeetCode 同集 split | Table 1 p3，Table 2 p6，p5 |
| 23 | RL Tango（2505.15034） | 权重（**两个模型**：generator Qwen2.5-Math-7B、verifier Qwen2.5-7B） | 弱（SFT 回答来自 Llama-3.1-70B） | **Eurus-2-SFT 113K 数学 prompt；Eurus-2-RL-Data 455K 数学问答对**（**没用其中代码题**） | 交替 RL，generator 200/300 步 | 数学 5 项 + BGQA、**CRUXEval**、StrategyQA、TableBench | **否**（无任何代码生成评测） | 未讨论 | p6–8，附录 B.1 p15 |

---

## 2. 训练 / 信号数据出现频次

按"作为训练题或信号来源的论文篇数"计数（一篇用多个来源时分别计）。括号里是规模。

| 训练数据来源 | 篇数 | 论文 |
|---|---|---|
| **APPS train** | 4（+1 模板） | ReST-EM（Intro 2,342）、CodeRL（5,000）、LEDEX（5,000）、ZeroCoder（5,000→约 1k）；Sol-Ver 只借题面模板 |
| **MBPP train / "MBPP"** | 5（+1 模板） | LEDEX（374）、SCoRe（未说明）、Self-MoE（100 种子）、SAGE（109 种子）、FTTT/OpTune（未说明）；CodeRL 附录（374）；Sol-Ver 模板 |
| **CodeContests** | 3（+1 模板） | CURE（4.5k）、ACE（约 4.5k）、LEDEX（6,627）；Sol-Ver 模板 |
| **TACO** | 1 | ReVeal（11,151） |
| **rStarCoder** | 1 | DuST（约 10K seed → 6.9K） |
| **LeetCode** | 1（非 RSI） | RLAR（LeetCode-Dataset 2,585）；另：AZR 附录 D 试过用 LeetCode 初始化种子，代码分前期高、最终持平、数学更低（p51–52） |
| **OSS-Instruct / Evol-Instruct / The Stack 等代码语料种子** | 5 | SelfCodeAlign（Stack 250k→74k）、InverseCoder（evol-codealpaca 111k）、Sol-Ver（OSS-Instruct 片段→103k 题）、CodeDPO（OSS-Instruct 式→114k 偏好）、AceCoder（87K，非 RSI） |
| **完全自生成 / 零外部题** | 2 | AZR、2606.28438 |
| **HumanEval（作为种子）** | 1 | SAGE（87 条）——污染 |
| **Eurus-2 / PRIME RL 数据** | 1（仅数学） | RL Tango |
| **KodCode** | 0 | 本轮读到的论文里没有任何一篇用 KodCode |
| 非题集环境 | 2 | SSR（仓库镜像）、CodeIt（ARC） |
| 测试时用评测题 | 1 | FTTT |

---

## 3. 按候选训练集列出：哪些 RSI 论文用它训练、并在 HE(+)/MBPP(+)/LCB 上评测

只列自改进程度为强 / 中的论文；非 RSI 参照（AceCoder、RLAR、Tango）单独标出。

| 候选训练集 | 用它训练且评测含 HE+/MBPP+ | 用它训练且评测含 LCB | 只评原版 HE/MBPP | 备注 |
|---|---|---|---|---|
| **APPS train** | **LEDEX**（HE+、MBPP+；同时用了 MBPP train 与 CC） | **ZeroCoder**（LCB 2024-10 至 2025-02） | ReST-EM（HE）、CodeRL（MBPP）、ZeroCoder（MBPP） | 唯一同时有"无标签自演化 + LCB"证据的是 ZeroCoder |
| **CodeContests** | LEDEX（同上） | **CURE**（LCB v2）、**ACE**（LCB v2） | CURE、ACE 的 MBPP（非 plus） | CURE 与 ACE 都是约 4.5k 题规模 |
| **TACO** | — | **ReVeal**（LCB v6、v5） | — | 只有 1 篇，且是 32B |
| **rStarCoder** | — | **DuST**（LCB v6、v5） | — | 训练一次性构造 |
| **MBPP train** | **LEDEX**；SAGE（但有 HE 种子污染） | SAGE（LCB v1–v5） | SCoRe（HE）、Self-MoE（HE）、FTTT | 若评测含 MBPP+，需按题号与 MBPP+ 去重（见第 5 节） |
| **LeetCode** | — | — | RLAR（MBPP 全集，非 RSI） | 本轮没有 RSI 论文用 LeetCode 训练后评 HE+/MBPP+/LCB |
| **Eurus-2 / PRIME 代码数据** | — | — | — | Tango 只用了数学部分，且无代码生成评测；本轮无 RSI 证据 |
| **KodCode** | — | — | — | 无 |
| **代码语料种子（Stack / OSS / Evol）** | **SelfCodeAlign**、**InverseCoder**、**CodeDPO**；AceCoder（非 RSI） | **SelfCodeAlign**（2023-05 至 2024-02）、**Sol-Ver**（版本未说明）、**CodeDPO**（未说明）；AceCoder（V4） | Sol-Ver（MBPP） | 主要是 SFT/DPO 型自生成，数据量大（74k–114k） |
| **零外部数据** | **AZR**、**2606.28438** | **AZR**（v1-5）、**2606.28438**（未说明） | — | AZR-14B HE+ 下降；2606.28438 LCB 归零 |

LCB 版本汇总（只列论文写明的）：ZeroCoder 2024-10 至 2025-02；CURE、ACE 为 v2（CURE 写 511 题，日期窗口两篇都没写）；ReVeal v6（2025-02 至 05）+ v5；DuST v6（2025-02 至 05，131 题）+ v5（2024-08 至 2025-02）；SAGE、AZR 为 v1–v5（2023-05 至 2025-02）；SelfCodeAlign 2023-05 至 2024-02（400 题）；AceCoder V4（日期未写）；Sol-Ver、CodeDPO、2606.28438 未说明。

---

## 4. 数据选择相关发现

1. **训练池太小会过拟合，迭代多了 test 回退。**
   - ReST-EM：APPS-Intro 只有 2,342 题，第 2 轮 APPS test 和 HumanEval 都回退；train 准确率近似线性上升而 test 不跟（Fig.3–4 p8–9）。
   - ZeroCoder：过滤阈值 τ=4 时只剩 217–428 题，代码和测试生成都下降，作者归因为数据太少而过拟合（Table 4 p10，p8）。
   - CodeRL：只用 Lce 继续微调，训练 loss 降但 test 降（p13）。
   - ACE：去掉对抗测试或偏好优化时，训练集分数相近，但 CC held-out 降 4–5 点（p8）。训练集分数不能代替 held-out。
2. **常见规模**：RL 类自改进用 **1k–11k 道带测试的题**（ZeroCoder 约 1k、CURE/ACE 4.5k、DuST 6.9K、ReVeal 11k）；SFT/DPO 类自生成数据 **74k–114k 条**（SelfCodeAlign 74k、AceCode 87K、CodeDPO 114k、Sol-Ver 103k 题）。
3. **污染与来源**：
   - SAGE 把 87 条 HumanEval 当种子，又把 HE+ 当 ID 评测（附录 B p13）——HE+ 的提升不可信。
   - Sol-Ver 用 MBPP/APPS/CC train 的题面模板生成题，再评 MBPP（p5），未做去污染。
   - 显式去污染的只有 SelfCodeAlign（子串匹配，p18）、InverseCoder（沿用 Magicoder，p4）；CodeRL 量化了 APPS train 与 MBPP 的代码行重叠（p16）。
   - ZeroCoder、ReVeal、DuST 都选**新时间窗的 LCB** 来降低泄漏；SelfCodeAlign 用"LCB 起始日期后移分数是否下降"做污染检查（Table 2 p5）。
   - TACO 与 CodeContests 都含 Codeforces 题，ReVeal 用 TACO 训练、用 CC 评测，但没讨论重叠【推断：有风险】。
4. **HE+/MBPP+ 与 LCB 经常走向相反，必须同时看。**
   - AceCoder：从 Qwen2.5-Coder-7B-Base 出发，HE+ +25.0，LCB V4 −0.2；从 Instruct 出发，HE+ 86.0→84.8（Table 4 p6）。【推断】HE+ 大涨主要是格式 / 指令跟随被激活。
   - SAGE：7B 上 MBPP+ 65.3→64.0、4B 上 62.4（下降），LCB 却 +8.9 / +9.1（Table 1 p6）。
   - AZR-14B：HE+ 78.0→70.7，同时数学大涨（笔记 表 5 p34）。
   - 2606.28438：LCB 最先归零（0.238→0），比 HE+/MBPP+ 更敏感（附录 M.2 p29）。
5. **纯自生成数据的天花板**：
   - InverseCoder 第二轮无显著提升（p7）；
   - LEDEX 自举数据上 RL 只再加 ≤0.78 分（Table 8 p9）；
   - 2606.28438 只用编译 / 长度这类弱外生门时，所有方案都低于第 0 轮（Table 2 p8）；
   - 反例：SelfCodeAlign 发现能力相近时，自数据比略强教师的数据更好（Table 8 p8）；ZeroCoder 在无标签条件下仍有提升，但 1.5B 的 LCB 几乎不动（Table 1 p7）。
   - 共同点：有效的自改进都有**执行级外部判定**（单元测试 / 执行器）。没有执行验证的（InverseCoder 自评）或只有弱门的（2606.28438），收益很小或发生崩塌。
6. **选择偏差**：SCoRe 按训练奖励最高选 checkpoint（p11）；ReST-EM 的迭代数是对照 test 曲线讨论的（已有笔记）。这两篇都没有独立 gate 集。

---

## 5. 用户点名的三篇：核实结果

| 论文 | 是否存在 / arXiv | 训练数据（全文核实） | 评测 | 结论 |
|---|---|---|---|---|
| ZeroCoder | 存在，**2604.07864**（v2，2026-06） | **APPS training split，5,000 题**（p6），过滤后约 1k 题 | LCB（2024-10 至 2025-02）、MBPP、APPS test 300、CF 300 | 说法成立。属于强自改进，无标签 |
| RLAR | 存在，**2603.00724** | **LeetCode-Dataset（Xia et al., 2025）train 2,585 / test 223**（Table 1 p3） | LeetCode test、**MBPP 974 题（全集）**、数学、翻译、对话 | 说法成立。但奖励 agent 由 GPT-5 驱动，**不是模型自我改进**；评测没有 HE/HE+/MBPP+/LCB |
| RL Tango | 存在，**2505.15034**（NeurIPS 2025） | **Eurus-2-RL-Data 中 455K 数学问答对**；SFT 用 Eurus-2-SFT-Data 113K 数学 prompt，回答由 Llama-3.1-70B 生成（p6，p15） | 数学 5 项 + CRUXEval 等 4 项 OOD | "用 Eurus-2/PRIME RL 数据"成立，但**只用数学部分**。没有代码训练，也没有任何代码生成评测，不能作为代码训练集的证据 |

---

## 6. 对本项目的含义（建议）

- **建议 1（训练 / 信号集）**：首选 **APPS train（5,000）或 CodeContests 训练部分（参照 CURE 的难度≤2、4.5k 切分）**，备选 **TACO 过滤版（约 11k）**。理由：这三者是 RSI 代码论文里最常见、带人工测试、能执行打分的题集（第 2 节），而且已有"用它训练、在 MBPP / LCB 上评测"的先例（第 3 节）。
- **建议 2（评测）**：评测同时放 HE+/MBPP+ 和**新时间窗的 LCB**（如 2024-08 之后的 v5/v6 窗口），因为两类指标常相反（第 4.4 节）；LCB 也是最敏感的退化哨兵。
- **建议 3（去重）**：
  - 不要用 HumanEval 做任何种子（SAGE 的教训）。
  - 若用 MBPP train，要按题号排除 MBPP+ 评测题。【需核实：MBPP+ 的题目可能跨 MBPP 的多个原始 split，本文献集里没有论文给出对应关系】
  - APPS / TACO / CodeContests 与 LCB 之间按发布时间去重，TACO 与 CC 之间按题目去重。
- **建议 4（规模与迭代）**：训练池至少数千题；不要在两三千题的池子上多轮反复选择（ReST-EM、ZeroCoder τ=4 的过拟合证据），并报告训练池与 held-out 的差距曲线。
- 未覆盖：本轮没有找到用 KodCode 或 Eurus-2 代码部分做自改进训练的全文证据。

---

## 7. 只看到标题 / 摘要的线索（未读全文，不用于结论）

- CoSPlay（2605.23491）：测试时代码-测试协同自博弈，据摘要**不更新权重**，所以排除。
- BenchEvolver（2606.01286）：把种子题演化成更难的题，再训练 gpt-oss-20b，并做了 LiveCodeBench-Plus。未读。
- R-Zero、Self-Questioning LM：据了解以数学为主，未读，未纳入。
- PSV / Propose-Solve-Verify（2512.18160）：形式化验证（Verus）自博弈，未读。
