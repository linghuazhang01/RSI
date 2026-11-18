# DATA_CONFIG：两篇论文的数据配置（2026-11-18）

> 规则：每个领域 **一份训练集 + 1–3 个留出集**。自量化 / 自编辑循环**只看训练集**；benchmark 全部留出，每一代结束后评测，只用于报告。
> 实现：`data/splits/split_config.yaml`（`role: train` = 训练集）；计数、去污染和偏差见 `data/README.md`；评测入口 `eval/run_eval.py`。
> 文献依据：`plan/research/method-research/rsi-data/RSI_DATA.md`（37 篇 RSI / 自我改进论文全文）。下表"依据"一列只列在该文献集中跑通过的配置；标【推断】的没有 RSI 先例。

## 1. 池的含义

| 池 | 来自 | 谁能看到 | Paper A（自量化） | Paper B（自编辑 LoRA） |
|---|---|---|---|---|
| `search` | 训练集，200 题 | 编辑者：分数 + 出错样例 | L1 信号 | dev |
| `gate` | 训练集，另 200 题 | 编辑者：只有通过 / 不通过，限次 | L2 验收 | dev2 |
| `train` | 训练集其余 | 训练 / 文本 | KL 上下文、校准文本 | 训练 raw LoRA |
| `test` | benchmark | 编辑者永远看不到 | L3 留出 | L3 留出 |

## 2. 各领域配置

| 领域 | 训练集（去污染后条数） | 留出集（主表） | 依据 |
|---|---|---|---|
| 数学 | GSM8K train 7,471；MATH train 7,466；DAPO-Math-17K 16,329 | MATH-500；GSM8K test 1,319；AIME 2024+2025（60） | ReST-EM（MATH train → MATH test、GSM8K）；WSP 博客 / WSP-Scope（GSM8K+MATH train → MATH-500 等）；WHALE（DAPO → AIME） |
| 代码 | APPS 入门级 train 2,397（带测试用例和参考解） | HumanEval+ 164；MBPP+ 378；LiveCodeBench 131 | ReST-EM（APPS-Intro train 2,342 → APPS test、HumanEval） |
| 通用推理 / 知识 | ARC-Easy+Challenge train 3,370；GPQA Main 去掉 Diamond 250；TriviaQA train 73,725 | GPQA-Diamond 198；MMLU-Pro 2,000；TriviaQA 2,000 | Transformer²（ARC-Easy → ARC-C）；WSP-Scope（GPQA Main 去 Diamond → Diamond）；TriviaQA train【推断】 |
| 指令遵循 | IFEval 的 270 条（信号 135 / 验收 135） | IFEval 其余 271；IFBench 300（领域外） | WSP-Scope（IFEval 270/271）；唯一先例 |
| 工具调用 | Seal-Tools train 12,019 | Seal-Tools test（in-domain 700 + out-domain 654）；API-Bank test 584；BFCL 4,696 | TT-SI 的 SFT 基线（SealTool train → SealTool test：66.4 → 70.2） |

额外留出集（附录用）：AMC23、OlympiadBench、FreebaseQA、MetaTool、T-Eval；MT-Bench、AlpacaEval 2（需 LLM judge，不进主结果）。
Paper B 的可选知识注入任务：SQuAD（SEAL 设定，train 段落 → SEAL 的 200 段）。
只作 Paper A KL 文本：ToolACE（工具调用校准，参照 2608.06564）。
不再使用（保留在 `reserve`）：Magicoder-OSS（无 RSI 先例、无测试用例）、Tulu-3 IF（无可判分参数）、API-Bank train（被 Seal-Tools 取代）。

**没选的配置及原因**：
- WHALE 的 HotpotQA + NQ train：它是检索增强问答，不是闭卷；
- MBPP train：与 MBPP+ 重叠；
- 从 benchmark 切 dev（如 OPRO 的 BBH 20%、EvoGM 的 70 题 MMLU-Pro）：小池过拟合严重（EvoGM 验证 0.429 / 测试 0.224；OPRO 的 BBH causal_judgement 搜索 84.8 / 测试 58.0）。

## 3. 反馈信号怎么用这些数据

**Paper A（自量化）**，每次 `evaluate(config)` 返回：
1. 相对 BF16 的 KL：上下文取自训练集 `train` 池，加上 BF16 模型在训练集题目上的自生成推理文本（2504.04823：3-bit GPTQ 用推理类校准数据比 WikiText2 平均 +9.8）；按领域分别报。
2. 各领域信号切片上的任务分数和出错样例：数学（MATH train + DAPO 各 100）、代码（APPS 100，沙箱执行）、知识（TriviaQA + ARC 各 100）、指令遵循（IFEval 135）、工具调用（Seal-Tools 200）。
3. 预先算好的诊断表（基于训练集）和 `bytes(config)`。

L2 验收用各领域验收切片，门槛相对第 0 代；E_t 和自压缩深度只在留出集上算。

**Paper B（自编辑 LoRA）**，每个候选编辑返回：
1. 目标任务信号切片上的分数和错题；
2. 保留信号：基座与候选在保留领域训练集 prompt 上的逐 token KL，以及 TriviaQA / ARC / IFEval 信号切片的分数；
3. adapter 诊断表（范数、谱、intruder 分数、相对改动量、checkpoint 轨迹）。

L2 验收用目标任务和保留集的验收切片；四分支成绩和 E_t（E-Seal：Seal-Tools train → Seal-Tools test，BFCL 作领域外）只在留出集上算。

## 4. 去污染

训练集在切分前去掉两类样本（放进 `train_excluded`）：与任何评测样本完全相同的；与同类别某条留出样本共享的 13-gram 占任一方 ≥50% 的。

| 训练集 | 精确匹配 | 13-gram 近似 |
|---|---|---|
| DAPO-Math-17K | 984（968 条是 MATH test 中 MATH-500 以外的题；4 条 MATH-500、5 条 AMC23、1 条 OlympiadBench） | 85 |
| TriviaQA train | 2,428 | 370 |
| MATH train | 1 | 33 |
| Seal-Tools train | 0 | 3 |
| GSM8K train | 0 | 2 |
| APPS、ARC、GPQA Main | 0 | 0 |

去污染后，主表留出集中被单条训练样本覆盖 ≥50% 13-gram 的题数为 0。

## 5. 证据薄弱、需要在论文里写明的地方

1. **指令遵循**：信号和留出集都来自 IFEval，只能防止记住具体题目，防不住适应 IFEval 本身，所以 IFBench 必须报。
2. **TriviaQA train** 当闭卷知识信号没有 RSI 先例。
3. **工具调用的"不该调用"判断**：Seal-Tools 每题都有正确工具。Paper A 的主张 3 依赖这一判断，需要从 Seal-Tools train 构造"金标 API 不在候选列表里、正确答案为空"的信号样例（待实现）；留出评测看 BFCL 的 irrelevance 类别。
4. **APPS 参考解**还没在沙箱里验证（Docker 未启动）；ReST-EM 用的是 2,342 题，我们是 2,397 题。
5. **DAPO 题数**：去重后 17,398 题，WHALE 写 17,917 题，差异来源未知。
