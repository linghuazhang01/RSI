# 2505.19433 阅读笔记：Can Compressed LLMs Truly Act?（ACBench）

## 元信息
- 标题：Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression
- 作者：Peijie Dong*, Zhenheng Tang*, Xiang Liu, Lujun Li, Xiaowen Chu, Bo Li（HKUST-GZ / HKUST）
- 发表：ICML 2025（PMLR 267）；arXiv v2，2025-06-01
- 代码：github.com/pprp/ACBench
- 页数：共 34 页。精读 p1–9（正文）、p24–34（实验设置 B.1–B.6、附录结果 C.1–C.7 及表 3–10）。略读 p10–17（参考文献）、p18–23（目录与扩展相关工作 A.1–A.7，只是综述性文字）。
- 源文本：`src_2505.19433/2505.19433.pages.md`

## 研究问题
现有压缩评测大多只看困惑度（WikiText-2 PPL）和 GLUE/MMLU 这类单轮 NLU 指标。本文问的是：经过训练后压缩（PTQ 量化、一次性剪枝）的 LLM，在 agent 必需的能力上还剩多少？他们把这些能力分成四类：动作执行（工具调用/函数调用）、工作流生成、长上下文理解、真实场景应用。另外还问了两件事：量化和剪枝哪个更适合 agent 场景；R1 蒸馏出的小模型在 agent 任务上表现怎样。

## 方法细节
### 被压缩的模型
- 中等规模：InternLM-2.5-7B、Qwen2.5-7B、Mistral-7B（v0.3）、Llama-3.1-8B（表 1 列出了它，但正文主要结果表里很少出现）
- 更大的 Qwen：Qwen2.5-14B（出现在 T-Eval 表 8 和 WorfBench 表 10）、**Qwen2.5-32B（只在 WorfBench 表 10 里测了 GPTQ/AWQ INT4，算是本文唯一落在 20B–35B 区间的模型）**。没有 20B–35B 模型的剪枝结果，也没有它在 T-Eval、LongBench、AgentBoard 上的结果。
- 小模型：Qwen2.5-1.5B/3B、Gemma-2-2B-it、Phi-3.5-mini-3.8B、MiniCPM3-4B、Megrez-3B
- 蒸馏模型：DeepSeek-R1-Distill-Qwen-1.5B/7B、DeepSeek-R1-Distill-Llama-8B（直接拿官方 checkpoint 和 Qwen2.5-7B 等对比，作者自己没做蒸馏）
- 统计分析部分（第 3 节）还用了 LLaMA-2-7B 和 OPT-125M 到 6.7B

### 压缩方法与配置（p4，B.1 p24）
- 量化：GPTQ、AWQ（主要是 INT4 weight-only，表 10 写的是 INT4；T-Eval 表 8 只写方法名，没写比特数）、GPTQ INT8、SmoothQuant W8A8、FP8。用 LLMC 框架统一实现。统计分析部分还扫了 W2/W3/W4/W8/W16 和 KV4/KV8/KV16。
- 剪枝：Magnitude、Wanda、SparseGPT，每种都有非结构化和 2:4 半结构化两个版本，**稀疏率都是 50%**。代码来自 Wanda 仓库。
- 校准数据：沿用 LLMC 的设置，取 Pile 验证集的一个子集，**128 条样本，序列长 512**。
- **group size 等超参都用默认值，没有扫描**（局限一节自己承认了）。
- 解码温度设为 0。
- **没有任何恢复训练**：没有 LoRA，也没有 QAT（B.1 写明压缩后不做微调）。
- 方法选择的原则是能在 vLLM 上部署，所以 QuaRot 这类方法被排除。

## 评测设置（表 1 p2，B.3–B.6 p24–26）
| 能力 | Benchmark | 测什么 | 规模 | 指标 |
|---|---|---|---|---|
| 动作执行 | T-Eval | 工具使用的 6 个子能力：Instruct、Plan、Reason、Retrieve、Understand、Review；每项分 String 和 JSON 两种输出格式 | 553 条 query，共 23,305 个测试用例，平均每题 5.8 步工具调用，涉及 15 个工具 | 各子项分数和 Overall |
| 工作流生成 | WorfBench | 生成图结构工作流，场景包括函数调用、具身、问题求解、开放式规划；子集有 Alfworld、Lumos、OS、ToolAlpaca、ToolBench、Webshop | 2146 条测试 + 723 条 held-out（表 1 列的子类分别是 1803/4048/4257/2281） | 子序列/子图匹配得到的 P/R/F1 |
| 长上下文 | LongBench | 单文档 QA（NarrativeQA、Qasper、MF-en）、多文档 QA（HotpotQA、2WikiMQA、Musique、DuReader）、摘要（GovReport、QMSum、MultiNews、VCSum）、few-shot（TREC、TriviaQA、SAMSum、LSHT）、合成任务（PassageRetrieval、PassageCount）、代码（LCC、RepoBench-P） | 每个子任务数百条 | 各任务原生指标 |
| 长上下文 | LongGenBench | 超长生成，多个 shot 拼在一个长 prompt 里（GSM8K、MMLU） | GSM8K 8000 / MMLU 15908 | 准确率 |
| 长上下文 | Needle-in-the-Haystack | 长文中检索单条事实，最长 40K token | — | 热力图（只给图） |
| 真实场景 | AgentBoard | ScienceWorld（具身）、Jericho 和 PDDL（游戏）、Tool-Query 和 Tool-Operation（工具） | 环境数 90/20/60/60/40 | Progress rate（P）和 Success rate（S） |

- shot 数：只有 LongGenBench 明确是多 shot，其他 benchmark 用各自框架的默认设置，文中没写具体 shot 数。
- **没测真实推理速度、吞吐和显存。** 第 4.2 节和图 7 的同 bit 预算比较只是按参数量和比特数估算显存，不是实测。
- 另外提出了三个分析指标：ERank（权重矩阵的有效秩）、Top-k 排名一致性（Jaccard、Kendall、Spearman）、Energy（logit 能量，借自 OOD 检测）。

## 关键实验与数字
### (a) 量化 vs 剪枝 vs 蒸馏（50% 稀疏 vs 4bit 量化，压缩率大致相当）
- **T-Eval 工具使用（表 8，p32）**：
  - InternLM2.5-7B：Base 71.4，GPTQ 71.8，AWQ 68.6，Wanda(Un) 64.7，SparseGPT(Un) 62.2，Wanda(2:4) 47.7，SparseGPT(2:4) 35.7，Mag(2:4) 24.8。
  - Qwen2.5-7B：FP16 70.0，AWQ 70.1，GPTQ 71.0，Wanda(Un) 62.3，SparseGPT(Un) 62.2，Wanda(2:4) 53.1，SparseGPT(2:4) 44.6，Mag(Un) 7.3，Mag(2:4) 3.9。
  - Qwen2.5-14B（只测了剪枝）：FP16 77.5，SparseGPT(Un) 77.0，Wanda(2:4) 66.1，SparseGPT(2:4) 50.0。从这组看，模型越大，非结构化剪枝越能保住性能。
  - 蒸馏模型：DS-Qwen-7B 只有 43.6，原版 Qwen2.5-7B 约 70（p6–7 正文写的是从 70% 降到 43.6%）。
  - 作者的推荐：4bit GPTQ 或 FP8 最好；如果一定要剪枝，选 Wanda 2:4（p7）。
- **WorfBench 工作流（表 10，p33）**：
  - Qwen2.5-7B：各种量化和 Wanda、SparseGPT 的平均 F1 都在 0.44–0.46，只有 Mag(2:4) 掉到 0。
  - InternLM-2.5-7B 剪枝损伤很大：Wanda(Un) 0.12，SparseGPT(Un) 0.20，Mag(Un) 0.08。
  - **Qwen2.5-32B：base 0.72，AWQ INT4 0.71，GPTQ INT4 0.69**，几乎没有损失。
  - Qwen2.5-14B：base 0.69，AWQ 0.69，**但 GPTQ INT4 只有 0.36**，这是一个明显的异常值。
  - 正文 p7 的说法是多数方法掉点在 5% 以内。
- **AgentBoard 真实场景（表 9，p33）**：
  - Qwen2.5-7B 的 progress rate：
    - PDDL：base 33.56%，AWQ 27.14%，GPTQ 22.89%，Wanda(Un) 14.00%，Wanda(2:4) 0.56%。
    - ScienceWorld：base 26.79%，GPTQ 26.41%，AWQ 17.33%，Wanda(Un) 14.98%，2:4 只剩 1.8%。
    - Tool-Query：base 44.00%，AWQ 47.16%，GPTQ 42.27%。
  - Magnitude 剪枝在各任务上基本全是 0。
  - 三个 DS-Distill 模型在所有任务上都接近 0。
  - 摘要的说法是 4bit 量化在真实场景应用上掉 10%–15%。
- **LongGenBench（表 6，p28）**：
  - Qwen2.5-7B 的 GSM8K/MMLU：base 61.33/64.65，AWQ 52.67/64.39，GPTQ 47.50/37.28，Wanda(Un) 35.00/57.54，SparseGPT(Un) 18.00/58.86，Wanda(2:4) 2.17/37.02，Mag 0/0。
  - InternLM2.5-7B 只有 AWQ 保住了 MMLU（59.12，base 61.49），其他剪枝方法的 MMLU 都在 10 以下。
- **LongBench（表 3、4，p27–28）**：
  - Qwen2.5-7B 上，magnitude 剪枝几乎全线归零；Wanda 和 SparseGPT 的非结构化版本大致保住 few-shot 和代码任务。
  - 小模型 Qwen2.5-3B 用 AWQ 后几乎全线归零，用 GPTQ 则基本不掉（表 5）。

### (b) 哪些能力最容易被压缩伤到
1. **结构化输出（JSON）比自然语言字符串更脆弱**（图 3、表 8，p6）。例如 InternLM 的 Plan-JSON 子项：Wanda(2:4) 11.2，SparseGPT(2:4) 0.1，FP16 73.7。
2. **长输出里的多步数学推理最脆弱**：Qwen2.5-3B 压缩后，LongGenBench 的 GSM8K 从 61% 掉到 11%，但 MMLU 还能和 7B 相当（p9）。
3. **真实环境里的多轮交互（AgentBoard）明显比单步工具调用和工作流规划更脆弱**（p9 第 7.2 节）。
4. Needle-in-a-Haystack：所有压缩模型在 32K 左右都出现性能边界；magnitude 剪枝最差；Wanda 和 SparseGPT 在 20K 以上开始退化（p9、C.5 p29）。
5. 工作流生成和 OS、Webshop 类任务相对稳健，压缩后大约还能保留 80%（p8）。

### (c) 断崖点
- 本文**没有扫稀疏率**，只测了 50%。在 50% 这一点上，2:4 普遍比非结构化差很多，magnitude 剪枝直接崩溃。
- 比特数方面，主实验基本只有 INT4 和 INT8/W8A8 两档。W2/W3 只出现在 ERank 分析里（图 2，p4），没有下游任务结果。
- 模型规模是另一条断崖：在 Alfworld 和 Lumos 上，Qwen2.5-3B 用 GPTQ/AWQ 最多掉约 50%，而 Qwen2.5-32B 量化后不掉（p8）。

### 其他
- **同 bit 预算比较**（图 7，p7）：在 1.5B–3B 区间，原生 BF16 小模型更好；到 7B 及以上，大模型 INT4（例如 32B AWQ）开始占优。
- **ERank 和规模**（表 2，p6）：在 OPT 上，ERank 随量化比特降低而减小。作者认为大模型压缩后结构保持得更好，但表里只有 4 个 OPT 规模，样本很少。
- **Top-k 一致性**（图 4–5，p5）：k 越小越不稳定，而且这个指标和 PPL 的 Spearman 相关性很高。

## 作者自己声明的局限（p9 第 9 节）
- 只研究了 PTQ 和一次性剪枝，没有 QAT。
- 只选了 vLLM 兼容的方法，排除了 QuaRot 等。
- 全部用默认超参，group size 等没有扫描，所以结论只在这些配置下成立。

## 方法问题与替代解释
- 【推断】基线有明显异常。LongBench 表 3 里 InternLM2.5-7B 的 base 在 NarrativeQA 上是 0，AWQ 后变成 25.08；TriviaQA 从 48.1 升到 84.69；PRE 从 50 升到 99.5。Qwen2.5-7B 用 GPTQ 后 Qasper 从 13.12 升到 38.94。量化不太可能真的让能力大幅提升，更可能是 base 的 prompt 或模板、截断、输出解析出了问题。所以 LongBench 上量化和剪枝的相对比较不太可信。
- 【推断】表 8 有疑似重复填表。Mistral-7B 的 AWQ、FP8、GPTQ 三行完全相同（都是 25.1）；InternLM 的 Base 行和 FP8 行完全相同；Qwen2.5-7B 的 FP8 和 FP16 完全相同。另外 Mistral 的 FP16 只有 25.7，JSON 子项全是 0，说明这个模型的分数主要反映格式失败，不是压缩造成的。
- 【推断】Qwen2.5-14B 在 WorfBench 上 GPTQ INT4 只有 0.36，同模型 AWQ 有 0.69。这更像 GPTQ 实现或校准出了问题（例如校准集太短或不匹配），说明单次 GPTQ 运行的方差可能很大。
- 【推断】WorfBench 表里 Qwen2.5-7B 没有未压缩基线行，所以 p7 说的“掉点在 5% 以内”在这个模型上没法直接核对。
- 【推断】蒸馏的比较不是同源比较。按 DeepSeek-R1 技术报告（外部知识），R1-Distill-Qwen-7B 的基座是 Qwen2.5-Math-7B，不是 Qwen2.5-7B-Instruct，而且它是用推理链 SFT 训出来的。所以“蒸馏损害 agent 能力”混入了基座不同、对齐数据不同、输出格式不同（会输出很长的 think 段）等因素，不能直接当成压缩方法之间的比较。
- 【推断】只有一个稀疏率（50%）和两档比特，没有剂量-反应曲线，找不到断崖的具体位置。没有方差，也没有多次运行，而 AgentBoard 每个任务只有 20–90 个环境，几个百分点的差距很可能在噪声范围内。
- 【猜测】ERank、Energy 这些分析指标和下游 agent 分数之间没有建立定量关系，表 2 的 4 个 OPT 点不足以支持“可以用来预测”的说法。

## 对“约 27B 模型自我压缩（压缩后还要继续当 agent 做下一轮）”研究设计的启示
1. **首选量化，而且优先 4bit AWQ 或 GPTQ（或 FP8/INT8）。** 本文证据：7B 和 32B 上 INT4 量化在工具调用（T-Eval）和工作流（WorfBench）上几乎不掉点（表 8、表 10）。Qwen2.5-32B AWQ INT4 在 WorfBench 上从 0.72 降到 0.71，是和 27B 规模最接近的直接证据。
2. **如果要剪枝，只考虑非结构化的 Wanda 或 SparseGPT，避免 2:4 和 magnitude。** 50% 的 2:4 在 PDDL、ScienceWorld 这类真实交互任务上几乎全部崩溃（表 9）。不过 14B 的 SparseGPT(Un) 在 T-Eval 上基本无损（77.5 → 77.0），提示更大的模型对非结构化剪枝更耐受；20B–35B 本身没有剪枝数据【推断：27B 大概率介于 14B 和 32B 的表现之间】。
3. **蒸馏或 LoRA：本文没有 LoRA 恢复实验。** R1-Distill 的负面结果来自基座和训练数据不同的模型，不能直接推到“自蒸馏或 LoRA 修复”上。
4. **每一轮必须用真实交互类 agent 基准把关，不能只看 PPL 或静态 MCQ。** 本文最关键的一个现象是：工具调用和工作流掉 1%–3%，但 AgentBoard 掉 10%–15%（摘要、表 9）。所以“压缩后的模型还能不能当 agent 去执行下一轮压缩”应该用多轮交互任务来衡量。建议的评测组合：T-Eval（特别是 JSON 子项）、WorfBench、AgentBoard（PDDL、ScienceWorld、Tool-Query）、LongGenBench-GSM8K（长输出推理）、长上下文检索（NIAH 或 LongBench）。
5. **JSON 和结构化输出要单独监控。** 自我压缩的 agent 需要输出严格格式的工具调用和配置，而本文显示 JSON 格式比 String 格式掉得多（图 3）。【推断】多轮迭代时，格式错误可能逐轮放大，变成整条流水线失败。
6. 【推断】GPTQ 有单次实现失败的风险（14B 的 0.36），流水线里应该设一个回滚机制：如果某轮压缩后的能力探针掉点超过阈值，就撤回这一轮。
7. 【猜测】本文没有研究多轮叠加压缩，例如 INT8 之后再 INT4，或者剪枝之后再量化。误差会不会累积，需要自己做实验。
