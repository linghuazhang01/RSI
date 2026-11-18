# 已有工作的实验任务与评测数据集对照

日期：2026-11-14。内容全部取自已全文精读的论文笔记（`plan/research/*/papers/<id>.notes.md`）。表中"模型"指被训练或被压缩的模型，"谁决策"指由谁决定怎么改权重。

---

## 表 1：训练 / 编辑 LoRA 的工作

### 1a. 模型自己（或系统内部）给自己做 LoRA

| 论文 | 被训练的模型 | 谁决策 | 任务（训练用什么） | 评测数据集与指标 |
|---|---|---|---|---|
| **SEAL** 2506.10943 | Llama-3.2-1B（ARC）；Qwen2.5-7B（知识注入） | 同一模型生成训练数据和超参 | ① ARC 少样本抽象推理；② 知识注入：SQuAD 段落 → 自生成"推论" | ARC：11 个训练任务、8 个评估任务，成功率；SQuAD 无原文 QA：200 篇文章、974 题，GPT-4.1 判分 |
| **Search over Self-Edit** 2601.14532 | Qwen3-8B | 同一模型写数据模板，并自选 6 个 LoRA 超参 | SQuAD 知识注入（50 段训练） | SQuAD 无原文 QA 验证集 200 段，Claude Haiku 判分 |
| **MetaRSI** 2609.06396 | Qwen3.5-35B-A3B | 同一模型出题、提议 LoRA 配置 | 自生成的数据 | Terminal-Bench 2.1（89 题）、SWE-bench Pro（731 题）、GPQA-D-hard100、AIME 2025+2026（60 题），pass@1，5 个 seed 平均 |
| **MASS** 2610.12176 | Qwen3.6-27B | 同一模型生成并自评轨迹，LoRA 配置人工固定 | 自建的研究型 agent 任务轨迹 | ScienceAgentBench、MLR-Bench、AstaBench、DSBench、Terminal-Bench 2.0、SWE-bench |
| **Self-Specialization** 2310.00160 | MPT-30B（生物医学）；LLaMA-2-7B（金融） | 同一模型从 80–90 条种子自产约 5K 条数据，QLoRA 配置人工固定 | 生物医学、金融领域指令 | 生物医学 10 个数据集（含 BioASQ 等），金融 10 个基准，F1；附录 MMLU |
| **Self-MoE** 2406.12034 | Gemma-7B 等 | 同一模型为 4 个领域自产数据，训练 LoRA 专家和路由器 | 知识、推理、数学、代码，种子来自各基准训练集 | MMLU、BBH、GSM8K、HumanEval |
| **TT-SI** 2510.07841 | Qwen2.5-1.5B / 7B | 同一模型为每个测试样本自产 1 条数据，LoRA 配置固定 | 工具调用 | 4 个 tool-calling 基准（含 SealTool、API-Bank、ToolAlpaca） |
| **Self-Developing** 2410.15639 | openchat-3.5（7B） | 同一模型的克隆写合并代码，合并的是外部专家 | 合并数学、代码、科学专家的任务向量 | GSM8K（dev 100 / test 1220）、MATH（dev 600 / test 4400），5-shot |

### 1b. 外部 agent 训练别的模型

| 论文 | 被训练的模型 | 谁决策 | 评测数据集 |
|---|---|---|---|
| **PostTrainBench** 2603.08640 | Qwen3-1.7B/4B-Base、SmolLM3-3B、Gemma-3-4B | Claude Code / Codex / Gemini CLI 等 | AIME 2025、GSM8K、GPQA、HumanEval、BFCL v3、ArenaHard-Writing、HealthBench-Easy |
| **FT-Dojo** 2603.01712 | Qwen2.5-7B-Instruct | GPT-5.2（agent） | 13 个任务：AIME25、PANORAMA 的 3 个子任务、ChemCoTBench 的 4 个子任务、FinanceIQ、TableBench 的 4 个子任务 |
| **SIA** 2605.27276 | gpt-oss-120b（LoRA r=32） | Claude Sonnet 4.6 | LawBench（191 类）、AlphaEvolve TriMul（CUDA kernel）、MAGIC scRNA 去噪 |
| **WHALE** 2609.00196 | Qwen3.5-2B / 4B | 外部 Claude 写 harness | SearchQA（7 个 QA 集，700 题）、AIME24/25（60 题）、Lichess 国际象棋谜题（256 题） |

### 1c. 直接生成 adapter 的工作（需要单独训练的生成器）

| 论文 | 模型 | 训练任务 | 评测数据集 |
|---|---|---|---|
| **Text-to-LoRA** 2506.06105 | Mistral-7B 等 | Lots-of-LoRAs / SNI 的 479 个任务 | ARC-C/E、BoolQ、GSM8K、HellaSwag、OpenBookQA、PIQA、WinoGrande、HumanEval、MBPP |
| **Generative Adapter** 2411.05877 | Mistral-7B、Llama2 | 预训练式适配 | SQuAD、StreamingQA（知识注入）、MetaICL 26 个任务、MSC（个性化） |
| **Transformer²** 2501.06252 | LLaMA3-8B/70B、Mistral-7B | GSM8K、MBPP、ARC-Easy、TextVQA 上训练 SVF 专家 | 未见任务：MATH、HumanEval、ARC-Challenge |

### 1d. 训练后编辑 / 合并 LoRA（非 LLM 的方法，是 Paper B 的基线）

| 论文 | 模型 | 操作 | 评测数据集 |
|---|---|---|---|
| **SLoRA**（ACL 2026） | Llama3.1-8B、Qwen2.5-7B/32B | 每个任务训完 raw LoRA 后做 SVD 去噪 | **TRACE 8 个任务**（C-STANCE、FOMC、MeetingBank、Py150、ScienceQA、NumGLUE-cm、NumGLUE-ds、20Minuten）；标准持续学习 4 个任务；15 个任务的长序列；遗忘指标 AFR |
| **SDFT recovery** 2604.15794 | Qwen2.5-3B、Qwen3.5-9B、Gemma-4-12B | 压缩后用 LoRA 自蒸馏恢复 | Science（2,674 / 507 条）、Tooluse（4,046 / 97 条）、Math（GSM8K）、Code；通用能力：MMLU、Winogrande、ARC-C、HellaSwag、GSM8K、TruthfulQA |
| **Self-Data Distillation** 2410.09982 | Llama3.1-8B、Mistral-7B | 剪层后用 LoRA 恢复 | OpenLLM Leaderboard v1 的 6 项（ARC-C、HellaSwag、TruthfulQA、MMLU、Winogrande、GSM8K） |
| **LoraHub** 2307.13269 | FLAN-T5-large/xl | CMA-ES 搜索 20 个 LoRA 的组合系数 | BBH 27 个任务，5-shot |
| **EvoGM** 2605.29295 | Qwen2.5-1.5B、Qwen3-8B | 生成器搜索合并系数 | Tulu 多任务（验证集 200、测试集 1000）、GLUE |
| **TaDA** 2606.05016 | Llama-2-7B、ViT-L/16 | 探针门控 + SVD 过滤后合并 | 6 个科学 QA（含 MedMCQA、MMLU-CS 等）、6 个视觉分类数据集 |
| **PARA** 2604.27796 | 视觉、NLU、LLM | 全局奇异值阈值截断 | 视觉分类、GLUE、常识推理、数学 |
| **Audit Before You Merge** 2611.10684 | Llama-3.1-8B | 依次合并现成 LoRA | GSM8K（probe 50 / 选择 200 / test 1319）、HumanEval |
| **Sakana 进化合并** 2403.13187 | Mistral-7B、Llama-2-13B | CMA-ES 合并全模型 | MGSM-JA、JP-LMEH |

---

## 表 2：自量化 / agent 驱动压缩的工作

### 2a. 由 LLM 决定怎么压缩（被压缩的都是别的模型）

| 论文 | 被压缩的模型 | 谁决策 | 压缩方式 | 搜索时的信号 | 最终评测数据集 |
|---|---|---|---|---|---|
| **CompressionBench** 2611.06858 | **Qwen3.8 27B**、Nemotron-3.5 120B、Qwen3.5 122B | 6 个前沿 coding agent | 不限，agent 自己写算法和 kernel | BF16 自己生成的续写（Dolma，256 段 × 512 token）上的 PPL 比 | GSM8K（4-shot）、HumanEval（0-shot）、TriviaQA（5-shot）各 50 题，需 ≥0.8×BF16；本地测试用 MATH-500；校准可用 FineWeb、UltraChat、WikiText-103 |
| **AutoQS** 2611.11975 | Qwen2.5-7B、LLaMA-7B 等 | 外部 Qwen-Plus | 每层的 group size | ARC-C、PIQA、BoolQ 的平均（Macro3） | 7 项零样本（Macro7） |
| **Self-Pruner** 2502.14413 | LLaMA 7B–70B | GPT-4o | 逐层结构化剪枝率 | WikiText-2 PPL | WikiText-2 PPL；7 项常识零样本（WinoGrande、HellaSwag、BoolQ、ARC-e、ARC-c、OpenBookQA、PIQA）；vLLM 实测吞吐 |
| **Adaptive Pruning by Agents** 2601.09694 | Qwen3-4B / 8B | Gemini-3-flash | 逐轮逐层非结构化剪枝 | PPL 变化 + 回滚 | MMLU（5-shot）、FreebaseQA、WikiText-2 PPL；校准用 C4 |
| **ProfilingAgent** 2509.05584 | ResNet、ViT 等视觉模型 | GPT-4o | 剪枝 + 动态 int8 | 精度 > 参数量 > 延迟 | Imagenette、CIFAR-10/100、ImageNet-1K |
| **APQF** 2608.05499 | ResNet、ViT、Swin 等视觉模型 | Claude Opus 4.8 | 剪枝率 + 逐层比特 + 微调恢复 | 确定性评估 agent | ImageNet、CIFAR-10；理论 BOPs |

### 2b. 模型参与压缩自己，或非 LLM 的分配基线

| 论文 | 模型 | 做法 | 评测数据集 |
|---|---|---|---|
| **Self-calibration** 2410.17170 | Gemma 2B、Phi-2、OPT 6.7B、Mistral 7B、Llama 3.1 8B | 模型自己生成校准数据，方法由人固定（AWQ/GPTQ 4-bit、SparseGPT/Wanda 2:4） | 10 项零样本（ARC-e/c、BoolQ、HellaSwag、LAMBADA、OBQA、PIQA、RTE、StoryCloze、WinoGrande） |
| **GAMMA** 2605.18475 | Llama3.1-8B、Qwen3-8B/14B/32B | 非 LLM 算法给每个模块分配 {2,3,4} bit | 6 项零样本（ARC-e/c、PIQA、HellaSwag、WinoGrande、BoolQ）、GSM8K、RULER、Wiki2/C4 PPL；校准用 RedPajama |

### 2c. 压缩后能力评测研究（Paper A 选数据集的依据）

| 论文 | 模型（含 20–35B） | 评测数据集 | 关键发现 |
|---|---|---|---|
| **ACBench** 2505.19433 | Qwen2.5 3B–32B 等 | T-Eval（工具调用）、WorfBench（工作流）、AgentBoard（多轮环境）、LongBench、LongGenBench、Needle-in-a-Haystack | 4-bit 量化在工具调用上几乎无损，剪枝掉分明显；多轮真实环境最脆弱 |
| **UniComp** 2602.09130 | LLaMA-3.1-8B、Qwen3 等 | MMLU、ARC、HellaSwag、PIQA、WinoGrande、Global-MMLU-Lite、GSM8K、MATH-500、GPQA-Diamond、IFBench；TrustLLM 30 个可靠性数据集 | 量化综合最优；推理和指令遵循最容易受损 |
| **LLM-KICK** 2310.01382 | Vicuna 7B/13B/33B | FreebaseQA、MMLU、TriviaQA（闭卷 / 开卷）、CNN/DailyMail、MT-Bench | PPL 掩盖了能力下降；闭卷知识问答最敏感 |
| **Quantization Hurts Reasoning?** 2504.04823 | R1-Distill-Qwen-32B、QwQ-32B 等 | AIME-120、MATH-500、GSM8K、GPQA-Diamond、LiveCodeBench | W4 近乎无损，W3 开始明显掉分；GSM8K 不敏感 |
| **When Reasoning Meets Compression** 2504.02010 | R1-Distill-Qwen-32B 等 | AIME 2024、FOLIO、BBH Temporal Sequences、MuSiQue（闭卷） | 剪枝和蒸馏伤知识多于伤推理 |
| **Which Decisions Low-Bit Quantization Breaks** 2608.06564 | Qwen3 0.6B–32B 等 16 个模型 | BFCL（多个子集）、MetaTool、XSTest、MMLU、BoolQ | "要不要调用工具"最脆弱，3-bit 下的翻转率取决于量化器 |
| **Flat Score, Amplified Failures** 2607.27275 | Gemma-4-31B、Qwen-3.6-27B、Qwen-3.5-27B 等 | τ²-bench（retail、telecom） | 最终分数会掩盖损伤，要按失败类型报错误率 |

---

## 小结：哪些数据集最常用、最适合我们

| 用途 | 出现最多 / 最合适的数据集 | 用过它的论文 |
|---|---|---|
| 数学推理 | GSM8K（不敏感，只作可比性）、MATH-500、AIME | CompressionBench、PostTrainBench、Self-Developing、Quant Hurts Reasoning、MetaRSI |
| 代码 | HumanEval(+)、MBPP(+)、LiveCodeBench | CompressionBench、Self-MoE、T2L、PostTrainBench |
| 知识 | TriviaQA 闭卷、FreebaseQA、MMLU(-Pro) | CompressionBench、LLM-KICK、Adaptive Pruning、UniComp |
| 知识注入（LoRA） | SQuAD 无原文 QA | SEAL、Search over Self-Edit、Generative Adapter |
| 工具调用 / agent | BFCL、τ²-bench、T-Eval | PostTrainBench、Which Decisions、Flat Score、ACBench、TT-SI |
| 持续学习（LoRA） | TRACE 8 任务 | SLoRA |
| 领域任务 | TableBench、FinanceIQ、PANORAMA、Science / Tooluse | FT-Dojo、SDFT recovery |
| 常识零样本（压缩论文的惯例） | ARC、PIQA、BoolQ、HellaSwag、WinoGrande | Self-Pruner、Self-calibration、GAMMA、AutoQS |

---

## 表 3：按能力类别给数据集分类

"对压缩是否敏感"一列来自已读的压缩评测论文（LLM-KICK、UniComp、Quant Hurts Reasoning、ACBench、Which Decisions、Flat Score、CompressionBench）。"我们怎么用"对应 `EXPERIMENT_PLAN.md`。

| # | 类别 | 数据集 | 用过的论文 | 对压缩是否敏感 | 我们怎么用 |
|---|---|---|---|---|---|
| 1 | **数学推理** | GSM8K、MATH / MATH-500、AIME（2024 / 2025 / 2026、AIME-120）、AMC、Minerva、OlympiadBench、NumGLUE-cm/ds、MGSM-JA；训练集 DAPO-17K | CompressionBench、PostTrainBench、Self-Developing、MetaRSI、Quant Hurts Reasoning、R-Zero、Absolute Zero、Sakana | 量化下：32B 推理模型的 GSM8K 几乎不变，越难越敏感（AIME、MATH Level 5 掉得最多）；但 7B 模型 GPTQ 后 GSM8K 也会掉 14 分（ACBench），剪枝下 GSM8K 会崩（UniComp：76.8 → 19.5）| A：MATH-500 和 AIME 作主信号，GSM8K 只作可比性；B：T2 任务（GSM8K 训练，MATH-500 测试） |
| 2 | **代码** | HumanEval / HumanEval+、MBPP / MBPP+、LiveCodeBench、CruxEval、Py150、CodeMMLU | CompressionBench、Self-MoE、T2L、PostTrainBench、Absolute Zero、递归代码崩塌论文 | 中到高；LiveCodeBench 区分度高 | A：验收门 + 最终评测；B：T3 任务 |
| 3 | **闭卷知识 / 学科选择题** | MMLU、MMLU-Pro、Global-MMLU-Lite、GPQA / GPQA-Diamond、SuperGPQA、TriviaQA（闭卷）、FreebaseQA、MuSiQue（闭卷）、ScienceQA、MedMCQA、MMLU-CS | LLM-KICK、UniComp、Adaptive Pruning、CompressionBench、MetaRSI、TaDA | **选择题最稳**（UniComp）；**闭卷生成式问答最敏感**，FreebaseQA 在 30–35% 稀疏时就超出容忍线，8-bit GPTQ 也掉 8–10% | 两篇都用 MMLU-Pro 作对照、TriviaQA 闭卷作敏感探针 |
| 4 | **知识注入 / 阅读理解** | SQuAD（无原文 QA）、StreamingQA、TriviaQA 开卷（ICRA-QA） | SEAL、Search over Self-Edit、Generative Adapter、LLM-KICK | 开卷检索对剪枝相对稳定（LLM-KICK） | B：T1 任务，可直接对比 SEAL 47.0 和 Search 45.1 / 50.3 |
| 5 | **常识 / 语言理解（零样本选择题）** | ARC-Easy / Challenge、PIQA、BoolQ、HellaSwag、WinoGrande、OpenBookQA、LAMBADA、RTE、StoryCloze、COPA、CommonsenseQA、GLUE | Self-Pruner、Self-calibration、GAMMA、AutoQS、T2L、EvoGM、PARA | **不敏感**，是压缩论文的老惯例，会掩盖能力下降 | 只作和老论文对比的附表，不作主指标 |
| 6 | **复杂 / 逻辑推理** | BBH（27 个子任务）、BBEH、BBH Temporal Sequences、FOLIO；ARC 抽象推理（Chollet 版，与上面的 ARC 选择题不同） | Self-MoE、LoraHub、R-Zero、When Reasoning Meets Compression、SEAL | 中等 | 可选 |
| 7 | **指令遵循 / 对话** | IFEval、IFBench、MT-Bench、AlpacaEval 2.0、ArenaHard-Writing、MSC（个性化对话） | UniComp、LLM-KICK、Self-Rewarding、PostTrainBench、Generative Adapter | 取决于方法：4-bit 量化基本无损（UniComp：AWQ 保留 100%，GPTQ 89%）；剪枝明显受损（Wanda 50% 保留 73%；LLM-KICK 中 MT-Bench 只在约 25% 稀疏以内可接受）| 两篇都用 IFEval 作保留集 |
| 8 | **工具调用 / 函数调用** | BFCL（v3 / v4，irrelevance、live_simple、live_multiple、multi_turn）、T-Eval（含 JSON 子项）、MetaTool、ToolAlpaca、API-Bank、SealTool、Tooluse（SDFT 划分） | PostTrainBench、Which Decisions、ACBench、TT-SI、SDFT recovery | **"要不要调用"最脆弱**，3-bit 翻转率取决于量化器；JSON 输出比字符串输出更脆弱 | **A 的核心**：压缩后的模型还要当 agent；B：编辑能力测试任务之一 |
| 9 | **多轮 agent / 环境** | τ²-bench（retail、telecom）、AgentBoard（ScienceWorld、Jericho、PDDL、Tool-Query、Tool-Operation）、WorfBench（Alfworld、Webshop、OS、Lumos、ToolBench）、Terminal-Bench、SWE-bench / SWE-bench Pro、ScienceAgentBench、MLR-Bench、AstaBench、DSBench、SearchQA | Flat Score、ACBench、MetaRSI、MASS、Self-HPO、WHALE | **最脆弱**（ACBench：真实应用掉 10–15%）；最终分数会掩盖损伤（Flat Score） | A：τ²-bench，按失败类型报错误率和 S(K) |
| 10 | **长上下文** | LongBench、LongGenBench、Needle-in-a-Haystack、RULER | ACBench、GAMMA | 长输出推理会崩（ACBench：3B 从 61% 掉到 11%） | 可选，第三阶段再加 |
| 11 | **领域任务** | 生物医学（BioASQ 等 10 个）、金融（FinanceIQ、FOMC 等 10 个）、法律（LawBench）、专利（PANORAMA）、化学（ChemCoTBench）、表格（TableBench / TableInstruct）、医疗（HealthBench）、科学（SDFT 的 Science） | Self-Specialization、FT-Dojo、SIA、PostTrainBench、SDFT recovery | — | B：T4 Science、T5 TableBench；FinanceIQ、PANORAMA 留作编辑能力测试任务 |
| 12 | **持续学习序列** | TRACE 8 任务（C-STANCE、FOMC、MeetingBank、Py150、ScienceQA、NumGLUE-cm、NumGLUE-ds、20Minuten）、标准 CL 4 任务、15 任务长序列 | SLoRA | 用遗忘指标 AFR 衡量 | B：可加 TRACE 作为和 SLoRA 直接对比的附加实验 |
| 13 | **安全 / 可信** | XSTest、BBQ、TrustLLM 的 30 个数据集（含 Jailbreak Trigger、Do-Not-Answer、HaluEval、TruthfulQA 等） | UniComp、Which Decisions | 性能保留与安全无关（UniComp ρ=−0.04）；GPTQ 可能明显削弱越狱抵抗 | A：XSTest 作轻量安全探针（可选） |
| 14 | **语言建模 PPL / 校准数据** | WikiText-2、WikiText-103、C4、FineWeb、UltraChat-200k、RedPajama、Pile、Dolma（BF16 续写） | 几乎所有压缩论文 | PPL **会掩盖能力下降**（LLM-KICK、CompressionBench） | A：BF16 续写上的 KL 作搜索信号；C4 / FineWeb / 自生成数据作校准数据选项。不用 WikiText-2 测试集做搜索目标 |
| 15 | **训练用的指令 / 任务集合** | SNI / Lots-of-LoRAs（479 个任务）、FLAN、Tulu、Alpaca、OpenMathInstruct、Open Assistant、Dolly | T2L、LoraHub、EvoGM、Self-Data Distillation、Self-Rewarding | — | 一般不用；B 如需额外训练数据再考虑 |
| 16 | **视觉 / 其他（不在范围内）** | ImageNet、CIFAR-10/100、Imagenette、PathMNIST、TextVQA；Lichess 谜题、TriMul（CUDA kernel）、scRNA 去噪 | ProfilingAgent、APQF、TaDA、PARA、Transformer²、WHALE、SIA | — | 不用 |

### 按两篇论文汇总

| 用途 | Paper A 自量化 | Paper B 自编辑 LoRA |
|---|---|---|
| 搜索时的反馈（L1） | 类别 14（BF16 续写 KL）+ 类别 1（MATH-500 100 题）+ 类别 8（BFCL 100 题） | 当前任务的 dev + 保留集探针（类别 3、7） |
| 验收门（L2） | 类别 1、2、3（GSM8K / HumanEval / TriviaQA 各 200 题）+ 类别 8、9 的错误率 | 当前任务的 dev2 |
| 最终评测（L3） | 类别 1、2、3、7、8、9 全部，加上效率 | 每个任务的 test + 类别 3、7 保留集 |
| 训练任务 | 无（不训练） | 类别 4（SQuAD）、1（数学）、2（代码）、11（Science、TableBench） |
| 编辑能力测试 E_t | 固定任务：在 12 GiB 下压缩 BF16 源 | 类别 11（FinanceIQ、PANORAMA）+ 类别 8（BFCL） |
