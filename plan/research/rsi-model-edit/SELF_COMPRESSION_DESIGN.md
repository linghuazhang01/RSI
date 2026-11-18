# 27B 模型自压缩：路线、数据集与流程（基于文献的方案）

日期：2026-11-13
依据：本轮新精读 11 篇，加上之前精读的 19 篇。中文笔记都在 `papers/<id>.notes.md`，每篇至少抽查 2 个关键数字，均与 `papers/src_<id>/<id>.pages.md` 原文一致。

新读的 11 篇：

| ID | 论文 | 主要用途 |
|---|---|---|
| 2505.19433 | ACBench（ICML 2025） | 压缩后 agent 能力 |
| 2602.09130 | UniComp | 量化 / 剪枝 / 蒸馏统一对比，40+ 数据集 |
| 2310.01382 | LLM-KICK（ICLR 2024） | PPL 掩盖能力下降；量化 vs 剪枝 |
| 2504.04823 | Quantization Hurts Reasoning?（COLM 2025） | 32B 推理模型的比特安全区 |
| 2504.02010 | When Reasoning Meets Compression（ICLR 2026） | 32B 量化与剪枝的断崖 |
| 2609.01587 | The Structure of Quantization Damage | 比特该怎么分配 |
| 2608.06564 | Which Decisions Low-Bit Quantization Breaks | 工具调用在哪个比特数崩 |
| 2607.27275 | Flat Score, Amplified Failures | **27B–31B** 模型 INT4 后的 agent 损伤 |
| 2606.28438 | When AI Reviews Its Own Code | 递归自训练崩塌，需要外部验证 |
| 2410.09982 | Self-Data Distillation | 剪枝后用自生成数据做 LoRA 恢复 |
| 2604.15794 | SDFT as Recovery | 压缩后 on-policy 自蒸馏 LoRA 恢复 |

这 11 篇都没有做"模型压缩自己"，之前的新颖性判断不变。它们的作用是告诉我们该选什么压缩方式、在哪个范围里做决策、用什么来评测。

---

## 1. 结论：做量化，剪枝和 LoRA 作为后续选项

### 1.1 主线选量化（weight-only，混合精度分配）

证据很一致：
- **相同压缩率下，量化全面优于剪枝。**
  - UniComp（Table 1，p6）：LLaMA-3.1-8B 用 AWQ W4A16，推理保留率 87.4%；Wanda 50% 只剩 40.2%，2:4 约 30%。
  - When Reasoning Meets Compression（Table 1/2，p6–7）：R1-Distill-Qwen-32B 原版 82.2，4-bit AWQ/GPTQ 83.1/83.0；SparseGPT 50% 降到 67.1，AIME 降到 30.0。
  - LLM-KICK（p9）：Vicuna-13B 剪到 7B 参数量后，MMLU 不如稠密的 7B。对"27B 剪成约 14B"这条路是负面证据。
- **剪枝在部署上不划算。** 非结构化 50% 剪枝在 vLLM 上显存和速度都不变；W4A16 把显存从 20.39 GB 降到 10.53 GB（UniComp Table 3，p8）。2:4 能提速约 1.7 倍，但会毁掉推理和工具调用（ACBench T-Eval：Qwen2.5-7B SparseGPT 2:4 只有 44.6，FP16 70.0）。
- **工具调用对量化相对耐受，对剪枝很敏感。** 这一点对自压缩是决定性的，因为压缩后的模型还要接着当 agent。ACBench（Table 8，p32）中，Qwen2.5-7B 量化后 T-Eval 不掉（70.0 → 70.1），Wanda 剪枝掉到 62.3。

### 1.2 研究价值在 3–4 bit 之间

| 比特范围 | 损伤程度 | 证据 | 对研究的意义 |
|---|---|---|---|
| 8-bit（W8A16） | 基本无损 | Flat Score：26–35B 上没有任何单元格显著变化（Table 4，p10）；Damage Structure：RTN8 与 fp16 差 −0.001 | 太容易，没有决策空间 |
| 4-bit | 最终分数几乎不变，过程损伤因模型而异 | 32B W4 −0.4（Quant Hurts Reasoning Table 1，p5）；Qwen-3.6-27B INT4 工具名幻觉 0.38% → 0.47%，不显著；Gemma-4-31B 从 19.5% 翻到 38.3%（Flat Score Table 1/5） | 适合作为第一轮目标 |
| 3-bit | 断崖，但取决于量化器和比特分配 | 32B W3 −3.9 到 −5.1；3-bit GPTQ 72.5 vs 原版 82.2；"要不要调用工具"的翻转率：RTN 47%，GPTQ 21%，GGUF iq3_xxs 10%（Which Decisions Table 1，p5） | **核心决策区**：好的分配和坏的分配差别巨大 |
| 2-bit | 工具调用决策被擦除 | Which Decisions §3：margin 收缩系数中位数为 0 | 只作为压力测试 |

所以，"模型自己决定怎么分配比特、用什么量化器、用什么校准数据"在 3–4 bit 区间最有意义。这和 CompressionBench 的观察一致：27B 在 12 GiB 下统一 GPTQ 的准确率只有 46%，前沿 agent 全部通过。

额外约束：激活和 KV cache 保持 16/8 bit。A4 会毁掉工具调用 margin（Which Decisions Fig.5，p15），W4A4KV4 时 32B 也掉 2.9（Quant Hurts Reasoning Table 1）。

### 1.3 LoRA：第二阶段作为"压缩后自我恢复"动作加入，不做独立主线

- 作为恢复手段，证据只到"部分恢复"，没有"超过原模型"的证据：
  - Self-Data Distillation（Table 1，p6）：删 6 层后自蒸馏 LoRA 把 6 项平均从 51.82 恢复到 63.33，但收益几乎全来自 GSM8K，而训练数据就是数学数据。轻度剪枝（删 2 层）时净增益是 −0.71。
  - SDFT（Table 2，p8）：Wanda 50% 后 MMLU 从 77.6 掉到 55.21，自蒸馏只补回 60–63。
- 在 4-bit 附近本来就近乎无损，LoRA 没有东西可恢复。它只在第二阶段、比特数压到 3-bit 区间后作为一个可选动作才有意义。
- 符合"不借助更大模型"的做法是 SDFT 式自蒸馏：压缩后的模型自己生成补全，由**它自己压缩前的 BF16 版本**给 token 分布当老师。老师是同一个模型，不是外部更强的模型。
- 下游任务 LoRA（原方向 b）离 SEAL 和 Search over Self-Edit 太近，而且后者的结果是模型自写模板不如人写模板（45.1 vs 50.3）。建议与压缩线分开，不要混在一篇里。

### 1.4 剪枝：放到第三阶段，只做结构化的层删除

如果后面要扩大动作空间，只加"删除整层"这类结构化剪枝（有真实加速）。不做非结构化剪枝和 2:4。LLM-KICK、UniComp、ACBench 都显示这两种对知识、推理、JSON 输出伤害最大。

---

## 2. 流程

### 2.1 每一轮在做什么

```
第 t 轮
  编辑者 = M_t（第 0 轮是 BF16 原模型；之后是上一轮压缩出的模型）
  被压缩对象 = 同一模型的 BF16 源权重（只读）
  目标 = 存储预算 B_t（例如 16 → 12 → 10 GiB）

  1. 编辑者调用工具：
     profile(bits)       -> 每层 / 每个模块在给定比特下的 KL 敏感度
     quantize(config)    -> 用 GPTQ / AWQ / RTN 执行（llm-compressor / GPTQModel）
     evaluate(candidate) -> 返回搜索信号
     submit(candidate)
  2. config 包括：每层或每个模块的比特数 {3, 4, 8}、group size、
     量化器、embedding 和输出头的精度、校准数据来源（C4 / FineWeb / 自生成）
  3. 冻结的外部评测器验证：存储、能力门槛、agent 门槛；不通过就回滚
  4. 通过的候选成为 M_{t+1}，当下一轮的编辑者
```

**为什么每轮都从 BF16 源权重出发，不在已量化的权重上再量化：** 重复量化会叠加舍入误差，而这种误差和"编辑者变笨了"混在一起，无法分开。从同一份源权重出发，"递归"体现在编辑者换成了压缩后的自己，被编辑的始终是"自己的权重"。另一种设计是在 M_t 上继续压缩，可以作为对照组。

### 2.2 防止递归崩塌（When AI Reviews Its Own Code 的教训）

- 判断"合格与否"的信号必须是**外部、固定**的：冻结的评测集、固定的 BF16 参考输出、确定性检查。不能用压缩后模型自己的 PPL 或自评。该文定理 2.3 说明自评分在模型自己的输出上会退化成常数；Which Decisions 也显示 PPL 只升到 1.01 倍时，工具调用率已经从 .910 掉到 .450（Table 4，p21）。
- 只检查格式或语法的弱外部门不够：编译门保住了可执行率，保不住 HumanEval 正确率（该文 Table 3，p8）。
- 用户模拟器和评测器不能跟着一起被压缩。Flat Score 的模拟用户固定为全精度 Qwen-3.6-27B。

### 2.3 必须有的对照组

| 对照 | 回答的问题 |
|---|---|
| 统一 GPTQ / AWQ，同预算 | 自己做决策是否有用 |
| 非 LLM 搜索（按敏感度贪心 / EvoPress 式），同样的评估次数 | 是 LLM 的推理在起作用，还是任何搜索都行 |
| 每轮都用冻结的 M_0 当编辑者 | 编辑者能力下降对递归的影响（**核心对照**） |
| 从 M_0 一步压到最终预算 | 递归累积的损伤 vs 单步压缩的损伤（2606 缺这个对照，所以无法归因） |
| 外部强 agent（若用 Qwen3.8 27B，可直接引用 CompressionBench Table 1） | 上界 |
| 随机合法配置 | 下界 |

---

## 3. 数据集

### 3.1 搜索时的反馈信号（便宜、固定、外部）

- **主信号**：在 BF16 自己生成的续写上算 KL/PPL（CompressionBench 的做法：256 段上下文 × 512 token，续写 1,536 token）。只需要 BF16 参考，不需要外部模型。
- **辅助的小探针**：MATH-500 抽 100 题，BFCL 抽一组"该调用 / 不该调用"的题。
- **不要用 WikiText-2 测试集做搜索目标**（Self-Pruner 的问题），也不要只看 PPL（LLM-KICK、CompressionBench 都显示 PPL 会掩盖能力下降）。
- **校准数据**也是决策变量。推理类校准对 GPTQ 很重要：换成推理数据后平均 +9.8（Quant Hurts Reasoning Table 2，p9）。但同样的校准用在 AWQ 上会伤 HellaSwag（UniComp Table 12）。

### 3.2 最终能力评测（每轮都测）

| 能力 | 数据集 | 选它的理由 |
|---|---|---|
| 知识（多选） | MMLU-Pro | 对比基准；多选知识对压缩最不敏感（UniComp），只作对照 |
| 知识（闭卷生成） | TriviaQA 闭卷、FreebaseQA | 对剪枝和量化都比多选敏感得多（LLM-KICK；8-bit GPTQ 也掉 8–10%） |
| 数学推理 | MATH-500、AIME（2024–25） | 任务越难越敏感；GSM8K 几乎不敏感（Quant Hurts Reasoning §3.4） |
| 代码 | HumanEval+、MBPP+、LiveCodeBench | 编辑者要写配置和脚本；LiveCodeBench 区分度高 |
| 指令遵循 | IFEval 或 IFBench | 压缩伤指令遵循（UniComp） |
| 可比性 | GSM8K（4-shot）、HumanEval（0-shot）、TriviaQA（5-shot） | 与 CompressionBench 协议一致（Table A2），能直接对比前沿 agent；建议题量远大于 50 |

同时监控输出长度和重复率：3-bit 和剪枝崩溃时会反复输出同一段（When Reasoning Meets Compression App P；Quant Hurts Reasoning Fig.3）。

### 3.3 Agent 能力评测（自压缩特有，最重要）

| 数据集 | 测什么 | 怎么报 |
|---|---|---|
| **自压缩任务本身** | 压缩后的模型还能不能完成下一轮压缩 | 通过率、最终 KL/PPL、工具调用失败数 |
| BFCL v3/v4 | 函数调用 | "该调用 / 不该调用"两侧分开报（irrelevance 与 live_simple），加上 live_multiple、multi_turn_base；报配对翻转率（Which Decisions） |
| τ²-bench（retail、telecom） | 多轮 agent | 按失败通道报错误率，以及成功率随失败预算 K 的曲线 S(K)；最终分数会掩盖损伤（Flat Score） |
| T-Eval（含 JSON 子项） | 工具使用的子能力 | JSON 输出比字符串输出脆弱得多（ACBench Fig.3） |

可选：XSTest 作为轻量安全探针，因为 GPTQ 可能明显削弱越狱抵抗（UniComp Table 17）。

### 3.4 效率（必须实测）

在 vLLM 上测显存峰值、batch=1 的 decode TPS、batch=32 的吞吐和 prefill。CompressionBench 的 agent 产物大多比 BF16 慢；UniComp 的吞吐测试只测了 prefill。建议 config 只允许 vLLM 有高效 kernel 的格式（例如 Marlin 的 GPTQ/AWQ 4-bit；3-bit 和混合精度要先确认 kernel 支持），或把 TPS 放进奖励。

---

## 4. 建议的 pilot 顺序（每一步都有判断条件）

1. **Pilot 0：先测"编辑者"在哪里变笨（约 1–2 天）。** 把 27B 用统一 W8 / W4 / W3（GPTQ 与 AWQ）压一遍，只跑 §3.3 的 agent 评测和自压缩任务。
   - 如果 W4 下 agent 能力已明显受损，递归会很快失败，那么这本身就是发现，预算要设得宽一些。
   - 如果到 W3 才受损，16 → 12 GiB 的递归区间可行。
2. **Pilot 1：单轮决策质量。** 让 M_0 在 12 GiB 下做配置级压缩，与统一量化和非 LLM 搜索比较。不如非 LLM 搜索的话，要先改动作空间或工具。
3. **Pilot 2：两到三轮递归。** 加上"冻结 M_0 当编辑者"和"一步压到底"两个对照，看编辑者能力的变化是否影响压缩质量。
4. **第二阶段**：加 SDFT 式自蒸馏 LoRA 作为可选动作，做有 / 无对照。
5. **第三阶段**：加层删除，或放开到代码级动作。

模型建议仍然是 Qwen3.8 27B，原因是可以直接对比 CompressionBench 的前沿 agent 结果。注意 Flat Score 显示，同为 27B 的 Qwen-3.6 和 Qwen-3.5 对 INT4 的耐受度不同，所以 Pilot 0 必须在你们实际选的那个 checkpoint 上跑，不能套用别的模型的结论。

## 5. 证据的薄弱处（如实说明）

- 20B–35B 上的直接证据只有：Flat Score（全部 26–35B）、32B 的推理量化（2504.04823、2504.02010）、Qwen2.5-32B 的 WorfBench（ACBench）、CompressionBench 的 Qwen3.8 27B。剪枝和 LoRA 恢复在 27B 上没有直接证据，最大只到 12B。
- 所有论文都没研究"多轮叠加压缩是否累积误差"，也没有"压缩后的模型当编辑者"的数据。这正是本工作要回答的问题，也是风险所在。
- UniComp 中 Qwen3-32B 的 dense 基线（GSM8K 71.72）低于 14B（89.31），评测可能有问题，没有引用这组数字作为证据。
- 硬件需求（一张 H200 或两张 80 GB 卡）是估算，没有实测。
