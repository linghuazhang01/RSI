# 已读论文中的量化 / 剪枝实验设置对照（面向 27B 自压缩）

日期：2026-11-13
来源：本目录下 19 篇全文笔记中，与压缩直接相关的 6 篇。关键数字已在 `papers/src_<id>/<id>.pages.md` 原文中抽查核对。

## 1. 总览

| 论文 | 被压缩的模型 | 谁做决策 | 压缩方式 | 模型能决定的范围 | 校准数据 | 搜索时的反馈信号 | 最终评测 | 是否实测加速 |
|---|---|---|---|---|---|---|---|---|
| **CompressionBench** (2611.06858) | **Qwen3.8 27B（dense）**；Nemotron-3.5 120B-A12B、Qwen3.5 122B-A10B（MoE） | Sonnet/Opus 5.5、GLM 5.3、Kimi K3.1、MiniMax M3.1、DeepSeek V4.1（Claude Code 外壳） | 不限：GPTQ、Hadamard 旋转、trellis codes、混合精度、专家/MLP 剪枝、低秩残差、蒸馏 | 全部（算法、代码、推理 kernel） | 提供 FineWeb、UltraChat-200k、WikiText-103，由 agent 自选 | 在 BF16 自己生成的续写上的 PPL 比（等价于逐 token KL） | GSM8K、HumanEval、TriviaQA 各 50 题，需 ≥0.8×BF16；外加完整性审查 | 测 batch-1 decode TPS，但不计分；多数产物比 BF16 慢 |
| **Self-Pruner** (2502.14413) | LLaMA-1/2/3/3.1、Vicuna（7B–70B） | GPT-4o | 结构化剪枝（Wanda-sp） | 每层剪枝率向量 | 未单列（Wanda-sp 默认） | WikiText-2 PPL | WikiText-2 PPL；7 个零样本常识任务 | 是：vLLM，A100，最高 1.82× |
| **Adaptive Pruning by Agents** (2601.09694) | Qwen3-4B、Qwen3-8B | Gemini-3-flash | 非结构化剪枝（Wanda），逐轮加稀疏 | 每轮选层及额外稀疏率（0.01–0.15）、是否停止 | C4 128 条 × 2048 | PPL 变化 + 文字反思；单轮 PPL 涨 >15% 回滚 | MMLU（5-shot）、FreebaseQA、WikiText-2 PPL | 否 |
| **ProfilingAgent** (2509.05584) | ResNet-101、ViT-B/16、Swin-B、DeiT-B（视觉模型） | GPT-4o | 结构化/head 剪枝 + PyTorch 动态 int8 量化 | JSON：层正则、剪枝比例、dtype | — | 精度 > 参数量 > 延迟 | Imagenette、CIFAR-10/100、ImageNet-1K | 是：延迟、内存 |
| **Self-calibration** (2410.17170) | Gemma 2B、Phi-2、OPT 6.7B、Mistral 7B、Llama 3.1 8B | 人（模型只负责生成校准数据） | AWQ、GPTQ（4-bit，g128）；SparseGPT、Wanda（2:4） | 无 | **模型自己生成**，128 × 2048 | — | 10 个零样本任务（lm-eval-harness） | 否 |
| **Anosognosia** (2610.06174) | Qwen3-8B、Llama-3.1-8B、**Gemma-3-27B-IT**、Qwen3-32B | 人（研究的是模型能否察觉自己被量化） | GPTQ（g64/128/256）、AWQ、RTN；8/4-bit | 无 | C4 128 × 2048 | — | MMLU-Pro、BBH（作为量化损伤的参照） | 否 |

**最值得注意的一点**：CompressionBench 里有一个 dense 任务用的正是 **27B 模型（Qwen3.8 27B）**，存储预算分 8 GiB 和 12 GiB 两档，而且给出了 6 个前沿 agent 在这个模型上的完整结果。如果你们的 27B 选同一个 checkpoint，可以直接复用它的评测协议，并把它的前沿 agent 结果当作"外部强模型"上界，不用自己再跑。

## 2. 每篇具体怎么做

### 2.1 CompressionBench（与 27B 最相关）
- **任务**：给 agent 一个只读的源 checkpoint、存储上限、一张 H200 NVL、vLLM 0.32.0、8 小时墙钟；在离线容器里，它可以查阅压缩论文、使用校准数据、在 MATH-500 上本地测试（p.3、p.5）。
- **两个存储约束**：磁盘包大小（含代码和 tokenizer）和推理时 GPU 常驻权重（含 scale、codebook、反量化副本）都要 ≤ 上限，KV cache 不计（p.3）。之所以分开算，是因为 pilot 里 QTIP 包只有 6.96 GiB，推理时却保留了 BF16 权重（p.17）。
- **奖励**：在 Dolma 3 Dolmino 的 256 段 × 512 token 上下文后，由 BF16 以温度 1 生成 1,536 token 续写；压缩模型对这些续写打分，奖励为 10 × PPL(BF16)/PPL(压缩)（p.4、p.6）。
- **最终验证**：在独立容器、用不相交的上下文重测 PPL；GSM8K（4-shot）、HumanEval（0-shot，单元测试）、TriviaQA（5-shot）各 50 题，greedy 解码，平均准确率需 ≥0.8×BF16，这个阈值对 agent 隐藏（Table A2，p.16）。
- **27B 上的结果（Table 1、Table A3，p.6、p.18）**：

| Qwen3.8 27B | PPL ↓ | Acc ↑ | TPS ↑ | 是否通过 |
|---|---|---|---|---|
| BF16 | 3.221 | 74.00 | 61.4 | — |
| **12 GiB** 统一 GPTQ | 3.546 | 46.00 | 11.1 | 不通过 |
| 12 GiB 已发布 GSQ-RCO | 3.334 | 66.00 | 74.6 | 通过 |
| 12 GiB Sonnet 5.5 | 3.253 | 76.67 | 16.7 | 通过 |
| 12 GiB Opus 5.5 | 3.272 | 78.67 | 31.4 | 通过 |
| 12 GiB 其余 4 个 agent | 3.333–3.399 | 72.00–80.67 | 8.0–18.4 | 全部通过 |
| **8 GiB** 统一 GPTQ | 7.480 | 4.67 | 12.6 | 不通过 |
| 8 GiB GSQ-RCO | 3.764 | 65.33 | 75.5 | 通过 |
| 8 GiB Sonnet / Opus / DeepSeek | 3.363 / 3.370 / 4.283 | 68.67 / 66.00 / 61.33 | 13.7–26.2 | 通过 |
| 8 GiB GLM / Kimi / MiniMax | 6.061 / 4.176 / 8.163 | 23.33 / 54.00 / 9.33 | — | 不通过 |

  12 GiB 档 6 个前沿 agent 全部通过，8 GiB 档只有 3 个通过。
- **27B pilot（Table A4，p.19，C4 PPL，BF16 = 9.9227）**：16 GiB 下 4-bit GPTQ g128 PPL +3.19%；12 GiB 下统一 3-bit +13.32%，3/4-bit 混合 +10.53%；8 GiB 下各种 2/3/4-bit 方案在 +48% 到 +1739% 之间。
- **Sonnet 在 27B/12 GiB 上的完整方案（Fig 9a，p.10）**：Hadamard 旋转 + trellis codes（QTIP 式）+ 按敏感度分配比特（mixer 约 3.8–4.5 bit，MLP 约 3.7–3.8 bit），embedding 和输出头按 token 频率给 3–8 bit，再写对应的 vLLM kernel。它试过 Fisher 加权分配，KL 变差后放弃。
- **坑**：PPL 下降不等于能力保持。Opus 在 27B/8 GiB 上做蒸馏，PPL 一路下降，GSM8K 却从 88% 掉到 72%（p.9、p.17）。

### 2.2 Self-Pruner
- 让 GPT-4o 跑进化搜索：先一次生成 30 个每层剪枝率配置（平均值等于目标），再按 fitness 做选择、交叉、变异，共 20 代（p.5–7）。
- 层内剪哪些神经元由 Wanda-sp 决定，fitness 是 WikiText-2 PPL。
- **数据集**：WikiText-2 PPL；7 个零样本任务：WinoGrande、HellaSwag、BoolQ、ARC-e、ARC-c、OpenBookQA、PIQA（lm-eval-harness）。
- **结果**：LLaMA-2-70B 剪 30% 后，零样本平均 73.81 → 73.01；vLLM 实测 LLaMA-2-7B 剪 50% 后 1.82× 加速（Table 2、Table 7，p.8–9）。
- **问题**：搜索和汇报用的是同一个 WikiText-2 PPL，相当于在测试集上搜索；也没有和同等预算下的非 LLM 进化算法比较。

### 2.3 Adaptive Pruning by Agents
- 每轮先算每层的 Wanda 分数和梯度重要性（z-score），然后让 Gemini 读层画像、当前 PPL 和上轮反馈，输出给哪些层加多少稀疏（p.5–7）。
- **数据集**：C4 128 条校准；MMLU 5-shot、FreebaseQA（20,358 题，用来测事实知识）、WikiText-2 PPL。
- **结果**：Qwen3-8B 稀疏约 43% 时 MMLU 56.67，FreebaseQA 25.16（dense 77.38 / 50.56）（Table 2，p.10）。
- **问题**：对比的基线是 2:4/4:8 magnitude，缺"均匀稀疏的 Wanda"和 OWL 这两个关键对照；非结构化稀疏没有实测加速。

### 2.4 ProfilingAgent
- 先用 profiler 测每层延迟、内存、MACs，交给 GPT-4o 出 JSON 方案，再由 DependencyGraph 剪枝、quantize_dynamic 量化（p.4–5）。
- 对象是视觉模型，对 27B LLM 的参考价值主要在"把 profiling 结果交给决策模型"这个设计。

### 2.5 Self-calibration
- 被压缩的模型只以 BOS 为条件、温度 1 自由生成 128 × 2048 token 作校准集，然后跑标准的 AWQ/GPTQ/SparseGPT/Wanda（p.3–5）。
- **结果**：剪枝时收益较明显，例如 SparseGPT 剪 Llama 3.1 8B，自生成数据 53.8，Cosmopedia 50.9，C4 54.8；量化对校准数据不太敏感（Table 2，p.5）。
- **对你们的用处**："用自己生成的数据给自己校准"可以作为自压缩动作空间里的一个选项，零成本，也天然符合"不借助外部"。

### 2.6 Anosognosia（给的是约束，不是方法）
- Gemma-3-27B-IT 用 AWQ 量化后，MMLU-Pro 8-bit 降 18.8、4-bit 降 25.5（Table 2/3，p.19），说明 27B 量级对量化方法很敏感，不同方法差别大。
- 模型从自己的输出看不出自己被量化过，训练出的识别能力换个量化方法就失效（p.4、p.8）。所以自压缩的评估信号必须来自外部测试，不能让模型自评。

## 3. 对 27B 自压缩实验的具体建议

1. **模型选 Qwen3.8 27B**：CompressionBench 的 BF16 参考、统一 GPTQ、已发布量化和 6 个前沿 agent 的结果都能直接引用，外部强模型上界不用自己跑。如果换别的 27B（如 Gemma-3-27B），这些都要重跑。
2. **预算从 16 GiB 和 12 GiB 开始**：16 GiB 统一 4-bit 已经接近无损（+3%），可以当 sanity check；12 GiB 统一 GPTQ 不过关（46%）而前沿 agent 全部通过，这一档最能看出"决策质量"；8 GiB 前沿 agent 也有一半失败，留到后面。
3. **第一阶段用配置级动作**（参照 Self-Pruner / Adaptive Pruning 的做法，执行交给现成库）：
   - 量化：每层或每个模块的比特数 {2, 3, 4, 8}、group size、方法 {GPTQ, AWQ, RTN}、embedding 和输出头的精度；
   - 剪枝：每层稀疏率，层内用 Wanda / SparseGPT；如果要实测加速，用结构化剪枝（Self-Pruner 在 vLLM 上测到了 1.82×）；
   - 校准数据：C4 / FineWeb / 自己生成（Self-calibration）。
4. **评估信号**：
   - 搜索时：在 BF16 自己生成的续写上算 PPL/KL（CompressionBench 的做法）。不要用 WikiText-2 测试集做 fitness，这正是 Self-Pruner 的问题。
   - 最终：GSM8K、HumanEval、TriviaQA，建议每个用远多于 50 题（CompressionBench 只有 50 题，噪声大）；再加 lm-eval 7 个零样本常识任务（便于对比 Self-Pruner）和 MMLU-Pro（便于对比 Anosognosia 的 27B 数字）。
   - **自压缩特有的一项**：压缩后的模型还能不能继续当压缩 agent，比如重跑同一个压缩任务，看通过率和 PPL。这正是 CompressionBench 结论里列为未来工作的内容。
5. **效率要真测**：CompressionBench 的产物大多比 BF16 慢（27B/12 GiB 下最快 31.4 TPS，BF16 是 61.4），因为自定义格式的反量化开销。建议要么把 TPS 放进奖励，要么只允许有现成高效 kernel 的格式（例如 vLLM 支持的 GPTQ/AWQ 4-bit），否则"提升推理效率"这个卖点站不住。
6. **硬件参考**：CompressionBench 每个 run 用一张 H200 NVL、8 小时。27B BF16 权重约 54 GB，自压缩时模型既要作为 agent 推理，又要被量化和评估，建议一张 H200（141 GB）或两张 80 GB 卡，把 agent 推理和压缩评估放在不同的卡上（这是估算，没有实测）。
