# SIGNALS_QUANT：量化决策用了什么信息、成本多少、哪些真正有用（含 LLM-agent 压缩工作里 agent 看到什么）

日期：2026-11-18
服务对象：Qwen3.8 27B 自量化项目。模型为自己选择仅权重量化配置：每模块 bits ∈ {2,3,4,8}、GPTQ/AWQ/RTN、group size、embedding/head 精度、校准来源，受存储预算约束；工具形如 `profile(module, bits, quantizer)` → KL 增量；反馈来自固定的外部评测器。

页码约定："p.N" 指对应 `.pages.md` 中的 `## Page N`。逐篇笔记在 `papers/<id>.notes.md`（新读）以及 `../../self-quantization/papers/`、`../../rsi-model-edit/papers/`（已有）。凡是我的推断都标【推断】。

阅读状态：
- **FULL-new**：本轮新下载全文。由并行子 agent 按统一 brief 逐篇通读并写笔记，主 agent 对每篇至少用 grep 核对了 2 个数字。
- **FULL-existing**：已有全文笔记，必要时重开 `.pages.md` 核对。
- 特殊情况：
  - HAWQ-V2 抽取到的是 arXiv v1（2019），不是 NeurIPS 2020 版。
  - TAP 的 arXiv PDF 只有 9 页，正文引用的附录 A–G 不在其中。

---

## 1. 总表

| 方法 | 决定什么 | 用的信息/信号 | 怎么算 | 成本 | 决策过程 | 关键证据（表/页） | 状态 |
|---|---|---|---|---|---|---|---|
| **GPTQ** 2210.17323 | 每个权重如何舍入，并对剩余列做误差补偿；bits/group 由人给定 | 层 Hessian H=2XXᵀ（完整输入协方差） | C4 上 128×2048 token；阻尼 λ 取对角均值的 1%；逐块量化，后层看到的是前层已量化后的输入 | 单张 A100：OPT-13B 20.9 min，175B 4.2 h（p.7） | 逐列闭式 OBQ 更新 | Tab.3 p.7；group size 与额外比特的关系（p.8，Tab.5）；2-bit OPT-175B：g128 9.58 → g32 8.94 | FULL-new |
| **AWQ** 2306.00978 | 每个输入通道的缩放 s=s_X^α，以及 clipping；bits/group 给定 | 每通道平均激活幅值 s_X | 用少量 Pile 样本缓存层输入，对 α 做 20 点网格，最小化层输出误差 | 一次校准前向 + 每层 20 次评估，无反传（论文未报告具体时间，【推断】） | 逐层 1-D 网格 | **Tab.1 p.4**：OPT-6.7B INT3，保留 1% 通道为 FP16 时，按激活选 11.39，按权重选 22.37，随机选 24.23（RTN 为 23.54）。**Fig.8b p.10**：跨域校准时 GPTQ 多掉 2.33–4.89 PPL，AWQ 只多掉 0.50–0.60 | FULL-new |
| **OWQ** 2306.02272 | 每层哪些输入列保留 fp16，其余用 3/4-bit OPTQ | sens_j = λ_j·‖ΔW_:,j‖²，λ_j 是 H 的对角，即该通道激活能量 | 复用 GPTQ 的 Hessian 对角；C4 128×2048 | 几乎免费；66B 在单卡上 <3 h | 每层取 top-k，k 由预算公式给出 | **Tab.7 p.8**：只用 Hessian 对角 11.25，完整指标 11.23（OPT-6.7B），只看权重误差 13.08。**Tab.13 p.12**：同预算下保护 {Wk,Wq} 得 11.26，保护 {Wq,Wv} 得 111.57，模块间强交互 | FULL-new |
| **SqueezeLLM** 2306.07629 | 每行的非均匀码本，以及 dense+sparse 中留 FP16 的权重（0.05% 敏感权重 + 0.4% 离群权重）；**不做比特分配** | 对角 Fisher F_ii=E[g_i²]，g 为最终 loss 的梯度 | 100 条样本（10 条已足够）；Fisher 计算 7B 0.3 min、65B 2.5 min（A100） | k-means 在 CPU 上 11–80 min；7B 峰值显存 33 GB | 闭式加权 k-means + 固定百分比阈值 | **Tab.D.2 p.15**：3-bit LLaMA-7B 的 C4 PPL，RTN 28.26 → 普通 k-means 18.08 → Fisher 加权 7.75；加 0.45% 稀疏后两者为 7.61 vs 7.56，二者可互相替代 | FULL-new |
| **SliM-LLM** 2405.14917 | 层内每个 128 列 group 的比特（N−1/N/N+1），层平均固定为 N，**不做跨层分配** | group 显著性 ≈ w²/[H⁻¹]²_jj（约等于激活×权重）；混合比例由**层输出 KL** 决定 | WikiText2 128×2048；搜索时用 RTN 伪量化，最终权重来自 GPTQ+SQC | 无反传；7B 每层 16 步搜索 | 按显著性排序 + 一维双指针搜索 | **Tab.1 p.7**：2-bit LLaMA-7B 152.31→14.58，3-bit 只有 6.55→6.40。**Tab.4 p.8**：ILP 17.55 vs SBA 14.58。**Tab.6 p.15**：目标函数用 MSE 21.94，用 KL 14.58 | FULL-new |
| **HAWQ-V2** 1911.03852（CNN） | 每层/块的权重与激活比特，之后做 QAT | 平均 Hessian trace Tr(H_i)/n_i | Hutchinson 估计，约 50 步、512 样本收敛 | ResNet50 的 54 块在 4 卡上 30 min | 按 trace 单调约束剪枝搜索空间（1.3e30→2.3e4），再按 Ω=Σ Tr·‖Q(W)−W‖² 选 | Tab.1–3 p.8–9：相对 V1 +0.16 到 +0.36 top-1；**可加性是假设，从未验证** | FULL-new（v1） |
| **EvoPress** 2410.14649 | 每个线性层选 2–6 bit GPTQ-g128 变体（也支持稀疏、删块） | **端到端 KL(全精度 ‖ 拼接模型)**，在 Fineweb-Edu 上算 | 多阶段选择：128 个子代先用 2K token 评，留 16 个用 16K 评，再留 4 个用 131K 评（Tab.8 p.22）；约 64K token 足以可靠排序（p.4） | 8B 模型在 RTX 3090 上约 2 GPU-h 拿到大部分收益，约 8 GPU-h 收敛（p.32）；另需预先建好变体库 | (1+λ) 进化：一层升一档、同尺寸的另一层降一档 | **Tab.3 p.8**：3-bit Llama-3-8B Wiki2，均匀 12.19，NMSE-DP 29.00，EvoPress 7.49。**Tab.7 p.21**：3-bit 时 PPL 作 fitness 得 8.17，KL 作 fitness 得 7.49。**Tab.1 p.2**：单调性反例（删块场景） | FULL-new |
| **HIGGS / 线性定理** 2411.17525 | 每层的比特/网格（2–4 bit FLUTE 网格 + 8-bit） | 预测 ΔPPL=Σ_l α_l·t_l²，t_l² 为相对权重 MSE | α_l：向单层注入 15 档高斯噪声，测 ΔPPL（287K token），过原点拟合直线；data-free 版用随机 token 上的 KL | 约 L×15 次全模型评估（论文未报告，【推断】8B 约 3.4K 次）；求解只需秒级 | 多选背包，CP-SAT 精确求解 | **p.9**：b>3 bit 时预测吻合，低于约 3 bit 发散。**Tab.4 p.10**：3.25 bit 时 6.388 vs GPTQ 7.133。**Tab.10 p.28**：70B 上 data-free 信号失效（3.133，差于均匀的 2.986）；NF 4.02 bit 时 PPL 2084 | FULL-new |
| **CoopQ/IMPQ** 2509.15455 | 每个 block 选 2 或 4 bit | NLL 变化的 Shapley 值 + 置换协方差构成交互矩阵 | M=100 个置换，在部分量化的上下文中逐层降到 2 bit；用 Quanto 估计 | 8B 在单张 A40 上约 18 h | 二次规划 → MILP（SCIP） | **Tab.3 p.7**：交互项 α=0/0.5/1 → 2.83e3/2.79e3/2.81e3，差约 1%；只评 PPL；接近 4 bit 时输给简单梯度基线 | FULL-new |
| **MixQuant** 2607.23047 | 每个线性模块 3–8 bit，并重校准 AWQ/GPTQ | **以上游比特为条件**的输出 NMSE，取 K=10 个随机上游配置的平均 | WikiText-2 128×2048 | 离线成本未报告 | 可加背包 + 贪心（对最低档加惩罚） | Fig.2a p.4：上下文让敏感度绝对值相差 2–3 个数量级。1.25 GB：PPL 12.43→10.70。**p.10**：按模块大小、不用任何敏感度的分配得 49.0，高于所有 FP16 上下文代理（46.1–48.5） | FULL-new |
| **Q-Strata** 2608.30564（MoE） | 专家 FFN 选 1–4 bit GPTQ-g128 | 块内：逐层孤立重建误差；块间：**组装后整模型的 JSD** | 外层每次评估一次整模型前向（32×4096 token） | 1.5K–1.9K 次外层评估，23–89 H100 GPU-h | 两级：块内 ILP + 外层 lazy greedy | **Tab.8 p.13**：同预算 32 个随机分配，整模型 JSD 与测试 PPL 的 Spearman ρ=0.98/0.99，可加梯度代理只有 0.42/0.28。**Tab.2/3/5**：孤立整模型边际 + ILP，在 ≥2.25 bit（MoE）/3.5 bit（稠密）时与组合贪心差 ≤2%，2.5 bit 稠密时差 26% | FULL-new |
| **GAMMA** 2605.18475 | 每个模块选 2/3/4 bit（QTIP/AQLM 后端） | teacher-forced 逐层隐藏状态重建 MSE，经 Gumbel-softmax 学出分数 | RedPajama 1024×2048 | 8B 约 1–1.5 h（不含预量化候选的时间）；HAWQ-V2 重实现 1.3–2.9 h | 可微分数 + ILP（换预算只需重解） | Tab.6/7 p.13：比 HAWQ-V2 高 1.4–5.9。Tab.8：重建损失优于 CE。分配在不同预算/校准集间的余弦相似度 0.9755/0.9913；k_proj 分高、v_proj 分低 | FULL-existing |
| **Structure of Q-Damage** 2609.01587 | 用于分析：额外比特该花在哪里 | 因果金标准：把单层升到 8-bit，看收回多少差距；同时检验电路/激活修补/权重统计/重建误差 | per-row RTN4；CORE 22 任务 | 每层一次评测 | — | 便宜定位器全部失败；损伤是弥散的（收回 50% 差距需恢复约 20% 的层）；**两层同时恢复的效果 ≈ 两个单层之和（0.91–1.10）**；等比特下全局 g128 比局部升层高 21–52 分（Tab.1 p.4） | FULL-existing |
| **Which Decisions** 2608.06564 | 用于分析：哪些决策会翻转 | 配对决策 margin：m′=c·m+b+ε，翻转率 | 首 token 的 logit 差，量化前后各前向一次 | 每个配置一次配对前向 | — | 3-bit 时 whether-to-call 先崩；**Tab.4 p.21**：blocks 16–19 让调用率 .910→.450，PPL 只有 1.01×；PPL 最高的块伤害最轻 | FULL-existing |
| **Flat Score** 2607.27275 | 用于分析：INT4 下 agent 的损伤 | 按失败通道统计的错误率、S(K) 曲线 | τ²-bench 日志 | — | — | 最终分持平，但 Gemma-4-31B 的工具名幻觉 19.5%→38.3%；K=2 时差距扩大到 16.7 分 | FULL-existing |
| **Quant Hurts Reasoning** 2504.04823 | 用于分析 | 比特 / 校准域 / 任务难度 | — | — | — | W3 出现断崖；GPTQ-W3 换成推理数据校准后 +9.8，AWQ ≤1.8（Tab.2 p.9）；GSM8K 低估损伤 | FULL-existing |
| **Reasoning Meets Compression** 2504.02010 | 用于分析：哪些模块对推理关键 | 推理行为重要性热图 | — | — | — | 只把最后一层 up_proj（0.7% 权重）量化到 3-bit RTN，平均分 −16.3（Tab.3 p.9）；恢复最后一层 MLP（约 2% 权重）+6.57（Tab.4） | FULL-existing |
| **Self-calibration** 2410.17170 | 校准数据来源 | 模型自生成文本 | 采样温度 1 | — | — | 量化对校准源不敏感（Gemma 上 AWQ 差 0.6、GPTQ 差 2.0），剪枝更敏感 | FULL-existing |
| **UniComp / LLM-KICK / ACBench** 2602.09130 / 2310.01382 / 2505.19433 | 用于分析 | 多维能力 | — | — | — | 推理和长生成最脆弱；PPL 掩盖知识与 agent 能力损失；JSON 结构化输出更脆弱 | FULL-existing |
| **CompressionBench** 2611.06858（agent） | coding agent 自由设计压缩方案 | 评测器返回：PPL 比奖励 + 包/常驻存储 + decode 计时；agent 本地可自测 recon/KL/NLL 和一套问题 | 自采样续写上的 CE = KL | 8 h 墙钟，单卡 | agent 自由搜索 | 79 次比较中，奖励推翻了 17 次本地排序（p.8）；Fig.5：weight MSE 与 PPL 反向，output MSE 更接近；PPL 不变时 GSM8K 88→72 | FULL-existing |
| **AutoQS** 2611.11975（agent） | 每层 4 个组的 GPTQ group size | 便宜代理（NLL、敏感度、存储）+ 每 5 轮一次真实 Macro3 | Qwen-Plus 输出 SEARCH/REPLACE 编辑 | 100 次迭代，20 次真实评估 | LLM 提议 + 真实评估 | Tab.2 p.8：完整 73.85，无反馈 72.58，随机 73.45（单种子）；Macro3 与 Macro7 r=0.135 | FULL-existing |
| **APQF / ProfilingAgent** 2608.05499 / 2509.05584（agent，视觉模型） | 逐层剪枝率、比特、恢复策略 | profiling CSV（MAC、时延、逐组剪枝敏感度扫描）→ LLM 汇总成简报 | ptflops、torch.profiler、DepGraph | 未报告 | 多 agent 单路径规划 | APQF Fig.6：去掉 profiling 后变差，但好于均匀（无数值） | FULL-existing |
| **Self-Pruner** 2502.14413（agent） | 逐层剪枝率 | 种群 + 适应度（WikiText-2 PPL） | GPT-4o 执行选择、交叉、变异 | 未报告 | LLM 进化 | Tab.3：去掉 LLM 初始化 PPL 17.43 vs 16.25 | FULL-existing |
| **Adaptive Pruning by Agents** 2601.09694（agent） | 每轮各层额外稀疏率 + 是否停止 | Wanda 与梯度重要性的 z-score、当前/目标稀疏率、PPL、上一轮反馈（含定性评语） | Gemini-3-flash 输出 JSON | 21–40 轮 | LLM + PPL 涨幅 >15% 回滚 | Tab.2–4；**没有特征消融** | FULL-existing |
| **HAQA** 2601.03484（agent） | QAT 超参、整模型统一比特、kernel 配置；**不做逐层分配** | 硬件规格 JSON、搜索空间、代码、当前配置、验证精度与 loss 曲线 / lm-eval 字典、kernel 时延、剩余轮数 | GPT-4-0613 + ReAct，完整对话历史 | 每次调用 2.34 s；约 15 万 token | 最多 10 轮 | Tab.1 p.6（ResNet20 88.38 vs BO 82.29）；Tab.2 p.7；**无消融** | FULL-new |
| **TAP** 2512.07419（LLM，CNN） | LLM 写出"代理程序"，由程序给出逐层比特 {2,4,8} | 适应度 ϕ=α·ρ_sens+(1−α)·Acc_quant | 执行代理 → 分配 → 量化 → ImageNet 评测；16 条校准样本 | 约 0.4–0.5 GPU-h | 冻结 LLM 做进化式程序搜索 | **Tab.8 p.7**：α=1（只看排序相关）ResNet18 得 62.48，α=0.1（加入真实精度）得 70.26 | FULL-new（无附录） |

---

## 2. 信号类型分类与证据

### 2.1 纯权重统计（weight magnitude/std、权重重建 MSE、t_l²）
- **作为"选哪些通道/层"的信号基本无效**：
  - AWQ Tab.1：按权重幅值选 1% 通道（22.37）≈ 随机（24.23）≈ 不保护（23.54）。
  - OWQ Tab.7：只看权重误差 13.08，含激活能量 11.23。
  - 2609.01587：权重 std 只在 LLaMA 家族有效；重建误差在 Qwen3-8B 上与可恢复性反相关。
  - CompressionBench Fig.5：k-means 码本的 weight MSE 更低，PPL 却更高；GPTQ 正好相反。
- **唯一的正面用法**：HIGGS 的 t_l² 乘以每层校准过的 α_l，在 3–8 bit、data-free 网格量化下能预测 ΔPPL。但作者说它和 GPTQ 这类数据感知目标"没有直接关系"（p.3）。
- 成本：零前向。**结论：只用于计费和格式选择，不用于分配。**

### 2.2 激活 / 离群统计（每通道激活幅值、XXᵀ 对角、离群比例）
- **性价比最高的局部信号**：
  - AWQ 按激活选通道 11.39 vs 随机 24.23。
  - OWQ 只用 Hessian 对角（λ_j=2Σx_j²）就拿到几乎全部收益。
  - SliM-LLM 的显著性实质上就是激活×权重。
- 一次校准前向即可得到（GPTQ 本来就要算）。
- 局限：它决定"层内如何量化"（scale、保留列），**不能直接告诉你跨层该给几 bit**。离群模块（HIGGS 中 70B 第一块的 v_proj）会让单模块出现灾难性损伤，值得做成一个"红旗"检测。

### 2.3 二阶 Hessian / Fisher（trace、对角 Fisher、Shapley-NLL）
- HAWQ-V2 的 trace 加权：LLM 上作为基线"便宜且不差"。GAMMA 重实现后，Qwen3-8B 3-bit 下 75.24，高于 AMQ 的 74.16。但它被学习式/端到端方法超出 1.4–5.9 分，可加性也从未验证。
- Fisher：
  - SqueezeLLM 中 Fisher 加权主要在**没有**离群提取时起作用（18.08→7.75）。离群提取后只多 0.05 PPL，而且所有 Fisher 消融都只看 PPL，下游任务没有证据。
  - MixQuant：Fisher 在 GPTQ 下常常最差。
  - CompressionBench：Sonnet 实测 Fisher 加权分配让 KL 变差，因此放弃（p.10）。
- 成本：Fisher 需要反传（27B 显存压力大，【推断】）。trace 需要多次 Hessian-vector 乘积。
- **结论：不值得作为首选工具。**

### 2.4 逐层输出误差 / 逐层 KL（层输出 NMSE、层 softmax-KL、逐层隐藏状态重建）
- 比纯权重统计好：CompressionBench Fig.5 中 output MSE 比 weight MSE 更接近 PPL。SliM-LLM 中层 KL 优于层 MSE（21.94→14.58，但 LLaMA-2-7B 上只是 16.86→16.01）。
- **把逐层误差简单加总后做分配会出问题**：EvoPress Tab.3 中，3-bit 下 NMSE-DP 在三个模型上都**差于均匀分配**（Llama-3-8B 29.00 vs 12.19）。
- 敏感度依赖测量时的上下文：MixQuant 显示上游配置让数值相差 2–3 个数量级。孤立地在 FP16 上下文中测量，会低估低预算下的边际损伤（Q-Strata Fig.2）。
- GAMMA 用 teacher-forced 逐层重建作为可微目标是有效的，但那是学习过程的损失函数，不是直接的排序信号。

### 2.5 整模型输出分布差异（端到端 KL / JSD / 自采样 CE）
- **在"配置排序"上证据最强**：
  - Q-Strata：同预算随机配置下，JSD 与测试 PPL 的 ρ=0.98/0.99。
  - EvoPress：3-bit 下 KL 作 fitness 优于 PPL（7.49 vs 8.17）；约 64K token 就能稳定排序。
  - CompressionBench 用自采样续写上的 CE（期望等于逐 token KL）作奖励，agent 用它做出了有效修订：24 次固定存储修订中 21 次降低了 PPL。
- 成本：每次评估 = 一次整模型前向（N 个 token）+ 一次 BF16 参考（可缓存 logits/top-k）。
- **但它仍然不等于能力**：
  - CompressionBench 中 PPL 几乎不变，GSM8K 却 88→72，150 题的对数在 104–115 之间波动。
  - 2608.06564 Tab.4：PPL 1.01× 时调用率已经腰斩。
- 【推断】KL 是"分布保真"的必要条件，不是"行为保真"的充分条件，对工具调用、长推理这类稀疏关键 token 尤其如此。

### 2.6 端任务探针（配对决策翻转、短推理题、工具调用通道错误率）
- 只有它能看到 PPL/KL 看不到的损伤：
  - whether-to-call 在 3-bit 先崩，且方向偏向"不调用"（2608.06564）。
  - INT4 下工具名幻觉翻倍，但被 error budget 掩盖（2607.27275）。
  - 难题和长输出掉得最多，GSM8K 会低估（2504.04823）。
  - 最后一层 up_proj 对推理特别关键（2504.02010）。
- **配对 margin 测量很便宜**：每个配置只需一次配对前向。2608.06564 显示，同一模型内用一半样本拟合 c、b、σ 来预测另一半，中位误差约 1 pp；但常数不能跨模型迁移，3-bit 下误差 26.8 pp。
- 风险：
  - 小探针噪声大。AutoQS 的 Macro3 与 Macro7 只有 r=0.135；CompressionBench 中 50 题×3 的方差也不小。
  - 拿多选常识题当搜索反馈会过拟合（AutoQS 中 ARC-C 超过 FP16 5.5 分）。

### 2.7 时延 / 内存实测
- 对仅权重量化，真实约束是**字节数**。group scale/zero 等元数据必须计入：
  - GPTQ：3-bit 下 g128 约多 0.15 bit。
  - 2609.01587：per-row→g128 是最大的单一杠杆。
  - AutoQS 的"收益"很可能来自更小 group 带来的更多元数据（【推断】）。
- decode 速度在 CompressionBench 中几乎都低于 BF16（解包开销），HAQA 的 INT8 发现是 prompt 中预先告诉它的。所以速度数据只有在实际 kernel 上测才有意义。

### 2.8 校准数据作为一个决策维度
- **GPTQ 对校准域敏感，AWQ 不敏感**：
  - AWQ Fig.8b：跨域时 GPTQ 多掉 2.33–4.89，AWQ 只多掉 0.5–0.6。
  - 2504.04823：GPTQ-W3 换推理数据校准 +9.8，AWQ ≤1.8。
  - 2410.17170：量化整体对来源不敏感，自生成数据可用。
- data-free（随机 token）的敏感度信号在 70B 上失效（HIGGS Tab.10）。
- **结论：校准源只在选 GPTQ 且 ≤3 bit 时是一阶变量；KL 的测量集必须是真实文本或自生成文本，并且与最终评测集不相交。**

### 2.9 总结：哪些信号预测端任务损伤

| 信号 | 预测排序（PPL/KL） | 预测推理/工具损伤 | 成本 | 判定 |
|---|---|---|---|---|
| 权重统计 / weight MSE | 差，甚至反向 | 无证据 | 0 | 避免用于分配 |
| 激活/离群统计 | 层内选择很强 | 间接 | 1 次校准前向 | 保留（廉价辅助 + 红旗） |
| Hessian trace / Fisher | 中等，可加性未验证 | 无证据 | 高（HVP/反传） | 只作基线 |
| 逐层输出误差（求和） | 中等，低 bit 时求和失效 | 无证据 | 每模块 1 次层前向 | 可做粗筛，不做最终依据 |
| **整模型 KL/JSD** | **强（ρ≈0.98）** | 不充分（PPL 不变而 GSM8K/调用崩） | 每配置 1 次全模型前向 | **主信号** |
| **配对行为探针**（翻转率、短推理、工具调用） | — | **唯一直接证据** | 每配置 1 次配对前向 + 少量生成 | **必须有** |
| 外部端任务评测 | — | 金标准，但噪声大、易过拟合 | 高 | 稀疏使用，保持 held-out |

---

## 3. LLM-agent 压缩工作：agent 看到什么、能做什么、消融与失败

| 工作 | 决策者 → 被压缩者 | agent 看到的诊断（格式） | 动作空间 | 反馈与历史 | 消融 | 报告的失败模式 |
|---|---|---|---|---|---|---|
| CompressionBench | Claude/GLM/Kimi 等 → Nemotron/Qwen 27–122B | 论文库、starter 代码、校准数据、本地问题集；**自己写代码**测 recon error、KL、NLL；评测器返回 PPL 比奖励、包/常驻存储、decode 计时（数值） | 任意：量化格式、分配、剪枝、低秩、KD、kernel | 墙钟 8 h，奖励可反复查询；准确率门槛**对 agent 隐藏** | 无正式消融；79 次本地 vs 奖励比较中 17 次被推翻，其中 8 次回滚 | 优化 PPL 而丢失 GSM8K；检测到生成退化却保留设计；宣布完成后仍改权重（17 个 run）；专家索引 bug |
| AutoQS | Qwen-Plus → 7B | 父策略文本 + 便宜代理分数（NLL、跨域统计、敏感度、存储）+ 周期性的真实 Macro3；敏感度**不**作为显式逐层指令 | 每层 4 组的 group size，以 SEARCH/REPLACE 编辑 | 100 次迭代，20 次真实评估 | **无反馈 72.58 vs 完整 73.85 vs 随机 73.45**（单种子）；种子方差 0.94 > 增益 0.70 | 搜索期指标与最终指标几乎不相关（r=0.135）；对搜索任务过拟合 |
| Self-Pruner | GPT-4o → LLaMA 7–70B | 种群配置 + PPL 适应度（文本列表） | 逐层剪枝率向量 | 20 代 | 去掉 LLM 初始化 / 变异 / 交叉各掉约 1 PPL；**没有与非 LLM 进化算法同预算比较** | 在测试集 PPL 上搜索 |
| Adaptive Pruning by Agents | Gemini-3-flash → Qwen3-4/8B | 按 z-score 排序的层画像（Wanda、梯度、当前稀疏率）、当前/基线 PPL、上一轮决策与定性评语 | (层, δ∈[0.01,0.15])、停止 | Reflexion 式文字反馈；PPL 涨幅 >15% 回滚 | **无**（没有贪心 z-score 基线） | 约 10% 的轮次回滚 |
| APQF / ProfilingAgent | Claude/GPT-4o → 视觉 CNN/ViT | profiling CSV（MAC、时延、逐组剪枝扫描）→ LLM 写成的架构简报 | 剪枝率、比特、PEFT 及其超参（JSON） | 多阶段；每轮从 M0 重新开始（ProfilingAgent） | APQF：去掉 profiling 掉点少于均匀分配（无数值）；ProfilingAgent：GPT-4o vs 4-Turbo | 量化退化为"全部 int8"；无效 JSON 允许修复一次；便宜模型失败更多 |
| HAQA | GPT-4-0613 → ResNet / LLaMA2 | 硬件 JSON、搜索空间、代码、当前配置、精度/loss 曲线、kernel 时延、"剩余 N 轮" | 超参 JSON、kernel 配置/代码、整模型比特 | ReAct，完整历史，≤10 轮 | **无** | 格式错误、越界、上下文溢出 |
| TAP | 冻结 LLM → CNN | 优秀代理库（推理 + 代码 + 适应度）+ 操作模板 | 写代理程序 | 标量适应度；30 轮内不可执行代码比例 40%→10% | α 消融：去掉真实精度、只看排序相关时大幅变差 | 声称"直接 prompt 分配不稳定"，但正文无数据 |

**跨工作的共同观察**
1. **没有一项工作把 agent 看到的诊断信息做成干净的消融**，比如"有/无逐层敏感度表"或"有/无行为探针"。最接近的三项：
   - AutoQS 的"无反馈"组：差 1.27 Macro3，单种子。
   - TAP 的 α 消融：真实量化后精度这一反馈很关键。
   - CompressionBench：整模型反馈推翻了 21.5% 的本地排序。
   - 【推断】三者一致指向：**整模型的真实反馈比本地代理更有价值**，而本地代理主要用于缩小候选集。
2. **LLM 相对非 LLM 搜索的增益几乎没被证明**：
   - AutoQS 比随机只高 0.41（单种子）。
   - Self-Pruner、Adaptive Pruning、APQF 都没有同预算的进化/贪心/BO 基线。
   - HAQA 没说明基线是否也用了 10 次评估。
3. **反馈指标的选择决定 agent 会被什么误导**：
   - 只给 PPL：CompressionBench 中 GSM8K 下降，Self-Pruner 在测试集上搜索。
   - 只给小规模 MCQ：AutoQS 过拟合。
4. **格式**：
   - 成功的系统都给结构化数值，例如 JSON、z-score 排序表、配置差异与指标的历史。
   - 动作约束在合法空间内，例如预算校验、去重、novelty repair、PPL 回滚。
   - 自由文本动作需要修复机制，如 ReAct 或 validator。
5. **所有工作的决策者都不是被压缩的模型**；"自己量化自己"在已有工作中未出现。CompressionBench 结论把它列为 future work。

---

## 4. 可加性：孤立测得的逐层敏感度能否预测组合配置？

**支持"近似可加"的证据（中低压缩，3–8 bit）**
- HIGGS：b>3 bit 时 Σα_l t_l² 能预测混合配置的 PPL（Fig.1/3，p.9），低于约 3 bit 时发散。
- Q-Strata：用孤立的**整模型**边际 + 可加 ILP，在 MoE ≥2.25 bit、稠密 3.5 bit 时与组合贪心差 ≤2%；稠密 3.0 bit 差约 7%，2.5 bit 差 26%。
- 2609.01587：4-bit 下"两层同时恢复 ≈ 两个单层之和"（比值 0.91–1.10）。
- EvoPress D.4：删 2–6 个块时，单块 PPL 的均值能预测组合；到 8–12 块时失效。
- CoopQ：显式两两交互项只带来约 1% 的改变；它的收益主要来自"在部分量化的上下文里测边际"。

**反对 / 需修正的证据**
- EvoPress：3-bit 下按逐层 NMSE 求和做 DP，结果差于均匀分配；单调性反例。但这同时混杂了"NMSE 本身是坏指标"，论文没有加权 DP 对照。
- MixQuant：上游配置让敏感度绝对值相差 2–3 个数量级，ILP 分配随之改变。但它没有报告排序相关；而且不用敏感度、只按模块大小分配的方案反而胜过 FP16 上下文代理。
- Q-Strata Fig.2：边际损失随整体压缩加深近单调**增大**（超可加），所以孤立测量会**系统性低估**低预算下的损伤。
- OWQ Tab.13：同层 q/k/v 的保护存在强交互（{Wk,Wq} 11.26 vs {Wq,Wv} 111.57）。【推断】attention 内 q 与 k 共同决定 logit，单独 profile 其中一个会误导。
- 2608.06564 / 2504.02010：即使 KL 可加，**行为损伤也不随 KL 线性变化**，个别模块对推理或工具调用的伤害远超其 KL 份额。

**对 `profile` 工具的结论**
1. 当配置主要落在 {3,4,8}、且平均比特 ≥3.5 时，用**整模型 ΔKL**（不是层输出误差）孤立测量再求和，作为预算分配的一阶模型基本够用，误差约 ≤2–7%。
2. 一旦较多模块用 2 bit，或平均比特 ≤3：
   - 求和会低估损伤；
   - 必须在**当前组合配置的上下文中重新 profile**（greedy 每一步都重测），并对完整候选做整模型评估。
3. 无论 KL 是否可加，**行为探针都要在组合配置上测**，因为没有证据表明行为损伤可加。
4. q/k（以及同一 attention 内的模块）应当作为一组来 profile。

---

## 5. 建议（以下全部为【建议】，基于上面的证据加推断）

### 5.1 最小工具集
1. **`profile(module_group, bits, quantizer, group_size, context="current")` → {ΔKL_mean, ΔKL_p99, top-1 一致率变化, Δbytes}**
   - 测量方式：整模型逐 token KL，对照缓存的 BF16 logits（可只存 top-k）；用约 64K token 的自生成或混合文本，与评测集不相交（依据 EvoPress p.4、CompressionBench 式 3）。
   - 默认上下文 = 当前配置，而不是全 FP16（依据 MixQuant、Q-Strata Fig.2）。q/k/v 默认作为一组。
   - 成本控制：
     - 先用 RTN 伪量化做粗筛（SliM-LLM 搜索时用 RTN、最终用 GPTQ 的做法）；
     - 再对候选用真实 GPTQ/AWQ 复测；
     - 或先按"层×模块类型"聚合。
   - 【推断】27B 若有约 64 层×7 个模块×4 档比特×3 种量化器，全量约 5K 次 profile 不现实，必须分层粗筛。
2. **`evaluate_config(config)` → {整模型 KL, 字节数（含 scale/zero/码本元数据与 embedding/head）, 是否满足预算}**
   - 用于验证组合配置，不限制用 2 bit，但要有调用预算。
3. **`probe_behavior(config)` → 相对 BF16 的配对指标**
   - (a) 工具调用决策翻转率：whether-to-call 正负两侧分开、which-tool，用首 token margin，可同时报告拟合的 c、b（依据 2608.06564）。
   - (b) 小规模难推理题的生成正确率与输出长度变化（依据 2504.04823）。
   - (c) 一小组多步工具调用 episode 的按通道错误率，K 取较紧的值（依据 2607.27275）。
   - 结果附置信区间或样本数，防止 agent 追噪声。
4. **`activation_stats(module)` → 每通道激活二阶矩、离群通道比例、最大/中位比**
   - 一次前向即可得到。用于 AWQ vs GPTQ 的选择、group size 的选择，以及对"灾难性模块"（HIGGS 中的 v_proj、Qwen3 早期层）发红旗。
5. **预算计算器 `bytes(config)`**：精确计入 group 元数据。最大的全局杠杆往往是 group 粒度（依据 2609.01587 Tab.1），应让 agent 能直接比较"降 group size"和"给某层升比特"在相同字节下的效果。

**外部评测器反馈**：
- 返回端任务分数，并带**按能力/通道的分解**（推理、工具调用决策、工具名/参数错误、拒答），外加字节数。
- 调用次数要稀疏，评测集严格 held-out。不要像 CompressionBench 那样把门槛隐藏后只奖励 PPL。
- 历史以结构化表格呈现：配置差异 → ΔKL / 探针 / 评测器结果，并保留可回滚的最佳配置（依据 Adaptive Pruning 的回滚、CompressionBench 的回滚行为）。

### 5.2 应避免或降级的信号
- 权重幅值/std、权重重建 MSE，作为分配依据（AWQ Tab.1、2609.01587、CompressionBench Fig.5）。
- 对逐层输出 NMSE 简单求和后做 DP（EvoPress Tab.3）。
- 把 WikiText PPL 当唯一目标或评测集：既有在测试集上搜索的问题，又掩盖推理和工具损伤。
- 用小规模多选常识题作为搜索反馈（AutoQS r=0.135）。
- 可解释性定位器（电路漂移、激活修补）；基于随机 token 的 data-free 敏感度（HIGGS 70B 失效）。
- Hessian trace / Fisher 作为主工具：成本高，下游证据缺失。可作为非 LLM 基线保留（HAWQ-V2 + ILP）。
- 在 FP16 上下文里测的孤立敏感度，用于 ≤3 bit 的决策。

### 5.3 必须同时跑的非 LLM 基线（否则无法回答"LLM 推理是否有用"）
- 均匀 GPTQ/AWQ g128 4-bit。
- 基于 profile ΔKL 的可加 ILP / 贪心（同样的工具，不需要 LLM）。
- EvoPress 式 KL 进化搜索，在相同评估次数下比较。
- 可选：HAWQ-V2 trace+ILP、GAMMA。
- 依据：现有 agent 工作几乎都缺少这一对照（§3 第 2 点）。

### 5.4 Pilot 必须回答的开放问题
1. **可加性实测**：在 Qwen3.8 27B 上用 GPTQ/AWQ，随机采约 20–30 个组合配置，覆盖平均 3.0/3.5/4.0 bit，以及有/无 2-bit 模块。比较 Σ孤立 ΔKL 与实际整模型 KL 的 Spearman 与绝对误差；再比较"当前上下文重测"能把误差降多少。
2. **KL 能否预测行为**：在同一批配置上，看 ΔKL 与工具调用翻转率、推理正确率的相关性。重点看是否存在"KL 份额小、行为伤害大"的模块（最后一层 up_proj、中后段 block、k_proj），以决定 probe 是否需要模块级版本。
3. **校准源**：GPTQ 的校准源（通用文本、自生成、推理/工具对话混合）在 3-bit 和 4-bit 下对行为探针的影响；KL 测量集的来源是否改变排序。
4. **RTN 伪量化粗筛与真实 GPTQ/AWQ 的排序一致性**：这决定 profile 的成本能否降一个数量级。
5. **探针噪声地板**：BF16 自身重复运行（vLLM 非确定性）时，翻转率与通道错误率的波动有多大；多少样本才能可靠区分配置。
6. **自指风险**：被量化后的模型若继续担任决策者，其工具调用（whether-to-call）是否先退化，从而让 agent"停止行动"或伪造调用（依据 2608.06564 的方向性崩塌）。要记录每轮真实发出的工具调用数。
7. **LLM 的增益**：在相同评测器调用预算下，LLM agent 能否超过 KL 贪心/ILP 与 EvoPress。如果超不过，应把贡献重新定位为"自我决策 + 行为感知"，而不是"更好的分配器"。
