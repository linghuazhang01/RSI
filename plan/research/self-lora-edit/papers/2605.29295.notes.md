# 2605.29295 EvoGM 阅读笔记（novelty check：Topic B'）

## 元信息
- 标题：EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization
- 作者：Tao Jiang, Xinmeng Yu, Chenhao Yi, Yiling Wu, Yan Li, Ran Cheng, Dongmei Jiang†, Jianguo Zhang†（南方科技大学、鹏城实验室、PolyU 等）
- 日期/版本：arXiv v1，2026-05-28，cs.NE；页眉标注 ICML 2026（PMLR 306）
- 代码：作者声明开源（github JiangTao97/evogm），专家权重放在 HF
- 阅读范围：全文 22 页，正文 p1–9，参考文献 p9–14，附录 A–C p15–22（含表 4–13、图 6–8），已全部读完
- 源文件：src_2605.29295/2605.29295.pages.md（1551 行）

## 研究问题
现有的进化式模型合并（CMA-ES、PSO-Merging、Model Swarm）在合并系数空间里靠随机扰动和手工算子来搜索，验证集小、评估预算紧时效率低。作者想把"系数搜索"改成一个可学习的生成式过程：用历史评估结果训练一个生成器，让它直接提出更好的合并系数。

## 方法细节
- **被合并的对象**：按 task arithmetic 的写法，每个专家的任务向量是 τ_i = θ_i − θ_pre，合并结果为 θ(λ) = θ_pre + Σ λ_i τ_i（p3）。Qwen2.5-1.5B 的 10 个专家是 Tulu-v2 十个 SFT 子集上各自训练的 **LoRA**，但合并发生在 ΔW（任务向量）层面，并不是在 LoRA 因子上做逐层操作。FLAN-T5 的 8 个专家直接用了公开的 GLUE 微调模型（表 10，p19）。
- **操作**：只搜索**每个专家一个全局标量系数** λ ∈ R^N（N = 8、10 或 20），由 tanh 限制在 [−1, 1]（p4）。不做逐层缩放、不做 SVD 过滤、不剪枝 rank、不做正交化，也不做 TIES/DARE 式稀疏化。
- **谁做决策**：两个 5 层 MLP 生成器 G−→+ 和 G+→−。把历史档案 H 里前 ρ = 0.3 的候选当作 winner、其余当作 loser，做成配对；损失由循环一致性损失 Lcyc 加上把 loser 拉向 winner 质心的 Lopt 组成（p3–4）。训练好的 G−→+ 作用在整个种群上产生新候选，再按验证分数保留 top-P。**决策者是从零训练的小 MLP，不是 LLM，更不是被合并的模型本身。**
- **多轮"basis shift"**：每轮结束时取历史中最好的 N 组 λ 合成 N 个新专家，重算 τ，下一轮在这组新基上继续搜索（Algorithm 1，p3–4）。每轮开始时**重置**生成器（Algorithm 1 中的 "Reset Dual-Generators"，pages.md 第 144 行），所以没有任何"编辑者能力"跨轮积累。
- **预算**：种群 P = 2N。主实验是 2 轮 × 3 次迭代（p6）。Qwen3-8B 上每个任务 160 次候选前向评估，Model Swarm 为 100–160 次（表 7，p16），**两者预算并不相等**。
- **所需数据**：每个任务一个小验证集，通常 200 条（MMLU-Pro 70 条，GLUE 每任务 50 条），测试集 1000 条（表 9，p18）。

## 数据集、评测与模型规模
- Seen 设定：FLAN-T5-base + 8 个 GLUE 专家，在 8 个 GLUE 任务上评测（表 2，p5）。基线数字直接抄自 PSO-Merging 论文（p5）。
- Unseen 设定：Qwen2.5-1.5B + 10 个 Tulu LoRA 专家，在 MMLU、MMLU-Pro、HellaSwag、K-Cross、GSM8K、NLGraph、TruthfulQA、AbstainQA 上评测，分 single-task（每个任务单独搜一组 λ，表 1）和 multi-task（所有任务共用一个模型，表 3）两种。
- 扩展实验：Qwen3-8B + 10 个专家（表 6–7）；ViT-B-16 + 20 个专家，对比 CMA-ES（表 8）。
- 硬件：A100-40GB；附录 B 的实验改用 Ascend 910 NPU。

## 关键实验与数字
- **表 1（p5）single-task，Qwen2.5-1.5B**：EvoGM 在 8 个任务中的 5 个拿到最高测试分。例如 NLGraph 测试 0.537，PSO-Merging 0.465，TIES 0.316，TA 0.286；GSM8K 测试 0.434，PSO 0.354。
- **表 2（p5）GLUE**：EvoGM 平均 82.4，PSO-Merging 81.2，TA 78.9，DARE 78.7，TIES 77.4，CMA 79.4。
- **表 3（p6）multi-task**：EvoGM 平均 0.380，TA 0.364，DARE 0.364，TIES 0.356，CMA 0.375，PSO 0.372，Model Swarm 0.372，Base 0.352，MTL 0.330。**Single Best 为 0.405，比 EvoGM 高。** 正文没有正面讨论这一点。
- **表 4（p15）4 个种子 + Welch t 检验**：EvoGM 0.4214±0.0073，PSO 0.4018±0.0019（p=0.0100），Model Swarm 0.3867±0.0076（p=0.0006）。按数值范围看这是 single-task 设定。
- **表 5（p16）basis shift 公平性检验**：把同样的多轮 basis reset 加到 PSO 上，PSO 平均从 0.4018 降到 0.3425，EvoGM 为 0.4214。
- **表 6（p16）Qwen3-8B**：EvoGM 0.603，Single Best 0.600，Model Swarm 0.593，TA 0.513，TIES 0.500。对 Single Best 的领先只有 0.003。
- **表 7（p16）搜索成本**：EvoGM 共 385.5 min、1280 次前向；Model Swarm 共 428.9 min、900 次前向。
- **表 8（p17）ViT-B-16，20 个专家**：EvoGM 0.6363，CMA-ES 0.6268。逐任务波动很大，例如 SVHN 上 EvoGM 0.4556、CMA 0.8956，EuroSAT 上 0.8106 对 0.4828。
- **图 4 / 表 11（p7、p19）消融**：去掉多轮、去掉循环损失、改成单生成器，都会掉分，NLGraph 上最明显。
- **表 12（p21）超参**：种群从 10 增到 30，测试 +0.0305；轮数为 1 时测试 −0.0165。
- **表 13（p21）专家数量**：合并 8 个模型时，平均测试分从 0.433 升到 0.460；合并 2 个模型时 final test 0.429，略低于 initial 0.433。

## 作者自己声明的局限（p9）
- 只处理同构的任务向量合并（所有专家来自同一个基座）；异构架构需要先做对齐。
- multi-task 只优化单一标量验证目标，只返回一个模型，不提供 Pareto 前沿；偏好感知的多目标合并留作未来工作。

## 方法问题与替代解释
- 【推断】multi-task 设定下 Single Best（0.405）高于 EvoGM（0.380），说明它在 multi-task 场景的价值有限。表 6 中对 Single Best 的领先也只有 0.003。
- 【推断】表 7 显示 EvoGM 用了 1280 次前向，Model Swarm 只用了 900 次，"同等预算"并没有严格对齐。因此不能直接拿来支持"等预算下更好"这类结论。
- 【推断】200 条验证集既用于选择，又用于生成器训练的排序信号，存在对验证集过拟合的风险：表 1 中多处 val 很高、test 落差很大，例如 MMLU-Pro val 0.429 / test 0.224。
- 【推断】表 1 与表 11 中 EvoGM 的 K-Cross val 分别是 0.430 和 0.450，同一配置数字不一致，可能是排版或复现差异。
- 【猜测】Lopt 把 loser 拉向 winner 质心，本质上接近"向精英均值靠拢"的 CEM/EDA 变体。生成器的贡献可能主要来自这种均值回归，而不在于"学到的分布结构"。
- 【推断】GLUE 的基线数字是从他文抄来的，并非同一环境重跑。

## 与 core claims 的关系
- **C1（同一模型对自己的 LoRA 做权重空间自编辑）**：**未覆盖。** 决策者是从零训练的 MLP 生成器，不是 LLM。被合并的是多个外部专家，不是编辑者自己的 adapter；没有诊断工具，没有写代码，只搜 N 维全局系数。
- **C2（等预算下胜过 TIES/DARE/TA 和非 LLM 搜索）**：本文本身就是 C2 中"非 LLM 搜索"类别的最强候选基线之一。它同时提供了与 TA、TIES、DARE、CMA、PSO-Merging、Model Swarm 的对比，但预算没有严格对齐（表 7）。
- **C3（递归、编辑者能力 E_t）**：**只有形式上相似。** basis shift 让"被合并的专家池"跨轮演化，但编辑者（生成器）每轮重置，并且没有在新任务上测量编辑者能力是否随代际提升。
- **C4（持续多任务、遗忘控制、无更强外部模型、固定外部评估）**：**未覆盖。** 没有持续学习的时间序列，也没有遗忘指标或正交化。它确实使用固定的 val/test 划分，也没有用到外部大模型，但这与"自编辑"设定无关。
- **作为 C2 基线如何复现**：
  1. 在本项目里，把当前 adapter 栈中的旧 adapter、新训练的 raw LoRA，以及训练轨迹上的 checkpoint 当作 N 个"专家"，在各自的 ΔW 上搜索 λ ∈ [−1, 1]^N。可以按层扩展为 λ ∈ R^{N×L}，这时要说明这一项是本项目自己加的扩展。
  2. 种群 P = 2N，ρ = 0.3，生成器为 5 层 MLP，学习率 1e-4（表 12 中最优）、训练 200–400 epoch，2 轮 × 3 次迭代。用官方代码（github JiangTao97/evogm）。
  3. 关键是把总评估次数和 LLM 编辑者的候选评估次数对齐，统一用 dev 打分、dev2 验收、sealed test 汇报。
  4. 同时汇报 Single Best（直接选最好的单个 adapter 或 checkpoint）作为必须超过的下限，因为本文表 3、表 6 显示这个下限很难打败。
