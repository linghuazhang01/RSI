# Sparks of Weight-Space Programming：阅读笔记

## 元信息
- 标题：Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights
- 作者：Rikiya Takehi*、Jayden Teoh*、Alex Zhang、Omar Khattab（MIT CSAIL；* 为共同一作）
- 发布：博客，2026-11-02；2026-11-12 追加了 "Response to Comments" 一节
- URL：https://rikiyat.github.io/blog/weight-space-programming/ ；日志在 github.com/RikiyaT/lm_understand-logs
- 读的范围：src_wsp_blog/wsp_blog.txt 全文第 1–455 行（TL;DR、Exp 1–8、Additional experiment、Related Works、What Comes Next、Response to Comments、Logs）。文本抽取完整，只有交互图（teaser、Figure 1 热力图）的数值抽丢了。抽取文本里看不到 LoRA 或 adapter 相关内容，HTML 里也一次都没出现 "lora"/"adapter"。

## 研究问题
通用的前沿 LLM 不经专门训练、不用梯度，能不能像读写代码一样“原生地”读懂并改写神经网络权重。作者把这个概念叫 Weight-Space Programming (WSP)，并在 FAQ 里明确说 WSP 是“概念/能力展示”，不是一种方法（"Is Weight-Space Programming a method?" 小节）。

## 方法细节
- **编辑者**：始终是 GPT-6 Astra（闭源前沿模型）。
- **被编辑者**：每个实验都是另一个更小的网络，跟编辑者不是同一个 checkpoint：
  - Exp 1：PPO 训练的 grid 游戏 MLP（17,253 个权重）；只读不写
  - Exp 2：DDPM MLP（23,824 个权重）；只读
  - Exp 3：7.2M 参数的字符级 transformer，只给中间 4 个 block；只读，可以用 Python 分析，不能前向
  - Exp 4：Kinetix 的 actor（171,664 个权重）；一次性改 3.5% 的权重
  - Exp 5：MinAtar 网络（600→32→32→6），从零写出全部权重
  - Exp 6：9×9 Go 网络（148 万个权重），在 Python sandbox 里从零写
  - Exp 7：两个 Gemma 3 270M 微调副本（全量微调，embedding 和输出层冻结），判断哪个是 sycophant
  - Exp 8：Qwen3.5-0.8B 的**全量基座权重**。最终编辑落在第 15、19 层 MLP 输出的各 64 列上，共 131,072 个权重（0.016%）
  - Additional：Gemma 3 270M，没能提升
- **编辑方式**：小网络是直接用文本写出权重数值（Exp 4、5）；大网络是在 Python kernel 里写 numpy/PyTorch 代码改张量（Exp 6、8）。禁止训练循环、梯度和拟合。
- **诊断信息**：权重本身；环境描述和一条录制的 episode（Exp 4）；有限次前向，Exp 6 是 0–50 次（每次 128 局对弈），Exp 8 是 10/25/50 次（每次最多 512 题或 1,024 条文本）；Exp 7 放宽后可以拿到注意力模式（最多 20 次）。Exp 8 中 Astra 能看到生成文本、准确率和激活，并据此定位“循环”方向。
- **多轮/递归**：Exp 6、8 里有“编辑—前向—保留或回退”的会话内迭代。**被编辑后的模型从不担任下一轮编辑者**。递归自我改进只出现在 "What Comes Next" 的展望里，说 LLM 检查并改进“自己的权重”是更原生的 RSI 路径，没有实验。
- **对照**：Exp 5、6 和 PPO 比，FAQ 明确说这不是等预算的方法比较（"Aren't the baselines too weak?" 小节）。Exp 8 只和未编辑的基座比。**没有同预算下的梯度微调、随机或进化搜索、LoRA 或 merge 规则作对照。**

## 数据集与评测、模型规模
- 小网络：自建的 grid 游戏、Kinetix、MinAtar（Breakout/Seaquest/Asterix）、9×9 Go、DDPM 形状。
- 语言模型：Qwen3.5-0.8B（thinking on）。练习数据是 GSM8K 训练集 7,469 题和 MATH 训练集 7,275 题；评测 MATH-500、GSM8K、IFEval、AMC23、Minerva、OlympiadBench、GPQA-D、MMLU-Pro（1,000 题子集），Astra 看不到评测集。Gemma 3 270M 只评 MATH-500。
- 采样：Qwen 用 temperature 0.6 和 presence penalty 1.5，上限 32,768 token；作者说明该模型容易陷入思考循环。

## 关键实验与数字（小节名）
- Exp 1：4 个游戏网络的训练游戏全部认对（txt 第 115 行）。
- Exp 3：语言两次尝试全部认对；copying heads 与实测完全一致（表格在第 163–169 行）。
- Exp 4：一次编辑，改 3.5% 的权重，1,024 个 episode 全部成功，原来是 0%（第 215 行）。
- Exp 5：Astra 一次写成的网络，Breakout 517.4、Seaquest 53.1、Asterix 53.2；PPO 训练 100M 步分别是 15.8/5.6/25.8，500M 步是 25.3/88.9/55.2（第 241–247 行）。
- Exp 6：0 次前向时对两个小 PPO 胜率过半；50 次前向时胜过同尺寸、训练了 2,000 万局的 PPO（第 279 行）。
- Exp 7：只看权重时 4 次尝试都失败；允许看注意力模式后 3 对全部认对（第 303 行）。
- Exp 8（核心）：50 次前向下，MATH-500 从 7.9 升到 48.1，GSM8K 从 13.8 升到 67.1，IFEval 从 21.0 升到 62.3，撞 token 上限的比例从 93.0% 降到 33.1%（第 352、356–362 行）；10 次和 25 次前向时 MATH-500 只有 12.3 和 14.1。只改了 131,072 个权重，占 0.016%（第 364 行）。
- Additional（Gemma 3 270M）：MATH-500 基线 6.8，5/10/25/50 次前向后是 7.0/7.6/7.7/7.0，Astra 自己也报告提升不成立（第 368–380 行）。

## 作者自己声明的局限
- 能力有限，主要卡在上下文长度和 harness 效率（"What Comes Next"）。
- 编辑能移除掩盖已有能力的障碍，但加不出模型本来没有的能力（Gemma 实验）。
- 只看权重识别不出 sycophancy（Exp 7）。
- baseline 不是用来公平排名的（FAQ）；也没有和可解释性研究者比较过（FAQ）。

## 方法问题与替代解释
- 【推断】Exp 8 的大幅提升基本是在修“思考循环/截断”：撞上限比例从 93% 降到 33%，作者自己也说这不等于数学能力提升。在采样设置固定、thinking 打开的前提下，任何抑制 "Wait" 类重复方向的编辑都可能得到类似收益；关掉 thinking 也能提高基线（FAQ 承认了这一点）。
- 【推断】每组设置只报告单次运行（n=1 条轨迹），没有方差。10/25/50 次前向的结果不单调（25 次时 IFEval 反而比 10 次低），说明结果对轨迹很敏感。
- 【推断】编辑者（GPT-6 Astra）远大于被编辑者，属于“强模型编辑弱模型”，跟“同一模型编辑自己”是不同的设定。
- 【猜测】小网络实验里编辑者可能利用了训练数据中关于这些环境的先验知识（例如 MinAtar 源码已经给了它），所以“从零写网络”更像把已知策略编译进网络。

## 与 core claims 的关系
- **C1（同一模型自编辑其 LoRA）**：**未覆盖**。编辑者是 GPT-6 Astra，被编辑的是 Qwen/Gemma/小网络，不是同一 checkpoint；改的是全量基座权重，没有 adapter。“LLM 改自己的权重”只在展望里作为 RSI 方向提了一句（第 402 行）。
- **C2（同评测预算下胜过 no-edit、merge 规则、非 LLM 搜索）**：**未覆盖**。只和 no-edit 比，PPO 对照也被作者明确说明不是等预算比较；没有 TIES/DARE、随机/进化搜索或 LoraHub 对照。
- **C3（递归，被编辑模型成为下一轮编辑者，并测量 E_t）**：**未覆盖**。只有会话内的迭代编辑，编辑者固定；递归只是展望。
- **C4（持续多任务、adapter 空间里控制遗忘、不用更强的外部模型）**：**未覆盖，而且相反**：依赖更强的外部模型。只在展望里提到小幅编辑“可能避免灾难性遗忘”（第 400 行），没有持续学习实验。
- 总体：这篇博客是 B' 的概念先驱（“LLM 写代码直接改权重、不用梯度、靠诊断和前向反馈”），但在编辑对象（基座权重而非自己的 LoRA）、编辑者身份（外部强模型）、对照和递归这几方面都不重叠。
