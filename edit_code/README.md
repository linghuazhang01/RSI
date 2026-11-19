# edit_code/：Paper A 固定体积下的迭代自量化

计划见 `plan/PLAN.md` §3、`plan/research/RSI_LOOP_DESIGN.md` §2、`plan/research/METHOD_DESIGN.md` §2；共用运行时接口见 `eval/rsicore/INTERFACES.md`（本目录只通过它调用 rsicore）。

**设计（2026-11-19 修订）**：改造者的权重冻结，不断迭代的是被量化的模型。
- 改造者：原始 BF16 模型 M_0（`loop.editor: frozen`）；消融 `loop.editor: current` 让当前最好的量化模型来写。
- 每个候选都从 **M_0 的原始权重**出发量化，所有代用**同一个体积**（默认 = 统一 3 bit GPTQ g128、词嵌入和输出层 8 bit 那份配置的打包大小，它也是起点 Q_0）。
- 第 g 代的目标是超过当前最好的量化 Q_{g-1}（incumbent）：在固定验收集上逐题配对比较，平均提升 ≥ `gate.min_delta`，至少一个领域显著变好、没有领域显著变差，才换成新的 incumbent；否则保留。连续 `loop.patience` 代没有新的 incumbent 就停。
- 改造者能看到：incumbent 的配置和它在本代题目上的分数、往代每代最好的几个候选（经验通道）、诊断表。

## 单领域实验（`scope`）

**默认 `scope: single`（2026-11-19 起）**：只用这次运行的 `--domain`。信号和验收只用这个领域，KL / 校准文本（`kl.domains: run`）和留出评测（`heldout.domains: run`）也只用这个领域，不考虑别的领域。运行名默认带 `_single`。原设计（KL / 校准文本和留出评测覆盖五个领域）用 `--set scope=all`（`launch.sh --all-domains`）；`--domain all` 总是 `scope=all`。

## 启动

```bash
./run_rsi.sh --paper A --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id A_math_s42
./run_rsi.sh --paper A --domain all  --gens 5 --run-id A_all_s42
./run_rsi.sh --paper A --domain math --gens 5 --gpus 0,1 --run-id A_math_s42 --set hardware=h20   # H20 80GB
./run_rsi.sh --paper A --domain math --gens 2 --mock            # 本机，无 GPU，约 1–2 分钟
# 等价：python -I edit_code/run.py ...；附加参数 --budget-gib 11.5（固定体积），--set key=value（覆盖 configs，写入锁定的 run_config.yaml）
```

同一个 run-id 再运行会跳过已完成的步骤（`state.json` 的键：`baseline/{draw,gate,noise,heldout,prep,budget,q0,q0_heldout}`、`genXX/{draw,refs,search,baselines,gate,accepted,mark_shown,export,heldout,done}`、`e_t/XX`、`stopped`）。

## 流程（run.py）

- **第 0 代**（`baseline/`）：切固定验收集 → M_0 在验收集上的分数 → screen 子集重复 3 次得噪声带 → M_0 五个领域留出评测 → 对 M_0 做一次校准（Hessian + 激活统计）、诊断表、逐层量化缓存（`prep/`，之后每代复用）→ 定体积（`budget.json`）→ Q_0 = 参照配置，在验收集上打分（第一个 incumbent，`incumbent.json`）并做留出评测。
- **第 g 代**（`gen_XX/`）：draw → M_0（卡 0）在本代 screen/signal 题上的参照分数和 KL 参照 logprob → incumbent 在同样的题上打分（不扣预算，`incumbent_scores.json`）→ `rsicore.search.inner_search`（改造者 = M_0；4 线程 × 6 步，每代 ≤24 次 evaluate；前 3 名跑完整 signal；候选在卡 1 上服务）→ 可选基线 → 只把在本代题目上分数高于 incumbent 的候选交验收（≤3 次）→ 通过：导出新 incumbent（fake-quant 检查点）+ 五个领域留出评测；不通过：incumbent 不变 → `summary.json`。
- **主要指标**（`summary.json` 的 `final`）：验收集平均分 M_0 / Q_0 / 最终 incumbent、相对 Q_0 的提升 `gain_vs_q0`、追回的差距 `gap_closed` =（最终 − Q_0）/（M_0 − Q_0）、哪些代换了 incumbent；每代的留出分数和相对 M_0 的比例。
- **E_t**（默认关）：新会话、不带档案，用第 g 代的改造者在同一体积下从 Q_0 开始做一遍；只在 `loop.editor: current` 时有意义（冻结的改造者不会变）。

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 编排（上面的流程）、断点续跑、汇总 |
| `problem.py` | 一个搜索问题（源模型 + 预算 + 数据 + 参照）：prepare（缓存/诊断）、配置校验与大小、候选构建、fake-quant 落盘、候选服务、screen/full/gate 评测与 L1 反馈 |
| `tools.py` | 程序可调用的工具（注册到 `rsicore.sandbox.ToolAPI`）：`model_info`、`profile_table`、`profile_summary`、`activation_table`、`bytes_table`、`bytes(config)`、`evaluate(config)`（扣 1 次）、`solve_allocation(...)`（可在配置里关掉做消融） |
| `allocation.py` | 多选背包（每组 (字节, 损伤) 下凸包 + 贪心），给 `solve_allocation`、KL 贪心基线和 mock 改造者用 |
| `prompts.py` | 系统提示（配置 schema、工具、规则）和 `render_context(history, thread)`：incumbent 及其本代分数、候选表（大小、KL、各领域分数及相对原模型的差和显著性）、最好配置、本线程上次反馈、2–3 个退化样例、剩余预算、往代档案 |
| `diagnostics.py` | `profile_table`（只把一个模块组 RTN 到 b 比特时整模型 ΔKL，参照=原模型，KL 文本子样本）、`activation_table`、`bytes_table`；只算一次，存在 `prep/diagnostics/` |
| `baselines.py` | 统一量化、KL 贪心、EvoPress 式进化搜索（同一构建/评测路径，不占改造者预算）；`--set 'baselines.methods=[uniform,kl_greedy,evo]'` |
| `mock.py` | mock 模式：合成"真实损伤"模型、合成诊断表、mock registry 登记候选质量、mock 改造者策略 |
| `quant/spec.py` | 从 safetensors 头读模块名和形状（不加载张量）；Qwen3.5 dense 的分组 |
| `quant/schema.py` | 配置 schema、校验（列出所有错误）、解析成逐组设置 |
| `quant/sizes.py` | 解析大小 `bytes(config)` |
| `quant/quantizers.py` | RTN / GPTQ / AWQ-lite（非对称 min-max、按输入维分组），bit 打包 |
| `quant/hessian.py` | 校准前向（transformers）：每层每种输入的 Hessian 与激活统计，按层窗口分批 |
| `quant/cache.py` | 逐层量化缓存（打包整数 + fp16 scale + 打包 zero point；缺的条目按需计算并落盘） |
| `quant/export.py` | fake-quant BF16 检查点导出（结构、配置、tokenizer 文件原样复制，分片布局和张量名不变） |
| `quant/kl.py` | 本地 transformers 精确 token KL（诊断表和测试用） |
| `configs/base.yaml`、`configs/domains/*.yaml` | 全部参数（固定体积 `budget`、`loop`（改造者冻结与否、patience）、动作空间、搜索预算、验收门槛、KL、校准、缓存、E_t、基线、硬件档 `hardware_profiles`、mock 覆盖） |

## 硬件（`hardware`，27B）

角色：editor（vLLM：M_0 / 改造者、参照）、candidate（候选服务；校准、缓存、诊断在它的第一张卡上）。

**改造者 1 张卡，其余卡全部用来同时评测候选**（`common.gpu_layout` / `candidate_groups`，不设 `gpu_roles` 时）：
- 改造者在候选评测时是空闲的，所以默认只给 1 张卡；
- 其余卡按 `gpu_layout.candidate_group_size`（默认 1）分组，每组起一个候选 vLLM 服务（组内张量并行）。2 张卡 → 1 个候选服务，4 张 → 3 个，8 张 → 7 个；
- 搜索的每一步里，各线程的程序同时提出、构建、评测（`rsicore.search.inner_search(parallel=组数)`），完整评测的前 3 名也同时跑。每个线程看到的是上一步结束时的全部候选；结果按线程顺序记录；
- `QuantProblem.serve` 从服务池里取一个空闲组：在这组的卡上导出该候选的 fake-quant 检查点（导出共用一个逐层缓存，一次一个），再在这组卡上起 vLLM，用完归还。`evaluate` 线程安全：同一个配置只评一次（其他线程等结果），计费的快速筛先预留预算，不会超过每代上限；
- 磁盘：每个候选服务各留一个检查点（约 54 GB），8 张卡时约 7 × 54 GB；
- 只有 1 组时（2 张卡）就是原来的逐个评测。手动指定：`--set 'gpu_roles.editor=[0,1]'`（改造者张量并行 2），候选用剩下的卡。

H20 档：至少 2 张（每张卡只放一个 27B，约 52 GB），vLLM 显存比例 0.92、`--max-num-seqs 32`，校准每次 4 层 Hessian。B300 / mock 只给 1 张时两个角色共用。

## 模块组（Qwen3.5 dense，含 GDN）

张量名 `model.language_model.layers.<i>.{self_attn,linear_attn,mlp}.<proj>.weight`（已用 transformers 5.9.0 的 `Qwen3_5ForConditionalGeneration` 小模型核实；`mtp.*` 不计大小，vLLM 跳过）。

- 全注意力层：`qk`={q_proj,k_proj}、`v`、`o`
- 线性注意力层（Gated DeltaNet）：`gdn_qkv`={in_proj_qkv}、`gdn_z`={in_proj_z}、`gdn_ba`={in_proj_b,in_proj_a}、`gdn_out`={out_proj}
- 每层：`gate_up`={gate_proj,up_proj}、`down`
- 另有 `embed` / `lm_head` 精度（bf16 / 8 / 4，RTN g128）。组键形如 `17.down`。

## 大小怎么算（`quant/sizes.py`）

每个量化权重 [out,in]，b 比特、group size g：codes `out·in·b/8` + scale `out·⌈in/g⌉·2`（fp16）+ zero point `out·⌈in/g⌉·b/8`；AWQ 另加 `in·2`。16 比特按 2 字节。其余张量（norm、conv1d、A_log、dt_bias、视觉塔）按存储 dtype 计；`mtp.*` 不计。

## 验证状态（2026-11-19，本机 macOS 无 GPU）

- `edit_code/tests/test_quant.py`：13 个 CPU 测试（transformers 5.9.0 临时环境全部通过）。覆盖：打包往返、RTN/GPTQ/AWQ 误差随比特单调下降、GPTQ/AWQ 的输出误差 ≤ RTN、配置大小公式与单调性、schema 解析与报错、小型 Qwen3.5（含 GDN、视觉塔）的模块发现、Hessian 收集、缓存拼装 = 直接量化、fake-quant 导出后 transformers 重新加载 logits 有限、KL 随比特下降、真实 `profile_table` 路径、`QuantProblem.prepare()` 真实路径（CPU）+ 工具 + 两代递归量化。eval/.venv 里的 transformers 5.23 会把 qwen3_5 强制映射到 Qwen3_5Tokenizer，测试用的词表 tokenizer 不能用，这一项会跳过。
- `edit_code/tests/test_mock_flow.py`：固定体积循环（2 代：第 1 代 mock 改造者的比特交换超过统一 3 bit 被接受；续跑全部跳过）；`loop.editor=current` + patience 1 时第一次没有新 incumbent 就停；体积小于参照配置时拒绝运行。
- **只能在 GPU 上跑、本机没有验证**：vLLM 加载 27B fake-quant 检查点、vLLM `prompt_logprobs` 的 KL、27B 的校准/诊断/缓存耗时与显存（H20 上尤其要实测）、候选服务（`problem.serve` 真实分支）、`kl.self_generated`。

## 尚未做 / 后续

- 真实打包格式（compressed-tensors）导出和速度测量：llm-compressor 装不进 vLLM 0.24.0 + Transformers 5.9.0，没有依赖它。真实打包时，vLLM 的融合模块（qkv、gate_up、in_proj_qkvz、in_proj_ba）内部要求设置相同，目前 fake-quant 下不受此限制。
- `behavior_sensitivity` 诊断表（METHOD_DESIGN §2.2）、`profile_in_context` 工具、GAMMA / HAWQ 基线。
- 对照：`archive.enabled=false`（不带经验）、`loop.editor=current`（不冻结改造者）都已支持。
- solver 里 group size 对损伤的缩放（g64 = 0.8、g32 = 0.65）是启发式，要在 A-P0 中实测。
