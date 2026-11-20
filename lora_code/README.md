# lora_code/：Paper B 自写 LoRA（递归）

计划见 `plan/PLAN.md` §4、`plan/research/RSI_LOOP_DESIGN.md` §3、`plan/research/METHOD_DESIGN.md` §3。模型 Qwen3.8-27B（`Qwen3_5ForConditionalGeneration`）。共用运行时在 `eval/rsicore/`（契约 `eval/rsicore/INTERFACES.md`），本目录只写 Paper B 特有的部分。

## 两种写 LoRA 的方式（`method.mode`，2026-11-19）

**2026-11-20 起没有 SFT 基线，也没有 free 模式。** 整个运行里唯一的训练是 `edit` 模式在第 0 代训练一次 raw LoRA；除此之外永远不训练。改造者在任何模式下都没有 `sgd` / `grad` / `from_grad`。

| 模式 | LoRA 从哪来 | 改造者能用的工具 | 验收的目标参照 | 都不过时 |
|---|---|---|---|---|
| `edit`（默认） | **只在第 0 代**按固定 `raw` 配方训练一次 raw LoRA，之后不再训练；每一代由 M_g = M_0 + 当前 LoRA 编辑它出发的那个 LoRA（句柄 `RAW`：第 0 代是 raw，之后是当前 LoRA），结果替换当前 LoRA；权重不合并，当前 LoRA 作为 adapter 挂在 M_0 服务上 | 编辑算子、SVD、激活统计、构造工具、`raw_lora` | 第 0 代：目标 ≥ raw − ε；之后：和当前模型逐题配对：平均提升 ≥ `gate.min_gain` 且 95% 置信区间下界 > 0 | 第 0 代合并 raw；之后保留当前 LoRA |
| `scratch` | 从零写，没有任何训练 | 同上，但没有 raw LoRA | 和当前模型逐题配对：平均提升 ≥ `gate.min_gain` 且 95% 置信区间下界 > 0 | 保持原样 |

raw LoRA 的配方和大小在 `raw` 里设置（默认 rank 16、lr 1e-4、1 个 epoch、batch 4，除 `in_proj_a/b` 外的全部投影层；`raw.n_train` 空 = 第 0 代全部 train 题）：

```bash
--set method.mode=edit --set raw.n_train=400 --set raw.steps=50 --set raw.rank=16   # 用 400 条 train 题、50 步训练 raw
--set method.mode=scratch                                                            # 从零写
```

训练答案（`train_data`：拒绝采样 M_g 自己答对的回答；4 个都错的题**跳过**，不用金标，`train_data.mode: self_correct`，2026-11-20）只在训练 raw LoRA 时（edit 第 0 代）采样。训练样本的题目部分就是评测时发给模型的那条请求：评测任务自己生成的聊天消息 + Qwen 对话模板、关闭思考，和 vLLM 评测时完全一致，只在回答部分算损失；其余情况 train 题直接用金标答案、不采样，只给激活统计和 `train_example` 用。

`gate.reference` / `gate.fallback` 可以覆盖上表的默认值（`raw` | `parent`，`raw` | `keep`）。非 LLM 分支随模式变化：edit = 对 raw LoRA 随机组合编辑规则（截断 / 剪枝 / intruder / RegMean / 正交化 / 缩放），scratch = 随机的无梯度构造（奇异方向缩放）。ΔE_write 的参照分支：edit = M_t ⊕ raw（自动加 `raw` 分支），scratch = parent。E_t 会话用同一个模式（edit 时为 θ0 在 E 任务上按同一 raw 配方训练一个 raw LoRA，在 `e_ref` 里只算一次）；E-ref 是 θ0 本身在 E-test 上的分数。

## 单领域实验（`scope`）

**默认 `scope: single`（2026-11-19 起）**：只用这次运行的 `--domain`，训练、验收、评测都只在这个领域上做，不考虑别的领域。
- 训练数据只来自目标领域的 train 题；没有保留探针：验收、反馈、token-KL 都只看目标领域（`retention.domains: []`）。改造者的提示词里说明是单领域运行，`regmean_preserve`（需要保留集）不提供；
- 留出评测只测这个领域的基准（`heldout.domains: run`）；
- E_t 关闭（E 任务是另一个领域），不起原模型服务；
- 运行名默认带 `_single`（例如 `B_math_s42_single`），和旧的五领域运行分开。
- 原设计（其余四个领域作保留探针、五个领域留出评测、E_t）：`--set scope=all`（`launch.sh --all-domains`）。`--set` 显式给的值仍然优先，例如 `--set e_t.enabled=true`。

## 每一步的数据量和固定噪声带（2026-11-20）

每一步用多少题都可以在 config 里设（`--set` 或 `configs/base.yaml`），空着就用 `data/draw/paper_b.yaml` 的值。`data:` 里的抽题数会写成 `<run>/draw_config.effective.yaml`，和运行一起锁定；同一个 run-id 改了数据量会被拒绝，要换 run-id。

| 步骤 | 键 | 默认 |
|---|---|---|
| 每代抽的目标领域题数（train + signal） | `data.n`（一个数 = 每个领域；或 `{math: 1000}`） | math 2000 |
| 其中用来训练的比例 | `data.target_train_frac` | 0.5（`data/draw/paper_b.yaml`，2026-11-20 起；原为 0.8：除 edit 第 0 代外都不训练，题目多给评测） |
| 快速筛的题数 | `data.screen_n` | 400（2026-11-20 起；原为 200） |
| 固定验收集题数 | `data.gate_n`（同 `data.n`） | math 1000 |
| 保留探针 / token-KL（scope=all） | `data.retention_n` / `data.retention_kl` | 400 / 64 |
| edit 的 raw LoRA 训练题数 | `raw.n_train` | 空 = 第 0 代全部 train 题 |
| 噪声重评用的题数 | `baseline.noise_n` | 空 = 全部验收题 |
| 每个任务的最大生成长度 | `max_tokens.<任务>`（覆盖 `eval/configs/tasks.yaml`） | dapo_math 4096、taco_verified 4096、ifeval_train 2048、sealtool_train 1024、triviaqa_train 1024 |
| 训练答案的采样参数 | `train_data.temperature` / `top_p` / `top_k` / `samples` | 0.7 / 0.8 / 20 / 4（指令遵循 8） |
| 噪声重评的解码参数 | `baseline.noise_temperature` / `noise_top_p` | 0.0 / 1.0（贪心：不重跑，噪声带 = 0） |
| 留出评测的解码参数 | `heldout.decoding`（`temperature` / `top_p` / `top_k` / `seed` / `n` / `n_by_task`） | 0.7 / 0.8 / 20 / 42，avg@n |
| 训练序列长度 | `train.max_len` | 4608 |
| 每个候选实际评测的题数上限 | `eval.screen_limit` / `full_limit` / `gate_limit` / `kl_limit` | 空 = 不截 |
| 留出评测 / E_t | `heldout.limit`、`e_t.n_train` / `n_signal` / `test_limit` | |

`raw.n_train` 用固定种子抽子集。train 题最多 `data.n × data.target_train_frac` 条，想用更多就调大 `data.n`。

**固定噪声带**：`baseline.noise_fixed` 给了值就不再做第 0 代的采样重评，直接用这个数当噪声带（原样使用，不再和 2×SE 取大）：

```bash
--set baseline.noise_fixed=0.03                     # 所有数据集
--set 'baseline.noise_fixed={math: 0.03}'           # 按领域 / 数据集名 / default；没给值的数据集照常测
--set baseline.noise_n=200                          # 照常测，但只用 200 道验收题重评（更省；题少噪声带偏大，偏保守）
--set data.n=500 --set raw.n_train=200 --set data.gate_n=300
```

测出来的值在 `baseline/gate_scores.json`（`noise_band`、`noise_source`）里，下次可以直接填进 `noise_fixed`。

**解码（2026-11-20）**：搜索、验收、噪声都用贪心（temperature 0、每题 1 个回答、关闭思考），保证结果确定、可以逐题配对；留出评测用采样，按 Qwen 非思考模式的推荐（temperature 0.7、top_p 0.8、top_k 20，种子 42），每题采 n 个取平均（avg@n：AIME 24/25、AMC23 各 8 个，MATH-500、GPQA-Diamond 各 4 个，其余 1 个；`heldout.decoding`，null = 用 tasks.yaml 的贪心）。噪声重评用贪心时结果完全一样，所以不重跑，噪声带 = 0。训练答案（只在 edit 第 0 代）用同一组采样参数（0.7 / 0.8 / top_k 20），每题采 4 个，取第一个判对的，4 个都错就跳过这道题。只有改造者写程序时用自己的采样参数（`editor.temperature` 0.6）。最大生成长度按任务，在运行配置的 `max_tokens` 里设（覆盖 `eval/configs/tasks.yaml`，对这次运行的所有模型调用生效）：math 4096、code 4096、指令遵循 2048、工具调用 1024、知识（triviaqa_train）1024 token，例如 `--set max_tokens.dapo_math=2048`。训练序列长度 `train.max_len` = 4608（回答最多 4096 + 题目留 512），KL 文本里 M_0 回答的长度见 Paper A 的 `kl.self_generated_max_tokens`。

## 启动

```bash
./run_rsi.sh --paper B --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id B_math_s42
./run_rsi.sh --paper B --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1,2 --run-id B_math_s42 --set hardware=h20   # H20（按每张卡 110 GB 规划）
./run_rsi.sh --paper B --domain math --gens 2 --mock          # 本机 mock，约 5 分钟
# 改配置：--set search.max_evaluate=16 --set gate.delta=0.03（写进 run_config.yaml 并锁定）
```

同一个 run-id 再运行会跳过已完成的步骤（`state.json`）。**提前停止**（2026-11-20）：连续 `loop.patience` = 3 代验收都保留当前模型（`accepted.json` 的 source 为 keep）就停（`state.json` 的 `stopped`）；null = 总是跑满 `--gens` 代。产物在 `experiment_records/<run-id>/`。

## 一代（第 g 代：改造者 M_g → 接任者 M_{g+1}；目标领域每代都是 `--domain`）

| 步骤（state key） | 内容 | 产物 |
|---|---|---|
| `baseline/gate` | M_0 在固定验收集（五个领域）上评测，噪声带 = max(重复评测波动, 2×SE) | `baseline/gate_scores.json` |
| `baseline/heldout` | M_0 留出评测 | `baseline/heldout.json` |
| `e_ref` | E 任务参照：θ0 本身在 E-test 上的分数；edit 模式另按 raw 配方为 θ0 训练一次 E 任务 raw LoRA | `e_ref/result.json` |
| `genXX/draw` | `draw_gen(target=domain)`：目标 train / signal / screen，其余领域 signal / screen / kl | `draws/gen_XX/` |
| `genXX/train_data` | train 题的训练目标：edit 第 0 代拒绝采样 M_g 自己答对的回答，4 个都错的题没有训练目标、不参与训练（`train_data.mode: self_correct`）；其余情况只用金标、不采样 | `train_data.json` |
| `genXX/diagnostics` | 固定诊断表（每个 (层, 模块)：范数、奇异值、已合并 adapter 的范数 / 有效秩 / intruder / 相对变化分位 / 激活比） | `diagnostics.json` |
| `genXX/raw` | 仅 `edit`：第 0 代按 `raw` 配方训练 raw LoRA（整个运行唯一的训练），screen / full / gate 评测；之后各代记录改造者出发的 LoRA = 当前 LoRA，不训练 | `baselines/raw.json`、`baselines/raw_lora.*` |
| `genXX/search` | `rsicore.search.inner_search`：4 线程 × 8 步、≤32 次 evaluate；改造者 = gpu0 上的 M_g（vLLM 多 LoRA），候选 LoRA 在 gpu1 上构建后热加载评测 | `candidates.jsonl`、`programs/`、`feedback/`、`loras/` |
| `genXX/gate` | `submit_to_gate`（≤3 次）：只送在信号题上分数 > 0（赢了当前模型）的候选（2026-11-20）；目标对参照（edit 第 0 代：raw − ε；edit 之后和 scratch：在验收集上和当前模型逐题配对，平均提升 ≥ min_gain 且置信区间下界 > 0）；每个保留领域 ≥ 第 0 代 − max(δ, 噪声带)；已学目标 ≥ 学完时 − δ；沙箱检查。都不过 → edit 第 0 代兜底合并 raw LoRA（要能过验收）→ 否则保留 M_g | `gate.json`、`accepted.json`、`accepted_lora.*`、`parent_gate.json`（scratch） |
| `genXX/merge` | LoRA 合并进权重 → M_{g+1}（mock：更新 registry 中 `m{g+1}` 的 quality）；`edit` 模式不合并，`model.json` 记录 M_{g+1} = M_0 + 哪个 LoRA | `model/`、`model.json` |
| `genXX/nonllm` | 非 LLM 分支（同 evaluate 预算，按模式）：edit 随机编辑规则作用于 raw；scratch 随机无梯度构造 | `baselines/nonllm.json` |
| `genXX/e_t/<branch>` | E_t：parent / llm / nonllm（edit 另加 raw）分支模型各开新会话，为 θ0 在 Seal-Tools 上写 LoRA（16 次 evaluate），在 Seal-Tools test + BFCL 上测；`--domain tool_calling` 时换成代码任务（TACO → LiveCodeBench） | `e_t/<branch>/result.json` |
| `genXX/heldout` | M_{g+1} 五个领域留出评测 | `heldout.json` |
| `genXX/summary` | 记录展示过的错例（`draw.py --mark-shown`）；汇总 ΔE_write、ΔE_nonLLM（按题配对 bootstrap） | `summary.json` |

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 编排（上表）；`Servers` 管理 vLLM / mock 服务（GPU 分工见下面"硬件"） |
| `configs/base.yaml`、`configs/domains/*.yaml` | 预算、秩上限 64、范数上限 τ、raw 配方、GPU 秒预算、搜索规模、ε / δ、λ、E_t、mock 缩小版（`mock_overrides`） |
| `primitives/naming.py` | Qwen3.5 模块名、vLLM 打包映射、PEFT 键名 |
| `primitives/weights.py` | `WeightStore`：按需读 safetensors 分片，top-k SVD（缓存） |
| `primitives/lora.py` | `LoRA`（A、B 因子，缩放并入 B）、沙箱检查、PEFT 保存 / 读取（vLLM 可加载）、合并进权重 |
| `primitives/construct.py` | `rank1`、`from_direction`、`scale_singular` |
| `primitives/ops.py` | `scale`、`interpolate`、`combine`、`drop`、`truncate_rank`、`intruder_scale`、`relchange_prune`、`regmean_preserve`、`orthogonalize`、`ties` |
| `primitives/train.py` | `TrainData`（只允许本代 train 题）、`HFModel`（transformers + peft：loss / sgd（只用于 raw LoRA）/ acts / grams / act_ratio） |
| `tools.py` | `Workspace`：把原语注册成 `ToolAPI` 工具（句柄、预算、train 限制、evaluate），`program_build`（沙箱执行 → 检查 → 保存候选） |
| `evaluator.py` | 候选评测（screen / full / gate、token-KL、配对差、错例、行为统计、总分） |
| `diagnostics.py` | 诊断表计算与渲染 |
| `prompts.py` | 改造者 system prompt、`ContextRenderer`（候选表、本线程最佳程序、最近反馈、档案） |
| `baselines.py` | raw LoRA 配方与训练（edit 第 0 代）、非 LLM 分支（随机编辑 / 随机构造） |
| `etask.py` | E_t 固定任务 |
| `mock.py` | mock 模型几何、mock LoRA 与 quality 模型、mock 诊断、mock 改造者策略 |

## 改造者看到的工具

程序形如 `def main(api): ...; return {"lora": handle}`（`{"lora": None}` = 保持原样），工具只能用关键字参数调用：
`modules`、`diagnostics`、`train_ids`、`train_example`、`svd`、`acts`、`new_lora`、`rank1`、`from_direction`、`scale_singular`、`scale`、`interpolate`、`combine`、`drop`、`truncate_rank`、`intruder_scale`、`relchange_prune`、`regmean_preserve`、`orthogonalize`、`ties`、`inspect`、`stack`、`raw_lora`、`candidate`、`evaluate`、`budget`（说明见 `tools.py` 的 `TOOL_DOCS`）。

按模式过滤（`tools.tools_for_mode`）：`edit` 有 `raw_lora`，`scratch` 没有。任何模式都没有训练 / 梯度工具（`sgd` / `grad` / `from_grad` 已删除）。没注册的工具调用会被 ToolAPI 拒绝。

2026-11-20 新增两个工具（`tools.compare` / `tools.train_loss` 可关掉）：
- `compare(cid, other='parent', level='screen')`：两个**已评测过**的模型逐题对比（坏掉 / 修好的题数和题号、按输出长度分组、截断率），只给题号和统计，不给题目内容，免费，从不触发评测。
- `train_loss(lora, ids=None, n=64)`：在 train 题上算当前模型 + 这个 LoRA 下答案的平均 NLL，并和不加 LoRA 比（只做前向，不训练，消耗 gpu_seconds）。scratch 里改造者可以先用它便宜地筛掉明显变差的构造，再花 evaluate。

- `acts` / `train_example` 只接受本代 `train` 题号，其他题号抛 `PermissionError`。
- `regmean_preserve` 的保留集 Gram 矩阵由可信运行时用保留领域的 `kl` 题面算出，改造者只拿到结果 LoRA。
- 预算：evaluate（相同 LoRA 只计一次）、gpu_seconds（`acts` / `regmean_preserve`，实测；mock 中估计）。

## LoRA 格式（vLLM 0.24.0）

- 键名用未打包的 HF 名：`base_model.model.model.language_model.layers.<i>.<self_attn|linear_attn|mlp>.<leaf>.lora_{A,B}.weight`；vLLM 自己打包成 `qkv_proj`、`gate_up_proj`、`in_proj_qkvz`、`in_proj_ba`，并用 `hf_to_vllm_mapper` 映射 `model.language_model.` 前缀。
- 全局统一秩 r（不足的模块补零，r 取 vLLM 支持的值 1/8/16/32/64…），`lora_alpha = r`，缩放恒为 1；`target_modules` 写叶子名。
- 合并：`W' = W + B A`，fp32 累加后转回原 dtype；只重写含改动张量的分片，其余硬链接；`merge_report.json` 记录 bf16 舍入后的实际相对误差。

## 测试

```bash
# 原语（tiny Qwen3.5：4 层，3 GDN + 1 全注意力；需要 torch / transformers 5.9 / peft / safetensors / pytest）
<venv>/bin/python -I -m pytest -q lora_code/tests/test_primitives.py
# mock 流程（需要 pytest；端到端部分用 eval/.venv 跑 run.py）
<venv>/bin/python -I -m pytest -q lora_code/tests/test_mock_flow.py
```

`eval/.venv` 没有 pytest / transformers / peft；本机测试用临时 venv（`uv venv --python 3.11 && uv pip install torch transformers==5.9.0 peft safetensors pyyaml numpy pytest`）。

## 硬件（`hardware`，27B）

角色：main（vLLM：M_t + 候选 LoRA，改造者）、theta0（vLLM：原模型，只给 E_t / e_ref 用）、train（HF 模型：训练、权重缓存、诊断、合并）。vLLM 角色多张卡 = 张量并行；train 多张卡 = 按层拆开（`device_map`）。

**给多少张卡就用多少张**（`common.gpu_layout`，不设 `gpu_roles` 时）：main 和 train 各一张，E_t 开着且 `gpu_layout.theta0_own_gpu`（H20）时 theta0 也单独一张，否则 theta0 和 train 共用第一张；多的卡依次给 main（TP 2）→ train（2）→ main（TP 4）→ theta0（TP 2）→ main（TP 8），剩下的全给 train，不留空卡。

| 卡数 | 单领域（默认） | 五领域（`scope=all`，H20） |
|---|---|---|
| 2 | main 1 · train 1 | 不够（要 3 张） |
| 3 | main 2 · train 1 | main 1 · theta0 1 · train 1 |
| 4 | main 2 · train 2 | main 2 · theta0 1 · train 1 |
| 5 | main 2 · train 3 | main 2 · theta0 1 · train 2 |
| 6 | main 4 · train 2 | main 2 · theta0 2 · train 2 |
| 8 | main 4 · train 4 | main 4 · theta0 2 · train 2 |

手动指定：`--set 'gpu_roles.train=[2,3]'`（--gpus 里的下标）把这个角色固定下来，其余角色在剩下的卡上自动分。B300 上 theta0 和 train 共用一张卡（显存比例 0.45）。

H20 档（按每张卡 110 GB 规划，卡约 140 GB）另外：vLLM 显存用 `--rsi-gpu-memory-gib 110`（启动时换算成显存比例），`--max-num-seqs 128`、上下文 20480、8 个 LoRA 槽；训练 `train.micro_batch: 2`【试点值】（每次前向 2 条序列，梯度累积，结果等于整批的平均损失）。原模型服务和训练模型放不进同一张 110 GB 的卡，所以五领域（有 E_t）时原模型单独一张。80 GB 的卡用 `h20_80`（0.92、32 条序列、4 个 LoRA 槽、每次 1 条）。

## 只能在 GPU 上验证的部分（本机未实跑）

真实 vLLM 加载我们的 PEFT adapter（Qwen3.5 的线性注意力投影）、27B 的 peft 短训练（transformers 没有 flash-linear-attention 时走 GDN 的 torch 实现，慢且占显存）、27B 上的 SVD / 诊断耗时、合并后的 bf16 舍入误差、各卡显存分配（`server.main_args` / `theta0_args`；H20 上 27B 训练的显存峰值要第一周实测）。
