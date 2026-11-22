# edit_code/：Paper A 固定体积下的迭代自量化

计划见 `plan/PLAN.md` §3、`plan/research/RSI_LOOP_DESIGN.md` §2、`plan/research/METHOD_DESIGN.md` §2；共用运行时接口见 `eval/rsicore/INTERFACES.md`（本目录只通过它调用 rsicore）。

**设计（2026-11-19 修订）**：改造者的权重冻结，不断迭代的是被量化的模型。
- 改造者：原始 BF16 模型 M_0（`loop.editor: frozen`）；消融 `loop.editor: current` 让当前最好的量化模型来写。
- 每个候选都从 **M_0 的原始权重**出发量化，所有代用**同一个体积**（默认 = 统一 4 bit GPTQ g128（2026-11-20 起；原为 3 bit）、词嵌入和输出层 8 bit 那份配置的打包大小，它也是起点 Q_0）。
- 第 g 代的目标是超过当前最好的量化 Q_{g-1}（incumbent）：在固定验收集上逐题配对比较，平均提升 ≥ `gate.min_delta`，至少一个领域显著变好、没有领域显著变差，才换成新的 incumbent；否则保留。连续 `loop.patience` 代没有新的 incumbent 就停。
- 改造者能看到：incumbent 的配置和它在本代题目上的分数、往代**每个**候选一行（经验通道，2026-11-22 起与 Paper B 一致）、诊断表。

## 单领域实验（`scope`）

**默认 `scope: single`（2026-11-19 起）**：只用这次运行的 `--domain`。信号和验收只用这个领域，KL / 校准文本（`kl.domains: run`）和留出评测（`heldout.domains: run`）也只用这个领域，不考虑别的领域。`kl_contexts` 段 KL / 校准文本全部从本领域抽（2026-11-20 起；之前平均分给五个领域，单领域只用到约 1/5）。运行名默认带 `_single`。原设计（KL / 校准文本和留出评测覆盖五个领域）用 `--set scope=all`（`launch.sh --all-domains`）；`--domain all` 总是 `scope=all`。

## 每一步的数据量和固定噪声带（2026-11-20）

空着就用 `data/draw/paper_a.yaml` 的值；`data:` 里的抽题数写成 `<run>/draw_config.effective.yaml`，和运行一起锁定（同一个 run-id 改数据量会被拒绝）。

| 步骤 | 键 | 默认 |
|---|---|---|
| 每代每个领域的 signal 题数（完整评测） | `data.n`（一个数 = 每个领域；或 `{math: 1000}`） | math 2000 |
| 快速筛的题数 | `data.screen_n` | 400（`data/draw/paper_a.yaml`，2026-11-20 起；原为 200） |
| 每代 KL / 校准文本数 | `data.kl_contexts` | 400（`data/draw/paper_a.yaml`，2026-11-20 起；原为 256。单领域时全部来自本领域；五领域时平均分给五个领域） |
| 固定验收集题数 | `data.gate_n` | math 1000 |
| 每个候选算 KL 用的文本数 | `kl.eval_texts` | 400（= 快速筛题数；原为 64。每条文本的 KL 存在 `evals/<cid>/<stage>/kl_per_text.json`） |
| 噪声重评用的题数 | `noise.n` | 空 = 第 0 代全部 screen 题 |
| 每个任务的最大生成长度 | `max_tokens.<任务>`（覆盖 `eval/configs/tasks.yaml`） | dapo_math 4096、taco_verified 4096、ifeval_train 2048、sealtool_train 1024、triviaqa_train 1024 |
| 噪声重评的解码参数 | `noise.temperature` / `noise.top_p` | 0.0 / 1.0（贪心：重跑结果完全一样，所以不重跑，噪声带 = 0；> 0 才做采样重评） |
| 留出评测的解码参数 | `heldout.decoding` | temperature 0.7、top_p 0.8、top_k 20、种子 42，avg@n（AIME / AMC 8 个，MATH-500 / GPQA 4 个，其余 1 个）；搜索、验收、噪声默认贪心（`decoding:` 见下文；`simple` 用 thinking 采样） |
| 校准文本 / 诊断表 / 留出评测 | `calibration.max_texts`、`diagnostics.profile_contexts`、`heldout.limit`、`e_task.heldout_limit` | |

**固定噪声带**：`noise.fixed` 给了值就跳过 M_0 的采样重评，直接写进 `baseline/noise.json`：

```bash
--set noise.fixed=0.03                       # 所有数据集
--set 'noise.fixed={math: 0.03}'             # 按领域 / 数据集名 / default；没给值的数据集照常测
--set noise.n=100                            # 照常测，只用 100 道 screen 题
```

## 直接在评测集上优化（`objective: eval_overfit`，2026-11-20）

不做训练 / 信号 / 验收切分，直接在评测集上过拟合：`--set objective=eval_overfit`（或 `./launch.sh --eval-overfit`），运行名带 `_evalfit`。
- 数据：`math_eval` = MATH-500 + GSM8K test，共 1819 道（`data/scripts/build_eval_target.py`，首次用到时自动生成；评测任务 `math_eval`，math 评分）。
- 快速筛、完整评测、验收都是这全部 1819 道，贪心打分；完整评测和验收直接复用快速筛的逐题结果，不重复评测。不抽题，不标记展示过的题。
- 候选分数 = 评测集准确率 / incumbent 的（`kl_weight: 0`，不算 KL）；**评测集分数严格高于 incumbent 就接受**（`gate.min_delta: 1e-6`，不做显著性检验）。
- KL / 校准文本 = 400 道评测题的题目 + M_0 的回答（GPTQ / AWQ 的 Hessian 也在评测题上算）；反馈例题额外给出标准答案。
- 留出评测也直接在这 1819 道上做（`heldout.domains: eval_target`，贪心，2026-11-21）：报告的就是过拟合的那份分数，不再单独测 MATH-500 / GSM8K / AIME。

**自定义数据角色 `data_roles`（2026-11-21）**：三个角色各自指定用哪份数据（`--set data_roles.train=... --set data_roles.gate=... --set data_roles.eval=...`，或 `./launch.sh --train-set / --gate-set / --eval-on`），由 rsicore 解析（`rsicore.evalsets`、`rsicore.draws.train_dataset / gate_dataset`）。
- `train`：改造者用来工作的数据：快速筛 / 完整评测的题、本代 incumbent 的对照分数、噪声带、KL / 校准文本（kl 角色）、反馈例题、`compare`。
- `gate`：决定是否接受候选的数据：M_0 / Q_0 的验收基线分（`baseline/gate.json`）、incumbent 的验收记录、候选的验收评测（`gen_XX/gate.json`）。
- `eval`：留出评测报告（逗号分隔）：领域（`math` = math500,gsm8k,aime24,aime25）、留出任务（`math500`、`humanevalplus`、`ifeval`……，test 池，按 `heldout.decoding` 采样）、评测集（`math`、`humanevalplus`……的 K_eval 副本，即 `math_eval`、`humanevalplus_eval`）、或 `train` / `gate`（该角色是评测集时）。
- `train` / `gate` 可取：本领域的训练集（`dapo_math`、`taco_verified`、`triviaqa_train`、`ifeval_train`、`sealtool_train`：照常每代抽题 / 固定验收切分）或评测集（`eval/configs/tasks.yaml` 的 `eval_targets` 键，如 `math`、`math500`、`gsm8k`、`humanevalplus`、`ifeval`，或其数据集名 `math500_eval`：用全部题，不抽题，不标记展示过的题，反馈例题额外给出标准答案）。
- 默认（不设 `data_roles`）：`train` = `gate` = 本领域的训练集；`objective: eval_overfit` 时都是该评测集；`eval` 按 `heldout.domains`。和原来完全一样。
- 限制：`train` / `gate` 必须属于本次的 `--domain`；评测集做 `train` 或 `gate` 需要 `scope: single`；`--domain all` 不能用 `data_roles`。和默认不同的 `train` / `gate` 会让运行名多出 `_train-<x>` / `_gate-<x>`。
- 验收和训练在不同数据上时，验收一定真的评测（只有数据集和题号都和快速筛一样时才复用快速筛的逐题结果）。

```bash
# 照常用 dapo_math 抽题训练，用全部 MATH-500 验收，只报告 MATH-500
--domain math --set data_roles.gate=math500 --set data_roles.eval=math500
# 在 MATH-500 全部题上训练，用 dapo_math 的固定验收集验收
--domain math --set data_roles.train=math500 --set data_roles.gate=dapo_math
# 指令遵循：ifeval_train 训练，IFEval 评测集验收和报告
--domain instruction_following --set data_roles.gate=ifeval --set data_roles.eval=gate
```

**`simple`（2026-11-21 v2，`--set objective=simple` / `./launch.sh --simple`）**：用 `simple_data/<领域>/`（`data/scripts/make_simple_data.py`，从去污染、去重后的训练集池子里抽），配置在 `objective_profiles.simple`：
- **数据**：训练角色 = `simple_<领域>_pool`（整个干净池子减去验收 / 测试题）。**每代重新抽** `simple.screen_n`（默认 500）道当快速筛（= 完整评测，`search.top_full: 0`，完整评测直接复用快速筛结果），种子只含 `--seed`、领域和轮次（`rsicore.draws`，Paper A 第 g 代和 Paper B 第 g-1 代抽到同一批题）；KL / 校准文本 = 本代池子里另外 400 道（kl 角色）+ M_0 的回答。各代的快速筛可以重叠，不标记展示过的题；题号写在 `draws/gen_XX/ids/`。验收 = `simple_<领域>_eval` 的 500 道（照常的配对验收，与 incumbent 比）；**留出评测 = `simple_<领域>_test` 的另外 500 道**（`data_roles.eval: test`）。
- **打分解码**：快速筛、验收、留出评测全部用 Qwen3 thinking 采样（temperature 0.6 / top_p 0.95 / top_k 20，max_tokens 16384，种子 = 运行的 `--seed`），三者都是 n 4（每题分数 = 4 次的平均，avg@4）。incumbent、Q_0、M_0 的验收基线和所有候选用完全相同的解码和种子。候选服务的 `--max-model-len` 提到 32768（`vllm.candidate_max_model_len`），保证 prompt + 16384 放得下。
- **最终流程（2026-11-21）**：每代先让**最新模型（incumbent）**回答本代的快速筛题一次（n 4，所有线程 / 步共用）；**候选不跑快速筛**（`search.screen_candidates: false`）。改造者在每步上下文里看到 incumbent 快速筛的摘要（avg@4、每题答对次数分布、截断率、几道失败题的题目 / 标准答案 / 抽取答案），并可以用程序和 run_python 里的只读工具 `api.screen_results(ids, filter='all'|'wrong'|'unstable'|'right', limit, offset, gen)`、`api.screen_output(id, sample, offset, gen)`（整段输出，含 thinking，分页）看全部数据。**没有 `api.evaluate`**（`program_evaluate_calls: 0`：工具、文档和提示词里都去掉；`compare` 也不注册）。每个候选建好后做**一次验收评测**（`search.gate_every_candidate: true`，500 题 × 4 次，和 incumbent 的验收记录逐题配对、同一种子；超出预算的配置不评测直接拒绝），结果按候选写进 `gen_XX/gate.json`（submissions 列表，每个候选一条：准确率、配对增益、CI、是否通过 / 原因、broken / fixed 个数和题号，不含题目内容），后面的步和后面的代都能看到（候选表、自己的反馈、`search_history`、`read_record`、存档；存档里还有往期 incumbent 的快速筛摘要，`screen_results(gen=k)` 可读往期数据）。一代结束时，在通过验收规则（尺寸、`gate.min_delta`、`require_significant` 时的 CI）的候选里接受**增益最大**的那个（平分时 cid 靠前者）；都不通过就保持。然后新 incumbent 跑测试集（500 × 4，唯一无偏的数字）。代价：每代 线程 × 步 × 500 × 4 个 thinking 采样花在验收集上。
- **搜索**：8 个线程 × 1 步（`search.threads` / `search.steps` 都可配置，提示词用配置的数字）：1 步时提示"你是本轮 N 个会话中的第 k 个、彼此独立"，并让它用 `search_history` / 存档避开往期思路；多步时保留"改进本线程最好的配置"。
- 改造者每个候选最多 `editor.tool_rounds` 轮工具调用（simple 为 20，可配置，提示词里的轮数跟着配置走；上下文 `editor.context_tokens` 131072）；`editor.explore.persistent: true`：run_python 每个提案一个持久会话，变量跨轮保留（超时则重启、变量丢失；最终程序仍须自包含）。五个领域都可以用；运行名带 `_simple`。profile 改了，旧的 `*_simple` 运行不能续跑，要换 `--run-id` / `--tag`。各领域合并后的配置见 `plan/configs/A_<领域>_simple.yaml`（由协调方更新）。服务器上需要先有 `simple_data/<领域>/` 的 `eval.jsonl`、`test.jsonl`、`pool_ids.json` 和 `data/processed/` 里对应的训练集。
- mock：`mock_overrides` 优先于 profile；`mock_objective_overrides.simple` 再把 mock 的 simple 运行改回"3 线程 × 1 步、候选不筛、每个候选一次验收、没有 api.evaluate"的布局，并用真实 simple 的验收规则（`gate.min_delta: 0.0`）、不跑 baselines（2026-11-22）；快速筛题数用 `--set simple.screen_n=20` 缩小。

## 与 Paper B 的流程对齐（2026-11-22，`objective: simple`）

每代流程和 Paper B 相同：从训练池抽 500 道 → 当前模型（incumbent）在这 500 道上采样（n 4，thinking）→ 8 个会话共享这批题分析、各写一个配置 → 每个配置在验收集上评测一次（与 incumbent 逐题配对）→ 通过且增益最大的成为新 incumbent，再跑测试集。这次改的是和 B 不一致的细节：
- **incumbent 的快速筛分到所有空闲的候选 GPU 上**（`QuantProblem.serve_all` + `score_split`；原来只用一张卡），和 B 用所有副本采样一样。检查点只写一次（有导出的 incumbent 模型就直接用）。候选的验收仍是一张卡一个。
- **不再做每代的 KL 工作**：simple 的 `search.kl_weight: 0.0`，验收只看配对增益，所以每代不再让 M_0 生成 KL 文本的回答、不再算 M_0 的参考 logprob 和 incumbent 的 KL（`Runner.kl_used`；E_t 或 baselines 打开时照旧；E_t 的留出 KL 参考在 `kl_weight: 0` 时也照常计算）。第 0 代的校准文本（含 M_0 回答，用于 Hessian）不变。
- **候选的 note**：程序在返回的配置里写 `"note"`（最多 600 字：改了哪些组、为什么、试过什么被否了及数字），存进候选记录，`search_history` 和存档里都显示（`{"config": ..., "note": ...}` 也接受；`solve_allocation` 自动写的那段去掉，编辑者在后面追加的保留）。
- **存档列出往代每个候选**（含失败的），每行：配置一行描述（覆盖按设置分组，带层号范围，如 `8b/g128/gptq: down L0-3 x4`）、GiB、平均位数、验收结果（增益、CI、未通过原因）、报错第一行、note；`archive.max_chars`（默认 30000）超长时先删最老几代的行；有 `search_history` / `read_record` 时不再贴完整配置（`archive.top_candidates` 只在没有这两个工具时用）。
- **和 incumbent 完全相同的配置不再评测验收**：直接记为增益 0、不通过（B 的空 LoRA 同样处理），省一次 500 × 4 的验收，也不会给后面的会话显示一个噪声造成的"通过"。
- `loop.patience: 3`（simple，和 B 一样：连续 3 代没有新 incumbent 才停）；系统提示写明程序时限（`sandbox.timeout_s`）和工具调用总数上限（`editor.max_tool_calls`）。
- simple profile 改了，旧的 `*_simple` 运行不能续跑（锁定的配置不同），要换 `--run-id` / `--tag`。
- 没有照搬的：B 每代重测当前模型的验收（A 的 incumbent 就是那份配置本身，同一种子重测只会复现原记录）；B 的改造者跑在多个副本上（A 的 8 个会话共用编辑 GPU 上的一个 M_0 服务）。

## 打分解码与搜索预算（2026-11-21）

**打分解码 `decoding: {screen, gate}`**（顶层，`full` 可选，空 = 跟 screen）：每个阶段一个 dict（temperature、top_p、top_k、min_p、n、thinking、max_tokens、seed），经 `rsicore.config.stage_decoding(cfg, stage, seed)` 传给每一次 `score` / `score_split`：候选和 incumbent 的快速筛 / 完整评测 / 验收、Q_0 的验收、M_0 的验收基线（`baseline/gate.json`）。null = `eval/configs/tasks.yaml`（贪心），即默认行为不变。种子为空时用运行的 `--seed`。评测记录（`evals.jsonl`）里记下 `decoding`；缓存的评测（同一配置、同一阶段）以及"完整评测 / 验收复用快速筛逐题结果"都要求解码相同（n 4 的验收不会复用 n 1 的快速筛）。incumbent 在本代题上只采样一次（cid `incumbent`），所有候选都和这一份记录配对比较。

**搜索预算键**（`edit_code.common.search_settings`，默认值 = 原来的行为）：

| 键 | 默认 | 作用 |
|---|---|---|
| `search.charge_screens` | true | true：每个返回配置的自动快速筛扣 `max_evaluate`（和 `api.evaluate` 共用；预算用完就不再运行程序）。false：自动快速筛免费、程序总是运行，`api.evaluate` 走单独的池子，本代已评测过的同一配置再评测免费，失败的调用（配置无效、池子空了）不扣 |
| `search.program_evaluate_total` | null | 免费快速筛时每代的 `api.evaluate` 池子（null = threads × program_evaluate_calls；没有每程序上限时 = max_evaluate） |
| `search.program_evaluate_calls` | 2 | 每个程序最多几次 `api.evaluate`（null = 不设每程序上限） |
| `search.max_candidates` | null | 每代候选数上限（null = threads × steps；扣快速筛时同时不超过 max_evaluate，和原来一样） |

steps = 1 时，提示里写明每个线程是本轮 N 个独立会话之一、看不到同代其他候选，并删掉"在最好的候选（你的或别的线程的）上改进"；免费快速筛时预算行写"返回的配置自动免费评测；api.evaluate 池子还剩 N 次、每个程序最多 k 次"。E_t 用 `e_task` 的 threads / steps / evaluate，候选数和池子按它自己的形状推出。

## 分析工具与保存的数据（2026-11-20）

改造者除了评测，还可以用几个**免费**（不扣 evaluate）的分析工具：

| 工具 | 回答什么问题 | 怎么算的 |
|---|---|---|
| `profile_table()` / `profile_summary()` | 哪些组对量化敏感（整模型 KL） | 第 0 代：每次只把一个组 RTN 量化到 2/3/4 bit，在 64 段 KL 文本上前向，和 BF16 比全词表 KL |
| `hessian_table()` | 同上，但噪声小、覆盖 rtn / gptq / awq | 第 0 代：每组单独量化的 Hessian 加权误差 tr(ΔW H ΔWᵀ)，用缓存里的量化权重和全部校准 token 的 Hessian，不做前向 |
| `activation_table()` | 哪些层的输入激活离群严重（适合 AWQ / 更高比特） | 校准时顺便统计 |
| `group_error(config)` | 一个配置的误差集中在哪些组 | 同 `hessian_table` 的算法，逐组查表 / 按需计算（结果缓存在 `prep/diagnostics/herr_cache.json`） |
| `compare(cid, other="incumbent")` | 这个候选和 incumbent（或另一个候选）比，坏了哪些题、修好哪些题 | 逐题对比两份评测记录：翻转数和题号、按输出长度分组、截断率、平均输出长度；**只给题号和统计，不给题目内容**，所以不会用掉信号题 |

每个候选每个阶段保存在 `gen_XX/evals/<cid>/<stage>/`：逐题记录（`<数据集>.jsonl`，含原始输出、解析答案、对错）、每段文本的 KL（`kl_per_text.json`）、每组 Hessian 加权误差（`group_error.json`）。incumbent 的 cid 是 `incumbent`。量化后的模型文件不保留（每个约 54 GB；配置已存，可从缓存几秒钟重新拼出）。

## 改造者：thinking + 工具轮次（2026-11-21）

改造者写每个候选时开 Qwen3 thinking（`editor.extra_body.chat_template_kwargs.enable_thinking: true`，temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0），并且可以先调用工具再交程序（`rsicore.editor` + `rsicore.chat_tools`）：
- **一轮** = 模型的一次回复里带了工具调用（并行的几个调用算一轮）。最多 `editor.tool_rounds` 轮（默认 50），之后再给一次 `tool_choice: "none"` 的最终回复。候选 = 不带工具调用、含一个 ```python 块的那次回复。
- 工具：`run_python(code)` 在同一个沙箱里跑探索代码，`api` 和最终程序的一样，但**没有 `api.evaluate`**（`make_tool_api(explore=True)`），`group_error` 只能查已缓存的设置（`hessian_table` 那些；`editor.explore.gpu_tools: true` 才允许现算）；`read_record(cid, part)` 看表里某个候选的完整程序 / 反馈；`search_history(gen, sort, offset)`（2026-11-21，默认开，`editor.history_tool=false` 关）列出所有试过的候选（本代可见的 + 往期所有代），每个一行：config 描述（默认设置 + 按设置分组的覆盖及其模块类型）、GiB、平均位数、KL、各 domain 相对 incumbent 的差、分数、验收结果，`same_as` = config hash 与更早候选相同；`read_record` 也能打开往期候选。E_t 只看自己的候选。系统提示里的验收规则按 `gate.min_delta` / `gate.require_significant` 生成（2026-11-21，关掉显著性时写明"不做显著性检验"），并写明程序里 `api` 只是 `main(api)` 的参数。探索**不扣**本代的 evaluate 预算，也不占候选槽。每次调用的完整输出在 `gen_XX/explore/<cid>/rNN_i/`。
- 记录：`gen_XX/editor/thread_K.jsonl`（format 2，每轮的推理、工具调用和结果都在）；`candidates.jsonl` 的 cost 里有 `editor_rounds / editor_tool_calls / editor_tool_s / editor_prompt_tokens / editor_completion_tokens / editor_stop`。
- E_t 用同一个改造者工厂，也一样。

| 配置（`editor.*`） | 默认 | 作用 |
|---|---|---|
| `tool_rounds` | 50 | 每个候选最多几轮工具调用；**0 = 以前的单次调用**（请求里没有 tools，提示词逐字不变） |
| `context_tokens` | 131072（`h20_80`: 49152） | **上下文长度的唯一旋钮**：客户端按它限制每轮 max_tokens、触发压缩，同时作为改造者 vLLM 服务的 `--max-model-len`（覆盖 `vllm.editor_args` 里写的值；候选服务的 20480 不变，`simple` 用 `vllm.candidate_max_model_len` 改成 32768）。null = 用 `editor_args` 原样 |
| `turn_max_tokens` / `max_tokens` | 8192 / 16384 | 工具轮 / 最终回复的 max_tokens |
| `max_tool_calls` / `proposal_tokens` / `proposal_wall_s` | 100 / 160000 / 2400 | 任一到顶就强制进入最终回复 |
| `tool_result_chars` | 6000 | 工具结果截断长度 |
| `compact_at` / `keep_tool_rounds` / `keep_reasoning_rounds` | 0.6 / 8 / 2 | 提示超过 0.6 × context_tokens 时压缩：只留最近 8 轮的工具输出、最近 2 轮的推理 |
| `reasoning_passback` | think_tag | 轮内推理怎么传回（think_tag / field / drop） |
| `history_keep` | 2 | 跨步历史：第 0 步 + 最近 2 步完整，其余只留代码（null = 全部；tool_rounds 0 时总是全部） |
| `explore` | timeout_s 300, mem_gb 8, gpu_tools false | run_python 的限制 |
| `mock_tool_rounds` | 1 | `--mock` 时 mock 改造者先调几次 run_python（`api.profile_summary()`） |

怎么改：
```bash
--set editor.tool_rounds=20              # 最多 20 轮
--set editor.tool_rounds=0               # 关掉工具（单次调用）
--set editor.context_tokens=65536        # 上下文 64K（改造者服务 --max-model-len 也跟着变）
--set editor.extra_body={}               # 不显式开 thinking（用模板默认）
```
`context_tokens` 的默认值写在 `hardware_profiles`（b300 / h20: 131072，h20_80: 49152；硬件档在 `editor:` 块之后合并），所以改默认值要改那里，或用 `--set`（`--set` 优先）。真实服务器上 `rsicore.servers` 会把 `--max-model-len` 压到模型 `config.json` 的 `max_position_embeddings` 以内（打警告，不加 YaRN），工具解析器按聊天模板自动选（qwen3_coder / hermes）。这些键都在锁定的配置里：改了要用新的 `--run-id`。

## KL / 校准文本 = 题目 + M_0 自己的回答（2026-11-20）

每段 KL / 校准文本是"题目（聊天格式）+ M_0 对这道题的贪心回答"（`kl.self_generated: true`，回答最多 `kl.self_generated_max_tokens` = 2048 token）。这样 KL 和 GPTQ / AWQ 的校准统计都覆盖答题部分，不只是读题。
- 回答每代生成一次，缓存在 `<run>/kl_selfgen/gen_XX.jsonl`；续跑时直接读缓存，不再起服务。回答被长度上限截断时不加 `<|im_end|>`。
- 第 0 代的校准用前 `calibration.max_texts` = 256 段（每段截到 `calibration.max_len` = 2560 token）。
- 每个候选的 KL 在 `kl.eval_texts` = 400 段上算（和快速筛题数一致）。
- 回到只用题目：`--set kl.self_generated=false`。

## 启动

```bash
./run_rsi.sh --paper A --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id A_math_s42
./run_rsi.sh --paper A --domain all  --gens 5 --run-id A_all_s42
./run_rsi.sh --paper A --domain math --gens 5 --gpus 0,1 --run-id A_math_s42 --set hardware=h20   # H20（按每张卡 110 GB 规划）
./run_rsi.sh --paper A --domain math --gens 2 --mock            # 本机，无 GPU，约 1–2 分钟
# 等价：python -I edit_code/run.py ...；附加参数 --budget-gib 11.5（固定体积），--set key=value（覆盖 configs，写入锁定的 run_config.yaml）
```

同一个 run-id 再运行会跳过已完成的步骤（`state.json` 的键：`baseline/{draw,gate,noise,heldout,prep,budget,q0,q0_heldout}`、`genXX/{draw,refs,search,baselines,gate,accepted,mark_shown,export,heldout,done}`、`e_t/XX`、`stopped`）。

## 流程（run.py）

- **第 0 代**（`baseline/`）：切固定验收集 → M_0 在验收集上的分数 → screen 子集重复 3 次得噪声带 → M_0 五个领域留出评测 → 对 M_0 做一次校准（Hessian + 激活统计）、诊断表、逐层量化缓存（`prep/`，之后每代复用）→ 定体积（`budget.json`）→ Q_0 = 参照配置，在验收集上打分（第一个 incumbent，`incumbent.json`）并做留出评测。
- **第 g 代**（`gen_XX/`）：draw → M_0（卡 0）只算 KL 参照 logprob（2026-11-20 起不再在本代题上给 M_0 打分）→ incumbent 在同样的题上打分（不扣预算，`incumbent_scores.json`），它是所有候选的参照：保持率 = 候选准确率 / incumbent 准确率，逐题配对差和例题（incumbent 对、候选错）都相对 incumbent → `rsicore.search.inner_search`（改造者 = M_0；4 线程 × 6 步，每代 ≤24 次 evaluate；前 3 名跑完整 signal；候选在卡 1 上服务）→ 可选基线 → 只把在本代题目上分数高于 incumbent 的候选交验收（≤3 次）→ 通过：导出新 incumbent（fake-quant 检查点）+ 五个领域留出评测；不通过：incumbent 不变 → `summary.json`。
- **主要指标**（`summary.json` 的 `final`）：验收集平均分 M_0 / Q_0 / 最终 incumbent、相对 Q_0 的提升 `gain_vs_q0`、追回的差距 `gap_closed` =（最终 − Q_0）/（M_0 − Q_0）、哪些代换了 incumbent；每代的留出分数和相对 M_0 的比例。
- **E_t**（默认关）：新会话、不带档案，用第 g 代的改造者在同一体积下从 Q_0 开始做一遍；只在 `loop.editor: current` 时有意义（冻结的改造者不会变）。

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 编排（上面的流程）、断点续跑、汇总 |
| `problem.py` | 一个搜索问题（源模型 + 预算 + 数据 + 参照）：prepare（缓存/诊断）、配置校验与大小、候选构建、fake-quant 落盘、候选服务、screen/full/gate 评测与 L1 反馈 |
| `tools.py` | 程序可调用的工具（注册到 `rsicore.sandbox.ToolAPI`）：`model_info`、`profile_table`、`profile_summary`、`hessian_table`、`activation_table`、`bytes_table`、`bytes(config)`、`group_error(config)`、`compare(cid, other)`、`evaluate(config)`（扣 1 次）、`solve_allocation(...)`（`tools.*` 可关掉做消融） |
| `allocation.py` | 多选背包（每组 (字节, 损伤) 下凸包 + 贪心），给 `solve_allocation`、KL 贪心基线和 mock 改造者用 |
| `prompts.py` | 系统提示（配置 schema、工具、规则）和 `render_context(history, thread)`：incumbent 及其本代分数、候选表（大小、KL、各领域分数及相对 incumbent 的配对差和显著性）、最好配置、本线程上次反馈、2–3 个退化样例、剩余预算、往代档案 |
| `diagnostics.py` | `profile_table`（只把一个模块组 RTN 到 2/3/4 bit 时整模型 ΔKL，参照 = M_0，前 64 段第 0 代 KL 文本做前向）、`activation_table`（校准时顺便统计的输入激活）、`bytes_table`、`hessian_table`（每组单独用 rtn/gptq/awq 量化到 2/3/4 bit 的 Hessian 加权误差，来自缓存和校准 Hessian，用到全部校准 token，不做前向）；只算一次，存在 `prep/diagnostics/` |
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

角色：editor（vLLM：M_0 / 改造者、参照）、candidate（候选服务，每个候选槽一张卡）。

**用满所有卡（2026-11-20，`gpu_layout.*`，改造者有自己的卡时才生效）**：
- `prep_all_gpus`：第 0 代准备分到所有卡上，每张卡一个 HF 模型：Hessian 按文本分给各卡后相加（结果和单卡一样）、敏感度表按组分、缓存预热和 Hessian 误差表按模块分。
  - 敏感度表提速（2026-11-21）：每张卡一个**独立进程**（以前是同一进程里的线程，共用 GIL，GPU 利用率约 10%）；上下文右填充后成批前向（每批 ≤ `diagnostics.profile_batch_tokens` 个 token，默认 8192；各层都是因果的，填充不改变真实 token 的 logits，结果和逐段前向只差 bf16 舍入）；每算完一组写 `prep/profile_rows/<组>.json`，中断后同一条命令续跑只算缺的组；校准 Hessian 都在 `prep/hessians/` 时跳过校准。GPU 环境没有 `flash-linear-attention` / `causal-conv1d` 时，transformers 对线性注意力层用纯 torch 回退实现（慢几倍），日志里会有 WARNING：`pip install flash-linear-attention causal-conv1d`。
- `m0_replicas`：M_0 的批量任务（第 0 代验收打分、留出评测、每代 KL 文本的回答）在所有卡上各起一份 M_0，按题目或任务分开跑（`rsicore.evaluate.score_split`、`run_heldout` 传多个地址）。
- `editor_sleep`：搜索时改造者写完这一轮的程序就进入 vLLM 睡眠模式（`--enable-sleep-mode`，权重移到 CPU 内存约 52 GB，释放 KV cache），它的卡临时当一个候选槽（4 张卡：同时评测 4 个候选）；下一轮开始前收回并唤醒。搜索结束后改造者关闭，验收和导出用所有槽。
- `parallel_heldout`：新 incumbent / Q_0 的留出评测在每组空闲卡上各起一份，任务分开跑。
- 验收：最多 3 个提交同时评测，再按排名顺序取第一个通过的（结果和逐个评测一样）。

**改造者 1 张卡，其余卡全部用来同时评测候选**（`common.gpu_layout` / `candidate_groups`，不设 `gpu_roles` 时）：
- 改造者在候选评测时是空闲的，所以默认只给 1 张卡；
- 其余卡按 `gpu_layout.candidate_group_size`（默认 1）分组，每组起一个候选 vLLM 服务（组内张量并行）。2 张卡 → 1 个候选服务，4 张 → 3 个，8 张 → 7 个；
- 搜索的每一步里，各线程的程序同时提出、构建、评测（`rsicore.search.inner_search(parallel=组数)`），完整评测的前 3 名也同时跑。每个线程看到的是上一步结束时的全部候选；结果按线程顺序记录；
- `QuantProblem.serve` 从服务池里取一个空闲组：在这组的卡上导出该候选的 fake-quant 检查点（导出共用一个逐层缓存，一次一个），再在这组卡上起 vLLM，用完归还。`evaluate` 线程安全：同一个配置只评一次（其他线程等结果），计费的快速筛先预留预算，不会超过每代上限；
- 磁盘：每个候选服务各留一个检查点（约 54 GB），8 张卡时约 7 × 54 GB；
- 只有 1 组时（2 张卡）就是原来的逐个评测。手动指定：`--set 'gpu_roles.editor=[0,1]'`（改造者张量并行 2），候选用剩下的卡。

H20 档（按每张卡 110 GB 规划，卡约 140 GB）：至少 2 张（每张卡只放一个 27B，约 52 GB）；vLLM 显存用 `--rsi-gpu-memory-gib 110`（启动时换算成显存比例），`--max-num-seqs 128`，校准每次 16 层 Hessian。80 GB 的卡用 `h20_80`（0.92、32 条序列、校准每次 4 层）。B300 / mock 只给 1 张时两个角色共用。

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
- `edit_code/tests/test_mock_flow.py`：固定体积循环（2 代：第 1 代 mock 改造者的比特交换超过统一 4 bit 被接受；续跑全部跳过）；`loop.editor=current` + patience 1 时第一次没有新 incumbent 就停；体积小于参照配置时拒绝运行。
- **只能在 GPU 上跑、本机没有验证**：vLLM 加载 27B fake-quant 检查点、vLLM `prompt_logprobs` 的 KL、27B 的校准/诊断/缓存耗时与显存（H20 上尤其要实测）、候选服务（`problem.serve` 真实分支）、`kl.self_generated`。

## 尚未做 / 后续

- 真实打包格式（compressed-tensors）导出和速度测量：llm-compressor 装不进 vLLM 0.24.0 + Transformers 5.9.0，没有依赖它。真实打包时，vLLM 的融合模块（qkv、gate_up、in_proj_qkvz、in_proj_ba）内部要求设置相同，目前 fake-quant 下不受此限制。
- `behavior_sensitivity` 诊断表（METHOD_DESIGN §2.2）、`profile_in_context` 工具、GAMMA / HAWQ 基线。
- 对照：`archive.enabled=false`（不带经验）、`loop.editor=current`（不冻结改造者）都已支持。
- solver 里 group size 对损伤的缩放（g64 = 0.8、g32 = 0.65）是启发式，要在 A-P0 中实测。
