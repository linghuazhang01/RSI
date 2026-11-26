# edit_code/：Paper A 固定体积下的迭代自量化

计划见 `plan/PLAN.md` §3、`plan/research/RSI_LOOP_DESIGN.md` §2、`plan/research/METHOD_DESIGN.md` §2；共用运行时接口见 `eval/rsicore/INTERFACES.md`（本目录只通过它调用 rsicore）。

**设计（2026-11-25 起：只看 KL）**：改造者的权重冻结，不断迭代的是被量化的模型；**唯一的评价指标是量化模型和原模型 M_0 之间的 KL**。没有快速筛（screen）、没有准确率验收（gate）、没有测试集留出评测、没有经典基线。
- 改造者：原始 BF16 模型 M_0（`loop.editor: frozen`）；消融 `loop.editor: current` 让当前最好的量化模型来写。
- 每个候选都从 **M_0 的原始权重**出发量化，所有代用**同一个体积**（默认 = 统一 4 bit GPTQ g128、词嵌入和输出层 8 bit 那份配置 Q_0 的打包大小）。
- **KL 题集（第 0 代定下，之后不变）**：从训练池 `simple_<领域>_pool` 第 0 代的抽题里（已去掉 eval / test 题和校准文本）按种子再抽 `kl_eval.n` = **256** 道。M_0 对每道题用 thinking 采样写一条回答（temperature 0.6 / top_p 0.95 / top_k 20，最长 `kl_eval.max_tokens` 16384 token，种子 = `--seed`；和 Paper B 的打分采样一致），并缓存它在这 256 条"题目 + 回答"上每个位置的 top-20 logprob（`baseline/kl_gate/{traces.json,answers.jsonl,ref.jsonl}`）。
- **一个配置的评测** = 量化模型读这 256 条固定文本（teacher forcing，不采样），在回答部分逐 token 算 KL(M_0 ‖ 量化模型)（top-20 近似，`rsicore/kl.py`），先每条取平均，再对 256 条取平均。越低越接近原模型。逐条结果存在 `gen_XX/evals/<cid>/kl.jsonl`。
- **每代**：8 个独立会话（`search.threads` 8 × `search.steps` 1）各写一个配置 → 每个配置评测一次 KL（写进 `gen_XX/gate.json`，后面的步和代都能看到）→ 预算内 KL **最低**的那个，**只有比 incumbent 的 KL 更低**才成为新 incumbent（不做显著性检验；KL 相等不算改进）；否则 incumbent 不变。连续 `loop.patience`（3）代没有新 incumbent 就停。
- **起点（`loop.start`，默认 `bf16`）**：第 1 代的 incumbent 是 M_0 本身（KL 0，体积超出上限），这一代直接接受预算内 KL 最低的候选（`first_quantization`）。`loop.start: q0` 是消融：从 Q_0 出发。Q_0 的 KL 在第 0 代总会算一次，作为统一 GPTQ 的对照行（`baseline/q0/record.json`）。
- 改造者能看到：incumbent 的配置和 KL（含逐条 KL 的分位数）、本代已有候选的 KL 表、往代**每个**候选一行（配置、大小、KL、note）、诊断表。和 incumbent 完全相同的配置不再评测（记为 KL 降低 0）。
- 已删除（2026-11-25）：`simple`、`decoding`、`max_tokens`、`gate`（含 `gate.metric` / `gate.kl`）、`feedback`、`scoring`、`heldout`、`reuse`、`baselines` 这些配置块，以及快速筛工具 `screen_results` / `screen_output`、`baselines.py` / `evopress.py`。KL 设置都在新的 `kl_eval` 块里。旧运行（准确率验收的）不能用新代码续跑，换新的 `--run-id`。

## 配置要点

| 键 | 默认 | 作用 |
|---|---|---|
| `kl_eval.n` | 256（mock 24） | KL 题集大小（第 0 代抽题里取） |
| `kl_eval.max_tokens` / `temperature` / `top_p` / `sample_top_k` | 16384 / 0.6 / 0.95 / 20（与 Paper B 相同） | M_0 写参考回答的采样 |
| `kl_eval.top_k` | 20 | 每个位置的 logprob 个数（≤ 服务的 `--max-logprobs 20`） |
| `kl_eval.max_num_batched_tokens` | null（`h20_80` 档自动 1024，`h20` 档 2048） | 每个 vLLM 服务的 `--max-num-batched-tokens`：prompt logprob 的显存在 vLLM 的显存估算之外（约 10 B × chunk × 248320，8192 → ~19 GiB，2048 → ~4.7 GiB）；不进运行锁 |
| `search.threads` / `steps` / `max_candidates` | 8 / 1 / 8 | 每代候选数 |
| `loop.start` / `loop.patience` | bf16 / 3 | 起点、停止条件 |
| `gpu_layout.gate_split` | true | 每代提案少于卡数时，一个候选的 KL 评测分到 ⌈槽位数 / 提案数⌉ 张卡上（8 × 1：每个 1 张） |
| 校准文本 | `rsicore.draws.KL_N` = 400 | 第 0 代训练池里另外 400 道 + M_0 的回答（GPTQ / AWQ 校准和敏感度表用） |

mock：`mock_overrides` 把布局缩小到 3 线程 × 1 步、KL 题集 24 道。

## 分析工具与保存的数据（2026-11-20）

改造者可以用几个**免费**的分析工具（都不评测配置）：

| 工具 | 回答什么问题 | 怎么算的 |
|---|---|---|
| `profile_table()` / `profile_summary()` | 哪些组对量化敏感（整模型 KL） | 第 0 代：每次只把一个组 RTN 量化到 2/3/4 bit，在 64 段校准文本上前向，和 BF16 比全词表 KL |
| `hessian_table()` | 同上，但噪声小、覆盖 rtn / gptq / awq | 第 0 代：每组单独量化的 Hessian 加权误差 tr(ΔW H ΔWᵀ)，用缓存里的量化权重和全部校准 token 的 Hessian，不做前向 |
| `activation_table()` | 哪些层的输入激活离群严重（适合 AWQ / 更高比特） | 校准时顺便统计 |
| `group_error(config)` | 一个配置的误差集中在哪些组 | 同 `hessian_table` 的算法，逐组查表 / 按需计算（结果缓存在 `prep/diagnostics/herr_cache.json`） |

每次评测保存在 `gen_XX/evals/<cid>/kl.jsonl`（逐条轨迹的 KL 和计入的位置数）和 `gen_XX/evals.jsonl`（均值、分位数、误差最集中的组）。量化后的模型文件只保留被接受的（`gen_XX/model`，约 54 GB；其余配置已存，可从缓存几秒钟重新拼出）。

## 改造者：thinking + 工具轮次（2026-11-21）

改造者写每个候选时开 Qwen3 thinking（`editor.extra_body.chat_template_kwargs.enable_thinking: true`，temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0），并且可以先调用工具再交程序（`rsicore.editor` + `rsicore.chat_tools`）：
- **一轮** = 模型的一次回复里带了工具调用（并行的几个调用算一轮）。最多 `editor.tool_rounds` 轮（base.yaml 50，2026-11-26 起；以前 20），之后再给一次 `tool_choice: "none"` 的最终回复。候选 = 不带工具调用、含一个 ```python 块的那次回复。
- 工具：`run_python(code)` 在同一个沙箱里跑探索代码，`api` 和最终程序的一样（`make_tool_api(explore=True)`；两边都没有 `api.evaluate`），`group_error` 只能查已缓存的设置（`hessian_table` 那些；`editor.explore.gpu_tools: true` 才允许现算）；`read_record(cid, part)` 看表里某个候选的完整程序 / 反馈；`search_history(gen, sort, offset)`（2026-11-21，默认开，`editor.history_tool=false` 关）列出所有试过的候选（本代可见的 + 往期所有代），每个一行：config 描述（默认设置 + 按设置分组的覆盖及其模块类型）、GiB、平均位数、KL、相对 incumbent 的 KL 降低量，`same_as` = config hash 与更早候选相同；`read_record` 也能打开往期候选。系统提示写明 KL 评测方式和接受规则（KL 最低且低于 incumbent），并写明程序里 `api` 只是 `main(api)` 的参数。探索不占候选槽。每次调用的完整输出在 `gen_XX/explore/<cid>/rNN_i/`。
- 记录：`gen_XX/editor/thread_K.jsonl`（format 2，每轮的推理、工具调用和结果都在）；`candidates.jsonl` 的 cost 里有 `editor_rounds / editor_tool_calls / editor_tool_s / editor_prompt_tokens / editor_completion_tokens / editor_stop`。

| 配置（`editor.*`） | 默认 | 作用 |
|---|---|---|
| `tool_rounds` | 50 | 每个候选最多几轮工具调用，**会话唯一的上限**（2026-11-26：不再限时间、token 和调用次数）；**0 = 以前的单次调用**（请求里没有 tools，提示词逐字不变） |
| `context_tokens` | 131072（`h20_80`: 49152） | **上下文长度的唯一旋钮**：客户端按它限制每轮 max_tokens、触发压缩，同时作为改造者 vLLM 服务的 `--max-model-len`（覆盖 `vllm.editor_args` 里写的值；候选服务由 `vllm.candidate_max_model_len` 设成 32768）。null = 用 `editor_args` 原样 |
| `turn_max_tokens` / `max_tokens` | 16384 / 16384 | 工具轮 / 最终回复的 max_tokens（2026-11-26 工具轮由 8192 提到 16384；第一个 editor 会话开始前改 `editor.*` 仍可续跑） |
| `max_tool_calls` / `proposal_tokens` / `proposal_wall_s` | null / null / null | 设了值时任一到顶就强制进入最终回复；null = 不限（2026-11-26 起，以前 100 / 160000 / 2400：限墙钟会让改造者写出的内容取决于解码速度）。每个请求仍有 `timeout_s` 3600 秒超时 |
| `tensor_parallel` | all | （2026-11-26）改造者 vLLM 服务跨几张卡做张量并行：all = `--gpus` 的全部卡（取不超过卡数的最大 2 的幂），1 = 只用自己的一张卡（以前的布局）。解码受显存带宽限制，TP 让每个会话都变快；评测候选时改造者睡眠，所有卡都给候选。第 0 代 M_0 的批量任务仍是每卡一个副本；`h20_80` 档固定为 1。只决定放在哪些卡上，不进运行锁：任何一次续跑都能改（TP 起不来就 `--set editor.tensor_parallel=1` 接着跑） |
| `tool_result_chars` | 6000 | 工具结果截断长度 |
| `compact_at` / `keep_tool_rounds` / `keep_reasoning_rounds` | 0.6 / 8 / 2 | 提示超过 0.6 × context_tokens 时压缩：只留最近 8 轮的工具输出、最近 2 轮的推理 |
| `reasoning_passback` | think_tag | 轮内推理怎么传回（think_tag / field / drop） |
| `history_keep` | 2 | 跨步历史：第 0 步 + 最近 2 步完整，其余只留代码（null = 全部；tool_rounds 0 时总是全部） |
| `explore` | timeout_s 300, mem_gb 8, gpu_tools false | run_python 的限制；`gpu_tools: true` 时探索里的 `group_error` 可以现算没缓存的设置（GPTQ / AWQ + Hessian 误差，用 GPU），放在 `gpu_layout.explore_tool_gpus` 那几张卡上 |
| `gpu_layout.explore_tool_gpus` | rest | （2026-11-26）写候选阶段给探索 GPU 工具的卡（只在 `gpu_tools: true` 时生效）：rest = 改造者 TP 组以外的所有卡；整数 k = `--gpus` 的最后 k 张，改造者的 TP 组从剩下的卡里取。每张卡同一时间算一个调用，几个会话分散到不同卡上；一次调用在 run_python 的时间上限前 30 秒停下，报告算好并缓存了几个设置，再调一次接着算；评测阶段所有卡照样都给候选。`rest` 而改造者 TP 组占满所有卡时启动就报错（要么调小 `editor.tensor_parallel`，要么设 k，要么设 0 接受和改造者共用第一张候选卡）。只决定放在哪些卡上，不进运行锁。例：8 张卡 + `editor.tensor_parallel: 4` → 改造者 TP 在 0–3，工具在 4–7（`plan/configs/run/A_math_kl_4x1_g10.yaml` 用的是 TP 8、不开 GPU 工具） |
| `mock_tool_rounds` | 1 | `--mock` 时 mock 改造者先调几次 run_python（`api.profile_summary()`） |

怎么改：
```bash
--set editor.tool_rounds=20              # 最多 20 轮
--set editor.tool_rounds=0               # 关掉工具（单次调用）
--set editor.context_tokens=65536        # 上下文 64K（改造者服务 --max-model-len 也跟着变）
--set editor.extra_body={}               # 不显式开 thinking（用模板默认）
```
续跑与 `editor.*`（2026-11-26）：还没有任何改造者会话写出 transcript（`gen_XX/editor/thread_*.jsonl`）时，改了 `editor.*`（`context_tokens` 除外）也能用同一条命令续跑；之后 `editor.*` 和其他配置一样锁定。2026-11-26 之前开的、已经跑过改造者会话的 run，续跑时要把旧的上限用 `--set` 传回去：`--set editor.tool_rounds=20 --set editor.turn_max_tokens=8192 --set editor.max_tool_calls=100 --set editor.proposal_tokens=160000 --set editor.proposal_wall_s=2400`（run config 里另设过的以 run config 为准）。

校准文本 16k（2026-11-26）：M_0 自己的贪心回答（`kl.self_generated_max_tokens`）由 2048 提到 16384，`calibration.max_len` 由 2560 提到 18432（prompt 约 2k token + 回答 16384）。这两项会改变 Hessian，所以在运行锁里：2026-11-26 之前开的 run 续跑时要加 `--set kl.self_generated_max_tokens=2048 --set calibration.max_len=2560`（例如之前 OOM 的 `kl_4x1_g10`），新开的 run 直接用新值。

`context_tokens` 的默认值写在 `hardware_profiles`（b300 / h20: 131072，h20_80: 49152；硬件档在 `editor:` 块之后合并），所以改默认值要改那里，或用 `--set`（`--set` 优先）。真实服务器上 `rsicore.servers` 会把 `--max-model-len` 压到模型 `config.json` 的 `max_position_embeddings` 以内（打警告，不加 YaRN），工具解析器按聊天模板自动选（qwen3_coder / hermes）。这些键都在锁定的配置里：改了要用新的 `--run-id`。

## 校准文本 = 题目 + M_0 自己的回答（2026-11-20）

每段校准文本是"题目（聊天格式）+ M_0 对这道题的贪心回答"（`kl.self_generated: true`，回答最多 `kl.self_generated_max_tokens` = 2048 token；块名仍叫 `kl`，已没有 KL 分数项）。这样 GPTQ / AWQ 的校准统计和敏感度表都覆盖答题部分，不只是读题。
- 只在第 0 代生成一次，缓存在 `<run>/kl_selfgen/gen_00.jsonl`；续跑时直接读缓存，不再起服务。回答被长度上限截断时不加 `<|im_end|>`。
- 第 0 代的校准用前 `calibration.max_texts` = 256 段（每段截到 `calibration.max_len` = 2560 token）。
- 回到只用题目：`--set kl.self_generated=false`。

## 启动

```bash
./run_rsi.sh --paper A --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id A_math_s42
./run_rsi.sh --paper A --domain math --gens 5 --gpus 0,1 --run-id A_math_s42 --set hardware=h20   # H20（按每张卡 100 GiB 规划）
./run_rsi.sh --paper A --domain math --gens 2 --mock            # 本机，无 GPU，约 1–2 分钟
# 等价：python -I edit_code/run.py ...；附加参数 --budget-gib 11.5（固定体积），--set key=value（覆盖 configs，写入锁定的 run_config.yaml）
```

同一个 run-id 再运行会跳过已完成的步骤（`state.json` 的键：`baseline/{draw,kl_gate,prep,budget,q0,start_bf16}`、`genXX/{search,gate,accepted,export,done}`、`stopped`）。

## 流程（run.py）

- **第 0 代**（`baseline/`）：抽题（`draws/gen_00/`）→ KL 题集：M_0 写 256 条参考回答并缓存 logprob（`kl_gate/`）→ 校准文本（题目 + M_0 的回答）→ 对 M_0 做一次校准（Hessian + 激活统计）、诊断表、逐层量化缓存（`prep/`，之后每代复用）→ 定体积（`budget.json`）→ Q_0 = 参照配置，算 KL（`q0/record.json`）→ 第 1 代的 incumbent（`incumbent.json`：M_0 或 Q_0）。
- **第 g 代**（`gen_XX/`）：`rsicore.search.inner_search`（改造者 = M_0；8 线程 × 1 步；每个候选建好就评测 KL，写进 `gate.json`；候选服务分在各候选卡上）→ `Runner.accept_lowest_kl`：KL 最低且低于 incumbent 的被接受（`accepted.json`）→ 导出新 incumbent（fake-quant 检查点 `model/`）；否则 incumbent 不变 → `summary.json`。
- **主要指标**（`summary.json` 的 `final`）：Q_0 和最终 incumbent 的 KL、两者之比 `kl_vs_q0`（< 1 = 比统一 GPTQ 更接近 M_0）、哪些代换了 incumbent；每代的 `candidate_kl`（每个候选的 KL）。

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 编排（上面的流程）、断点续跑、汇总 |
| `problem.py` | 一个搜索问题（源模型 + 预算 + KL 题集 + incumbent）：prepare（缓存/诊断）、配置校验与大小、候选构建、fake-quant 落盘、候选服务、KL 评测（`evaluate` / `kl_stats`） |
| `tools.py` | 程序可调用的工具（注册到 `rsicore.sandbox.ToolAPI`）：`model_info`、`profile_table`、`profile_summary`、`hessian_table`、`activation_table`、`bytes_table`、`bytes(config)`、`group_error(config)`、`solve_allocation(...)`（`tools.*` 可关掉做消融；没有 `evaluate` / `compare`；`tools.compare` 键已删除，给 true / false 都当作空操作，旧锁里有它也照常续跑） |
| `allocation.py` | 多选背包（每组 (字节, 损伤) 下凸包 + 贪心），给 `solve_allocation` 和 mock 改造者用 |
| `prompts.py` | 系统提示（配置 schema、工具、KL 评测和接受规则）和 `render_context(history, thread)`：incumbent 及其 KL、本代候选表（大小、KL、KL 降低量）、KL 最低的配置、本线程上次的 KL 结果、往代档案 |
| `diagnostics.py` | `profile_table`（只把一个模块组 RTN 到 2/3/4 bit 时整模型 ΔKL，参照 = M_0，前 64 段第 0 代校准文本做前向）、`activation_table`（校准时顺便统计的输入激活）、`bytes_table`、`hessian_table`（每组单独用 rtn/gptq/awq 量化到 2/3/4 bit 的 Hessian 加权误差，来自缓存和校准 Hessian，用到全部校准 token，不做前向）；只算一次，存在 `prep/diagnostics/` |
| `mock.py` | mock 模式：合成"真实损伤"模型、合成诊断表、mock registry 登记候选质量、mock 改造者策略 |
| `quant/spec.py` | 从 safetensors 头读模块名和形状（不加载张量）；Qwen3.5 dense 的分组 |
| `quant/schema.py` | 配置 schema、校验（列出所有错误）、解析成逐组设置 |
| `quant/sizes.py` | 解析大小 `bytes(config)` |
| `quant/quantizers.py` | RTN / GPTQ / AWQ-lite（非对称 min-max、按输入维分组），bit 打包 |
| `quant/hessian.py` | 校准前向（transformers）：每层每种输入的 Hessian 与激活统计，按层窗口分批 |
| `quant/cache.py` | 逐层量化缓存（打包整数 + fp16 scale + 打包 zero point；缺的条目按需计算并落盘） |
| `quant/export.py` | fake-quant BF16 检查点导出（结构、配置、tokenizer 文件原样复制，分片布局和张量名不变） |
| `quant/kl.py` | 本地 transformers 精确 token KL（诊断表和测试用） |
| `configs/base.yaml`、`configs/domains/*.yaml` | 全部参数（固定体积 `budget`、`loop`（改造者冻结与否、patience）、动作空间、搜索规模、KL 题集 `kl_eval`、校准文本、校准、缓存、硬件档 `hardware_profiles`、mock 覆盖） |

## 硬件（`hardware`，27B）

角色：editor（vLLM：M_0 / 改造者、参照）、candidate（候选服务，每个候选槽一张卡）。

**用满所有卡（2026-11-20，`gpu_layout.*`，改造者有自己的卡时才生效）**：
- `prep_all_gpus`：第 0 代准备分到所有卡上，每张卡一个 HF 模型：Hessian 按文本分给各卡后相加（结果和单卡一样）、敏感度表按组分、缓存预热和 Hessian 误差表按模块分。
  - 敏感度表提速（2026-11-21）：每张卡一个**独立进程**（以前是同一进程里的线程，共用 GIL，GPU 利用率约 10%）；上下文右填充后成批前向（每批 ≤ `diagnostics.profile_batch_tokens` 个 token，默认 8192；各层都是因果的，填充不改变真实 token 的 logits，结果和逐段前向只差 bf16 舍入）；每算完一组写 `prep/profile_rows/<组>.json`，中断后同一条命令续跑只算缺的组；校准 Hessian 都在 `prep/hessians/` 时跳过校准。GPU 环境没有 `flash-linear-attention` / `causal-conv1d` 时，transformers 对线性注意力层用纯 torch 回退实现（慢几倍），日志里会有 WARNING：`pip install flash-linear-attention causal-conv1d`。
- `m0_replicas`：M_0 的批量任务（第 0 代 KL 参考回答和 logprob、校准文本的回答）在所有卡上各起一份 M_0，按题目分开跑。
- `editor_sleep`：搜索时改造者写完这一轮的程序就进入 vLLM 睡眠模式（`--enable-sleep-mode`，权重移到 CPU 内存约 52 GB，释放 KV cache），它的卡临时当一个候选槽（4 张卡：同时评测 4 个候选）；下一轮开始前收回并唤醒。搜索结束后改造者关闭，KL 评测和导出用所有槽。
- KL 评测：每个候选的 KL 评测在各候选槽上同时跑（搜索中，每个候选建好就评测）；Q_0 的 KL 用所有槽。

**改造者 1 张卡，其余卡全部用来同时评测候选**（`common.gpu_layout` / `candidate_groups`，不设 `gpu_roles` 时）：
- 改造者在候选评测时是空闲的，所以默认只给 1 张卡；
- 其余卡按 `gpu_layout.candidate_group_size`（默认 1）分组，每组起一个候选 vLLM 服务（组内张量并行）。2 张卡 → 1 个候选服务，4 张 → 3 个，8 张 → 7 个；
- 搜索的每一步里，各线程的程序同时提出、构建、评测 KL（`rsicore.search.inner_search(parallel=组数)`）。每个线程看到的是上一步结束时的全部候选；结果按线程顺序记录；
- `QuantProblem.serve` 从服务池里取一个空闲组：在这组的卡上导出该候选的 fake-quant 检查点，再在这组卡上起 vLLM，用完归还。2026-11-23 起各组**同时导出**：每组在自己的卡上开一个逐层缓存（`QuantProblem._slot_cache`），缓存里缺的条目只算一次（`QuantCache` 按文件加锁，其他组等它算完再读），导出的张量和原来一次一个时逐位相同。`evaluate` 线程安全：同一个配置只评一次（其他线程等结果）；
- 磁盘：每个候选服务各留一个检查点（约 54 GB），8 张卡时约 7 × 54 GB。被接受的候选直接把它评测时的检查点移到 `gen_XX/model`（`QuantProblem.adopt_checkpoint`，不再重新导出一遍）；每代结束删掉 `gen_XX/ckpt`（以前没接受的代会留下）；
- 只有 1 组时（2 张卡）就是原来的逐个评测。手动指定：`--set 'gpu_roles.editor=[0,1]'`（改造者张量并行 2），候选用剩下的卡。

H20 档（2026-11-25 起按每张卡 **100 GiB** 规划（我们在一张卡上的所有进程加起来），卡约 140.4 GiB；以前 110 GiB 时第 0 代敏感度表 OOM）：至少 2 张（每张卡只放一个 27B，vLLM 权重约 52 GiB，HF 模型约 54.7 GiB）；改造者 `--rsi-gpu-memory-gib 90`（没有 nvidia-smi 时 0.64；TP 时每张卡都是 90），候选服务和 M_0 副本 `--rsi-gpu-memory-gib 82`（0.58；2026-11-26 由 84 下调，给睡眠的 TP 改造者在每张卡上的残留留空间），`--max-num-seqs 128`，`kl_eval.max_num_batched_tokens 2048`（prompt logprob 在 vLLM 预算外约 4.7 GiB），校准每次 12 层 Hessian。每张卡的估算（GiB）：

| 卡 | 组成 | 合计 |
|---|---|---|
| 第 0 代 M_0 副本 | GPU 0：90 + prompt logprob ~4.7；其他卡：82 + ~4.7 | ~95 / ~87 |
| 写候选阶段（TP 8 改造者） | 每张卡 vLLM 90（权重约 6.5，其余是 KV） | ~90 |
| 评测阶段，一般候选卡 | 睡眠的改造者 ~1–2 + 候选 82 + ~4.7 | ~89 |
| 评测阶段，第一张候选卡 | 上一行 + 本进程的 group_error / 现算 GPTQ 3–8 | ~97 |
| 第 0 代准备（无 vLLM） | 校准 ~78（HF 模型 + 12 × 1.57 fp32 Hessian + 激活）、敏感度表进程 ~60、缓存预热 ~8 | ≤ 78 |

改造者 KV：TP 8 时每张卡约 80 GiB，8 个线程跑满 131072 token 也放得下；`editor.tensor_parallel: 1` 时约 36 GiB，4 个线程 × 8 GiB 放得下。显存 / 批处理旋钮（vLLM 参数里的 `--rsi-gpu-memory-gib`、`--gpu-memory-utilization`、`--max-num-batched-tokens`，`kl_eval.max_num_batched_tokens`，`calibration.layers_per_pass`）不进运行锁：OOM 或中断的运行用同一条命令按新的显存规划续跑。80 GB 的卡用 `h20_80`（0.92、32 条序列、校准每次 4 层）。B300 / mock 只给 1 张时两个角色共用。

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
- `edit_code/tests/test_kl_flow.py`：KL 流程的单元测试（配置、工具、KL 评测与缓存、接受规则、提示词）。
- `edit_code/tests/test_mock_flow.py`：mock 端到端（KL 题集、每代 KL 最低且低于 incumbent 才接受、续跑全部跳过、`loop.start` 两种起点、patience 停止、4 卡、`search_history`、工具轮次）。
- **只能在 GPU 上跑、本机没有验证**：vLLM 加载 27B fake-quant 检查点、27B 的校准/诊断/缓存耗时与显存（H20 上尤其要实测）、候选服务（`problem.serve` 真实分支）、`kl.self_generated`。

## 尚未做 / 后续

- 真实打包格式（compressed-tensors）导出和速度测量：llm-compressor 装不进 vLLM 0.24.0 + Transformers 5.9.0，没有依赖它。真实打包时，vLLM 的融合模块（qkv、gate_up、in_proj_qkvz、in_proj_ba）内部要求设置相同，目前 fake-quant 下不受此限制。
- `behavior_sensitivity` 诊断表（METHOD_DESIGN §2.2）、`profile_in_context` 工具。
- 经典基线和测试集准确率已删除（2026-11-25）；如需报告准确率，要另外评测导出的 `gen_XX/model`。
- 对照：`archive.enabled=false`（不带经验）、`loop.editor=current`（不冻结改造者）都已支持。
- solver 里 group size 对损伤的缩放（g64 = 0.8、g32 = 0.65）是启发式，要在 A-P0 中实测。
