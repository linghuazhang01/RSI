# lora_code/：Paper B 自写 LoRA（递归）

计划见 `plan/PLAN.md` §4、`plan/research/RSI_LOOP_DESIGN.md` §3、`plan/research/METHOD_DESIGN.md` §3。模型 Qwen3.8-27B（`Qwen3_5ForConditionalGeneration`）。共用运行时在 `eval/rsicore/`（契约 `eval/rsicore/INTERFACES.md`），本目录只写 Paper B 特有的部分。

## 三种改模型的方式（`method.mode`，2026-11-21）

**默认不训练**：改造者在任何模式下都没有 `sgd` / `grad` / `from_grad`。只有保留下来的 `edit` 模式会在第 0 代训练一次 raw LoRA。

| 模式 | 改什么 | 改造者能用的工具 | 候选怎么评测 | 验收参照 | 都不过时 |
|---|---|---|---|---|---|
| `scratch`（**默认**） | 从零写一个 LoRA（低秩 ΔW） | SVD、激活统计、秩 1 / 方向 / 奇异方向构造、编辑算子、`compare`、`train_loss` | 热加载到 main 副本上 | 在固定训练集（500 题）上和当前模型逐题配对：平均提升 ≥ `gate.min_gain`（默认 0 = 严格为正）；只有 `gate.require_significant: true` 时还要 95% 置信区间下界 > 0 | 保持原样 |
| `weights` | **直接编辑权重**（任意秩的结构化修改） | 同上（用来构造低秩部分）+ `new_edit`、`w_scale`（整个模块缩放）、`w_scale_rows` / `w_scale_cols`（输出 / 输入通道缩放）、`w_add_lora`（把构造出的低秩 ΔW 稠密地加进权重）、`w_toward_original`（往原模型的权重拉回）、`inspect_edit`；每个模块 ‖W'−W‖/‖W‖ ≤ `weights.norm_cap_tau`（0.05） | 写出改动后的权重（只重写被改的分片，其余硬链接），在一个空闲候选槽上起一个 vLLM 服务评测，评完删除；同一轮的候选同时在各槽上评测 | 同上 | 保持原样 |
| `edit`（保留，非默认） | 第 0 代训练一次 raw LoRA，之后每代编辑它（M_g = M_0 + 当前 LoRA，不合并） | 同 scratch，另有 `raw_lora` | 热加载 | 同上（每个候选都和当前模型比） | 第 0 代：raw LoRA 过同样的验收就用它；之后保留当前 LoRA |

**weights 模式的卡**（4 张卡）：卡 0 是改造者（当前模型，vLLM 睡眠模式）；卡 1、2 是候选槽；卡 3 是 train（导出改动后的权重、SVD、激活统计、`train_loss`）。改造者写完这一轮的程序就睡眠，卡 0 临时当第 3 个候选槽，所以一轮里 3 个候选同时导出、加载、评测。每个候选要写出改动的分片并启动一个 vLLM 服务，比热加载 LoRA 多几分钟。接受后改动直接写进权重，得到 M_{t+1}。

```bash
--set method.mode=scratch      # 默认：从零写 LoRA
--set method.mode=weights      # 直接编辑权重
--set method.mode=edit         # 训练一次 raw LoRA 再编辑（保留的旧方式）
```

### scratch 模式接受后怎么变成下一代：`merge.mode`（2026-11-22）

| `merge.mode` | 接受后 | 下一代怎么服务 | 精度 |
|---|---|---|---|
| `merge`（**默认**） | W' = W + B A，fp32 累加后转回 bf16，写成 `gen_XX/model` | 新 checkpoint，main 上的 vLLM 重启成 `m{g+1}` | bf16 舍入会吞掉小的 ΔW（‖ΔW‖/‖W‖ ≲ 3e-3 时明显），gate 测的（基座 + adapter）和合并后的后继不完全一样；`model.json` 的 `max_realised_rel_err` 记录误差 |
| `adapter` | **不合并**：C_{g+1} = C_g ⊕ ΔW（秩拼接，fp32 存在 `gen_XX/cumulative_lora`） | 整个 run 只有一个 M_0 服务（`m0`），M_g = M_0 + C_g 作为一个 adapter，名字 = 被接受候选的 adapter 名 | 候选按 C_g ⊕ ΔW 注册并过 gate，接受的就是这个 adapter 本身：gate 测的 = 下一代 |

adapter 模式的细节：
- 改造者写的仍然是相对当前模型的增量 ΔW（`rank_cap`、`norm_cap_tau` 只约束 ΔW，τ 的分母是当前权重 ‖W_0 + B_C A_C‖）。组合只在 `Evaluator.register` 一处做；C_g、C_{g+1} 这类完整 adapter 用 `full=True` 注册，不会重复组合。
- 工具看到的是当前模型：`Workspace.store` 是 `OverlayStore`（读取时 fp32 计算 W_0 + B_C A_C，SVD 缓存按模块因子的 digest 分开），`acts` / `train_loss` / 诊断的 HF 前向都在 `hf.applied(C_g)` 下跑；`tensor_compute` 拿到的也是叠加后的张量。
- 秩预算：`merge.adapter_rank_budget`（默认 256）。拼接后某模块秩超过预算时，**在注册之前**用精确 SVD 重分解并截断，所以 gate 测的仍然就是保留下来的 C_{g+1}（`model.json` 的 `recompressed` 记录丢掉的能量）。main 服务的 `--max-lora-rank` = padded(max(预算, `lora.rank_cap`))，默认 256（vLLM 最大 512）。vLLM 按 `max_loras` × 这个秩预留显存，H20 上要核对显存或调小 `server.max_loras`；走 adapter 解码也比合并后的基座慢。
- `edit` 模式本来就是 adapter（忽略 `merge.mode`）；`weights` 模式不支持 adapter（启动时报错）。`merge.mode` 不写进 base.yaml 的实际配置（paper_cfg 被锁定，旧 run 照常续跑），默认值在代码里。

```bash
--set merge.mode=adapter                                   # scratch：不合并，累积 adapter
--set merge.mode=adapter --set merge.adapter_rank_budget=128   # 更小的秩预算（server --max-lora-rank 128）
```

raw LoRA 的配方和大小在 `raw` 里设置（默认 rank 16、lr 1e-4、1 个 epoch、batch 4，除 `in_proj_a/b` 外的全部投影层；`raw.n_train` 空 = 固定训练集的全部题）：

```bash
--set method.mode=edit --set raw.n_train=400 --set raw.steps=50 --set raw.rank=16   # 用固定训练集里的 400 题、50 步训练 raw
--set method.mode=scratch                                                            # 从零写
```

训练答案（`train_data`：拒绝采样 M_g 自己答对的回答；4 个都错的题**跳过**，不用金标，`train_data.mode: self_correct`，2026-11-20）只在训练 raw LoRA 时（edit 第 0 代）采样。训练样本的题目部分就是评测时发给模型的那条请求：评测任务自己生成的聊天消息 + Qwen 对话模板、关闭思考，和 vLLM 评测时完全一致，只在回答部分算损失；其余情况固定训练集的题直接用金标答案、不采样，只给激活统计和 `train_example` 用。

`gate.fallback` 可以覆盖上表的默认值（`raw` | `keep`）；验收参照没有开关（2026-11-23 删掉了 `gate.reference` / `gate.epsilon`：每个候选都和当前模型配对比较）。非 LLM 分支（随机编辑 / 随机构造）2026-11-25 已删除。

## 唯一的流程：固定训练集（2026-11-25；simple 流程 2026-11-22 起是唯一的流程，已并进 `configs/base.yaml`）

2026-11-22 删除了 `objective`（train_gate / eval_overfit）、`scope`（all）、`data_roles`、保留探针 / token-KL / `regmean_preserve`、E_t（θ0 服务器、`e_ref`、`e_t/<branch>`、ΔE）以及按代抽题的 draw.py；`objective_profiles.simple` 和 `scope_profiles.single` 的值直接写在 base.yaml 顶层，`mock_objective_overrides.simple` 并进了 `mock_overrides`。`--set objective=simple` / `scope=single`（`./launch.sh --simple`）仍然接受，但什么都不做；其他取值会被拒绝。运行名仍然默认带 `_single_simple`，以前的 `*_single_simple` 运行用同样的命令照常续跑（`rsicore.config.lock_view` 忽略删掉的键）。**只用这次运行的 `--domain`**：训练、评测、反馈都只在这个领域上，改造者的提示里写的是单领域运行。

2026-11-25 删除了：单独的验收集（`simple_<领域>_eval` 不再用来打分）、留出评测（`heldout`，`simple_<领域>_test`）、非 LLM 分支（`nonllm`）、`reuse.baseline_from`、已学目标 / δ 规则（`gate.delta`）、每代重新抽快速筛，以及 `eval.screen_limit` / `full_limit` / `gate_limit`（换成一个 `eval.limit`）。2026-11-25 之前锁定的运行不能用新代码续跑（锁比较会拒绝：`eval.*_limit`、`nonllm`、`gate.delta`、`gate.require_significant` 的默认值都变了；旧代目录里的 `parent_gate.json` 也是按 `simple_<领域>_eval` 记的），换新的 `--run-id` / `--tag` 重新开始。

用 `simple_data/<领域>/`（`data/scripts/make_simple_data.py`，从去污染、去重后的训练集池子里抽）。训练池 `simple_<领域>_pool`（`pool_ids.json` 里的全部题）；`simple_<领域>_eval`（500）和 `simple_<领域>_test`（500）只为了从池子里排除它们的题才构建，不再打分；运行名带 `_single_simple`。
- **一个固定训练集**：第 0 代抽一次（`rsicore.draws.simple_draw`）：候选 = 池子 − eval 500 − test 500，从中随机抽 `simple.screen_n` = 500 题（screen 角色，种子 `<seed>:simple:<领域>:<轮次>`，不含运行名和论文），写在 `draws/gen_00/ids/simple_<领域>_pool.screen.json`。**每一代、每件事都用这同一批 500 题**（`Run.fixed_ids`）：改造者的 train 题号（`train_example` / `acts` / `train_loss`、raw LoRA 的训练数据）、当前模型的作答（改造者读的 `screen_results` / `screen_output`）、每个候选的评测。之后各代不再抽题（`genXX/draw` 只记录这 500 题）。`eval.limit`（空 = 全部；mock 是 30）只截这一批题的前几道。`gen_XX/train_data.json` 就是这 500 题（gold 模式，不采样）。
- **screen 和 gate 是同一批题**（`Run.gds == Run.tds`）：screen / full / gate 三个级别装的是同样的数据集和题号，`lora_code/evaluator._cache_level` 让它们共用一个打分目录（`screen@<tag>`），所以每个模型在这 500 题上只采样一次。`decoding.screen`（以及显式设置的 `decoding.full`）必须和 `decoding.gate` 相同，否则启动时直接退出（`run.decoding_mismatch`）。
- **打分用 thinking 采样**（`decoding` 块，`rsicore.config.stage_decoding`）：temperature 0.6、top_p 0.95、top_k 20、thinking、max_tokens 16384，n 4（avg@4：每题分数是 4 个样本的平均，在 [0,1] 里；配对按每题平均配对），种子 = 运行的 `--seed`，当前模型和所有候选用同一个解码字典（同一种子）。缓存目录带解码标签：`evals/<name>/<level>@<hash>/`（贪心时仍是 `<level>/`）。当前模型（`__parent__`）在一代里只采样一次（每个 (模型, 级别) 一把锁，同时评测多个候选时也只算一次），所有候选共用；M_0 的基线和 weights 模式候选槽打分也用同一解码。压缩记录在 n > 1 时记 `ns`（样本数）、`len`（平均长度）、`trunc`（截断比例）；反馈里的错题 = 平均分 ≤ 0.5（当前模型答对的排在前面）。打分服务器都是 main 副本 / weights 候选槽，`--max-model-len` = `editor.context_tokens`（131072；h20_80 是 49152），提示 + 16384 放得下。启动时 `run.context_shortfall` 检查（max_tokens + 4096 > 某个打分服务器的 `--max-model-len` 就直接退出）。
- **每代的流程**：(1) `Run.get_ev()` 建好评测器后立刻调用 `gate_reference`：**当前模型**在 500 题上的记录先算好（`decoding.gate`，n 4），能复用就不重算：第 0 代用 `baseline/gate` 已经跑过的 M_0；上一代保留模型时用上一代当前模型的记录；上一代接受了候选、且下一代模型和评测时完全一样（edit 模式、`merge.mode: adapter`）时用那个候选自己的记录；merge / weights 模式合并出的新权重和评测时的 adapter 数值上不完全相同，重新评一次（复用的记录里有 `adopted_from`）。raw LoRA 的点和搜索都读这份记录，不重新采样。(2) 改造者在每步上下文里看到这次作答的摘要（`Run.reference_screen`：avg@4、每题答对样本数的分布、截断率、几道错题的题面 / 标准答案 / 抽取答案），完整数据用免费工具读：`api.screen_results(ids, filter='all'|'wrong'|'unstable'|'right', limit, offset)` 和 `api.screen_output(id, sample, offset)`（整段输出分页，**含 thinking**；记录里的 `reasoning`）；摘要写进 `gen_XX/reference_screen.json`，后几代的 `search_history` 和存档里也看得到。(3) 每个候选（`search.threads × search.steps` 个，默认 8 × 1）建好后**只做一次评测**：同样的 500 题 × 4，和当前模型同种子逐题配对（`Run.gate_check`），结果立即写进 `gen_XX/gate.json`（submissions 形状，每个候选一条），候选的 feedback 里是 `gate`（accuracy、reference_accuracy、配对 gain、CI、passed / reasons、broken / fixed 个数和题号，没有题目内容），候选分数 = 配对 gain；后面的步骤和后几代都看得到（上下文表格、`search_history`、`read_record`、存档）。(4) 代末（`accept_best_gate`）在配对 gain > 0 的候选里**取最大的**（`gate.min_gain` 默认 0 = 严格为正；`gate.require_significant` 默认 false，设 true 时还要 CI 下界 > 0；并列取 cid 早的），没有就 keep（edit 第 0 代先试 raw LoRA：过同样的规则就用它）；`gate.json` 里标 `accepted`。**没有 `api.evaluate`**（程序、探索 api、工具说明和提示里都没有），没有留出评测。代价：每代 8 × 500 × 4 个 thinking 样本。线程数 × 步数是参数，提示按配置的数字写（1 步时"你是第 k 个会话（共 N 个）"，多步时"改进本线程最好的程序"）。`search.parallel_builds: true`：有多个 main 副本时，最多副本数个候选同时构建 + 评测。
- **论文式工具**（`tools.paper_tools`，base.yaml 设为 true，不设也是开；程序和探索 api 都有）：`save_edit(handle, name)` / `load_edit(name)` / `list_edits()` / `drop_edit(name)`（按候选 cid 隔离，存在 `gen_XX/saved/<cid>/`，探索结束、构建和续跑后都还在；最终程序可以直接 `return {"lora": api.load_edit(name="best")["lora"]}`；载入时把构建它用到的工具记进程序的工具日志，`classify()` 照常分类）；`output_matrices(layer)`（o_proj / out_proj / down_proj）；`unit(ref)`、`direction(a, b)` = (a−b)/‖a−b‖（S / A / T 向量引用，返回 T 句柄）；`tensor_compute(..., vectors=[refs])` 把向量作为 `V` 传进代码；`acts(module, ids, k, source='own', sample=0|'all')` 在当前模型**自己的**作答上（题面 + 整段输出，含 thinking）算激活统计（只收固定训练集的题号，花 gpu_seconds；HF 模型按 max_len 截断长输出），`source='prompt'` 只用题面；`project_out(target, module, u, alpha)`：W ← W − α u uᵀ W（u 是输出侧向量，归一化；W = 当前权重加上 target 在该模块上的改动），LoRA 目标加秩 1 项 B = −αu、A = uᵀW，weights 模式的 edit 目标加一个 lowrank op，照常检查上限；`classify()`：LoRA 上的 project_out 算 construct，edit 上的算 weights op。提示里给了论文式做法的提示（两组自答 → direction → project_out）。
- **探索会话常驻**（`editor.explore.persistent: true`）：rsicore.chat_tools 每个候选保留一个 Python 会话，变量 / import / 句柄跨 `run_python` 调用保留；Paper B 的工厂返回 `(api, cleanup, before_call)`，每个候选只建一个探索 api，`before_call(timeout_s)` 设 tensor_compute 的截止时间，每次工具调用先绑定这个 tag 的线程状态（不管哪个线程在服务）；候选结束时 `cleanup` 释放句柄。`editor.explore.gpu_tools: true`（探索里也能用 `acts` / `train_loss`，每个会话单独的 600 GPU 秒预算）。
- `method.mode` 默认 scratch；改造者每个候选最多 `editor.tool_rounds` 轮工具调用（20，可配置，提示里写的是配置的数字）；`api.tensors` / `api.tensor_compute` 默认开（见下）。
- **这个流程是写死的**（2026-11-23）：`search.screen_candidates` / `gate_every_candidate` / `charge_screens` / `program_evaluate_calls` / `program_evaluate_total` / `max_evaluate` / `top_full` / `rerun_seeds`、`gate.max_submissions` / `reference` / `epsilon` / `use_noise_band`、`baseline.noise_*`、`submit_to_gate`、`api.evaluate` 及其预算都删了。`--set` / `--config` 这些键会报错（给的正好是现在写死的值时只提示一句、不生效；`rsicore.config.REMOVED_KNOBS`）。以前的 `*_single_simple` 运行用同样的命令照常续跑（锁比较时忽略这些键）；旧锁里这些键的值和写死的流程不同时拒绝续跑。`tools.search_settings` 只剩 threads、steps、`max_candidates`（空 = threads × steps）、`parallel_builds`。
- mock：`mock_overrides`（在 `--set` 之后生效）是 3 × 1、`eval.limit` 30、每个候选一次评测、没有 `api.evaluate`；mock 改造者的 1 号线程走论文式流程（`screen_results` / `screen_output`、`acts(source='own')` 两组、`direction` / `unit`、`tensor_compute(vectors=...)`、`project_out`、`save_edit` / `load_edit`）。测试：`python3 -I -m pytest -q lora_code/tests/test_paper_tools.py`，`lora_code/tests/test_mock_flow.py -k mock_simple_flow`。
- 服务器上需要 `simple_data/<领域>/train.jsonl`、`eval.jsonl`、`test.jsonl`、`pool_ids.json` 和对应的 `data/processed/` 源数据（`rsicore.draws.build_eval_set` 首次运行时构建 `simple_<领域>_pool / _eval / _test`，后两个只用来排除题目）。`plan/configs/B_<领域>_simple.yaml` 是 2026-11-25 之前的合并配置快照（标了 STALE），以 `configs/base.yaml` 为准。

## 每一步的数据量

每一步用多少题都可以在 config 里设（`--set` 或 `configs/base.yaml`）。

| 步骤 | 键 | 默认 |
|---|---|---|
| 固定训练集的题数（第 0 代抽一次，每代都用） | `simple.screen_n` | 500 |
| 实际用到的固定训练集题数（train、当前模型作答、每个候选的评测） | `eval.limit` | 空 = 全部（mock 30） |
| edit 的 raw LoRA 训练题数 | `raw.n_train` | 空 = 固定训练集的全部题 |
| 每个任务的最大生成长度（没有自己 max_tokens 的调用） | `max_tokens.<任务>`（覆盖 `eval/configs/tasks.yaml`） | dapo_math 4096、taco_verified 4096、ifeval_train 2048、sealtool_train 1024、triviaqa_train 1024 |
| 训练答案的采样参数 | `train_data.temperature` / `top_p` / `top_k` / `samples` | 0.7 / 0.8 / 20 / 4（指令遵循 8） |
| 当前模型 / 候选评测的解码 | `decoding.screen` = `decoding.gate`（两者必须相同） | thinking，0.6 / 0.95 / 20，max_tokens 16384，n 4，种子 = `--seed` |
| 接受规则 | `gate.min_gain` / `gate.require_significant` | 0（严格为正）/ false |
| 训练序列长度 | `train.max_len` | 4608 |

`raw.n_train` 用固定种子抽子集。

M_0 在固定训练集上的分数和标准误在 `baseline/gate_scores.json`（`score`、`se`、`n`）里，第 0 代的评测器直接采用这些记录（M_0 只采样一次）。噪声带（`baseline.noise_*`，2026-11-23）已删除。

**解码**：当前模型和候选的评测都用 thinking 采样（见上表）。训练答案（只在 edit 第 0 代）用 0.7 / 0.8 / top_k 20，每题采 4 个，取第一个判对的，4 个都错就跳过这道题。改造者写程序时用自己的采样参数（thinking 模式：`editor.temperature` 0.6、top_p 0.95、top_k 20）。训练序列长度 `train.max_len` = 4608（回答最多 4096 + 题目留 512）。

## 启动

```bash
./run_rsi.sh --paper B --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id B_math_s42
./run_rsi.sh --paper B --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1,2 --run-id B_math_s42 --set hardware=h20   # H20（按每张卡 110 GB 规划）
./run_rsi.sh --paper B --domain math --gens 2 --mock          # 本机 mock，约 5 分钟
# 改配置：--set search.threads=4 --set gate.min_gain=0.01（写进 run_config.yaml 并锁定）
```

同一个 run-id 再运行会跳过已完成的步骤（`state.json`）。**提前停止**（2026-11-20）：连续 `loop.patience` = 3 代都保留当前模型（`accepted.json` 的 source 为 keep）就停（`state.json` 的 `stopped`）；null = 总是跑满 `--gens` 代。产物在 `experiment_records/<run-id>/`。

## 一代（第 g 代：改造者 M_g → 接任者 M_{g+1}；目标领域每代都是 `--domain`）

`state.json` 里的 key 只有下表这些（另有提前停止时的 `stopped`）。

| 步骤（state key） | 内容 | 产物 |
|---|---|---|
| `baseline/gate` | 第 0 代抽固定训练集（`rsicore.draws.draw_gen(0)`），M_0 在这 500 题上评测（`decoding.gate`，n 4）；记录被第 0 代的评测器直接采用，M_0 只采样一次 | `draws/gen_00/`、`baseline/gate/`、`baseline/gate_scores.json` |
| `genXX/draw` | 不再抽题：只记录固定训练集的题数（第 0 代那次抽的 500 题，`Run.fixed_ids`） | — |
| `genXX/train_data` | 固定训练集的训练目标：edit 第 0 代拒绝采样 M_g 自己答对的回答，4 个都错的题没有训练目标、不参与训练（`train_data.mode: self_correct`）；其余情况只用金标、不采样 | `train_data.json` |
| `genXX/diagnostics` | 固定诊断表（每个 (层, 模块)：范数、奇异值、已合并 adapter 的范数 / 有效秩 / intruder / 相对变化分位；行里另有目标题上的激活比，不进提示） | `diagnostics.json` |
| `genXX/raw` | 仅 `edit`：第 0 代按 `raw` 配方训练 raw LoRA（整个运行唯一的训练），在固定训练集上评测（和当前模型同一批题、同一份当前模型记录）；之后各代记录改造者出发的 LoRA = 当前 LoRA，不训练 | `baselines/raw.json`、`baselines/raw_lora.*` |
| `genXX/search` | 先算好 / 复用当前模型在 500 题上的记录（`gate_reference`），然后 `rsicore.search.inner_search`：8 线程 × 1 步；改造者 = main 副本上的 M_g（vLLM 多 LoRA），候选 LoRA 在 train 卡上构建后热加载，每个候选在同样的 500 题上评测一次，和当前模型逐题配对（`gate.json`） | `candidates.jsonl`、`programs/`、`loras/`、`gate.json`、`reference_screen.json`、`parent_gate.json` |
| `genXX/gate` | 代末在通过规则的候选里取配对提升最大的（`accept_best_gate`）。规则：在固定训练集上和当前模型逐题配对，平均提升 ≥ `gate.min_gain`（0 = 严格为正），`gate.require_significant: true` 时还要置信区间下界 > 0（默认不要求）；沙箱检查。都不过 → edit 第 0 代兜底用 raw LoRA（要能过同样的规则）→ 否则保留 M_g | `gate.json`、`accepted.json`、`accepted_lora.*` |
| `genXX/merge` | LoRA 合并进权重 → M_{g+1}（mock：更新 registry 中 `m{g+1}` 的 quality）；`edit` 模式和 `merge.mode adapter` 不合并，`model.json` 记录 M_{g+1} = M_0 + 哪个 adapter（adapter 模式另存 `cumulative_lora`） | `model/`、`model.json`、`cumulative_lora` |
| `genXX/summary` | 记录反馈里展示过的错例（`shown.json`，只是记录）；汇总这一代（接受了谁、各候选的评测结果、raw、固定训练集的题数、当前模型的准确率） | `summary.json` |
| `stopped` | 连续 `loop.patience` 代都保留当前模型时写入，之后不再跑新的代 | — |

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 编排（上表）；`Servers` 管理 vLLM / mock 服务（GPU 分工见下面"硬件"） |
| `configs/base.yaml`、`configs/domains/*.yaml` | 预算、秩上限 64、范数上限 τ、raw 配方、GPU 秒预算、搜索规模、解码、接受规则（`gate.min_gain` / `require_significant`）、固定训练集大小（`simple.screen_n`、`eval.limit`）、mock 缩小版（`mock_overrides`） |
| `primitives/naming.py` | Qwen3.5 模块名、vLLM 打包映射、PEFT 键名 |
| `primitives/weights.py` | `WeightStore`：按需读 safetensors 分片，top-k SVD（缓存） |
| `primitives/lora.py` | `LoRA`（A、B 因子，缩放并入 B）、沙箱检查、PEFT 保存 / 读取（vLLM 可加载）、合并进权重 |
| `primitives/construct.py` | `rank1`、`from_direction`、`scale_singular` |
| `primitives/ops.py` | `scale`、`interpolate`、`combine`、`drop`、`truncate_rank`、`intruder_scale`、`relchange_prune`、`orthogonalize`、`ties` |
| `primitives/train.py` | `TrainData`（只允许固定训练集的题）、`HFModel`（transformers + peft：loss / sgd（只用于 raw LoRA）/ acts / act_ratio） |
| `tools.py` | `Workspace`：把原语注册成 `ToolAPI` 工具（句柄、GPU 预算、train 限制；`evaluate_lora` 只给 raw LoRA 打分），`program_build`（沙箱执行 → 检查 → 保存候选） |
| `evaluator.py` | 候选评测（screen / full / gate 三个级别是同一批题，`_cache_level` 共用一个打分目录；配对差、错例、行为统计、分数 = 配对提升） |
| `diagnostics.py` | 诊断表计算与渲染 |
| `prompts.py` | 改造者 system prompt、`ContextRenderer`（候选表、本线程最佳程序、最近反馈、档案） |
| `baselines.py` | raw LoRA 配方与训练（edit 第 0 代） |
| `mock.py` | mock 模型几何、mock LoRA 与 quality 模型、mock 诊断、mock 改造者策略 |

## 改造者看到的工具

程序形如 `def main(api): ...; return {"lora": handle}`（`{"lora": None}` = 保持原样），工具只能用关键字参数调用：
`modules`、`diagnostics`、`train_ids`、`train_example`、`svd`、`acts`、`new_lora`、`rank1`、`from_direction`、`scale_singular`、`scale`、`interpolate`、`combine`、`drop`、`truncate_rank`、`intruder_scale`、`relchange_prune`、`orthogonalize`、`ties`、`inspect`、`stack`、`raw_lora`、`candidate`、`budget`（说明见 `tools.py` 的 `TOOL_DOCS`）。

按模式过滤（`tools.tools_for_mode`）：`edit` 有 `raw_lora`，`scratch` 没有。任何模式都没有训练 / 梯度工具（`sgd` / `grad` / `from_grad` 已删除）。没注册的工具调用会被 ToolAPI 拒绝。

2026-11-20 新增两个工具（`tools.compare` / `tools.train_loss` 可关掉）：
- `compare(cid, other='parent', level='screen')`（工具说明里写 level='gate'：主搜索里候选只有验收结果）：两个**已评测过**的模型逐题对比（坏掉 / 修好的题数和题号、按输出长度分组、截断率），只给题号和统计，不给题目内容，免费，从不触发评测。
- `train_loss(lora, ids=None, n=64)`：在 train 题上算当前模型 + 这个 LoRA 下答案的平均 NLL，并和不加 LoRA 比（只做前向，不训练，消耗 gpu_seconds）。scratch 里改造者可以先用它便宜地筛掉明显变差的构造，再交候选。

- `acts` / `train_example` / `screen_results` / `screen_output` 只接受固定训练集的题号，其他题号抛 `PermissionError`。
- 预算：gpu_seconds（`acts` / `train_loss`，实测；mock 中估计）。没有 evaluate 预算（2026-11-23：没有 `api.evaluate`，每个返回的候选在程序外做一次验收评测）。`api.budget()` 在程序里还返回
  `program_seconds_left`（离 `sandbox.timeout_s` 还剩几秒）。

### 张量工具 `api.tensors` / `api.tensor_compute`（2026-11-21）

`tools.tensor_compute`（base.yaml 设为 true，不设也是开；`false` 同时去掉 `tensors`）。三种模式都有，探索 api 里也有（`tensor_compute.explore`）。
- `api.tensors(layer=None, pattern=None)` → `[{name, shape, dtype, editable}]`：语言模型的所有张量（不只是线性模块：`input_layernorm` / `post_attention_layernorm`、`self_attn.q_norm` / `k_norm`、`linear_attn.dt_bias` / `A_log` / `conv1d` / `norm` …）。`embed_tokens`、最终 norm、`lm_head` 只读、永远不能改。
- `api.tensor_compute(code, tensors)`：在新的沙盒子进程里跑一段 CPU torch 代码，`T` = 当前模型这些张量的 fp32 拷贝（改 `T` 不影响模型），可以设置 `result`（JSON，截到 `result_chars`）、`vectors`（注册成句柄 `T<n>`，`"T<n>.名字"` 可以用在 `rank1` / `from_direction` 的向量参数里，长度要对）、`lora = {模块: (A [r, in], B [out, r])}`（注册成 LoRA 句柄，线性模块，rank / 范数上限照常检查）；weights 模式还可以 `writes = {张量: 新的整个张量}`（`edit=` 可接在已有编辑后），变成该张量上的稠密增量（任意秩，也包括 1 维的 norm / dt_bias / A_log）。
- `loras=`（2026-11-25，三种模式都有，tensor_compute 开着就有）：读取 LoRA 句柄的精确因子，代码里是 `LORA = {句柄: {模块: (A [r, in], B [out, r])}}`（fp32，ΔW = B @ A，缩放已并进 B）。可以传一个句柄、句柄列表，或 `{句柄: [模块]}` 只取部分模块；句柄可以是 `api.stack()` 的 `H<g>`（历次接受的 adapter，merge 模式下已合进 `T` 读到的权重）、`CUR`（merge.mode adapter 下实际服务的累积 adapter C_g，有接受过的 LoRA 才有）、`RAW`（mode=edit，不在 `T` 里）、自己建的 `L<n>`、`api.candidate(cid)` 返回的句柄；整个工具返回值（带 `lora` 键）或 `api.stack()` 的一行也算一个句柄。工具说明和系统提示里列出的句柄按模式不同（edit / weights 模式下 `stack()` 为空，不再指向它）。每次最多 8 个句柄，因子大小计入 `max_input_mb`；只传 `loras=` 时 `tensors=` 可以为空。mock 的 LoRA 没有真实因子，给的是按记录的秩和 ‖BA‖/‖W‖ 生成的固定种子随机因子。adapter 模式下 `H<g>` 是每代接受时的原始增量；累积秩超过 `merge.adapter_rank_budget` 时实际服务的 C_g 是 SVD 截断后的，和 `H<g>` 之和会有差别，要读实际在模型里的改动用 `CUR`。写输入文件的时间也计入调用的时限。
- 上限（`tensor_compute:` 块）：每次最多 `max_tensors` 4 个张量、`max_input_mb` 8192、`timeout_s` 300（不超过调用它的程序 / run_python 剩余时间）、`mem_gb` 16、`threads` 4、全进程同时 `max_concurrent` 2 个；运行时不占工作区锁。
- weights 模式写入的上限：≥ 2 维 `||W'-W||_F/||W||_F <= weights.norm_cap_tau`；1 维 `rms(W'-W) <= weights.vector_cap_tau (0.5) * max(rms(W), 1)`（Qwen3.5 的 RMSNorm 权重以 0 为中心，层里乘的是 1 + w，纯相对上限没有意义）；几乎是整体缩放（`||D - cW||/||D|| < weights.rescale_tol` 0.05）的写入被拒（用 `w_scale`）。
- 系统提示（工具开着时）：向量引用规则里加上 `T` 句柄；模块名规则说明 `api.modules()` 只列线性模块，其他张量用 `api.tensors()` 看、用 `api.tensor_compute` 读（只有 weights 模式能写）；策略里加上 tensor_compute（`svd_lowrank`、CPU、时限）；weights 模式的上限里加上 1 维张量上限、零中心说明和不可改的张量。

系统提示与每步上下文（2026-11-21，根据 nosig 实验的失败候选改写）：
- 验收规则按配置生成（`gate.min_gain`、`gate.require_significant`、验收题数换算成"约多对几题"；edit 第 0 代另写"都不过时 raw 过同样的验收就用它"），
  不再写死"95% CI"；不提保留领域（只有目标领域）。
- 新增规则：程序里 `api` 只是 `main(api)` 的参数（run_python 里是全局变量），辅助函数要显式传入；模块名只能取自
  `api.modules()`（各层类型不规则）；rank1 / from_direction 的 u、direction 是输出侧向量（长度 out），v、key 是输入侧
  （长度 in），取自同一模块的 svd / acts；工具结果可能为空，索引前检查，可选步骤用 try/except 回退；墙钟时限包含所有工具调用。
- 每步上下文第一行附预算：没有 `api.evaluate`、每个候选一次验收，本代还剩几 GPU 秒、程序时限、已有程序的运行时间中位数和
  最大值、超时次数、一次评测大约几秒。
- 工具说明：`modules` 是唯一合法的模块名来源；`diagnostics` 可能返回空列表；`rank1` / `from_direction` 写明向量所在侧。

## 改造者：thinking + 工具轮次（2026-11-21）

改造者写每个候选时开 Qwen3 thinking（`editor.extra_body: {chat_template_kwargs: {enable_thinking: true}, top_k: 20, min_p: 0.0}`，temperature 0.6、top_p 0.95），并且可以先调用函数再交程序：
- 一轮 = 模型的一次回复里带了函数调用（同一回复里的多个调用算一轮）。最多 `editor.tool_rounds` = 20 轮（base.yaml），之后强制一次不带工具的最终回复（tool_choice none）。回复里**不带函数调用**、含一个 ```python 块，就是这次的程序。空回复、写到长度上限、把调用写成文本时最多提醒 2 次；最后还没有程序就用循环里写过的最后一个代码块（`stop: fallback`）。
- 两个函数：`run_python(code)` 在同一个沙盒里跑探索代码，`api` 是 `Workspace.explore_api`：和程序的工具一样，但没有 `raw_lora`（句柄 `RAW` 照样能用），GPU 工具 `acts` / `train_loss` 只在 `editor.explore.gpu_tools: true`（base.yaml 是 true）时可用，记在单独的探索预算 `explore.gpu_seconds`（每个会话 600）上。探索**不花本代的 gpu_seconds 预算**，句柄每次调用后释放，工具记录在 `<cid>~xRR_i` 下（不影响 `classify`），完整输出在 `gen_XX/explore/<cid>/rRR_i/`。`read_record(cid, part)` 读本步候选表里某个候选的完整程序或反馈；`search_history(gen, sort, offset)`（2026-11-21，默认开，`editor.history_tool=false` 关）列出所有试过的候选（本代可见的 + 往期所有代），每个一行：strategy、modules、max_rank、用过的工具、target_gain、分数、验收结果，`same_as` = LoRA digest 与更早候选相同；`read_record` 也能打开往期候选。
- 上限（都可配置，`editor:` 块）：`tool_rounds` 20、`max_tool_calls` 100、`turn_max_tokens` 8192（每个工具轮）、`max_tokens` 16384（最终轮，原来 8192）、`proposal_tokens` 160000、`proposal_wall_s` 2400、`tool_result_chars` 6000、`history_keep` 2，压缩 `compact_at` / `keep_tool_rounds` / `keep_reasoning_rounds` = 0.6 / 8 / 2，探索沙盒 `explore.timeout_s` 300、`mem_gb` 8。
- **上下文长度只看 `editor.context_tokens`**：客户端用它截每轮 max_tokens、决定何时压缩，同时写成 main 副本（改造者就在上面；weights 模式下候选槽的临时服务器也一样，保证所有打分服务器上下文相同）的 `--max-model-len`，覆盖 `server.main_args` 里写的值（`lora_code.common.main_server_args`）。b300 / h20 = 131072，h20_80 = 49152。设成 null 就按 `main_args` 原样。真实服务器上若模型的 `max_position_embeddings` 更小，`rsicore.servers` 会截到它并警告，不加 YaRN。
- `--set editor.tool_rounds=0`：回到每个候选一次请求，请求里没有 tools，system prompt 和以前逐字相同。
- mock：改造者先调用 `editor.mock_tool_rounds` = 1 次 `run_python("print(api.budget())")`，再给出和不用工具时完全相同的程序。
- 改了 `editor` / `server` 键，旧运行的 paper_cfg 锁会拒绝续跑，要用新的运行名。

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

角色：main（vLLM：M_t + 候选 LoRA，改造者）、train（HF 模型：训练、权重缓存、诊断、`train_loss`、合并）。2026-11-22 起没有 theta0 角色（它只给 E_t 用）。

**给多少张卡就用多少张，main 是数据并行副本**（2026-11-20，`gpu_layout.main_replicas`，不设 `gpu_roles` 时）：train 取最后 `gpu_layout.train_gpus` = 1 张；**其余的卡全给 main，每张卡一份独立的 vLLM 副本**（`gpu_layout.main_tp` = 每份几张卡）。
- 每个候选 LoRA 加载到所有副本上；一次评测的题目分给各副本（`rsicore.evaluate.score_split`）；改造者线程 k 用副本 k mod n，所以一轮里所有线程同时写程序，然后候选逐个评测、每次都用满所有副本；
- `search.parallel_builds: true` 时最多副本数个候选同时构建 + 评测；代末取配对提升最大的。

| 卡数 | 分工 |
|---|---|
| 2 | main 1 份 · train 1 |
| 3 | main 2 份 · train 1 |
| 4 | main 3 份 · train 1 |
| 8 | main 7 份 · train 1 |

旧的布局（main 一个张量并行服务，多的卡依次给 main / train）：`--set gpu_layout.main_replicas=false`。

手动指定：`--set 'gpu_roles.train=[2,3]'`（--gpus 里的下标）把这个角色固定下来，其余角色在剩下的卡上自动分。

H20 档（按每张卡 110 GB 规划，卡约 140 GB）另外：vLLM 显存用 `--rsi-gpu-memory-gib 110`（启动时换算成显存比例），`--max-num-seqs 128`、上下文 main 131072（`editor.context_tokens`）、8 个 LoRA 槽；训练 `train.micro_batch: 2`【试点值】（每次前向 2 条序列，梯度累积，结果等于整批的平均损失）。80 GB 的卡用 `h20_80`（0.92、32 条序列、4 个 LoRA 槽、每次 1 条，main 上下文 49152）。

## 只能在 GPU 上验证的部分（本机未实跑）

真实 vLLM 加载我们的 PEFT adapter（Qwen3.5 的线性注意力投影）、27B 的 peft 短训练（transformers 没有 flash-linear-attention 时走 GDN 的 torch 实现，慢且占显存）、27B 上的 SVD / 诊断耗时、合并后的 bf16 舍入误差、各卡显存分配（`server.main_args`；H20 上 27B 训练的显存峰值要第一周实测）。
