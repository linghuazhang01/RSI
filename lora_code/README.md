# lora_code/：Paper B 自写 LoRA（递归）

计划见 `plan/PLAN.md` §4、`plan/research/RSI_LOOP_DESIGN.md` §3、`plan/research/METHOD_DESIGN.md` §3。模型 Qwen3.8-27B（`Qwen3_5ForConditionalGeneration`）。共用运行时在 `eval/rsicore/`（契约 `eval/rsicore/INTERFACES.md`），本目录只写 Paper B 特有的部分。

## 三种改模型的方式（`method.mode`，2026-11-21）

**默认不训练**：改造者在任何模式下都没有 `sgd` / `grad` / `from_grad`。只有保留下来的 `edit` 模式会在第 0 代训练一次 raw LoRA。

| 模式 | 改什么 | 改造者能用的工具 | 候选怎么评测 | 验收参照 | 都不过时 |
|---|---|---|---|---|---|
| `scratch`（**默认**） | 从零写一个 LoRA（低秩 ΔW） | SVD、激活统计、秩 1 / 方向 / 奇异方向构造、编辑算子、`compare`、`train_loss` | 热加载到 main 副本上 | 和当前模型逐题配对：平均提升 ≥ `gate.min_gain` 且 95% 置信区间下界 > 0 | 保持原样 |
| `weights` | **直接编辑权重**（任意秩的结构化修改） | 同上（用来构造低秩部分）+ `new_edit`、`w_scale`（整个模块缩放）、`w_scale_rows` / `w_scale_cols`（输出 / 输入通道缩放）、`w_add_lora`（把构造出的低秩 ΔW 稠密地加进权重）、`w_toward_original`（往原模型的权重拉回）、`inspect_edit`；每个模块 ‖W'−W‖/‖W‖ ≤ `weights.norm_cap_tau`（0.05） | 写出改动后的权重（只重写被改的分片，其余硬链接），在一个空闲候选槽上起一个 vLLM 服务评测，评完删除；同一轮的候选同时在各槽上评测 | 同上 | 保持原样 |
| `edit`（保留，非默认） | 第 0 代训练一次 raw LoRA，之后每代编辑它（M_g = M_0 + 当前 LoRA，不合并） | 同 scratch，另有 `raw_lora` | 热加载 | 第 0 代：目标 ≥ raw − ε；之后同上 | 第 0 代合并 raw；之后保留当前 LoRA |

**weights 模式的卡**（4 张卡）：卡 0 是改造者（当前模型，vLLM 睡眠模式）；卡 1、2 是候选槽；卡 3 是 train（导出改动后的权重、SVD、激活统计、`train_loss`）。改造者写完这一轮的程序就睡眠，卡 0 临时当第 3 个候选槽，所以一轮里 3 个候选同时导出、加载、评测。每个候选要写出改动的分片并启动一个 vLLM 服务，比热加载 LoRA 多几分钟。接受后改动直接写进权重，得到 M_{t+1}。非 LLM 分支是随机的权重编辑（通道 / 模块缩放）。weights 模式不支持 E_t（单领域本来就关闭）。

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
- 工具看到的是当前模型：`Workspace.store` 是 `OverlayStore`（读取时 fp32 计算 W_0 + B_C A_C，SVD 缓存按模块因子的 digest 分开），`acts` / `regmean_preserve` / `train_loss` / 诊断的 HF 前向都在 `hf.applied(C_g)` 下跑；`tensor_compute` 拿到的也是叠加后的张量。
- 秩预算：`merge.adapter_rank_budget`（默认 256）。拼接后某模块秩超过预算时，**在注册之前**用精确 SVD 重分解并截断，所以 gate 测的仍然就是保留下来的 C_{g+1}（`model.json` 的 `recompressed` 记录丢掉的能量）。main 服务的 `--max-lora-rank` = padded(max(预算, `lora.rank_cap`))，默认 256（vLLM 最大 512）。vLLM 按 `max_loras` × 这个秩预留显存，H20 上要核对显存或调小 `server.max_loras`；走 adapter 解码也比合并后的基座慢。
- `edit` 模式本来就是 adapter（忽略 `merge.mode`）；`weights` 模式不支持 adapter（启动时报错）。`merge.mode` 不写进 base.yaml 的实际配置（paper_cfg 被锁定，旧 run 照常续跑），默认值在代码里。

```bash
--set merge.mode=adapter                                   # scratch：不合并，累积 adapter
--set merge.mode=adapter --set merge.adapter_rank_budget=128   # 更小的秩预算（server --max-lora-rank 128）
```

raw LoRA 的配方和大小在 `raw` 里设置（默认 rank 16、lr 1e-4、1 个 epoch、batch 4，除 `in_proj_a/b` 外的全部投影层；`raw.n_train` 空 = 第 0 代全部 train 题）：

```bash
--set method.mode=edit --set raw.n_train=400 --set raw.steps=50 --set raw.rank=16   # 用 400 条 train 题、50 步训练 raw
--set method.mode=scratch                                                            # 从零写
```

训练答案（`train_data`：拒绝采样 M_g 自己答对的回答；4 个都错的题**跳过**，不用金标，`train_data.mode: self_correct`，2026-11-20）只在训练 raw LoRA 时（edit 第 0 代）采样。训练样本的题目部分就是评测时发给模型的那条请求：评测任务自己生成的聊天消息 + Qwen 对话模板、关闭思考，和 vLLM 评测时完全一致，只在回答部分算损失；其余情况 train 题直接用金标答案、不采样，只给激活统计和 `train_example` 用。

`gate.reference` / `gate.fallback` 可以覆盖上表的默认值（`raw` | `parent`，`raw` | `keep`）。非 LLM 分支随模式变化：edit = 对 raw LoRA 随机组合编辑规则（截断 / 剪枝 / intruder / RegMean / 正交化 / 缩放），scratch = 随机的无梯度构造（奇异方向缩放）。ΔE_write 的参照分支：edit = M_t ⊕ raw（自动加 `raw` 分支），scratch = parent。E_t 会话用同一个模式（edit 时为 θ0 在 E 任务上按同一 raw 配方训练一个 raw LoRA，在 `e_ref` 里只算一次）；E-ref 是 θ0 本身在 E-test 上的分数。

## 直接在评测集上优化（`objective: eval_overfit`，2026-11-20）

不做训练 / 信号 / 验收切分，直接在评测集上过拟合：`--set objective=eval_overfit`（或 `./launch.sh --eval-overfit`），运行名带 `_evalfit`。
- 数据：`math_eval` = MATH-500 + GSM8K test，共 1819 道（`data/scripts/build_eval_target.py`，首次用到时自动生成）。
- train / signal / gate 都是这全部 1819 道：改造者的 `train_example`、`acts`、`train_loss` 直接看到评测题和标准答案；快速筛、完整评测、验收共用同一份逐题结果，贪心打分。不抽题，不标记展示过的题。
- **评测集分数严格高于当前模型就接受**（`gate.min_gain: 1e-6`，不做显著性检验）；只送评测集上提升 > 0 的候选。
- edit 模式第 0 代的 raw LoRA 也直接在评测题上训练（自己答对的回答，4 个都错就跳过）；scratch 不训练。
- 留出评测也直接在这 1819 道上做（`heldout.domains: eval_target`，贪心，2026-11-21）：报告的就是过拟合的那份分数，不再单独测 MATH-500 / GSM8K / AIME。

### 自定义数据角色 `data_roles`（2026-11-21）

每个角色可以单独指定数据集（`--set data_roles.train=... --set data_roles.gate=... --set data_roles.eval=...`，由 `rsicore.config` 校验、`rsicore.draws.train_dataset / gate_dataset` 解析）：
- **train**：改造者干活用的数据。train 题（`train_example` / `acts` / `train_loss`、edit 第 0 代 raw LoRA 的训练数据）、快速筛 / 完整评测的信号题、"只送比当前模型好的候选"这一步，都用它。
- **gate**：决定接不接受候选的数据。基线验收分数、和参照（parent / raw）的逐题配对比较、`gate_scores`，都用它。
- **eval**：留出评测报告（逗号分隔）：领域（`math` = math500,gsm8k,aime24,aime25）、留出任务（`math500`、`humanevalplus`……，用 test 池）、评测集（`math`、`math500`、`ifeval`……，即 K_eval 副本），或 `train` / `gate`（该角色的数据集，须是评测集）。
- train / gate 可填训练集名（`dapo_math` 等，按代抽题 / 固定验收切分，和以前一样）或评测集（`tasks.yaml` 的 `eval_targets` 键，如 `math500`、`ifeval`，或 `math500_eval`：全部题都用，不抽题，不标记展示过的题）。
- 不设时：train = gate = 该领域的训练集（`dapo_math` 等）；`eval_overfit` 时两者都是那份评测集；eval 用 `heldout.domains`。
- train / gate 必须属于这次运行的 `--domain`；用评测集当 train / gate 需要 `scope: single`。和默认不同的角色写进运行名，例如 `B_math_s42_single_gate-math500`、`B_math_s42_single_train-math500`。
- 例子：`--set data_roles.gate=math500 --set data_roles.eval=math500`（在 dapo_math 上改、在全部 MATH-500 上验收、只报 MATH-500）；`--set data_roles.train=math500 --set data_roles.gate=dapo_math`（看着 MATH-500 改、用 dapo_math 的固定验收集把关）。

### `simple` v2（2026-11-21，`--set objective=simple` / `./launch.sh --simple`）

用 `simple_data/<领域>/`（`data/scripts/make_simple_data.py`，从去污染、去重后的训练集池子里抽）。数据集名：训练池 `simple_<领域>_pool`（`pool_ids.json` 里的全部题）、验收 `simple_<领域>_eval`（500）、留出 `simple_<领域>_test`（500），三者两两不相交；运行名带 `_simple`。
- **每代重新抽题**（`rsicore.draws.simple_draw`，没有 draw.py）：候选 = 池子 − 验收 500 − 测试 500；快速筛 = signal = 从候选里随机抽 500（`simple.screen_n`，种子 `<seed>:simple:<领域>:<轮次>`，不含运行名和论文，A 的第 g 代和 B 的第 g−1 代是同一轮，两篇论文筛同样的题）；train = 候选里剩下的全部（数学 13,238，知识约 72k；`train_example` / `acts` / `train_loss` 只接受这些题号，看得到标准答案）。抽题写在 `draws/gen_XX/ids/simple_<领域>_pool.<role>.json`。各代的快速筛独立抽，可能和前几代的 train / 快速筛重叠（不标记 shown）。`gen_XX/train_data.json` 就是这一代的 train（gold 模式，不采样；数学约 8 MB，知识约 26 MB，构建 1–2 秒，本代内存里直接复用，不再读回）。
- **打分用 thinking 采样**（`decoding` 块，`rsicore.config.stage_decoding`）：temperature 0.6、top_p 0.95、top_k 20、thinking、max_tokens 16384，种子 = 运行的 `--seed`，当前模型和所有候选用同一个解码字典（同一种子）。当前模型的快速筛（只筛当前模型，见下）、验收和留出评测都是 n 4（avg@4：每题分数是 4 个样本的平均，在 [0,1] 里；配对验收按每题平均配对）。完整评测 = 快速筛（`search.top_full: 0`）。缓存目录带解码标签：`evals/<name>/<level>@<hash>/`（贪心时仍是 `<level>/`）；只有题目和解码都相同的级别才共享快速筛结果。当前模型（`__parent__`）在一代里的快速筛 / 验收结果只采样一次（每个 (模型, 级别) 一把锁，同时筛多个候选时也只算一次），所有候选共用；M_0 的基线验收和 weights 模式候选槽打分也用同一解码。压缩记录在 n > 1 时记 `ns`（样本数）、`len`（平均长度）、`trunc`（截断比例）；反馈里的错题 = 平均分 ≤ 0.5，回退题 = 当前模型 > 0.5 ≥ 候选（和 Paper A 一样）。打分服务器都是 main 副本 / weights 候选槽，`--max-model-len` = `editor.context_tokens`（131072；h20_80 是 49152），提示 + 16384 放得下；theta0（20480）只给 E_t 用，simple 里 E_t 关闭。启动时 `run.context_shortfall` 检查（max_tokens + 4096 > 某个打分服务器的 `--max-model-len` 就直接退出）。
- **最终流程（FINAL，2026-11-21）**：每代 (1) **当前模型**答本代 500 道快速筛题一次（`decoding.screen`：thinking，n 4；`Run.reference_screen`，评测器的父模型缓存按解码区分，所有会话共用），**候选不做快速筛**（`search.screen_candidates: false`）；(2) 改造者在每步上下文里看到这次作答的摘要（avg@4、每题答对样本数的分布、截断率、几道错题的题面 / 标准答案 / 抽取答案），完整数据用免费工具读：`api.screen_results(ids, filter='all'|'wrong'|'unstable'|'right', limit, offset)` 和 `api.screen_output(id, sample, offset)`（整段输出分页，**含 thinking**；记录里的 `reasoning`）；摘要写进 `gen_XX/reference_screen.json`，后几代的 `search_history`（每代一行的 `reference_screen`）和存档里也看得到；(3) 每个候选（`search.threads × search.steps` 个，这里 8 × 1）建好后**只做一次验收评测**（`search.gate_every_candidate: true`；`decoding.gate`，500 × 4，与当前模型同种子配对；当前模型的验收记录每代只算一次），结果立即写进 `gen_XX/gate.json`（submissions 形状，每个候选一条），候选的 feedback 里是 `gate`（accuracy、reference_accuracy、配对 gain、CI、passed / reasons、broken / fixed 个数和题号，没有题目内容），候选分数 = 配对 gain；后面的步骤和后几代都看得到（上下文表格、`search_history`、`read_record`、存档）；(4) 代末在通过验收规则（`gate.min_gain`，`require_significant` 时还要 CI 下界 > 0）的候选里**取配对 gain 最大的**（并列：快速筛分高、cid 早），没有就 keep（raw 回退规则不变）；`gate.json` 里标 `accepted`；然后测试集 500 × 4 出报告。**没有 `api.evaluate`**（`search.program_evaluate_calls: 0`：程序、探索 api、工具说明和提示里都去掉）。代价：每代 8 × 500 × 4 个 thinking 样本在验收集上。线程数 × 步数是参数，提示按配置的数字写（1 步时"你是第 k 个会话（共 N 个）"，多步时"改进本线程最好的程序"）。`search.parallel_builds: true`：有多个 main 副本时，最多副本数个候选同时构建 + 验收。
- **论文式工具**（`tools.paper_tools`，不设 = 只在 simple 下开；程序和探索 api 都有）：`save_edit(handle, name)` / `load_edit(name)` / `list_edits()` / `drop_edit(name)`（按候选 cid 隔离，存在 `gen_XX/saved/<cid>/`，探索结束、构建和续跑后都还在；最终程序可以直接 `return {"lora": api.load_edit(name="best")["lora"]}`；载入时把构建它用到的工具记进程序的工具日志，`classify()` 照常分类）；`output_matrices(layer)`（o_proj / out_proj / down_proj）；`unit(ref)`、`direction(a, b)` = (a−b)/‖a−b‖（S / A / T 向量引用，返回 T 句柄）；`tensor_compute(..., vectors=[refs])` 把向量作为 `V` 传进代码；`acts(module, ids, k, source='own', sample=0|'all')` 在当前模型**自己的**快速筛作答上（题面 + 整段输出，含 thinking）算激活统计（只收快速筛题号，花 gpu_seconds；HF 模型按 max_len 截断长输出），`source='prompt'` 也接受快速筛题号；`project_out(target, module, u, alpha)`：W ← W − α u uᵀ W（u 是输出侧向量，归一化；W = 当前权重加上 target 在该模块上的改动），LoRA 目标加秩 1 项 B = −αu、A = uᵀW，weights 模式的 edit 目标加一个 lowrank op，照常检查上限；`classify()`：LoRA 上的 project_out 算 construct，edit 上的算 weights op。提示里给了论文式做法的提示（两组自答 → direction → project_out）。
- **探索会话常驻**（`editor.explore.persistent: true`）：rsicore.chat_tools 每个候选保留一个 Python 会话，变量 / import / 句柄跨 `run_python` 调用保留；Paper B 的工厂返回 `(api, cleanup, before_call)`，每个候选只建一个探索 api，`before_call(timeout_s)` 设 tensor_compute 的截止时间，每次工具调用先绑定这个 tag 的线程状态（不管哪个线程在服务）；候选结束时 `cleanup` 释放句柄。simple 里 `editor.explore.gpu_tools: true`（探索里也能用 `acts`，单独的 600 GPU 秒预算）。
- 非 LLM 分支同样 8 个候选（`nonllm.evaluate: 8`，仍按快速筛分数选）；没有保留探针和 E_t；`method.mode` 仍是 scratch；改造者每个候选最多 `editor.tool_rounds` 轮工具调用（simple 设 20，可配置，提示里写的是配置的数字）；`api.tensors` / `api.tensor_compute` 默认开（见下）。
- 新键都有代码默认值（`tools.search_settings`）：不设 = 以前的行为（候选自动快速筛并花一次 evaluate，候选上限 = `max_evaluate`，串行构建，`submit_to_gate`），所以其他目标不变。改了 simple 配置后旧的 `*_simple` 运行锁会拒绝续跑，要用新的运行名 / `--tag`。
- mock：`mock_objective_overrides.simple` 在 `mock_overrides` 之后生效，保持 3 × 1、候选不筛、每个候选一次验收、没有 `api.evaluate`；mock 改造者的 1 号线程走论文式流程（`screen_results` / `screen_output`、`acts(source='own')` 两组、`direction` / `unit`、`tensor_compute(vectors=...)`、`project_out`、`save_edit` / `load_edit`）。测试：`python3 -I -m pytest -q lora_code/tests/test_paper_tools.py`，`lora_code/tests/test_mock_flow.py -k mock_simple_objective`。
- 服务器上需要 `simple_data/<领域>/train.jsonl`、`eval.jsonl`、`test.jsonl`、`pool_ids.json` 和对应的 `data/processed/` 源数据（`rsicore.draws.build_eval_set` 首次运行时构建 `simple_<领域>_pool / _eval / _test`）。各领域合并后的配置见 `plan/configs/B_<领域>_simple.yaml`。

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

**解码（2026-11-20）**：搜索、验收、噪声都用贪心（temperature 0、每题 1 个回答、关闭思考），保证结果确定、可以逐题配对；留出评测用采样，按 Qwen 非思考模式的推荐（temperature 0.7、top_p 0.8、top_k 20，种子 42），每题采 n 个取平均（avg@n：AIME 24/25、AMC23 各 8 个，MATH-500、GPQA-Diamond 各 4 个，其余 1 个；`heldout.decoding`，null = 用 tasks.yaml 的贪心）。噪声重评用贪心时结果完全一样，所以不重跑，噪声带 = 0。训练答案（只在 edit 第 0 代）用同一组采样参数（0.7 / 0.8 / top_k 20），每题采 4 个，取第一个判对的，4 个都错就跳过这道题。只有改造者写程序时用自己的采样参数（thinking 模式：`editor.temperature` 0.6、top_p 0.95、top_k 20）。最大生成长度按任务，在运行配置的 `max_tokens` 里设（覆盖 `eval/configs/tasks.yaml`，对这次运行的所有模型调用生效）：math 4096、code 4096、指令遵循 2048、工具调用 1024、知识（triviaqa_train）1024 token，例如 `--set max_tokens.dapo_math=2048`。训练序列长度 `train.max_len` = 4608（回答最多 4096 + 题目留 512），KL 文本里 M_0 回答的长度见 Paper A 的 `kl.self_generated_max_tokens`。

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
| `genXX/merge` | LoRA 合并进权重 → M_{g+1}（mock：更新 registry 中 `m{g+1}` 的 quality）；`edit` 模式和 `merge.mode adapter` 不合并，`model.json` 记录 M_{g+1} = M_0 + 哪个 adapter（adapter 模式另存 `cumulative_lora`） | `model/`、`model.json`、`cumulative_lora` |
| `genXX/nonllm` | 非 LLM 分支（同 evaluate 预算，按模式）：edit 随机编辑规则作用于 raw；scratch 随机无梯度构造。2026-11-21：`--set nonllm.enabled=false` 完全不跑；`nonllm.skip_if_rejected`（默认 true）在验收保留当前模型（`accepted.json` source keep）的那一代跳过（没有被接受的 LoRA 可比）。两个键不写在 base.yaml 里（只在注释里），按默认读取，这样之前锁定的运行配置还能续跑；跳过时 `nonllm.json` 里有 `skipped` | `baselines/nonllm.json` |
| `genXX/e_t/<branch>` | E_t：parent / llm / nonllm（edit 另加 raw）分支模型各开新会话，为 θ0 在 Seal-Tools 上写 LoRA（16 次 evaluate），在 Seal-Tools test + BFCL 上测；`--domain tool_calling` 时换成代码任务（TACO → LiveCodeBench） | `e_t/<branch>/result.json` |
| `genXX/heldout` | M_{g+1} 的留出评测（领域按 scope / data_roles.eval）。2026-11-21：验收保留当前模型时 M_{g+1} = M_g，直接复用上一代（第 0 代：baseline）的结果，不再重测（`state.json` 里记 `reused_from`） | `heldout.json` |
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
- 预算：evaluate（相同 LoRA 只计一次）、gpu_seconds（`acts` / `regmean_preserve`，实测；mock 中估计）。`search.charge_screens: false`（simple）时自动快速筛不花 evaluate，evaluate 只是程序里 `api.evaluate` 的池子（`program_evaluate_total`）。
- 每个程序自己调用 `api.evaluate` 最多 `search.program_evaluate_calls` 次新评测（2026-11-21，代码默认 2，`null` = 不限；
  已评测过的相同 LoRA 不算；返回候选后的自动评测不算），超了抛 `BudgetExceeded`。`api.budget()` 在程序里还返回
  `program_seconds_left`（离 `sandbox.timeout_s` 还剩几秒）和 `program_evaluate_calls_left`。

### 张量工具 `api.tensors` / `api.tensor_compute`（2026-11-21）

`tools.tensor_compute`（不设 = 只在 objective simple 下开；`false` 同时去掉 `tensors`）。三种模式都有，探索 api 里也有（`tensor_compute.explore`）。
- `api.tensors(layer=None, pattern=None)` → `[{name, shape, dtype, editable}]`：语言模型的所有张量（不只是线性模块：`input_layernorm` / `post_attention_layernorm`、`self_attn.q_norm` / `k_norm`、`linear_attn.dt_bias` / `A_log` / `conv1d` / `norm` …）。`embed_tokens`、最终 norm、`lm_head` 只读、永远不能改。
- `api.tensor_compute(code, tensors)`：在新的沙盒子进程里跑一段 CPU torch 代码，`T` = 当前模型这些张量的 fp32 拷贝（改 `T` 不影响模型），可以设置 `result`（JSON，截到 `result_chars`）、`vectors`（注册成句柄 `T<n>`，`"T<n>.名字"` 可以用在 `rank1` / `from_direction` 的向量参数里，长度要对）、`lora = {模块: (A [r, in], B [out, r])}`（注册成 LoRA 句柄，线性模块，rank / 范数上限照常检查）；weights 模式还可以 `writes = {张量: 新的整个张量}`（`edit=` 可接在已有编辑后），变成该张量上的稠密增量（任意秩，也包括 1 维的 norm / dt_bias / A_log）。
- 上限（`tensor_compute:` 块）：每次最多 `max_tensors` 4 个张量、`max_input_mb` 8192、`timeout_s` 300（不超过调用它的程序 / run_python 剩余时间）、`mem_gb` 16、`threads` 4、全进程同时 `max_concurrent` 2 个；运行时不占工作区锁。
- weights 模式写入的上限：≥ 2 维 `||W'-W||_F/||W||_F <= weights.norm_cap_tau`；1 维 `rms(W'-W) <= weights.vector_cap_tau (0.5) * max(rms(W), 1)`（Qwen3.5 的 RMSNorm 权重以 0 为中心，层里乘的是 1 + w，纯相对上限没有意义）；几乎是整体缩放（`||D - cW||/||D|| < weights.rescale_tol` 0.05）的写入被拒（用 `w_scale`）。
- 系统提示（工具开着时）：向量引用规则里加上 `T` 句柄；模块名规则说明 `api.modules()` 只列线性模块，其他张量用 `api.tensors()` 看、用 `api.tensor_compute` 读（只有 weights 模式能写）；策略里加上 tensor_compute（`svd_lowrank`、CPU、时限）；weights 模式的上限里加上 1 维张量上限、零中心说明和不可改的张量。

系统提示与每步上下文（2026-11-21，根据 nosig 实验的失败候选改写）：
- 验收规则按配置生成（`gate.min_gain`、`gate.require_significant`、本代参照 raw / parent、验收题数换算成"约多对几题"），
  不再写死"95% CI"；单领域实验不再提保留领域。
- 新增规则：程序里 `api` 只是 `main(api)` 的参数（run_python 里是全局变量），辅助函数要显式传入；模块名只能取自
  `api.modules()`（各层类型不规则）；rank1 / from_direction 的 u、direction 是输出侧向量（长度 out），v、key 是输入侧
  （长度 in），取自同一模块的 svd / acts；工具结果可能为空，索引前检查，可选步骤用 try/except 回退；墙钟时限包含所有工具调用。
- 每步上下文第一行附预算：本代还剩几次 evaluate / GPU 秒、每个程序的评测上限、程序时限、已有程序的运行时间中位数和
  最大值、超时次数、一次评测大约几秒；剩余 evaluate ≤ 4 时提醒不要在程序里评测。
- 工具说明：`modules` 是唯一合法的模块名来源；`diagnostics` 可能返回空列表；`rank1` / `from_direction` 写明向量所在侧。

## 改造者：thinking + 工具轮次（2026-11-21）

改造者（主搜索和 E_t 都一样）写每个候选时开 Qwen3 thinking（`editor.extra_body: {chat_template_kwargs: {enable_thinking: true}, top_k: 20, min_p: 0.0}`，temperature 0.6、top_p 0.95），并且可以先调用函数再交程序：
- 一轮 = 模型的一次回复里带了函数调用（同一回复里的多个调用算一轮）。最多 `editor.tool_rounds` = 50 轮，之后强制一次不带工具的最终回复（tool_choice none）。回复里**不带函数调用**、含一个 ```python 块，就是这次的程序。空回复、写到长度上限、把调用写成文本时最多提醒 2 次；最后还没有程序就用循环里写过的最后一个代码块（`stop: fallback`）。
- 两个函数：`run_python(code)` 在同一个沙盒里跑探索代码，`api` 是 `Workspace.explore_api`：和程序的工具一样，但没有 `evaluate`、`raw_lora`（句柄 `RAW` 照样能用），默认也没有 GPU 工具 `acts` / `train_loss` / `regmean_preserve`（`editor.explore.gpu_tools: true` 时可用，记在单独的探索预算 `explore.gpu_seconds` 上）。探索**不花本代的 evaluate / gpu_seconds 预算**，句柄每次调用后释放，工具记录在 `<cid>~xRR_i` 下（不影响 `classify`），完整输出在 `gen_XX/explore/<cid>/rRR_i/`。`read_record(cid, part)` 读本步候选表里某个候选的完整程序或反馈；`search_history(gen, sort, offset)`（2026-11-21，默认开，`editor.history_tool=false` 关）列出所有试过的候选（本代可见的 + 往期所有代），每个一行：strategy、modules、max_rank、用过的工具、target_gain、worst_retention_delta、KL、分数、验收结果，`same_as` = LoRA digest 与更早候选相同；`read_record` 也能打开往期候选。E_t 只看自己的候选。
- 上限（都可配置，`editor:` 块）：`tool_rounds` 50、`max_tool_calls` 100、`turn_max_tokens` 8192（每个工具轮）、`max_tokens` 16384（最终轮，原来 8192）、`proposal_tokens` 160000、`proposal_wall_s` 2400、`tool_result_chars` 6000、`history_keep` 2，压缩 `compact_at` / `keep_tool_rounds` / `keep_reasoning_rounds` = 0.6 / 8 / 2，探索沙盒 `explore.timeout_s` 300、`mem_gb` 8。
- **上下文长度只看 `editor.context_tokens`**：客户端用它截每轮 max_tokens、决定何时压缩，同时写成 main 副本（改造者就在上面；weights 模式下候选槽的临时服务器也一样，保证所有打分服务器上下文相同）的 `--max-model-len`，覆盖 `server.main_args` 里写的值（`lora_code.common.main_server_args`）。b300 / h20 = 131072，h20_80 = 49152；theta0 服务器不变（20480）。设成 null 就按 `main_args` 原样。真实服务器上若模型的 `max_position_embeddings` 更小，`rsicore.servers` 会截到它并警告，不加 YaRN。
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

角色：main（vLLM：M_t + 候选 LoRA，改造者）、theta0（vLLM：原模型，只给 E_t / e_ref 用）、train（HF 模型：训练、权重缓存、诊断、`train_loss`、合并）。

**给多少张卡就用多少张，main 是数据并行副本**（2026-11-20，`gpu_layout.main_replicas`，不设 `gpu_roles` 时）：train 取最后 `gpu_layout.train_gpus` = 1 张；E_t 开着且 `theta0_own_gpu`（H20）时 theta0 单独一张，否则和 train 共用；**其余的卡全给 main，每张卡一份独立的 vLLM 副本**（`gpu_layout.main_tp` = 每份几张卡）。
- 每个候选 LoRA 加载到所有副本上；一次评测的题目分给各副本（`rsicore.evaluate.score_split`）；改造者线程 k 用副本 k mod n，所以一轮里所有线程同时写程序，然后候选逐个评测、每次都用满所有副本；
- 留出评测的任务分给各副本同时跑；验收的最多 3 个提交同时评测，再按排名取第一个通过的。

| 卡数 | 单领域（默认） | 五领域（`scope=all`，H20） |
|---|---|---|
| 2 | main 1 份 · train 1 | 不够（要 3 张） |
| 3 | main 2 份 · train 1 | main 1 份 · theta0 1 · train 1 |
| 4 | main 3 份 · train 1 | main 2 份 · theta0 1 · train 1 |
| 8 | main 7 份 · train 1 | main 6 份 · theta0 1 · train 1 |

旧的布局（main 一个张量并行服务，多的卡依次给 main / train / theta0）：`--set gpu_layout.main_replicas=false`。

手动指定：`--set 'gpu_roles.train=[2,3]'`（--gpus 里的下标）把这个角色固定下来，其余角色在剩下的卡上自动分。B300 上 theta0 和 train 共用一张卡（显存比例 0.45）。

H20 档（按每张卡 110 GB 规划，卡约 140 GB）另外：vLLM 显存用 `--rsi-gpu-memory-gib 110`（启动时换算成显存比例），`--max-num-seqs 128`、上下文 main 131072（`editor.context_tokens`）/ theta0 20480、8 个 LoRA 槽；训练 `train.micro_batch: 2`【试点值】（每次前向 2 条序列，梯度累积，结果等于整批的平均损失）。原模型服务和训练模型放不进同一张 110 GB 的卡，所以五领域（有 E_t）时原模型单独一张。80 GB 的卡用 `h20_80`（0.92、32 条序列、4 个 LoRA 槽、每次 1 条，main 上下文 49152）。

## 只能在 GPU 上验证的部分（本机未实跑）

真实 vLLM 加载我们的 PEFT adapter（Qwen3.5 的线性注意力投影）、27B 的 peft 短训练（transformers 没有 flash-linear-attention 时走 GDN 的 torch 实现，慢且占显存）、27B 上的 SVD / 诊断耗时、合并后的 bf16 舍入误差、各卡显存分配（`server.main_args` / `theta0_args`；H20 上 27B 训练的显存峰值要第一周实测）。
