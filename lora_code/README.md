# lora_code/：Paper B 自写 LoRA（递归）

计划见 `plan/PLAN.md` §4、`plan/research/RSI_LOOP_DESIGN.md` §3、`plan/research/METHOD_DESIGN.md` §3。模型 Qwen3.8-27B（`Qwen3_5ForConditionalGeneration`）。共用运行时在 `eval/rsicore/`（契约 `eval/rsicore/INTERFACES.md`），本目录只写 Paper B 特有的部分。

## 启动

```bash
./run_rsi.sh --paper B --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id B_math_s42
./run_rsi.sh --paper B --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1,2 --run-id B_math_s42 --set hardware=h20   # H20 80GB
./run_rsi.sh --paper B --domain math --gens 2 --mock          # 本机 mock，约 5 分钟
# 改配置：--set search.max_evaluate=16 --set gate.delta=0.03（写进 run_config.yaml 并锁定）
```

同一个 run-id 再运行会跳过已完成的步骤（`state.json`）。产物在 `experiment_records/<run-id>/`。

## 一代（第 g 代：改造者 M_g → 接任者 M_{g+1}；目标领域每代都是 `--domain`）

| 步骤（state key） | 内容 | 产物 |
|---|---|---|
| `baseline/gate` | M_0 在固定验收集（五个领域）上评测，噪声带 = max(重复评测波动, 2×SE) | `baseline/gate_scores.json` |
| `baseline/heldout` | M_0 留出评测 | `baseline/heldout.json` |
| `e_ref` | E 任务参照：固定配方 SFT LoRA 写给 θ0，在 E-test 上的分数 | `e_ref/result.json` |
| `genXX/draw` | `draw_gen(target=domain)`：目标 train / signal / screen，其余领域 signal / screen / kl | `draws/gen_XX/` |
| `genXX/train_data` | train 题的训练目标：先拒绝采样 M_g 自己答对的回答，否则用评测格式的金标（`train_data.mode`） | `train_data.json` |
| `genXX/diagnostics` | 固定诊断表（每个 (层, 模块)：范数、奇异值、已合并 adapter 的范数 / 有效秩 / intruder / 相对变化分位 / 激活比） | `diagnostics.json` |
| `genXX/sft` | SFT 基线（r16、同一梯度预算、25/50/75/100% checkpoint），screen / full / gate 评测 | `baselines/sft.json` |
| `genXX/search` | `rsicore.search.inner_search`：4 线程 × 8 步、≤32 次 evaluate；改造者 = gpu0 上的 M_g（vLLM 多 LoRA），候选 LoRA 在 gpu1 上构建后热加载评测 | `candidates.jsonl`、`programs/`、`feedback/`、`loras/` |
| `genXX/gate` | `submit_to_gate`（≤3 次）：目标 ≥ SFT − ε；每个保留领域 ≥ 第 0 代 − max(δ, 噪声带)；已学目标 ≥ 学完时 − δ；沙箱检查。都不过 → SFT LoRA（若能过）→ 否则保留 M_g | `gate.json`、`accepted.json`、`accepted_lora.*` |
| `genXX/merge` | LoRA 合并进权重 → M_{g+1}（mock：更新 registry 中 `m{g+1}` 的 quality） | `model/`、`model.json` |
| `genXX/nonllm` | 非 LLM 分支：SFT 超参 × checkpoint × 全局缩放的随机搜索（同 evaluate 预算、同梯度预算）；全局缩放前沿、checkpoint 前沿 | `baselines/nonllm.json` |
| `genXX/e_t/<branch>` | E_t：parent / sft / llm / nonllm 四个分支模型各开新会话，为 θ0 在 Seal-Tools 上写 LoRA（16 次 evaluate），在 Seal-Tools test + BFCL 上测；`--domain tool_calling` 时换成代码任务（TACO → LiveCodeBench） | `e_t/<branch>/result.json` |
| `genXX/heldout` | M_{g+1} 五个领域留出评测 | `heldout.json` |
| `genXX/summary` | 记录展示过的错例（`draw.py --mark-shown`）；汇总 ΔE_train、ΔE_write、ΔE_nonLLM（按题配对 bootstrap） | `summary.json` |

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 编排（上表）；`Servers` 管理 vLLM / mock 服务（GPU 分工见下面"硬件"） |
| `configs/base.yaml`、`configs/domains/*.yaml` | 预算、秩上限 64、范数上限 τ、梯度预算、搜索规模、ε / δ、λ、E_t、mock 缩小版（`mock_overrides`） |
| `primitives/naming.py` | Qwen3.5 模块名、vLLM 打包映射、PEFT 键名 |
| `primitives/weights.py` | `WeightStore`：按需读 safetensors 分片，top-k SVD（缓存） |
| `primitives/lora.py` | `LoRA`（A、B 因子，缩放并入 B）、沙箱检查、PEFT 保存 / 读取（vLLM 可加载）、合并进权重 |
| `primitives/construct.py` | `rank1`、`from_direction`、`scale_singular`、`from_lowrank_grad` |
| `primitives/ops.py` | `scale`、`interpolate`、`combine`、`drop`、`truncate_rank`、`intruder_scale`、`relchange_prune`、`regmean_preserve`、`orthogonalize`、`ties` |
| `primitives/train.py` | `TrainData`（只允许本代 train 题）、`HFModel`（transformers + peft：loss / grad / sgd / acts / grams / act_ratio） |
| `tools.py` | `Workspace`：把原语注册成 `ToolAPI` 工具（句柄、预算、train 限制、evaluate），`program_build`（沙箱执行 → 检查 → 保存候选） |
| `evaluator.py` | 候选评测（screen / full / gate、token-KL、配对差、错例、行为统计、总分） |
| `diagnostics.py` | 诊断表计算与渲染 |
| `prompts.py` | 改造者 system prompt、`ContextRenderer`（候选表、本线程最佳程序、最近反馈、档案） |
| `baselines.py` | SFT、全局缩放 / checkpoint 前沿、随机搜索 |
| `etask.py` | E_t 固定任务 |
| `mock.py` | mock 模型几何、mock LoRA 与 quality 模型、mock 诊断、mock 改造者策略 |

## 改造者看到的工具

程序形如 `def main(api): ...; return {"lora": handle}`（`{"lora": None}` = 保持原样），工具只能用关键字参数调用：
`modules`、`diagnostics`、`train_ids`、`train_example`、`svd`、`acts`、`grad`、`new_lora`、`rank1`、`from_direction`、`scale_singular`、`from_grad`、`sgd`、`scale`、`interpolate`、`combine`、`drop`、`truncate_rank`、`intruder_scale`、`relchange_prune`、`regmean_preserve`、`orthogonalize`、`ties`、`inspect`、`stack`、`candidate`、`evaluate`、`budget`（说明见 `tools.py` 的 `TOOL_DOCS`）。

- `acts` / `grad` / `sgd` / `train_example` 只接受本代 `train` 题号，其他题号抛 `PermissionError`。
- `regmean_preserve` 的保留集 Gram 矩阵由可信运行时用保留领域的 `kl` 题面算出，改造者只拿到结果 LoRA。
- 预算：evaluate（相同 LoRA 只计一次）、gpu_seconds（实测；mock 中按步数估计）。

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

| 档 | 每个运行的卡 | 分工（`gpu_roles`，--gpus 里的下标） |
|---|---|---|
| `b300`（默认，288 GB） | 2 | main [0]；theta0 [-1]；train [-1]（原模型服务和训练共用第 2 张） |
| `h20`（80 GB） | 3 | main [0]（M_t + LoRA，改造者）；theta0 [1]（原模型）；train [2]（HF 模型训练 / 梯度工具、权重缓存、诊断、合并） |

H20 档另外：vLLM 显存比例 0.92、`--max-num-seqs 32`、上下文 20480、4 个 LoRA 槽；训练 `train.micro_batch: 1`（每次前向 1 条序列，梯度累积，结果等于整批的平均损失）。4 张卡时 `--set 'gpu_roles.train=[2,3]'` 会把训练模型按层拆到两张卡上（`device_map`）。

## 只能在 GPU 上验证的部分（本机未实跑）

真实 vLLM 加载我们的 PEFT adapter（Qwen3.5 的线性注意力投影）、27B 的 peft 短训练（transformers 没有 flash-linear-attention 时走 GDN 的 torch 实现，慢且占显存）、27B 上的 SVD / 诊断耗时、合并后的 bf16 舍入误差、各卡显存分配（`server.main_args` / `theta0_args`；H20 上 27B 训练的显存峰值要第一周实测）。
