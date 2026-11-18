# data/：统一数据环境（Paper A 与 Paper B 共用）

本目录包含两篇论文共用的全部评测集与训练集：下载（固定版本）、统一格式化、统一划分，并附带校验脚本。

- Paper A：27B 模型（Qwen3.8 27B）递归量化自身权重，每一轮评测量化后的模型。
- Paper B：模型用代码编辑自己训练出来的 LoRA；LoRA 在一部分任务上训练，在另一部分任务上评测。

所有来源均为数据作者的官方 Hugging Face 仓库 / 官方 GitHub 仓库 / 官方 pip 包，并固定到确切版本（HF revision、git commit、包版本）。
下载的文件一律视为**不可信数据**：`data/raw/` 里的任何脚本都不会被执行；处理脚本只解析 json/csv/parquet，并以 `python -I` 运行。

## 目录结构

```
data/
  raw/<dataset>/            原始下载（每个来源一个目录；_provenance.json 记录 repo、revision、每个文件的 sha256）
  processed/<dataset>.jsonl 统一 schema（见下）
  splits/                   划分：<dataset>/<pool>.json（排序后的 id 列表）、split_config.yaml、MANIFEST.json、check_report.json
  scripts/                  全部代码（下载 / 规范化 / 划分 / 校验 / 生成表格）；脚本不放进 raw/
```

## 重现

```bash
PY="eval/.venv/bin/python -I"          # 先按 eval/README.md 建好 eval/.venv
$PY data/scripts/download.py           # 按 data/scripts/sources.yaml 下载（不使用任何 token）
$PY data/scripts/normalize.py          # raw -> processed（split_pool 先为 null）
$PY data/scripts/make_splits.py        # 生成 splits/ 与 MANIFEST.json，并回填 processed 的 split_pool
$PY data/scripts/check_data.py --ngram # 计数、互斥、manifest 哈希、训练/评测去污染；写 splits/check_report.json
$PY data/scripts/report_table.py       # 打印下表
```

`make_splits.py` 只依赖 item id、`split_config.yaml` 和种子 `seed=20261118`，可重复生成；改划分规模时只需改 `split_config.yaml` 后重跑。

## 当前配置（2026-11-18 修订）

每个领域：**训练集 + 1–3 个留出集**。自量化 / 自编辑循环**只看训练集**：
- `gate` 池 = 固定的**验收**集（每个领域 1,000 题，按权重分到该领域的训练集；每代相同，只返回通过 / 不通过）；
- `pool` 池 = **抽样池**（训练集其余部分）。每一代用 `scripts/draw.py` 从中抽 1,000–2,000 题/领域，作为这一代的信号（L1）、KL / 校准文本（Paper A）和梯度训练数据（Paper B）。训练数据可以跨代复用：先抽没用过的题，用完后按最久未用的顺序复用；训练过或展示给编辑者看过的题不再当信号。见下文"每代抽样"。

benchmark 全部是**留出集**（`test` 池），每一代结束后评测、只用于报告，不回传、不用来挑候选。依据见 `plan/research/DATA_CONFIG.md` 与 `plan/research/method-research/rsi-data/RSI_DATA.md`。

| 领域 | 训练集（每领域验收 1,000 固定 / 其余为抽样池） | 留出集（主表） | 来源论文 |
|---|---|---|---|
| 数学 | `gsm8k_train`、`math_train`、`dapo_math` | MATH-500（500）、GSM8K test（1,319）、AIME 2024+2025（60） | ReST-EM；WSP 博客 / WSP-Scope；WHALE（DAPO） |
| 代码 | `taco_verified`（TACO-verified 10,806 题，参考解都跑过测试；**唯一的代码训练集**） | HumanEval+（164）、MBPP+（378）、LiveCodeBench（131） | EvoTrainer（TACO-verified 11,897 → LCB v6）；ReVeal（TACO 11,151 → LCB v6） |
| 通用推理 / 知识 | `arc`（ARC-Easy+Challenge train）、`gpqa_main`（GPQA Main 去掉 Diamond，250 题全部作验收）、`triviaqa_train` | GPQA-Diamond（198）、MMLU-Pro（抽 2,000）、TriviaQA（抽 2,000） | Transformer²；WSP-Scope；TriviaQA train 无 RSI 先例 |
| 指令遵循 | `ifeval_train`（RLVR-IFeval，14,970 题，每题 1 个可判分约束；验收 200 / 抽样池 14,754）；IFEval 的 135 题验收集保留为 IF 验收（IFEval 的 135 题抽样池保留但不再抽样） | IFEval 其余 271、IFBench（300，领域外） | Tülu 3 RLVR；WSP-Scope |
| 工具调用 | `sealtool_train`（Seal-Tools train） | Seal-Tools test（in-domain 700 + out-domain 654）、API-Bank test（584）、BFCL（4,696） | TT-SI 的 SFT 基线 |

额外留出集（不进主表）：AMC23、OlympiadBench、FreebaseQA、MetaTool、T-Eval、MT-Bench、AlpacaEval。
保留但不再使用：`magicoder_oss`、`tulu3_if`、`apibank_train`（`reserve`）；IFEval 的 135 题抽样池（不再作为 IF 抽样来源）；`toolace` 只作 Paper A 的 KL 文本。
SQuAD（SEAL 设定）保留为 Paper B 的可选知识注入任务。

## 数据集总表

计数单位为 processed 记录数（SQuAD 为“段落”）。`expected` 为官方/预期条数，`actual` 为实际条数，二者全部一致。

| dataset | category | source | revision | license | pool（抽样池） | gate（验收） | test（留出） | reserve | train / search | expected | actual |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_train（训练集） | math | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `740312add88f` | MIT | 7221 | 250 |  |  |  (+2 train_excluded) | 7473 | 7473 |
| math_train（训练集） | math | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | `21a5633873b6` | MIT | 7116 | 350 |  |  |  (+34 train_excluded) | 7500 | 7500 |
| dapo_math（训练集） | math | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | `65877096c24f` | Apache-2.0 (per DAPO project) | 15929 | 400 |  |  |  (+1069 train_excluded) | 17398 | 17398 |
| math（留出集） | math | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | `21a5633873b6` | MIT |  |  | 500 | 4500 |  | 5000 | 5000 |
| gsm8k（留出集） | math | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `740312add88f` | MIT |  |  | 1319 |  |  | 1319 | 1319 |
| aime24（留出集） | math | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | `2fe88a2f1091` | unspecified on HF (problems (c) MAA; used for research eval) |  |  | 30 |  |  | 30 | 30 |
| aime25（留出集） | math | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) | `c94da77eb22b` | CC-BY-NC-SA-4.0 |  |  | 30 |  |  | 30 | 30 |
| amc23（留出集） | math | [math-ai/amc23](https://huggingface.co/datasets/math-ai/amc23) | `80815d37005f` | unspecified on HF (problems (c) MAA) |  |  | 40 |  |  | 40 | 40 |
| olympiadbench（留出集） | math | [Hothan/OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | `91184b52131e` | Apache-2.0 |  |  | 674 |  |  | 674 | 674 |
| apps（训练集（不再使用）） | code | [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps) | `21e74ddf8de1` | MIT |  |  |  | 4449 |  | 4449 | 4449 |
| taco_verified（训练集） | code | [likaixin/TACO-verified](https://huggingface.co/datasets/likaixin/TACO-verified) | `06dd53e0cfcf` | MIT tag on HF (upstream BAAI/TACO is Apache-2.0; card notes some content under CC-BY-4.0) | 9806 | 1000 |  |  |  | 10806 | 10806 |
| magicoder_oss（训练集（不再使用）） | code | [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | `5f839b1f368a` | MIT |  |  |  | 75197 |  | 75197 | 75197 |
| humanevalplus（留出集） | code | [evalplus/humanevalplus](https://huggingface.co/datasets/evalplus/humanevalplus) | `d32357cf319e` | Apache-2.0 |  |  | 164 |  |  | 164 | 164 |
| mbppplus（留出集） | code | [evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus) | `b2d74c91837c` | Apache-2.0 |  |  | 378 |  |  | 378 | 378 |
| livecodebench（留出集） | code | [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) | `0fe84c3912ea` | CC (per HF card; problems from LeetCode/AtCoder/Codeforces) |  |  | 131 |  |  | 131 | 131 |
| arc（训练集） | knowledge | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | `210d026faf99` | CC-BY-SA-4.0 | 2870 | 500 |  |  |  | 3370 | 3370 |
| gpqa_main（训练集） | knowledge | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | `83022cefff93` | CC-BY-4.0 (gated; requires accepting terms; do not redistribute publicly) |  | 250 |  |  |  | 250 | 250 |
| triviaqa_train（训练集） | knowledge | [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | `0f7faf33a390` | Apache-2.0 (per TriviaQA GitHub; HF card says unknown) | 73225 | 500 |  |  |  (+2798 train_excluded) | 76523 | 76523 |
| mmlu_pro（留出集） | knowledge | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | `b189ec765aa7` | MIT |  |  | 2000 | 10032 |  | 12032 | 12032 |
| gpqa_diamond（留出集） | knowledge | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | `83022cefff93` | CC-BY-4.0 (gated; requires accepting terms) |  |  | 198 |  |  | 198 | 198 |
| triviaqa（留出集） | knowledge | [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | `0f7faf33a390` | Apache-2.0 (per TriviaQA GitHub; HF card says unknown) |  |  | 2000 | 7960 |  | 9960 | 9960 |
| freebaseqa（留出集） | knowledge | [kelvin-jiang/FreebaseQA](https://github.com/kelvin-jiang/FreebaseQA) | `d16aae5149aa` | CC-BY-4.0 |  |  | 3996 |  |  | 3996 | 3996 |
| ifeval_train（训练集） | instruction_following | [allenai/RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval) | `47c03c73621c` | ODC-BY | 13954 | 1000 |  |  |  (+16 train_excluded) | 14970 | 14970 |
| tulu3_if（训练集（不再使用）） | instruction_following | [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | `fe0c7d350c9b` | ODC-BY |  |  |  | 29980 |  | 29980 | 29980 |
| ifeval（留出集） | instruction_following | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | `966cd89545d6` | Apache-2.0 | 135 | 135 | 271 |  |  | 541 | 541 |
| ifbench（留出集） | instruction_following | [allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test) | `ed7c12452594` | ODC-BY (per allenai/IFBench repo README; HF card has no license tag) |  |  | 300 |  |  | 300 | 300 |
| mt_bench（留出集） | instruction_following | [lm-sys/FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) | `587d5cfa1609` | Apache-2.0 |  |  | 80 |  |  | 80 | 80 |
| alpaca_eval（留出集） | instruction_following | [tatsu-lab/alpaca_eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) | `2edc6fad8be6` | CC-BY-NC-4.0 |  |  | 805 |  |  | 805 | 805 |
| sealtool_train（训练集） | tool_calling | [fairyshine/Seal-Tools](https://github.com/fairyshine/Seal-Tools) | `ce753ecd60ed` | Apache-2.0 (repo LICENSE) | 11019 | 1000 |  |  |  (+3 train_excluded) | 12022 | 12022 |
| toolace（训练集） | tool_calling | [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | `6bda777c88d2` | Apache-2.0 |  |  |  |  | train 11300 | 11300 | 11300 |
| apibank_train（训练集（不再使用）） | tool_calling | [liminghao1630/API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank) | `12e8158b7628` | MIT |  |  |  | 10376 |  | 10376 | 10376 |
| sealtool（留出集） | tool_calling | [fairyshine/Seal-Tools](https://github.com/fairyshine/Seal-Tools) | `ce753ecd60ed` | Apache-2.0 (repo LICENSE) |  |  | 1354 | 700 |  | 2054 | 2054 |
| apibank（留出集） | tool_calling | [liminghao1630/API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank) | `12e8158b7628` | MIT |  |  | 584 |  |  | 584 | 584 |
| bfcl（留出集） | tool_calling | [bfcl-eval](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | `bfcl-eval==2026.3.23` | Apache-2.0 |  |  | 4696 |  |  | 4696 | 4696 |
| metatool（留出集） | tool_calling | [HowieHwong/MetaTool](https://github.com/HowieHwong/MetaTool) | `35e81bb75768` | MIT |  |  | 5327 |  |  | 5327 | 5327 |
| teval（留出集） | tool_calling | [lovesnowbest/T-Eval](https://huggingface.co/datasets/lovesnowbest/T-Eval) | `af355ab2b62c` | Apache-2.0 |  |  | 4358 |  |  | 4358 | 4358 |
| squad_train（训练集） | knowledge_injection | [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) | `7b6d24c440a3` | CC-BY-SA-4.0 |  | 50 |  | 3887 | train 14904 / search 50 | 18891 | 18891 |
| squad_val（留出集） | knowledge_injection | [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) | `7b6d24c440a3` | CC-BY-SA-4.0 |  |  | 200 | 1867 |  | 2067 | 2067 |

BFCL 分类计数（全部与官方一致）：simple_python 400、simple_java 100、simple_javascript 50、multiple 200、parallel 200、parallel_multiple 200、live_simple 258、live_multiple 1053、live_parallel 16、live_parallel_multiple 24、irrelevance 240、live_irrelevance 884、live_relevance 16、multi_turn_{base,miss_func,miss_param,long_context} 各 200、memory 155、web_search 100。
T-Eval：instruct/plan_json/plan_str/reason_retrieve_understand_json/reason_str/retrieve_str/understand_str 各 553，review_str 487。
MetaTool：Task1 1040、Task2-Subtask1 995、Subtask2 1800、Subtask3 995、Subtask4 497。API-Bank：level-1 API 399、level-2 API 135、level-3 50。

## 统一 schema

`data/processed/<dataset>.jsonl` 每行：

```json
{"id": "...", "dataset": "...", "category": "math|code|knowledge|instruction_following|tool_calling|knowledge_injection",
 "split_pool": "search|gate|test|reserve|train|train_excluded",
 "prompt": "...或 null", "messages": [...] 或 null, "reference": ..., "metadata": {...}}
```

- `id`：优先使用官方 id（如 `HumanEval/0`、`Mbpp/2`、`lcb/abc392_a`、`mmlu_pro/70`、`triviaqa/tc_2`、`FreebaseQA-eval-0`、BFCL id、`teval/<subset>/<key>`）；无官方 id 时为 `<dataset>/<split>/<固定版本文件中的行号>`；SQuAD 段落为 `squad/<sha256(context)[:16]>`。
- `prompt`：与任务无关的题面（评测时的指令模板在 `eval/configs/tasks.yaml`）。BFCL/T-Eval/MT-Bench 等原生多轮或带 system 的数据放在 `messages`。
- `reference` 形状：数学为最终答案字符串（OlympiadBench 为列表，MATH 取 solution 中最后一个 `\boxed{}`，MATH-500 用其自带 answer）；选择题为字母；TriviaQA/FreebaseQA 为别名列表；IFEval/IFBench 为 `{instruction_id_list, kwargs}`；BFCL 为官方 possible_answer（irrelevance 类为 null）；T-Eval 为官方 ground_truth；MetaTool 为工具名 / `"None"` / 工具名列表 / yes|no；API-Bank 为 `API-Request: [...]`；SQuAD 为该段落的问题列表 `[{qid, question, answers}]`；代码题为参考解（LiveCodeBench 无参考解，测试用例留在 raw 中，只在沙箱内解码）。

## 统一划分方案（seed=20261118）

规则写在 `splits/split_config.yaml`：`role: train` 的数据集是训练集，其余都是留出集。

| 数据集 | pool（抽样池） | gate（验收，固定） | test（留出） | 其他 |
|---|---|---|---|---|
| gsm8k_train / math_train / dapo_math | 7,221 / 7,116 / 15,929 | 250 / 350 / 400（数学共 1,000） | — | 去污染移除的 → train_excluded |
| arc / triviaqa_train | 2,870 / 73,225 | 500 / 500（知识共 1,000） | — | |
| taco_verified（10,806） | 9,806 | 1,000（按难度分层） | — | 唯一的代码训练集 |
| apps（4,449） | — | — | — | 全部 reserve（2026-11-19 起不再使用） |
| ifeval_train（RLVR-IFeval，14,970） | 13,954 | 1,000（按 constraint_type 分层） | — | 去污染移除 16 → train_excluded |
| sealtool_train | 11,019 | 1,000 | — | |
| gpqa_main（250） | — | 250 | — | |
| IFEval（541） | 135 | 135 | 271 | 按首个指令类型分层（WSP-Scope 的 270/271） |
| MATH test | — | — | 500（MATH-500） | 其余 4,500 → reserve |
| GSM8K test / AIME24 / AIME25 / AMC23 / OlympiadBench | — | — | 全部 | |
| HumanEval+ / MBPP+ / LiveCodeBench（2025-02-01 起） | — | — | 全部 | |
| MMLU-Pro | — | — | 2,000 | 其余 10,032 → reserve |
| TriviaQA rc.nocontext validation | — | — | 2,000 | 其余 7,960 → reserve |
| GPQA-Diamond / FreebaseQA / IFBench / MT-Bench / AlpacaEval | — | — | 全部 | |
| Seal-Tools | — | — | test_in_domain 700 + test_out_domain 654 | dev 700 → reserve |
| BFCL / API-Bank test / MetaTool / T-Eval | — | — | 全部 | BFCL 多轮与 agentic 只能用官方 CLI 评 |
| SQuAD（SEAL 设定） | 50 段（train，50 个不同标题） | 50 段（train，另 50 个标题） | SEAL 的 200 段 | 见下 |
| toolace | — | — | — | 全部 train（只作 KL 文本） |
| magicoder_oss / tulu3_if / apibank_train | — | — | — | 全部 reserve（不再使用） |

说明：
- 采样方法：每个数据集（或层）内按 id 排序后用 `random.Random(f"{seed}:{dataset}:{stratum}")` 打乱，再依次切出固定池；分层按最大余数法分配名额。训练集先去污染、再切分，所以验收集和抽样池都是干净的。
- `eval/run_eval.py` 中训练集任务（`gsm8k_train`、`math_train`、`dapo_math`、`taco_verified`、`arc`、`gpqa_main`、`triviaqa_train`、`sealtool_train`、`ifeval_train`）只允许 `--pool pool|gate`；用 `--pool pool` 时配合抽样结果 `--ids-file`。

## 两个配置开关

| 开关 | 位置 | 默认 | 说明 |
|---|---|---|---|
| 固定验收集大小 | `splits/split_config.yaml` 的 `gate_domains.<领域>.size` | 每领域 1,000 | 按权重分到该领域各训练集；改动后要重跑 `make_splits.py`（已有抽样运行会被拒绝，需要新开运行） |
| 每代抽样量 | `draw/paper_a.yaml`、`draw/paper_b.yaml` 的 `domains.<领域>.n` | 数学 / 知识 / 指令 / 工具 2,000，代码 1,000 | Paper B 的目标领域再按 `target_train_frac` 分成训练 / 信号；其余领域用 `retention_n` |

## 每代抽样（`scripts/draw.py`）

每一代从抽样池里抽一批新题，配置在 `data/draw/paper_a.yaml`、`data/draw/paper_b.yaml`，结果写在运行目录（`experiment_records/<run_id>/draws/`）。

```bash
PY="eval/.venv/bin/python -I"
$PY data/scripts/draw.py --run experiment_records/<run_id> --gen 0 --config data/draw/paper_a.yaml   # 第一次：锁定配置
$PY data/scripts/draw.py --run experiment_records/<run_id> --gen 1                                 # 之后每代
$PY data/scripts/draw.py --run experiment_records/<run_id> --gen 0 --target math --config data/draw/paper_b.yaml  # Paper B 要指定目标领域
$PY data/scripts/draw.py --run experiment_records/<run_id> --status                                # 每代数量、剩余新题
$PY eval/run_eval.py --task gsm8k_train --pool pool --ids-file experiment_records/<run_id>/draws/gen_01/ids/gsm8k_train.signal.json ...
```

| 角色 | Paper A | Paper B |
|---|---|---|
| `signal` | 每个领域 N 题（数学 / 知识 / 工具 / 指令遵循 2,000，代码 1,000） | 目标领域 20%；其余领域各 400 题（保留探针） |
| `screen` | `signal` 中按数据集分层抽 200 题，用于第一阶段快速评测 | 同左 |
| `kl` | 共 256 段 KL / 校准上下文，各领域均分，与 `signal` 不重叠 | 其余领域各 64 条，算相对当前模型的逐 token KL |
| `train` | — | 目标领域 80%（梯度工具唯一可用的数据） |

规则：
- **复用规则**（2026-11-19 修订，配置键 `reuse`；`ledger.json` 记录每道题在哪一代、以什么角色用过）：
  - 每代内部各角色互不重叠（`screen` 是 `signal` 的子集）；
  - 跨代可以复用：先抽从没用过的题，用完后按最久未用的顺序复用（`across_generations: true`）；
  - 用作过 `train`（Paper B 梯度数据）或在反馈里展示给编辑者看过的题（`--mark-shown` 记录），之后不再当 `signal` / `screen` / `kl`，但可以再次当 `train`（`never_signal_after: [train, shown]`）。
  - 编辑循环每代结束时，把展示过的错例写成 `{"<dataset>": ["<id>", ...]}`，调用 `draw.py --run ... --gen t --mark-shown shown.json`。
- 同样的切分、配置、运行名和抽样顺序，得到完全相同的题号（`check_draw.py` 检查）。
- 抽样池在运行中途变化（重新切分）会被拒绝；`--force` 只能重抽最新一代。
- 有了复用，能跑几代不再受抽样池大小限制。小池子（ARC 2,870）在几代后开始复用，自测里 6 代都能抽满。
- 自测：`$PY data/scripts/check_draw.py`（Paper A、B 各抽 6 代，检查数量、互斥、先新后旧的复用顺序、训练过或展示过的题不再当信号、可复现、`--force` 和各种拒绝情况）。
- 代码训练集的测试用例在规范化时转换成 LiveCodeBench 评测器的格式，写在 `processed/aux/taco_verified_tests.jsonl`（APPS 的在 `apps_tests.jsonl`，已不再使用），只在 Docker 沙箱里执行。
- **SQuAD 的口径**：SEAL 代码中的 “article” 指一个段落，因此 search/gate 各取 50 个段落。SEAL 的 200 个评测段落覆盖了 SQuAD 验证集 48 个维基标题中的 47 个，所以 search/gate 段落取自 SQuAD **train**，每个标题只取 1 段，search、gate、train 三池按标题互斥；search/gate 标题下的其余段落放入 reserve。test 是 SEAL 官方仓库 `general-knowledge/data/synthetic_data/eval/base_val.json`（200 段、974 问）。

## 去污染

`make_splits.py` 对每个训练集在**切分前**做两步过滤，命中的样本放进 `train_excluded`，不进入信号 / 验收 / train 任何一池：
1. **精确匹配**：规范化后与任何评测样本（所有池，包括 reserve）题面相同；
2. **13-gram 近似重复**：与同类别某一条**留出**样本共享的 13-gram 占任一方 13-gram 的 ≥50%（双向，能抓到"长训练题里包含一道留出题"的情况）。Seal-Tools 只比较 task_instruction 一行，跳过共用模板和 api_list。

移除数量（精确 / 近似）：
- dapo_math：984 / 85
- gsm8k_train：0 / 2
- math_train：1 / 33
- sealtool_train：0 / 3
- ifeval_train：0 / 16（全部是约束模板句与 IFEval/IFBench 短题面共用 13-gram 造成的，去掉约束句后的指令部分与 IFEval/IFBench 零精确重合、零 13-gram 重合）
- triviaqa_train：2428 / 370
- 其余训练集均为 0 / 0。

要点：
- **DAPO-Math-17K 含有大量 MATH test 原题**：精确匹配 984 条，其中 968 条是 MATH test 中 MATH-500 以外的题，4 条 MATH-500、5 条 AMC23、1 条 OlympiadBench；近似重复再移除 85 条。
- 去污染后，按"留出样本被单条训练样本覆盖 ≥50% 的 13-gram"统计，所有主表留出集都是 0。
- `check_data.py` 独立断言训练集的 search/gate/train 三池与评测集零精确重合；`--ngram` 给出任意 13-gram 重合的参考数字（已去掉在 ≥5 条评测样本中出现的模板性 n-gram），完整数字见 `splits/check_report.json`。任意一个 13-gram 重合大多是常见措辞（如 "find the remainder when ... is divided by 1000"）。
- 评测集之间的精确重合（不是错误，仅提示）：FreebaseQA 与 TriviaQA 有 308 道题面相同；BFCL 与 TriviaQA 有 1 道。

## 需要用户手动获取

### GPQA Main（gated，同一仓库）
**2026-11-18 已完成**：`gpqa_main.csv`（448 题）用本机已登录的 HF 账号下载到 `data/raw/gpqa_main/`，去掉 Diamond 的 198 题后剩 250 题，作为知识领域的选择题训练集（信号 125 / 验收 125）。同样不要放进公开仓库：`data/raw/gpqa_main/`、`data/processed/gpqa_main.jsonl`。

### GPQA-Diamond（gated，需要接受条款）
**2026-11-18 已完成**：用户在 Hugging Face 上接受条款后，用本机已登录的 `hf` 下载了固定版本（198 条，全部为 test），判分程序自检和端到端测试均通过。数据作者要求不要公开传播题目，所以 `data/raw/gpqa_diamond/` 和 `data/processed/gpqa_diamond.jsonl` 不要放进公开仓库。以下是当时记录的步骤，留作复现参考。

原记录：本次没有登录、没有接受条款、没有使用任何 token，因此未下载。加载代码已就绪（`normalize.py` 中的 GPQA 分支：选项按 `seed-RecordID` 固定打乱，答案为字母）。步骤：
1. 用你自己的 HF 账号登录并在 https://huggingface.co/datasets/Idavidrein/gpqa 接受使用条款（数据作者要求不要公开传播题目）。
2. 下载固定版本的文件到 `data/raw/gpqa_diamond/`：
   ```bash
   huggingface-cli download Idavidrein/gpqa gpqa_diamond.csv --repo-type dataset \
     --revision 83022cefff930aea54f654c0b282e74b9eeda5c6 --local-dir data/raw/gpqa_diamond
   ```
3. 重跑：`eval/.venv/bin/python -I data/scripts/normalize.py --only knowledge && eval/.venv/bin/python -I data/scripts/make_splits.py && eval/.venv/bin/python -I data/scripts/check_data.py`（预期 198 条，全部为 test）。

## 偏差与注意事项

- **DAPO-Math-17K**：原始 parquet 有 1,791,700 行（每题重复多次），按题面去重后 17,398 题；WHALE 论文写的是 17,917 题，差异来源未知。题面去掉了 DAPO 自带的指令前后缀，评测时统一用 `\boxed{}` 模板。
- **TACO-verified**（2026-11-19 起唯一的代码训练集）：固定 HF 对原 JSON（commit 9e68575e）的 parquet 转换（refs/convert/parquet，06dd53e0）。12,898 题筛到 10,806 题：去掉带图片的 940 题、"输出任意一个正确答案"类的 837 题（没有检查器）、内部重复 312 题、测试不可用 3 题；每题测试序列化后上限 1 MB（超出截断）。和 LiveCodeBench 的 13-gram 重合全部来自 AtCoder 模板句，单条 TACO 题最多覆盖一道 LCB 题的 7%；和 HumanEval+、MBPP+ 基本无重合。先例：EvoTrainer（2606.03108）用 11,897 道 TACO-verified 题训练、在 LCB v6 上评测；ReVeal（2506.11442）用 11,151 道 TACO 题训练、在 LCB v6（2025-02 至 05）上评测。
- **APPS**（已不再使用）：只下载 `train.jsonl`（没有下载 1.3 GB 的 test 和 `apps.py` 加载脚本）。入门级 2,639 题中 2,397 题同时有测试用例和参考解（ReST-EM 写 2,342 题，可能还过滤了参考解跑不通的题；沙箱可用后应跑一遍参考解，剔除不通过的题）。2,263 题是函数调用式（fn_name），其余是标准输入输出。
- **ARC**：少数题有 3 或 5 个选项，标签统一按位置转成字母。
- **Seal-Tools**：使用官方 GitHub 仓库 `dataset_for_finetune/` 中的格式（每题带候选 api_list，金标是调用列表）。判分：预测与金标的 (api, 参数) 多重集合完全相同才算对，忽略 `responses` 占位符；同时记录格式正确率、API 名和参数级别的 P/R/F1（Seal-Tools 论文的指标）。TT-SI 用的测试划分（Lin et al. 2025，294 题）与官方 test 不同，数字不能直接对比。

- **MATH 来源**：hendrycks/math README 指向的 Berkeley 压缩包返回 403，README 链接的 HF 镜像 `qwedsacf/competition_math` 没有 train/test 划分列，因此使用 `EleutherAI/hendrycks_math`（同样的 7500/5000 划分，MIT）。MATH-500 通过题面（空白归一化）与 `HuggingFaceH4/MATH-500` 匹配，500/500 全部命中，`metadata.math500_unique_id` 记录原始 id。
- **AIME / AMC**：MAA 没有官方 HF 仓库，使用社区常用版本（H4 aime_2024、MathArena aime_2025、math-ai/amc23 的 40 题）。
- **TriviaQA**：HF 的 rc.* 配置按证据来源（web / wikipedia）把同一问题重复存放，rc.nocontext validation 17944 行只有 9960 个唯一 question_id（train 138384 行 → 76523 个），按 question_id 去重保留第一次出现。有 1 个 id（Kinect 题）两行题面略有差异，保留第一行。
- **LiveCodeBench**：只下载 `test6.jsonl`（release_v6 新增的 175 题，134 MB）。已用完整下载核实 `test5.jsonl` 的日期范围为 2024-09-22 至 2025-01-04，窗口内 0 题，所以不需要下载 test1–5（共约 4.3 GB）。私有测试用例在官方文件中是 base64(zlib(pickle))，**只在 Docker 沙箱内解码**。
- **BFCL**：数据来自 pip 包 `bfcl-eval==2026.3.23` 自带的 v4 数据（复制到 `raw/bfcl/`，不执行包内代码来获取数据）。`format_sensitivity` 是对已有 id 的重新组合，不作为独立条目收录。
- **MetaTool**：`dataset/tool_embedding.pkl` 是 pickle 文件，故意没有下载。Task2-Subtask3（可靠性）中的正确工具都不在候选列表里，参考答案为 `"None"`（按候选列表解析判断，而不是子串匹配）。
- **API-Bank**：test 使用 HF 上作者发布的 `test-data`（level-1/2 API 调用 + level-3）；训练使用 `training-data/lv{1,2,3}-api-train.json`。
- **IFBench**：使用 `allenai/IFBench_test`（300 条单轮）。
- **RLVR-IFeval（`ifeval_train`）**：14,973 行，去掉 3 条完全重复（题面与 ground_truth 都相同）后 14,970 题；24 种约束，每种 604–655 题。约束句约一半放在题面开头、一半放在结尾；去掉约束句后只有 1,807 个不同的基础指令（每个基础指令平均配 8.3 个不同约束），验收集与抽样池会共享基础指令（只判约束，不判内容）。`metadata.instruction` / `instruction_group` 记录去掉约束句后的指令及其哈希。判分见 `eval/evallib/graders/rlvr_if.py`（open-instruct 的两处问题已修正：`validate_choice` 的 options 是字符串，原实现逐字符匹配；`validate_repeat_prompt` 的 original_prompt 含"不要重复"的那句话本身）。
- **AlpacaEval 2**：参考输出为 `alpaca_eval_gpt4_baseline.json`（与 GitHub 上 results/gpt4_1106_preview/model_outputs.json 大小一致，即 AlpacaEval 2 的基线）；judge 提示词从 alpaca_eval 仓库固定 commit 下载。
- 处理后的文件较大的主要是训练集（magicoder 190 MB、tulu 126 MB）；总占用见 eval/README.md。
