# eval/：统一评测环境（客户端 + 评分器）

`eval/run_eval.py` 是唯一的推理客户端：向任意 OpenAI 兼容端点（HPC 上的 vLLM，含多 LoRA）发请求，缓存生成结果，用确定性评分器打分，输出逐条结果（用于后续 paired bootstrap）和汇总。
服务端（vLLM）不装在本机，见 `eval/hpc/README.md`。

## 1. 安装

```bash
bash eval/setup_env.sh      # uv 建 eval/.venv（Python 3.11.10），装 requirements.lock，克隆固定 commit 的第三方评测仓库，装 IFBench，取 NLTK 数据
```

已安装的关键版本（完整列表见 `requirements.lock`）：`bfcl-eval==2026.3.23`、`math-verify==0.9.0`（antlr4 4.13.2）、`ifbench==0.2.0`（allenai/IFBench@488918665d53）、`httpx==0.28.1`、`huggingface-hub==2.4.0`、`pyarrow==19.0.1`、`numpy==1.26.4`（bfcl-eval 固定）、`torch==2.15.0`（由 bfcl-eval→sentence-transformers 带入）。
有意**不装**：`datasets`（≥5 需要 pyarrow≥24 → numpy 2，与 bfcl-eval 的 numpy 固定冲突；本项目直接读 parquet）、`evalplus`（其 tree-sitter 版本与 bfcl-eval 冲突；放在 Docker 沙箱镜像里，`evalplus==0.3.1`）。

第三方仓库（`third_party/pins.yaml`）：

| 目录 | 仓库 @ commit | 用途 |
|---|---|---|
| google-research-ifeval | google-research/google-research @ 7117df5f368e（sparse: instruction_following_eval） | IFEval 官方检查器 |
| IFBench | allenai/IFBench @ 488918665d53 | IFBench 官方检查器 |
| T-Eval | open-compass/T-Eval @ 58f22406404d | T-Eval 官方 evaluator |
| DAMO-ConvAI-apibank | AlibabaResearch/DAMO-ConvAI @ faf754671f5b（sparse: api-bank） | API-Bank 官方 API 调用解析器 |
| MetaTool | HowieHwong/MetaTool @ 35e81bb75768（sparse: src） | 仅参考（官方未发布评分脚本） |
| LiveCodeBench | LiveCodeBench/LiveCodeBench @ 28fef95ea8c9 | `lcb_runner.evaluation`，只拷进沙箱镜像 |
| evalplus | evalplus/evalplus @ 26d6d00bb1fd | 参考；沙箱内用 PyPI `evalplus==0.3.1` |
| （.nltk_data） | nltk/nltk_data @ 550b6625bcef | punkt_tab / stopwords / tagger（均为非 pickle 资源） |

BFCL 不需要克隆，直接用 pip 包 `bfcl-eval==2026.3.23`（官方 AST checker）。

## 2. 运行

```bash
eval/.venv/bin/python -I eval/run_eval.py --task <name>[,<name>...] --pool <search|gate|test|reserve> \
  --endpoint http://NODE:8000/v1 --model <served-name> [--lora <lora-name>] --out eval/results/<run_id>
```

常用参数：`--thinking on|off`（Qwen3 思考模式，经 `chat_template_kwargs={"enable_thinking": ...}` 发送）、`--max-tokens`、`--temperature`、`--n`、`--concurrency`（默认 64）、`--retries`（默认 6，指数退避）、`--cache-tag`、`--no-cache`、`--limit`、`--ids-file`、`--no-sandbox`、`--judge-endpoint/--judge-model`。
解码配置在 `configs/tasks.yaml`：默认贪心（temperature 0），AIME24/AIME25/AMC23 为 n=8、temperature 0.6、top_p 0.95（可改）；优先级：命令行 > 任务条目 > defaults。

**缓存**：`eval/.cache/gen/<sha256>.json`，键 = 请求 payload（含 model 名、解码参数）+ `--cache-tag`，**不含端点地址**。因此每一轮量化模型 / 每个 LoRA 必须使用唯一的服务名（或给 `--cache-tag`），否则会读到旧生成。

### 输出
- `<out>/<task>.jsonl`：每个评分单元一行（多数任务一题一行；SQuAD 一问一行、MT-Bench 一轮一行），字段：`id`/`prompt_id`（题目 id，用于 paired bootstrap）、`unit_id`、`pool`、`raw_output`（n>1 时为列表）、`reasoning`、`parsed_answer`、`sample_scores`、`score`（n 个样本的平均）、`metrics`、`status`、`finish_reason`；BFCL 另有 `tool_calls`、`bfcl_category`、`decision`（call / no_call）。
- `<out>/summary.json`：每个任务 `accuracy`、`n`、`n_pending`、`status_counts`、解码配置、模型/LoRA/端点、包版本、processed 文件与 split MANIFEST 的 sha256；n>1 时有 `avg_at_n`/`pass_at_n`/`maj_at_n`；IFEval/IFBench 有 prompt/instruction × strict/loose；TriviaQA/FreebaseQA/SQuAD 有 em/f1/contains；BFCL 有 `per_category`、`call_decisions`、`irrelevance_vs_live_simple`（irrelevance 与 live_simple 的准确率及 call/no_call 计数）。

## 3. 任务与池

| task | 数据集 | 类别 | 可用池 | 评分器 |
|---|---|---|---|---|
| gsm8k | gsm8k | math | gate, test | math-verify（`\boxed{}` 抽取 + 等价判断，字符串兜底） |
| math | math | math | search, gate, reserve（test=MATH-500） | 同上 |
| math500 | math（test 池） | math | test | 同上 |
| aime24 / aime25 / amc23 | 同名 | math | test | 同上，n=8 → avg@8 / pass@8 / maj@8 |
| olympiadbench | olympiadbench | math | test | 同上（多答案需全部匹配） |
| humanevalplus | humanevalplus | code | gate | EvalPlus（base+plus 测试），**仅在 Docker 沙箱内执行** |
| mbppplus | mbppplus | code | search, test | 同上 |
| livecodebench | livecodebench | code | search, test | LCB 官方 `codegen_metrics`，**仅在沙箱内执行** |
| mmlu_pro | mmlu_pro | knowledge | search, gate, test | 选项字母抽取（MMLU-Pro 官方顺序：`answer is (X)` → `Answer: X` → 最后一个独立字母） |
| gpqa_diamond | gpqa_diamond | knowledge | test | 同上（需手动下载数据） |
| triviaqa / freebaseqa | 同名 | knowledge | triviaqa: search/gate/test/reserve；freebaseqa: test | 官方归一化后对别名精确匹配（另报 F1、contains） |
| squad_closed_book_test | squad_val | knowledge_injection | test（SEAL 200 段）, reserve | 不给段落直接回答，EM/F1 |
| squad_closed_book_dev | squad_train | knowledge_injection | search, gate | 同上 |
| ifeval | ifeval | instruction_following | search, gate, test | google-research 官方检查器（主指标 prompt-level strict） |
| ifbench | ifbench | instruction_following | search, gate, test | allenai/IFBench 官方检查器 |
| ifeval_train | ifeval_train | instruction_following | pool, gate（训练集） | `evallib/graders/rlvr_if.py`：RLVR-IFeval 每题 1 个约束，24 个 func_name；15 个直接调用 IFEval 官方检查器，9 个按 open-instruct 语义自写（映射见模块 docstring）；summary 另给 `per_constraint_type` |
| mt_bench | mt_bench | instruction_following | test | 可选：FastChat 单答打分（judge） |
| alpaca_eval | alpaca_eval | instruction_following | test | 可选：AlpacaEval 2 对 gpt4_1106_preview 胜率（judge，logprob 加权） |
| bfcl | bfcl | tool_calling | search, gate, test | 官方 `ast_checker`（FC 模式）；irrelevance=不调用，relevance=至少调用一次 |
| teval | teval | tool_calling | test | T-Eval 官方 evaluator：instruct（args EM）、review、retrieve（name）、understand（args） |
| metatool | metatool | tool_calling | test | 按论文定义复现（官方无评分脚本） |
| apibank | apibank | tool_calling | test | 官方 `parse_api_call` 解析后比对 API 名与参数（level-1/2） |

## 4. 代码执行沙箱（EvalPlus / LiveCodeBench）

生成的代码和 LCB 测试用例（pickle 格式）**绝不在宿主机执行**。`run_eval.py` 只抽取代码、写样本文件，然后启动 Docker 容器：
`--network none --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges --memory 8g --pids-limit 1024`，非 root 用户（uid 65534），数据目录只读挂载。镜像由 `sandbox/Dockerfile` 构建（python:3.11.10-slim-bookworm + evalplus==0.3.1 + LCB evaluator）。

**代码执行自检：PENDING**。本机 Docker daemon 未运行（按要求没有自行启动 Docker Desktop），所以代码题目前只完成了生成 + 代码抽取，记录为 `status=pending_sandbox`、`score=null`。启动 Docker 后执行：

```bash
cd /Users/linghuazhang/Desktop/Project/RSIModelEdit
docker build -t rsi-eval-sandbox:0.1 -f eval/sandbox/Dockerfile eval
# 用 oracle 端到端结果做执行自检：金标准解应接近 100%，错误解应为 0
eval/.venv/bin/python -I eval/grade_code.py --run-dir eval/results/e2e_oracle_gold --task humanevalplus
eval/.venv/bin/python -I eval/grade_code.py --run-dir eval/results/e2e_oracle_wrong --task humanevalplus
```

之后正常运行 `run_eval.py` 时如果 Docker 可用会自动在沙箱里评分；否则 summary.json 的 `sandbox.command` 字段会给出补评命令。

## 5. 验证（已完成，无需 GPU）

```bash
eval/.venv/bin/python -I data/scripts/check_data.py --ngram   # 计数 / 池互斥 / manifest / 去污染
eval/.venv/bin/python -I eval/tests/test_graders.py           # 评分器自检 -> tests/grader_selftest_report.json
eval/.venv/bin/python -I eval/tests/run_e2e.py                # oracle 服务器端到端 -> tests/e2e_report.json
```

- 评分器自检（全量数据，金标准答案 / 错误答案）：gsm8k、math（5000）、aime24/25、amc23、olympiadbench、mmlu_pro（12032）、triviaqa（9960）、freebaseqa（3996）、SQuAD SEAL 子集（974 问）、apibank（534 条 API 调用）均为 1.000 / 0.000；BFCL 单轮 3641 条 0.994 / 0.000；T-Eval 4 个确定性子集 2146 条 0.995 / 0.000；MetaTool 5327 条 1.000 / 0.000；IFEval 6 条、IFBench 3 条、ifeval_train 24 条（每个 func_name 一条，`tests/fixtures/rlvr_if_handmade.json`）手写通过/不通过回复全部判对；HumanEval+/MBPP+ 金标准代码抽取 164/164、378/378（执行 PENDING）。
  - BFCL 未满分的 21 条是 oracle 构造金标准调用时的格式问题（Java HashMap 字符串表示、个别 possible_answer 中必填参数只允许空值等），不是评分器问题。
  - T-Eval 未满分的 11 条：understand_str 中 ground truth 为布尔值 `False`，官方 evaluator 拿它与字符串比较，任何模型都不可能得分（官方行为，保持不变）。
- 端到端（`tests/oracle_server.py` 模拟 vLLM，gold / wrong 两种模式，每类至少一个任务）：全部通过，见 `tests/e2e_report.json`。gold：gsm8k gate 1.000（200）、aime24 1.000（n=8）、mmlu_pro search 1.000（500）、triviaqa search 1.000（500）、SQuAD SEAL test 0.999（974 问，1 条为题面冲突）、ifeval 1.000 / ifbench 1.000（手写样例）、bfcl search 0.990（irrelevance 50/50 no_call，live_simple 49/50）、metatool 1.000、apibank 1.000、teval 1.000、mt_bench 平均 10.0、alpaca_eval 0.990；wrong 模式全部 0.000（mt_bench 1.0、alpaca 0.010）。humanevalplus 两种模式均为 164 条 pending_sandbox，gold 代码抽取 164/164。另验证了缓存（第二次运行 200/200 命中）与 `--lora` 寻址（请求 model=LoRA 名）。oracle 用的是标准库 HTTP 服务器，高并发下会出现连接重置，客户端重试机制会处理。

## 6. 已知注意事项

- **IFEval 的唯一改动**：google-research 检查器在 `count_sentences` 中加载 pickle 版 punkt（`english.pickle`）。为避免加载 pickle，运行时替换为等价的非 pickle `nltk.tokenize.PunktTokenizer("english")`（punkt_tab，同一模型参数），其余逻辑原样调用。`.nltk_data/tokenizers/punkt/` 故意为空目录，以阻止 IFBench 自动下载 pickle 版 punkt。
- **T-Eval**：只把 instruct / review / retrieve / understand 四个确定性子集放进主任务。plan_*、reason_str、reason_retrieve_understand_json 的官方指标需要句向量模型 `all-mpnet-base-v2`（相似度打分），未下载，代码路径会返回 `needs_sentence_embedding_model`。官方 evaluator 依赖的 mmengine/evaluate/termcolor 只在我们用不到的路径里使用，用极小的 stub 代替以免安装 lagent 全家桶。
- **BFCL**：只评 13 个单轮 AST 类别；multi_turn（800）和 agentic（memory 155、web_search 100）需要 BFCL 的有状态执行环境，web_search 还需要 SerpAPI key，`run_eval.py` 将其记为 `not_supported_use_official_cli`（不计入准确率），请用官方 CLI（见 hpc/README.md）。工具名中的 `.` 按官方 OpenAI 风格替换为 `_`，评分前映射回原名再交给 `ast_checker`。vLLM 需开启 `--enable-auto-tool-choice --tool-call-parser hermes`。
- **API-Bank**：官方 evaluator 还会把调用在模拟 API 上执行再比较结果；对作者发布的 JSON 测试文件，我们用官方解析器比对 API 名和参数（空白、引号、大小写归一化）。日期/时间等参数的等价写法可能被判错，属于比官方更严格的一侧。level-3（50 条，需要交互式检索+执行）不评分。
- **MetaTool**：官方仓库没有评分脚本，按论文定义复现：Task1 取回答里第一个 yes/no；Task2 子任务 1/2 要求提到的候选工具集合恰好等于金标准；子任务 3（可靠性，正确工具不在列表中）要求回答 None 且不提任何候选；子任务 4 要求恰好提到两个金标准工具。
- **AlpacaEval 2**：未实现长度控制（LC）胜率，只报原始/加权胜率。MT-Bench / AlpacaEval 是可选项，主实验只用确定性评分。judge 的模型和端点固定在 `configs/tasks.yaml`，密钥通过环境变量（默认 `JUDGE_API_KEY`）读取，不写入任何文件。
- **SQuAD 闭卷**：同一标题下有题面完全相同但答案不同的问题，闭卷时不可区分（oracle 测试中因此有 1/974 冲突）。评分为确定性 EM/F1；SEAL 原文用 LLM 判分，数值不可直接对比。
- **思考模式**：客户端会剥离 `<think>...</think>`；若输出在思考中被截断，则视为没有最终答案。
- **缓存与复现**：采样任务（n>1）向 vLLM 传 `seed=20261118`；结果仍可能受 vLLM 批处理顺序影响而有微小差异。
- `eval/.cache/`、`eval/results/` 可以随时删除（e2e 结果在 `results/e2e_oracle_{gold,wrong}/`）。
