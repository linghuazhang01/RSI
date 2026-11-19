# rsicore 接口契约（三位子 agent 共同遵守）

2026-11-19。`eval/rsicore/` 是两篇共用的运行时；`edit_code/`（Paper A 自量化）和 `lora_code/`（Paper B 自写 LoRA）只写各自论文特有的部分，通过下面的接口调用共用运行时。**接口签名改动必须先改本文件，并在最终报告里写明。**

设计依据：`plan/PLAN.md`、`plan/RUN_PIPELINE.md`、`plan/research/RSI_LOOP_DESIGN.md`、`data/README.md`。

---

## 0. 目标环境与总规则

- GPU 环境：verl 0.9.1 · vLLM 0.24.0 · Transformers 5.9.0 · PyTorch 2.11.0+cu130，Python 3.11（`eval/hpc/requirements-gpu.lock`，`./setup_env.sh gpu`）。模型：Qwen3.8-27B（`Qwen3_5ForConditionalGeneration`，dense 64 层，线性注意力（GDN）与全注意力混合；vLLM 的 LoRA 打包映射见 `vllm/model_executor/models/qwen3_5.py`：`qkv_proj←q/k/v_proj`、`gate_up_proj←gate/up_proj`、`in_proj_qkvz←in_proj_qkv/in_proj_z`、`in_proj_ba←in_proj_b/in_proj_a`）。
- 本机（macOS，无 GPU）只能跑 **mock 模式**：`--mock` 时不加载真实模型，用 mock OpenAI 服务和 mock 构建器把整条流程走通。**所有代码都必须能在 mock 模式下端到端跑完**；真实 GPU 路径写好、做语法和单元检查，但无法在本机实跑，报告里要写明。
- Python 一律 `python -I` 运行；模型写的程序只在沙箱里执行；不执行任何下载来的代码。
- 一切随机性由 `RunConfig.seed`（默认 42）派生；数据切分种子 20261118 不变。
- 每一步都要可断点续跑：同一个 run-id 再运行，跳过已完成的步骤。

## 1. 启动

```bash
./run_rsi.sh --paper A --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id A_math_s42 [--mock]
./run_rsi.sh --paper B --domain code --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id B_code_s42 [--mock]
```

- `run_rsi.sh`（根目录，core 负责）：解析参数 → 选 Python（默认 `~/venvs/rsi-gpu/bin/python`，`--mock` 时用 `eval/.venv/bin/python`，可用 `RSI_PYTHON` 覆盖）→ `python -I edit_code/run.py ...` 或 `python -I lora_code/run.py ...`，参数原样透传。
- `--model` 可以是 `download_models.sh` 里的名字（解析为 `${MODELS_DIR:-$HOME/models}/<name>`）或路径。
- `--domain`：`math | code | knowledge | instruction_following | tool_calling`，Paper A 另有 `all`。

## 2. 运行目录 `experiment_records/<run-id>/`

```
run_config.yaml          合并后的配置（首次运行写入并锁定；之后参数不一致则报错）
state.json               已完成步骤 {key: info}（RunState）
draws/                   data/scripts/draw.py 的输出（gate/、gen_XX/、ledger.json）
baseline/                第 0 代：原模型在验收集和留出集上的结果、噪声带
gen_XX/
  editor/thread_K.jsonl  改造者对话记录
  candidates.jsonl       每个候选：cid, thread, step, program 路径, artifact 摘要, screen, full, score, cost
  programs/<cid>.py      模型写的程序
  feedback/<cid>.json    回给改造者的 L1 反馈
  gate.json              每次提交的结果
  accepted.json          接任的候选（或 null + 原因）
  model/                 接任后的模型（A：量化权重；B：合并 LoRA 后的权重）或指向它的路径
  heldout/               留出评测（run_eval.py 的 out 目录）
  e_t/                   改造者能力测量
summary.json             每代汇总（逐代追加）
```

## 3. `eval/rsicore` 模块与签名（core agent 实现）

```python
# config.py
@dataclass
class RunConfig:
    paper: str            # "A" | "B"
    domain: str
    gens: int
    run_id: str
    run_dir: Path
    model: str            # 名字或路径（原样）
    model_path: Path      # 解析后的路径（mock 时可不存在）
    seed: int
    gpus: list[int]
    mock: bool
    draw_config: Path     # data/draw/paper_a.yaml 或 paper_b.yaml
    paper_cfg: dict       # 论文 base.yaml + domains/<domain>.yaml 合并后的内容
def build_arg_parser(paper: str) -> argparse.ArgumentParser   # 通用参数；论文可再加参数
def load_run_config(paper: str, args) -> RunConfig            # 合并 <paper_dir>/configs/base.yaml、domains/<domain>.yaml、CLI；写入 / 校验 run_config.yaml

# state.py
class RunState:
    def __init__(self, run_dir: Path): ...
    def done(self, key: str) -> bool: ...
    def get(self, key: str) -> dict | None: ...
    def mark(self, key: str, info: dict | None = None) -> None: ...   # 原子写 state.json
    def step(self, key: str):  # context manager -> 返回 bool（True=需要执行）；正常退出才 mark

# records.py
def gen_dir(rc: RunConfig, gen: int) -> Path
def write_json(path: Path, obj) -> None          # 原子写
def append_jsonl(path: Path, obj) -> None
def update_summary(rc: RunConfig, gen: int, info: dict) -> None

# draws.py   （包装 data/scripts/draw.py，子进程调用）
DOMAIN_TRAIN = {"math": ["dapo_math"], "code": ["taco_verified"], "knowledge": ["triviaqa_train"],
                "instruction_following": ["ifeval_train"], "tool_calling": ["sealtool_train"]}
def ensure_gate(rc) -> dict          # {domain: {dataset: [ids]}}（draw.py --gate-only）
def draw_gen(rc, gen: int, target: str | None = None) -> dict    # 返回 manifest
def ids(rc, gen: int, dataset: str, role: str) -> list[str]      # role: signal|screen|train|kl
def gate_ids(rc, dataset: str) -> list[str]
def ids_file(rc, gen: int, dataset: str, role: str) -> Path
def gate_ids_file(rc, dataset: str) -> Path
def mark_shown(rc, gen: int, shown: dict[str, list[str]]) -> None

# servers.py
class VLLMServer:  # context manager
    def __init__(self, model_path, served_name: str, gpu_ids: list[int], *, port: int | None = None,
                 enable_lora: bool = False, max_loras: int = 8, max_lora_rank: int = 64,
                 extra_args: list[str] = (), mock: bool = False, mock_registry: Path | None = None,
                 seed: int = 42, log_dir: Path | None = None, ready_timeout_s: float = 3600,
                 host: str = "127.0.0.1"): ...      # 后四个为 core 增补的可选参数（2026-11-19）
    endpoint: str                      # "http://127.0.0.1:PORT/v1"
    served_name: str; loras: dict[str, str]; def start() / stop() / alive() / log_tail()   # 增补
    def load_lora(self, name: str, path: Path) -> None     # /v1/load_lora_adapter
    def unload_lora(self, name: str) -> None
# 真实路径：vllm serve（--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --seed <seed>），
# CUDA_VISIBLE_DEVICES=gpu_ids，等待 /v1/models 就绪，日志写 run_dir/logs/。
# mock 路径：启动 eval/rsicore/mock_server.py（见 §5）。

# evaluate.py
@dataclass
class EvalResult:
    summary: dict              # run_eval.py summary.json 里该任务的条目
    records: list[dict]        # 每题记录（含 id, score）
    out_dir: Path
def score(endpoint, served_name, task: str, *, pool: str, ids: list[str] | None, out_dir: Path,
          lora: str | None = None, extra: list[str] = (), mock: bool = False,
          seed: int | None = None, temperature: float | None = None) -> EvalResult
    # 子进程调用 eval/run_eval.py（--ids-file 临时文件、--no-cache、--quiet），代码任务在 Docker 不可用时状态为 pending
    # seed / temperature（2026-11-19 增补）→ run_eval --seed / --temperature，覆盖任务的解码设置；seed 只在 temperature > 0 时发给服务
def paired_delta(a: list[dict], b: list[dict], seed: int = 42, n_boot: int = 1000) -> dict
    # 按 id 配对，返回 {"delta", "ci": [lo, hi], "n"}
def reference_logprobs(endpoint, served_name, texts: list[str], cache: Path, top_k: int = 20) -> Path
    # vLLM prompt_logprobs，结果缓存到 cache（jsonl）
def token_kl(ref_cache: Path, endpoint, served_name, texts: list[str], lora: str | None = None,
             top_k: int = 20) -> dict   # {"mean": float, "per_text": [...]}（top-k 近似 KL(ref || cand)）
def domain_eval_tasks(domain: str) -> list[str]   # 训练集对应的 run_eval 任务名（与 DOMAIN_TRAIN 一致）

# sandbox.py
class BudgetExceeded(Exception): ...
class ToolAPI:
    def __init__(self, budget: dict[str, float]): ...     # 例如 {"evaluate": 24, "gpu_seconds": 7200}
    def register(self, name: str, fn, *, cost: dict[str, float] | None = None, doc: str = "",
                 cost_fn=None, dedupe_key_fn=None) -> None   # cost_fn / dedupe_key_fn：2026-11-19 增补
        # cost：每次调用的固定费用，调用前检查并扣除
        # cost_fn(kwargs, result) -> dict[str, float]：调用成功后按实测扣费；扣完超预算则抛 BudgetExceeded（结果作废，费用保留）。
        #   有 cost_fn 的工具在调用前还要求：cost 里声明的键（值为 0 表示纯实测键）和它以前实测扣过的键都没有用完，否则直接拒绝
        # dedupe_key_fn(kwargs) -> str | None：同一工具同一个非 None 键的后续调用直接返回第一次成功调用的缓存结果，免费（不检查预算）
    def call(self, name: str, kwargs: dict): ...          # 扣预算后调用；超预算抛 BudgetExceeded（见上）
    def describe(self) -> str                            # 给改造者看的工具说明
    spent: dict[str, float]
@dataclass
class ProgramResult:
    ok: bool
    result: object | None      # main(api) 的返回值（必须可 JSON 序列化）
    stdout: str
    stderr: str
    error: str | None
    seconds: float
    info: dict = {}            # 增补：limits（实际生效的 rlimit）、network_isolation、tool_calls、returncode
def run_program(code: str, api: ToolAPI, *, timeout_s: int, mem_gb: int, workdir: Path) -> ProgramResult
    # 模型写的程序定义 def main(api): ...；在子进程（python -I、清空环境变量、rlimit、Linux 上尽量 unshare -n 断网）里运行；
    # 子进程里的 api.<tool>(**kwargs) 通过管道以 JSON-RPC 转发给父进程的 ToolAPI.call；只有注册过的工具可用。

# editor.py
class Editor:
    def __init__(self, endpoint, served_name, *, system_prompt: str, seed: int, transcript: Path,
                 temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 8192,
                 mock_policy=None, extra_body: dict | None = None, timeout_s: float = 3600): ...  # 后两个为增补
    def propose(self, user_msg: str) -> str     # 发一轮对话，返回模型回答里最后一个 ```python 代码块（没有则返回 ""）
# mock_policy：callable(history: list[dict]) -> str，返回一段程序文本（mock 模式下代替真模型）

# search.py
@dataclass
class Candidate:
    cid: str; thread: int; step: int; program: str
    artifact: dict | None; error: str | None
    screen: dict | None; full: dict | None; score: float | None; cost: dict
def inner_search(rc, gen: int, *, editor_factory, render_context, build_fn, screen_fn, full_fn,
                 threads: int, steps: int, max_candidates: int, top_full: int = 3, rerun_seeds: int = 0,
                 out_dir: Path | None = None) -> list[Candidate]
    # editor_factory(thread)->Editor；render_context(history: list[Candidate], thread)->str（论文自己写反馈格式）
    # build_fn(program, cid)->(artifact|None, error|None)；screen_fn(artifact, cid)->(score, feedback)；full_fn 同
    # 写 <out_dir>/candidates.jsonl、programs/、feedback/（out_dir 默认 gen_dir(rc, gen)）；返回按 full（无则 screen）分数降序
    # out_dir（2026-11-19 增补）：同一代的额外搜索（E_t，Paper B 每个分支一个）传自己的目录，例如 gen_XX/e_t[/<branch>]，
    # 不要再用 dataclasses.replace(rc, run_dir=...)（会产生 gen_XX/e_t/gen_XX/ 嵌套）；续跑语义不变（按 out_dir 里的记录跳过）
def submit_to_gate(rc, gen: int, ranked: list[Candidate], gate_fn, max_submissions: int = 3,
                   out_dir: Path | None = None) -> Candidate | None
    # gate_fn(candidate)->(passed: bool, info: dict)；写 <out_dir 或 gen_XX>/gate.json
def load_candidates(rc, gen: int, out_dir: Path | None = None) -> dict[str, Candidate]   # 与 inner_search 用同一个 out_dir

# gate.py
def noise_band(results_runs: list[list[dict]]) -> float     # 同一模型重复评测的分数波动（max - min）
def measure_noise_band(endpoint, served_name, task: str, ids: list[str] | None, out_dir: Path, *, repeats: int = 3,
                       temperature: float = 0.6, base_seed: int = 42, mock: bool = False, lora: str | None = None,
                       pool: str = "pool") -> dict   # 2026-11-19 增补
    # 用 score(seed=base_seed+k, temperature=temperature) 重复评测 repeats 次（输出 out_dir/seed<seed>/）；
    # 返回 {"band": max-min, "se": sqrt(p(1-p)/n)（p=各次均值的平均，n=有分数的题数）, "scores", "seeds", "mean", "n", "temperature", "task"}
    # 贪心（temperature=0）重复评测在 vLLM 和 mock 里都完全相同，band=0
def relative_pass(score: float, baseline: float, ratio: float) -> bool

# heldout.py
HELDOUT = {"math": [("math500", "test"), ("gsm8k", "test"), ("aime24", "test"), ("aime25", "test")],
           "code": [("humanevalplus", "test"), ("mbppplus", "test"), ("livecodebench", "test")],
           "knowledge": [("gpqa_diamond", "test"), ("mmlu_pro", "test"), ("triviaqa", "test")],
           "instruction_following": [("ifeval", "test"), ("ifbench", "test")],
           "tool_calling": [("sealtool", "test"), ("apibank", "test"), ("bfcl", "test")]}
def run_heldout(endpoint, served_name, out_dir: Path, *, domains="all", lora=None, mock=False, limit=None) -> dict
```

### 3.1 core 实现说明（2026-11-19，均为增补，原签名不变）

- `config`：CLI 另有 `--runs-dir`（默认 `experiment_records/`）、`--paper-dir`（覆盖读取 configs/ 的目录）、`--set KEY=VALUE`（可重复，点号键，YAML 值，并入 paper_cfg）。`run_config.yaml` 锁定 paper/domain/model/model_path/seed/mock/draw_config/paper_cfg；`gens`、`gpus` 可改（延长运行、换机器）。`RunConfig.cli_extra` 记录论文自加的 CLI 参数（不锁定）。`config.eval_python()`：run_eval.py / draw.py 用的解释器（`RSI_EVAL_PYTHON` > `eval/.venv/bin/python` > 当前解释器）。真实模式下 model_path 不存在直接报错。
- `state.RunState`：另有 `unmark(key)`、`keys()`；`step()` 块内自己调用 `mark(key, info)` 不会被覆盖。
- `records`：另有 `read_json(path, default)`、`read_jsonl(path)`；`summary.json = {"run_id","paper","domain","gens": {"<g>": {...}}}`。
- `draws`：已抽过的代 / 已切过的 gate 直接读文件，不再起子进程；`ids()` 文件不存在返回 `[]`。
- `servers.VLLMServer` 真实命令：`vllm serve <model> --served-model-name N --host 127.0.0.1 --port P --seed S --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --max-model-len 40960 --tensor-parallel-size len(gpu_ids) [--enable-lora --max-loras --max-lora-rank] <extra_args>`（extra_args 里给了 `--max-model-len` / `--tensor-parallel-size` 就不加默认值）；vllm 可执行文件：`RSI_VLLM` > 解释器同目录的 `vllm` > PATH > `python -m vllm.entrypoints.cli.main`。日志 `<log_dir>/<served_name>.<port>.log`。mock 时 extra_args 原样传给 mock_server（如 `["--tasks", "dapo_math,math500"]`）。`load_lora` 同名同路径重复调用是空操作。
- `evaluate`：`EvalResult.accuracy` 属性；`ids=[]` 直接返回 n=0；mock 时自动加 `--no-sandbox`（不启动 Docker）。同一 out_dir 不要并发 score（run_eval 合并写 summary.json）。**`paired_delta(a, b)` 的 delta = mean(b − a)**（b=候选，a=参照）；同一 id 的多个单元先平均，score=None 跳过。`token_kl` 另返回 `token_mean`、`n_tokens`；KL 在"参照报告的 top-k token + 其余"划分上计算（数据处理不等式下是真 KL 的下界），候选没报告的 token 用候选最小 logprob 代替。
- `sandbox.ToolAPI`：另有 `charge(key, amount)`（直接记账，不检查）、`remaining()`、`names()`、`calls`、`log`（每条 `{tool, ok, seconds}`，实测费用另有 `measured`，去重命中为 `deduped: true`）。实测费用 / 去重见 §3 的 `register`；`describe()` 里标注 `measured: <keys>` 和 `repeated identical calls are free`。超时是总墙钟时间（含工具调用耗时，在两条 RPC 消息之间检查）。子进程 = 同一解释器 `python -I`；文件写只允许 workdir，读禁止仓库根 / $HOME / /proc（解释器自身前缀除外）；禁止 subprocess/fork/exec/kill/socket connect|bind|getaddrinfo（Python audit hook，C 扩展可绕过，属于纵深防御）。断网：Linux `unshare -rn`，macOS `sandbox-exec (deny network*)`，都不可用则 `info["network_isolation"]="none ..."`。程序里可用 `api.BudgetExceeded`、`api.ToolError`（全局名 `BudgetExceeded`、`ToolError` 也可用），工具只能用关键字参数。
- `editor.Editor`：transcript 在构造时重新载入；`propose()` 收到与下一条记录相同的 user 消息时直接重放记录的回答（不调模型），消息不同则从该处截断——所以续跑的内层搜索会得到完全相同的程序。`extra_body` 并入请求（如 `{"chat_template_kwargs": {"enable_thinking": true}}`）。历史里只放 content（reasoning 只写入 transcript）。
- `search`：cid = `g{gen:02d}_t{thread}_s{step:02d}`；步骤按 step 轮转所有线程；`screen = {"score", "feedback"}`，`full = {"first", "reruns", "score", "feedback"}`（score = first 与 reruns 的平均，rerun 调用为 `full_fn(artifact, f"{cid}_r{i}")`）；`cost` 含 propose_s/build_s/screen_s/full_s 并合并 `artifact["cost"]`。build_fn 抛异常记为该候选的 error；screen_fn / full_fn 的异常向上抛（视为基础设施故障，续跑时重做该候选）。程序为空（无 ```python 块）时不调用 build_fn。另有 `load_candidates(rc, gen) -> {cid: Candidate}`、`ranked(cands)`、`Candidate.to_record()`。`submit_to_gate` 跳过 error / 无分数的候选，gate.json = `{"submissions": [{cid, passed, info, t}]}`，续跑时已提交的不再提交，已通过的直接返回。
- `gate.measure_noise_band`：mock 下 temperature > 0 时答案依赖请求的 seed（见 §5），所以 band > 0；论文用它算第 0 代噪声带（Paper A：`baseline/noise.json` = {dataset: band}，详情 `noise_detail.json`；Paper B：gate 余量 = max(band, 2×SE)）。`run_eval.py --seed N` 覆盖任务 / 默认 seed（与 `--temperature` 同样写入 `decoding`）；seed 只在 temperature > 0 时进入请求体，因而也进入生成缓存键（缓存键 = 请求体）。
- `heldout`：另有 `heldout_tasks(domains)`；out_dir/summary.json 里已有同一服务名 / LoRA / limit 的任务会跳过。
- `mock_server`：默认索引全部训练集 + 留出任务（首次约 5 s，之后读 `eval/.cache/mock_index/` 缓存约 1 s）；`--tasks` 只索引指定任务——**不在索引里的请求回答 "UNKNOWN REQUEST"（得 0 分）**，所以 mock 下要跑留出评测的服务器不要把 `--tasks` 限制成只有训练集。ifeval_train 的"金标准"回复按约束构造（全部通过判分器）；留出的 IFEval/IFBench 只有手写样例，mock 下大多为 0 分。未知模型名返回 vLLM 式 404；`load_lora_adapter` 需 `--enable-lora`，且 lora_path 必须存在。
- 自测：`eval/.venv/bin/python -I eval/tests/rsicore/test_core.py`；玩具论文（不依赖 edit_code/lora_code 的整条闭环）：`eval/tests/rsicore/toy_paper.py`。

## 4. 论文侧（各自实现）

### Paper A：`edit_code/`（A agent）
- `edit_code/run.py`（2026-11-19 修订为固定体积）：`main()` 用 `rsicore.config.build_arg_parser("A")`（加 `--budget-gib`）。第 0 代：原模型的验收分、噪声带、留出评测，原模型的逐层量化缓存和诊断表（只算一次），定体积，Q_0 = 参照配置（第一个 incumbent）。逐代：draw → 原模型参照 → incumbent 在本代题上的分数 → `inner_search`（改造者默认是冻结的原模型）→ 只把分数高于 incumbent 的候选交 `submit_to_gate`（验收 = 在验收集上逐题配对超过 incumbent）→ 通过则导出新 incumbent 并做留出评测，否则保留；连续 `loop.patience` 代无提升就停。
- 配置：`edit_code/configs/base.yaml`、`edit_code/configs/domains/<domain>.yaml`（含 `all.yaml`）。`budget`（一个体积）、`loop.editor`（frozen / current）、`hardware`（b300 / h20）。
- 候选格式：**模拟量化（fake-quant）**——权重量化再反量化成 BF16 写成普通 checkpoint，vLLM 照常加载，用于测质量；大小按配置解析计算。速度只对 vLLM 有内核的格式另行测量。
- mock：不加载真实权重；候选 artifact 只记录配置与一个由配置算出的 `quality`（越接近原精度越高），写入 mock registry。

### Paper B：`lora_code/`（B agent）
- `lora_code/run.py`：逐代：draw（`--target <domain>`）→ 诊断表 → `inner_search`（工具：`svd`、`acts`、`grad`、`sgd`、LoRA 栈操作、`evaluate(lora)`）→ `submit_to_gate` → 把 LoRA 合并进权重得 M_{t+1} → 对照分支（SFT 基线、非 LLM 搜索）→ E_t（固定任务：为原始基座在 Seal-Tools 上写 LoRA；工具调用领域单独跑时换成代码领域）→ 留出评测。
- 配置：`lora_code/configs/base.yaml`、`lora_code/configs/domains/<domain>.yaml`。
- LoRA 存成 PEFT 格式（`adapter_config.json` + `adapter_model.safetensors`），目标模块名要能被 vLLM 0.24.0 的 Qwen3.5 LoRA 加载；梯度工具用 transformers + peft（只能用本代 `train` 角色的题）。
- mock：不加载真实权重；LoRA artifact 是一个小 JSON（含由程序参数算出的 gain / damage），登记到 mock registry 的是按领域的 dict：目标领域 = 父模型该领域 + gain，其余领域 = 父模型 − damage（2026-11-19），合并后的 M_{g+1} 继承这个 dict。

## 5. mock 模式（core agent 提供 `eval/rsicore/mock_server.py`）

- OpenAI 兼容：`/v1/models`、`/v1/chat/completions`、`/v1/completions`（支持 `prompt_logprobs` 的确定性假值）、`/v1/load_lora_adapter`、`/v1/unload_lora_adapter`。
- 答题：复用 `eval/tests/oracle_server.py` 的索引思路，对每个请求以概率 `quality` 返回标准答案、否则返回错误答案（用 `hash(seed, 题目) < quality` 决定，确定性；hash 里**不含**模型名，即所有模型对同一题用同一个随机数，同一批题上质量高的模型答对的题包含质量低的模型答对的题，分数随质量单调变化，2026-11-19）。贪心请求（temperature 为 0 / 缺省）不看请求 seed；temperature > 0 的请求再把请求的 `seed` 和 `temperature` 放进 hash，所以不同 seed 的重复评测会不同，同一 (seed, temperature, 题目) 仍确定（2026-11-19）。
- `quality` 从 mock registry（JSON 文件：`{served_name 或 lora 名: value}`）读取；没登记的名字用默认 0.7。value 可以是 float（所有请求同一个质量），也可以是按领域 / 任务的 dict，例如 `{"default": 0.6, "math": 0.7, "knowledge": 0.55, "dapo_math": 0.72}`，键为领域（math / code / knowledge / instruction_following / tool_calling，即 tasks.yaml 的 `category`）、run_eval 任务名或 "default"；查找顺序 任务名 > 领域 > default > 各值平均；也可包成 `{"quality": value, ...}`。假 prompt_logprobs / KL 用 `mock_quality.scalar(value)`（五个领域解析后的平均；只有 float / default 时就是它）。解析函数在 `eval/rsicore/mock_quality.py`（`resolve`、`scalar`、`as_map`、`unwrap`、`CATEGORY_DOMAIN`，只依赖标准库）。
- 改造者对话：请求里带 `"metadata": {"mock_editor": "<paper>"}` 或 served_name 以 `editor` 结尾时，返回由论文侧注册的 mock 程序（core 提供注册机制：论文在 mock 模式下把 `mock_policy` 交给 `Editor`，`Editor` 直接调用，不经过服务）。
- 要求：`./run_rsi.sh --paper A --domain math --gens 2 --mock` 和 `./run_rsi.sh --paper B --domain math --gens 2 --mock` 在本机几分钟内跑完，产出完整的运行目录；第二次运行同一个 run-id 应全部跳过。
