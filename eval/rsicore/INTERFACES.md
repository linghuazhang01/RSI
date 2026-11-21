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
def check_decoding(dec, where="decoding", extra_keys=()) -> dict | None   # 2026-11-21：校验解码 dict（去掉 null 值；None / {} -> None）
def stage_decoding(cfg: dict, stage: str, run_seed: int | None = None) -> dict | None
    # 2026-11-21：stage = screen | full | gate；读论文配置 decoding.<stage>（full 没有时用 screen），null = tasks.yaml（贪心）；
    # seed 为 null 时 = run_seed。参照（incumbent / parent / M_0）和候选必须用同一个 dict（共同随机数）

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
# objective simple（2026-11-21，不走 draw.py）：
def build_eval_set(name: str) -> list[str]       # build_eval_target.build(name)（每进程一次）+ 返回排序后的 id
def is_simple(rc) -> bool; def simple_pool(rc) -> str | None; def test_dataset(rc) -> str | None
def simple_screen_n(rc) -> int                    # simple.screen_n > data.screen_n > 500
def simple_draw(rc, gen: int) -> dict             # {signal, screen, train, kl: [ids]}；写 draws/gen_XX/ids/simple_<d>_pool.<role>.json

# servers.py
class VLLMServer:  # context manager
    def __init__(self, model_path, served_name: str, gpu_ids: list[int], *, port: int | None = None,
                 enable_lora: bool = False, max_loras: int = 8, max_lora_rank: int = 64,
                 extra_args: list[str] = (), mock: bool = False, mock_registry: Path | None = None,
                 seed: int = 42, log_dir: Path | None = None, ready_timeout_s: float = 3600,
                 host: str = "127.0.0.1", sleep_mode: bool = False,
                 tool_call_parser: str = "auto"): ...   # log_dir 起为 core 增补（2026-11-19/20）；tool_call_parser 2026-11-21
    endpoint: str                      # "http://127.0.0.1:PORT/v1"
    served_name: str; loras: dict[str, str]; def start() / stop() / alive() / log_tail()   # 增补
    def load_lora(self, name: str, path: Path) -> None     # /v1/load_lora_adapter
    def unload_lora(self, name: str) -> None
def max_model_len(args, default=40960) -> int | None          # 2026-11-21：从 vLLM 参数里读 --max-model-len
def set_max_model_len(args, n) -> list[str]                     # 替换（或追加）--max-model-len；n=None 原样返回
def model_max_positions(model_path) -> int | None               # config.json（或 text_config）的 max_position_embeddings
def detect_tool_call_parser(model_path) -> str                  # chat template 含 "<function=" -> qwen3_coder，否则 hermes
# 真实路径：vllm serve（--enable-auto-tool-choice --tool-call-parser <auto 检测> --reasoning-parser qwen3 --seed <seed>），
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
          seed: int | None = None, temperature: float | None = None, top_p: float | None = None,
          top_k: int | None = None, decoding: dict | None = None) -> EvalResult
def score_split(endpoints: list[str], served_name, task: str, *, pool, ids, out_dir, lora=None, extra=(), mock=False,
                seed=None, temperature=None, top_p=None, top_k=None, decoding: dict | None = None) -> EvalResult
    # decoding（2026-11-21）：{temperature, top_p, top_k, min_p（只能 0）, n, thinking, max_tokens, seed} 任意子集，None = tasks.yaml；
    # 显式的 seed/temperature/top_p/top_k 参数优先于它，它的 max_tokens 优先于 max_tokens 表，extra 里的 CLI 参数优先于一切。
    # n > 1：每条记录 score = n 个样本的平均（avg@n，[0,1]），"scores" = 样本分列表；summary accuracy = 记录分的平均
    # （score_split 另给 avg_at_n / pass_at_n）。thinking：--thinking on（chat_template_kwargs enable_thinking），只判最终答案；
    # 思考中被 max_tokens 截断的样本（finish_reason length、没有 reasoning、content 里没有 </think>）记 0 分（记录 truncated_thinking）
    # 记录保留每个样本的思考文本：rec["reasoning"]（n > 1 时为每样本一项的列表，n == 1 时为字符串，与 raw_output 同形；来自服务器的
    #   reasoning_content，qwen3 reasoning parser；mock_server 在 enable_thinking 时也返回）。screen_view.py（2026-11-21）读它：
    #   summary / rows / output(rec, sample, offset) 给改造者看参考模型的 screen 结果（含完整思考文本）。
    # 子进程调用 eval/run_eval.py（--ids-file 临时文件、--no-cache、--quiet），代码任务在 Docker 不可用时状态为 pending
    # seed / temperature（2026-11-19 增补）→ run_eval --seed / --temperature，覆盖任务的解码设置；seed 只在 temperature > 0 时发给服务；top_p（2026-11-20）→ --top-p
def set_max_tokens(spec: dict | None) -> dict[str, int]   # 2026-11-20：本次运行每个任务的最大生成长度
    # 论文配置的 max_tokens: {任务名: token}；之后每次 score() 对列出的任务加 --max-tokens（extra 里已有则不加）；未知任务名报错
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
def run_program(code: str, api: ToolAPI, *, timeout_s: int, mem_gb: int, workdir: Path,
                save_output: bool = False) -> ProgramResult
    # save_output（2026-11-21）：子进程结束后把完整 stdout / stderr 复制到 workdir/stdout.txt、stderr.txt（ProgramResult 只保留最后 64 KiB）
    # 模型写的程序定义 def main(api): ...；在子进程（python -I、清空环境变量、rlimit、Linux 上尽量 unshare -n 断网）里运行；
    # 子进程里的 api.<tool>(**kwargs) 通过管道以 JSON-RPC 转发给父进程的 ToolAPI.call；只有注册过的工具可用。
RESTARTED = "session restarted: variables lost"
class Kernel:                                         # 2026-11-21：持久沙箱内核（editor.explore.persistent）
    def __init__(self, workdir: Path, *, mem_gb: float, extra_allow_read=(), extra_env=None): ...
    def run(self, code: str, api: ToolAPI, *, timeout_s: float, subdir: str = "call") -> ProgramResult
        # 第一次 run 时懒启动一个长驻子进程（限制与 run_program 相同：rlimit、audit hook、断网、禁 subprocess；另禁 resource.setrlimit）；
        # 每次在同一个 globals 里 exec code（变量 / import / 函数保留；api 是持久全局，工具调用 RPC 到传入的 ToolAPI）；
        # 每次调用前清掉 result（设 result = ... 返回 JSON 值；本次新定义的 main(api) 在没设 result 时被调用）；
        # 输出写 workdir/<subdir>/program.py、stdout.txt、stderr.txt；RLIMIT_CPU 软限制每次调到「已用 CPU + timeout_s + 5」。
        # 超时（墙钟）或子进程崩溃 -> 杀掉内核，error 以 RESTARTED 结尾、info["restarted"] = True；下一次 run 启动新内核。
        # info 另有 kernel、fresh_kernel、kernel_starts。单个实例不是线程安全的（一个提案一个内核）。
    def close(self) -> None                           # 发 exit，必要时 SIGKILL 进程组；可重复调用
    alive: bool

# editor.py（2026-11-21 改为 thinking + 工具轮次；旧调用方式仍然有效）
class EditorReplyError(RuntimeError): ...       # 本次提案拿不到回答（压缩 + 强制最终轮之后仍是上下文错误）；search 只吞这一种
class Editor:
    def __init__(self, endpoint, served_name, *, system_prompt: str, seed: int, transcript: Path,
                 temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 8192,
                 mock_policy=None, extra_body: dict | None = None, timeout_s: float = 3600,
                 tools=None, tool_rounds: int = 50, turn_max_tokens: int = 8192, max_tool_calls: int = 100,
                 proposal_tokens: int | None = 160000, proposal_wall_s: float | None = 2400,
                 context_tokens: int | None = None, tool_result_chars: int = 6000, compact_at: float = 0.6,
                 keep_tool_rounds: int = 8, keep_reasoning_rounds: int = 2, reasoning_passback: str = "think_tag",
                 history_keep: int | None = None, clock=None): ...
    def propose(self, user_msg: str, ctx: dict | None = None) -> str
        # ctx = {"cid", "gen_dir", "visible_cids"}（search 传入，交给工具）；返回程序文本（没有则 ""）
    last_stats: dict     # 本次提案：rounds, tool_calls, tool_s, prompt_tokens, completion_tokens, wall_s, stop,
                         # model_calls, nudges, compactions, finish_reason, forced_by（重放时取记录里的值，另有 replayed=True）
    n_calls: int         # 本实例实际发出的模型调用数（每一轮都算；重放不算）
def editor_settings(cfg) -> dict                      # 论文配置 editor: 块叠加 EDITOR_DEFAULTS 后的完整设置（含 explore、mock_tool_rounds）
def editor_kwargs(cfg, server_args=None) -> dict      # Editor(...) 的关键字参数（采样、extra_body、max_tokens、timeout_s、循环键、context_tokens）
def context_tokens(cfg, server_args=None) -> int | None   # editor.context_tokens（默认 131072）；显式 null -> server_args 的 --max-model-len
def apply_context_tokens(args, cfg) -> list[str]      # 改造者所用服务器的 vLLM 参数，--max-model-len 换成 editor.context_tokens（覆盖 args 里原有的值）
def describe_stop(stats) -> str                       # 空程序的原因文字，例如 "final reply cut at max_tokens while thinking"
def extract_python(text: str) -> str
# mock_policy：callable(messages: list[dict]) -> str（程序文本，没有代码块时自动包一层）或 dict
#   {"content", "tool_calls": [{"name", "arguments"}], "reasoning"}；mock 下 id 合成为 call_{p}_{r}_{i}，守卫 / 强制最终轮同样生效

# chat_tools.py（2026-11-21 新增）：改造者写候选时可调用的对话工具
class ChatTools:
    def __init__(self, explore_api_factory, *, timeout_s: float = 300, mem_gb: float = 8, result_chars: int = 6000,
                 run_python_doc: str | None = None, read_record: bool = True,
                 history: SearchHistory | None = None, persistent: bool = False): ...
    def begin(self, ctx) -> None                      # 提案实时开始（Editor 调用）：先 end(ctx) 丢弃同 cid 的旧内核，再删 gen_dir/explore/<cid>/
    def end(self, ctx) -> None                        # 2026-11-21：提案结束（Editor 在 propose 的每条退出路径上调用）：关内核、跑 cleanup；从不抛异常
    def end_all(self) -> None                         # 关掉所有还开着的持久会话
    def schemas(self) -> list[dict]                   # OpenAI tool schema：run_python(code)、read_record(cid, part, offset=0)、
                                                      # 有 history 时再加 search_history(gen=None, sort="score", offset=0)
class SearchHistory:                                  # 2026-11-21：search_history / 跨代 read_record 的数据来源
    def __init__(self, past: dict | None = None, *, row_fn=None, key_fn=None, gen_fn=None): ...
    # past {代号: 该代搜索目录}；row_fn(候选记录) -> dict 论文自己的一行摘要；key_fn(记录) -> 结果身份（同值 = same_as）；
    # gen_fn(代号, 目录) -> dict 该代结果（接受了谁、验收原因）。SearchHistory({}) = 不给往期（E_t）
    def call(self, name, arguments, ctx, r, i, deadline_s=None) -> dict   # {"content": JSON 文本, "ok", "seconds", "args"}；从不抛异常
def chat_tools_kwargs(cfg) -> dict                    # {"timeout_s", "mem_gb", "result_chars", "persistent"}：editor.explore.* 与 editor.tool_result_chars
RUN_PYTHON_DOC / RUN_PYTHON_DOC_PERSISTENT; def run_python_doc(persistent=False) -> str   # 持久模式下以 RUN_PYTHON_DOC 开头的 doc 自动换成持久版前缀
def with_mock_exploration(policy, rounds: int, code: str)   # mock 策略：先做 rounds 次 run_python(code)，再把去掉循环轮次的消息交给 policy
def strip_loop(messages) -> list[dict]; def cut(text, n, head=0.25) -> str; FORCED_MARK: str
# explore_api_factory(tag: str, ctx: dict)：每次 run_python 在提案线程里调用一次，tag = f"{cid}~x{r:02d}_{i}"；
#   返回 ToolAPI、(ToolAPI, cleanup)（cleanup() 在运行后调用）、(ToolAPI, cleanup, before_call)（2026-11-21：before_call(timeout_s)
#   在每次运行前调用，cleanup 可为 None）或 __enter__ 返回 ToolAPI 的 context manager。
#   不得注册 evaluate 或任何花本代预算的工具（Paper B：begin_program(tag) 放在 factory 里，end_program 放在 cleanup 里）。
# 持久会话（2026-11-21，persistent=True <- editor.explore.persistent，默认 false = 上面的每次新进程；simple 配置为 true）：
#   每个提案（ctx["cid"]）一个 sandbox.Kernel，第一次 run_python 时懒启动；explore api 每个提案只造一次：
#   factory(f"{cid}~x", ctx)（返回形式同上；before_call(timeout_s) 每次 exec 前调用，Paper B 用它设 tensor_compute 截止时间）；
#   单次超时 = editor.explore.timeout_s：超时或崩溃 -> 杀内核，回复 error 含 "session restarted: variables lost"、"restarted": true，
#   下一次调用起新内核，api 对象（以及 workspace handle）不变。end(ctx) 关内核并跑 cleanup；begin(ctx) 也丢弃旧内核。
#   workdir gen_dir/explore/<cid>/session/，每次调用 r{r:02d}_{i}/（program.py、stdout.txt、stderr.txt、result.json）。
#   多个提案并发时线程安全（同一提案的调用是串行的）。with_mock_exploration 照常可用。

# search.py
@dataclass
class Candidate:
    cid: str; thread: int; step: int; program: str
    artifact: dict | None; error: str | None
    screen: dict | None; full: dict | None; score: float | None; cost: dict
def inner_search(rc, gen: int, *, editor_factory, render_context, build_fn, screen_fn, full_fn,
                 threads: int, steps: int, max_candidates: int, top_full: int = 3, rerun_seeds: int = 0,
                 out_dir: Path | None = None, parallel: int = 1, phases=None) -> list[Candidate]
    # editor_factory(thread)->Editor；render_context(history: list[Candidate], thread)->str（论文自己写反馈格式）
    # 2026-11-21：propose 收 ctx 时传 {"cid", "gen_dir", "visible_cids": [本次 history 里的 cid]}；EditorReplyError -> 该候选
    #   program ""、error "editor failed: ..."，搜索继续（其他异常照旧向上抛）；cost 增加 editor_rounds / editor_tool_calls /
    #   editor_tool_s / editor_prompt_tokens / editor_completion_tokens / editor_stop（来自 ed.last_stats）；
    #   空程序的 error 为 "no ```python block in the reply"，有原因时附在括号里（如 "(final reply cut at max_tokens while thinking)"）
    # parallel / phases（2026-11-20）：见 search.py 的 docstring（同一步内并行提案 / 改造者阶段与评测阶段分开）
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
                       temperature: float = 0.6, top_p: float = 1.0, base_seed: int = 42, mock: bool = False,
                       lora: str | None = None, pool: str = "pool") -> dict   # 2026-11-19 增补；top_p 2026-11-20
    # 用 score(seed=base_seed+k, temperature=temperature, top_p=top_p) 重复评测 repeats 次（输出 out_dir/seed<seed>/）；
    # 返回 {"band": max-min, "se": sqrt(p(1-p)/n)（p=各次均值的平均，n=有分数的题数）, "scores", "seeds", "mean", "n", "temperature", "task"}
    # 贪心（temperature=0）重复评测在 vLLM 和 mock 里都完全相同，band=0
def fixed_noise_band(spec, dataset: str) -> float | None   # 2026-11-20：config 里的固定噪声带
    # spec: None | 数字（所有数据集）| {数据集名 / 领域名 / "default": 数字}；没有值返回 None（照常测）
def noise_ids(ids, n: int | None, seed: int) -> list | None   # 2026-11-20：噪声重评用的固定子集（n=None 或 >= len 时全部）
def relative_pass(score: float, baseline: float, ratio: float) -> bool

# heldout.py
HELDOUT = {"math": [("math500", "test"), ("gsm8k", "test"), ("aime24", "test"), ("aime25", "test")],
           "code": [("humanevalplus", "test"), ("mbppplus", "test"), ("livecodebench", "test")],
           "knowledge": [("gpqa_diamond", "test"), ("mmlu_pro", "test"), ("triviaqa", "test")],
           "instruction_following": [("ifeval", "test"), ("ifbench", "test")],
           "tool_calling": [("sealtool", "test"), ("apibank", "test"), ("bfcl", "test")]}
def run_heldout(endpoint, served_name, out_dir: Path, *, domains="all", lora=None, mock=False, limit=None,
                decoding: dict | None = None) -> dict
def decoding_args(decoding: dict | None, task: str) -> dict
    # {"kw": {temperature, top_p, top_k, seed}, "extra": ["--n", n, (--thinking on|off), (--max-tokens N)], "key": 续跑键,
    #  "decoding": 同一设置的 decoding dict（2026-11-21）}；decoding 为 null 时 {"kw": {}, "extra": [], "key": None}
def ensure_built(domains="all") -> None                       # 2026-11-21：先构建 held-out 里的 eval set（eval:<dataset>）
```

### 3.1 core 实现说明（2026-11-19，均为增补，原签名不变）

- `config`：`hardware_profiles[hardware]`（`--set hardware=h20`）和 `scope_profiles[scope]`（`--set scope=single`：只用这次运行的领域；`--domain all` 时拒绝）依次在 base/domain 之后、`--set` 之前合并（`--set` 优先），之后 `heldout.domains: run` 解析成运行的领域，结果锁进 run_config.yaml。CLI 另有 `--runs-dir`（默认 `experiment_records/`）、`--paper-dir`（覆盖读取 configs/ 的目录）、`--set KEY=VALUE`（可重复，点号键，YAML 值，并入 paper_cfg）。`run_config.yaml` 锁定 paper/domain/model/model_path/seed/mock/draw_config/paper_cfg；`gens`、`gpus` 可改（延长运行、换机器）。`RunConfig.cli_extra` 记录论文自加的 CLI 参数（不锁定）。paper_cfg 的 `data:`（2026-11-20）覆盖抽题配置里的数量（Paper A：`n`、`gate_n`、`screen_n`、`kl_contexts`；Paper B：`n`、`gate_n`、`screen_n`、`target_train_frac`、`retention_n`、`retention_kl`；`n` / `gate_n` 可以是一个数或 {领域: 数}），有值时 `config.apply_data_sizes` 生成 `<run>/draw_config.effective.yaml`，`rc.draw_config` 指向它（锁检查通过后才写）。Paper A 单领域（`kl.domains` 不是 `all`、`--domain` 不是 all）时 `config.kl_domains_for` 还会在里面写 `kl_domains: [<domain>]`，draw.py 把 `kl_contexts` 只分给这些领域。`config.eval_python()`：run_eval.py / draw.py 用的解释器（`RSI_EVAL_PYTHON` > `eval/.venv/bin/python` > 当前解释器）。真实模式下 model_path 不存在直接报错。
- `state.RunState`：另有 `unmark(key)`、`keys()`；`step()` 块内自己调用 `mark(key, info)` 不会被覆盖。
- `records`：另有 `read_json(path, default)`、`read_jsonl(path)`；`summary.json = {"run_id","paper","domain","gens": {"<g>": {...}}}`。
- `draws`：已抽过的代 / 已切过的 gate 直接读文件，不再起子进程；`ids()` 文件不存在返回 `[]`。
- `servers.VLLMServer` 真实命令：`vllm serve <model> --served-model-name N --host 127.0.0.1 --port P --seed S --enable-auto-tool-choice --tool-call-parser <hermes|qwen3_coder，auto 检测，见下> --reasoning-parser qwen3 --max-model-len 40960 --tensor-parallel-size len(gpu_ids) [--enable-lora --max-loras --max-lora-rank] <extra_args>`（extra_args 里给了 `--max-model-len` / `--tensor-parallel-size` 就不加默认值）；vllm 可执行文件：`RSI_VLLM` > 解释器同目录的 `vllm` > PATH > `python -m vllm.entrypoints.cli.main`。extra_args 里的 `--rsi-gpu-memory-gib X` 在启动时换算成 `--gpu-memory-utilization = X / 卡的总显存`（nvidia-smi，上限 0.95；读不到时保留配置里的比例），mock 时去掉。日志 `<log_dir>/<served_name>.<port>.log`。mock 时其余 extra_args 原样传给 mock_server（如 `["--tasks", "dapo_math,math500"]`）。`load_lora` 同名同路径重复调用是空操作。
- `evaluate`：`EvalResult.accuracy` 属性；`ids=[]` 直接返回 n=0；mock 时自动加 `--no-sandbox`（不启动 Docker）。同一 out_dir 不要并发 score（run_eval 合并写 summary.json）。**`paired_delta(a, b)` 的 delta = mean(b − a)**（b=候选，a=参照）；同一 id 的多个单元先平均，score=None 跳过。`token_kl` 另返回 `token_mean`、`n_tokens`；KL 在"参照报告的 top-k token + 其余"划分上计算（数据处理不等式下是真 KL 的下界），候选没报告的 token 用候选最小 logprob 代替。
- `sandbox.ToolAPI`：另有 `charge(key, amount)`（直接记账，不检查）、`remaining()`、`names()`、`calls`、`log`（每条 `{tool, ok, seconds}`，实测费用另有 `measured`，去重命中为 `deduped: true`）。实测费用 / 去重见 §3 的 `register`；`describe()` 里标注 `measured: <keys>` 和 `repeated identical calls are free`。超时是总墙钟时间（含工具调用耗时，在两条 RPC 消息之间检查）。子进程 = 同一解释器 `python -I`；文件写只允许 workdir，读禁止仓库根 / $HOME / /proc（解释器自身前缀除外）；禁止 subprocess/fork/exec/kill/socket connect|bind|getaddrinfo（Python audit hook，C 扩展可绕过，属于纵深防御）。断网：Linux `unshare -rn`，macOS `sandbox-exec (deny network*)`，都不可用则 `info["network_isolation"]="none ..."`。程序里可用 `api.BudgetExceeded`、`api.ToolError`（全局名 `BudgetExceeded`、`ToolError` 也可用），工具只能用关键字参数。
- `editor.Editor`：transcript 在构造时重新载入；`propose()` 收到与下一条记录相同的 user 消息时直接重放记录的回答（不调模型），消息不同则从该处截断——所以续跑的内层搜索会得到完全相同的程序。续跑时 search 跳过已记录的 cid：`ctx` 带 cid 时，先把记录里开头那些 cid 不同的完成提案吸收进历史（等同重放，不返回），再与本 cid 的记录比较（2026-11-21 修复：之前第一次实时调用会清空整份 transcript 和跨步历史）；user 行带 `cid`，旧记录没有 cid 时按旧规则。`extra_body` 并入请求（如 `{"chat_template_kwargs": {"enable_thinking": true}}`）。历史里只放 content（reasoning 只写入 transcript）。
- `editor.Editor` 工具循环（2026-11-21，"改造者：thinking + 工具轮次"）：
  - **开关**：`tools` 不为 None 且 `tool_rounds > 0` 时进入循环；否则与之前完全一样只发一次请求（请求体没有 `tools` / `tool_choice` 键，键顺序 model, messages, temperature, top_p, max_tokens, seed + extra_body）。
  - **一轮** = 一次带工具调用的模型回答（同一回答里的多个并行调用只算一轮）。最多 `tool_rounds` 轮，之后再发一次 `tool_choice: "none"` 的强制最终轮；提示语（以 `FORCED_MARK` 开头）**追加在最后一条 tool 消息末尾**，不新增 user 消息（保持前缀缓存，mock 策略找最后一条 user 消息仍能找到 `[thread=K step=S]`）。剩 5 轮以内时每条工具结果末尾加 `[N tool rounds left]`。
  - **结束**：模型给出不带工具调用、含 ```python 块的回答（没有 submit 工具）。空回答、`finish_reason == "length"`、内容里有未解析的 `<tool_call>` / `<function=` 文本时最多提醒 2 次（user 消息），强制最终轮另有 1 次提醒（仍是 tool_choice none；`<tool_call>` 文本用 `NUDGE["tool_text_final"]`）；最终仍无代码块则用循环中最后见到的代码块（stop = "fallback"）。
  - **守卫**（任一触发即强制最终轮，stop = "forced:<原因>"）：rounds ≥ tool_rounds（rounds）、工具调用数 ≥ max_tool_calls（tool_calls；超出的并行调用得到 "skipped: tool-call cap" 错误结果）、completion tokens 累计 ≥ proposal_tokens（tokens）、墙钟 ≥ proposal_wall_s（wall；每次模型调用和工具调用前检查）、上下文剩余不足（context）、HTTP 400（http400）、工具轮回复超时（timeout）。HTTP 超时：工具轮 = max(60, min(timeout_s, 剩余墙钟 + 600))，最终轮 = timeout_s；循环内读超时不重试（event kind timeout）。
  - **上下文**：每轮 max_tokens = min(turn_max_tokens（最终轮用 max_tokens）, context_tokens − 估计 prompt − 512)；估计 = 请求字符数 × 上一次 usage 的 token/字符比（初始 1/3）。真实服务器第一次调用前读 `/v1/models` 的 `max_model_len`，比 context_tokens 小就用它（servers 已按 max_position_embeddings 截过）；LoRA adapter 的条目没有 max_model_len 时用其 parent（基座）条目的，再没有就用列表里最小的。估计超过 compact_at × context_tokens 时确定性压缩（只改本次提案的消息）：除最近 keep_reasoning_rounds 轮外去掉 reasoning，早于最近 keep_tool_rounds 轮的工具输出换成 `[elided: run_python output rN, X chars]`，仍超过 0.85 时再把这些轮里超过 1000 字符的调用参数换成 `{"code": "[elided]"}`；剩余 < 2048 时激进压缩（不留 reasoning、只留最近 2 轮工具输出），仍不够则强制最终轮。HTTP 400 且报错与上下文 / token 有关 → 激进压缩后重试一次 → 强制最终轮 → 仍失败（或最终轮超时）时有 fallback 代码块就用它（stop = "fallback"，stats.error 记原因），否则抛 `EditorReplyError`（transcript 记一条 `failed: true` 的 final 行，不进跨步历史；重放时再次抛出）。其他 4xx、连接错误和 5xx（重试 5 次后）仍抛 RuntimeError（致命）。
  - **reasoning 回传**（循环内的 assistant 轮）：`think_tag`（默认，content 前加 `<think>\n…\n</think>\n\n`，由 Qwen3 模板拆分并只保留最后一条 user 之后的推理）、`field`（`reasoning_content`）、`drop`。
  - **跨步历史**：永远是折叠后的（每个旧提案一条 user + 一条 assistant：最终回答；代码来自 fallback 时把代码块附在后面），不含工具轮次和 reasoning；`history_keep=K` 保留提案 0 和最近 K 个旧提案全文，中间的变成 `HISTORY_STUB` + 只含代码块的 assistant；None = 全部保留（构造函数默认 None，`editor_kwargs` 默认 2；`tool_rounds` 0 时 `editor_kwargs` 给 None，单次调用与之前的请求完全相同）。
  - **transcript v2**：system 行带 `"format": 2`；之后每行带 `p`（提案序号），循环内的行另带 `r`（轮次）：user、assistant（content, reasoning, tool_calls [{id, name, arguments}], usage, finish_reason）、tool（tool_call_id, name, args, content（发出去的、已截断）, full_chars, ok, seconds）、提醒用的 assistant（nudge）+ user（nudge: true）、event（kind: compact | nudge | forced_final | http400 | timeout, detail）、final（final: true, content, reasoning, code, usage, finish_reason, stats）。载入时按 p 分组，有 final 行才算完成；末尾未完成的提案（循环中途崩溃）丢弃，下次 propose 重新实时运行并重写文件。旧格式（严格 user/assistant 交替）按零轮的完成提案载入，重放后第一次实时提案时整份重写为 v2。重放既不调模型也不调工具，last_stats 取记录值。
  - **mock**：mock_policy 收到本次提案的完整消息（含工具轮次），可返回 dict 发起工具调用；`chat_tools.with_mock_exploration(policy, rounds, code)` 先做 rounds 次 run_python，再把去掉循环轮次的消息交给原策略，所以程序与不带工具时相同。
- `chat_tools.ChatTools`（2026-11-21）：`run_python(code)` 用 `sandbox.run_program(..., save_output=True)` 运行包装后的代码（`def main(api)` 里 `exec` 用户代码；顶层代码即可，`result = ...` 返回值，也可以定义 `main(api)`；返回值不能 JSON 序列化时给 repr），超时 `timeout_s`（默认 300，且不超过本次提案剩余墙钟、至少 5 s），内存 `mem_gb`（默认 8），workdir `gen_dir/explore/<cid>/r{r:02d}_{i}/`（program.py、完整的 stdout.txt / stderr.txt、result.json）。返回 JSON `{ok, seconds, tool_calls: {工具名: 次数}, error?, stderr?（失败时）, result?, stdout?}`，整体约 result_chars（6000）字符（stdout 头 1/4 + 尾部，中间 `[... N chars cut ...]`）；Editor 再按 tool_result_chars 硬截一次。`search_history(gen=None, sort="score", offset=0)`（2026-11-21，只在传了 `history` 时有）：列出已经试过的所有候选：往期（`history.past` 各目录的 candidates.jsonl）加本步可见的（`ctx["visible_cids"]`，同一步并行的其他线程看不到），每个一行：cid、gen、score / screen / full、错误首行、验收结果（该代 gate.json：passed + reason）、论文摘要（row_fn）、same_as（key_fn 相同的最早候选）；最新一代在前，代内按分数（`sort="order"` 按尝试顺序）；offset=0 时附每代一行（gen_fn）；按 result_chars 分页（`next_offset`）。有 history 时 `read_record` 也接受往期的 cid（在对应目录的 programs/、feedback/ 里读）。`read_record(cid, part, offset=0)`：只允许 `ctx["visible_cids"]` 里的 cid，读 `gen_dir/programs/<cid>.py` 或 `gen_dir/feedback/<cid>.json`，按页返回（`next_offset`）。参数不是 JSON、未知工具、参数不对、factory / 沙箱出错、超时都变成 `{"ok": false, "error": "<Type>: msg"}`（最多 2000 字符），循环继续；子进程因 CPU 时间上限（SIGXCPU）退出也报为 timeout。`begin(ctx)`：提案实时开始时（Editor 调用）删除 `gen_dir/explore/<cid>/`（被丢弃的上一次尝试留下的目录）。探索不花任何本代预算（由论文的 explore API 保证）。
- 配置键（`editor:` 块，`editor_settings` 的默认值）：`temperature` 0.6、`top_p` 0.95、`max_tokens` 16384（最终轮）、`extra_body` `{chat_template_kwargs: {enable_thinking: true}, top_k: 20, min_p: 0.0}`（给了这个键就原样使用，`{}` 即关闭）、`tool_rounds` 50（0 = 单次调用）、`turn_max_tokens` 8192、`max_tool_calls` 100、`proposal_tokens` 160000、`proposal_wall_s` 2400、`context_tokens` 131072、`tool_result_chars` 6000、`compact_at` 0.6、`keep_tool_rounds` 8、`keep_reasoning_rounds` 2、`reasoning_passback` think_tag、`history_keep` 2、`timeout_s` 3600、`mock_tool_rounds` 1、`explore: {timeout_s: 300, mem_gb: 8, gpu_tools: false, gpu_seconds: 300, persistent: false}`（persistent 2026-11-21：run_python 用每个提案一个的持久内核，见 chat_tools）（gpu_tools / gpu_seconds 由论文的 explore API 读取）。**`editor.context_tokens` 是上下文长度的唯一旋钮**：客户端用它截每轮 max_tokens、决定何时压缩；论文用 `apply_context_tokens(args, cfg)` 把它写进改造者所用服务器的 `--max-model-len`（Paper A `vllm.editor_args`，Paper B `server.main_args`；args 里原有的 `--max-model-len` 被覆盖；Paper A 的候选服务器不变，Paper B 按 main_args 启动的服务器都用它）；显式 null 时反过来用 `context_tokens(cfg, server_args)` 读服务器参数（没有则 40960）。
- `servers.VLLMServer`（2026-11-21）：`tool_call_parser="auto"`（默认）读模型的 chat template（tokenizer_config.json 的 `chat_template`，或 chat_template.jinja），含 `<function=` 用 `qwen3_coder`，否则 `hermes`；extra_args 里给了 `--tool-call-parser` 就不加。`--max-model-len`（extra_args 里的，否则 40960）大于模型 `max_position_embeddings`（config.json 或其 text_config）时截到后者并警告（stderr + 日志），从不加 YaRN。选择结果在 `srv.notes` 里，也写进日志开头。mock 路径不受影响（extra_args 里的 `--max-model-len` 会被去掉）。
- `search`：cid = `g{gen:02d}_t{thread}_s{step:02d}`；步骤按 step 轮转所有线程；`screen = {"score", "feedback"}`，`full = {"first", "reruns", "score", "feedback"}`（score = first 与 reruns 的平均，rerun 调用为 `full_fn(artifact, f"{cid}_r{i}")`）；`cost` 含 propose_s/build_s/screen_s/full_s 并合并 `artifact["cost"]`。build_fn 抛异常记为该候选的 error；screen_fn / full_fn 的异常向上抛（视为基础设施故障，续跑时重做该候选）。程序为空（无 ```python 块）时不调用 build_fn。另有 `load_candidates(rc, gen) -> {cid: Candidate}`、`ranked(cands)`、`Candidate.to_record()`。`submit_to_gate` 跳过 error / 无分数的候选，gate.json = `{"submissions": [{cid, passed, info, t}]}`，续跑时已提交的不再提交，已通过的直接返回。
- `gate.measure_noise_band`：mock 下 temperature > 0 时答案依赖请求的 seed（见 §5），所以 band > 0；论文用它算第 0 代噪声带（Paper A：`baseline/noise.json` = {dataset: band}，详情 `noise_detail.json`；Paper B：gate 余量 = max(band, 2×SE)）。`run_eval.py --seed N` 覆盖任务 / 默认 seed（与 `--temperature` 同样写入 `decoding`）；seed 只在 temperature > 0 时进入请求体，因而也进入生成缓存键（缓存键 = 请求体）。
- `heldout`：另有 `heldout_tasks(domains)`；out_dir/summary.json 里已有同一服务名 / LoRA / limit 的任务会跳过。
  2026-11-21：`heldout.decoding` 还可以给 `thinking`（bool）和 `max_tokens`（优先于 max_tokens 表），给了就进入续跑键（与
  summary 里 run_eval 记录的 decoding 比较）；`run_heldout` 先 `ensure_built(domains)`（eval:<dataset> 形式的任务，如
  simple_<d>_test，按需构建 data/processed + data/splits）。
- **objective simple（2026-11-21，第二版）**：数据集（tasks.yaml `eval_targets`，`build_eval_target.py` 按需构建，不进 git）：
  `simple_<d>_pool` = 训练集 processed 记录中 id 在 `simple_data/<d>/pool_ids.json` 里的（`{task, ids_file, dataset}` 形式，
  ids 文件或 processed 文件大小变化时重建）、`simple_<d>_eval`（500，gate）、`simple_<d>_test`（500，held-out）、
  `simple_<d>_train`（1,500，旧运行用，保留）。`evalsets.default_roles` = {train: simple_<d>_pool, gate: simple_<d>_eval,
  test: simple_<d>_test}；`data_roles.eval` 接受 `test`（-> `eval:simple_<d>_test`），train / gate / eval 接受简写
  simple_pool / simple_train / simple_eval / simple_test。每代抽样（`draws.simple_draw`，`draw_gen` / `ids` 都会触发，
  `ids()` 在 eval set 捷径之前返回它）：候选 = pool − gate − test；screen = signal = `Random(f"{seed}:simple:{domain}:{g}")`
  抽 simple_screen_n 个（种子不含 run id 和 paper：A / B 同一代 screen 完全相同；每代独立重抽，可与前几代重叠）；
  train = 其余候选；kl = train 中 `Random(f"{seed}:simple_kl:{domain}:{g}")` 抽 eval_target.kl_n（默认 400）个；写
  `draws/gen_XX/ids/simple_<d>_pool.<role>.json` + `manifest.json`（simple: true）。`ensure_gate` / `gate_ids` 不变（全部 500）；
  `mark_shown` 对 simple 的数据集是空操作；`ensure_gate` 顺带构建 test 集。论文配置 `decoding: {screen, full, gate}` 用
  `config.stage_decoding` 读。改了 simple profile 的运行不能续跑旧的 _simple 运行（paper_cfg 锁定）：用新 run id / `--tag`。
- `mock_server`：默认索引全部训练集 + 留出任务（首次约 5 s，之后读 `eval/.cache/mock_index/` 缓存约 1 s）；`--tasks` 只索引指定任务——**不在索引里的请求回答 "UNKNOWN REQUEST"（得 0 分）**，所以 mock 下要跑留出评测的服务器不要把 `--tasks` 限制成只有训练集。ifeval_train 的"金标准"回复按约束构造（全部通过判分器）；留出的 IFEval/IFBench 只有手写样例，mock 下大多为 0 分。未知模型名返回 vLLM 式 404；`load_lora_adapter` 需 `--enable-lora`，且 lora_path 必须存在。
- 自测：`eval/.venv/bin/python -I eval/tests/rsicore/test_core.py`；工具循环：`python -I -m pytest -q eval/tests/rsicore/test_editor_loop.py -p no:cacheprovider`（假 OpenAI 服务器）；玩具论文（不依赖 edit_code/lora_code 的整条闭环）：`eval/tests/rsicore/toy_paper.py`。

## 4. 论文侧（各自实现）

### Paper A：`edit_code/`（A agent）
- `edit_code/run.py`（2026-11-19 修订为固定体积）：`main()` 用 `rsicore.config.build_arg_parser("A")`（加 `--budget-gib`）。第 0 代：原模型的验收分、噪声带、留出评测，原模型的逐层量化缓存和诊断表（只算一次），定体积，Q_0 = 参照配置（第一个 incumbent）。逐代：draw → 原模型参照 → incumbent 在本代题上的分数 → `inner_search`（改造者默认是冻结的原模型）→ 只把分数高于 incumbent 的候选交 `submit_to_gate`（验收 = 在验收集上逐题配对超过 incumbent）→ 通过则导出新 incumbent 并做留出评测，否则保留；连续 `loop.patience` 代无提升就停。
- 配置：`edit_code/configs/base.yaml`、`edit_code/configs/domains/<domain>.yaml`（含 `all.yaml`）。`budget`（一个体积）、`loop.editor`（frozen / current）、`hardware`（b300 / h20）。
- 候选格式：**模拟量化（fake-quant）**——权重量化再反量化成 BF16 写成普通 checkpoint，vLLM 照常加载，用于测质量；大小按配置解析计算。速度只对 vLLM 有内核的格式另行测量。
- mock：不加载真实权重；候选 artifact 只记录配置与一个由配置算出的 `quality`（越接近原精度越高），写入 mock registry。

### Paper B：`lora_code/`（B agent）
- `lora_code/run.py`：逐代：draw（`--target <domain>`）→ 诊断表 → `inner_search`（工具：`svd`、`acts`、构造工具、LoRA 栈操作、`evaluate(lora)`；没有训练 / 梯度工具）→ `submit_to_gate` → 把 LoRA 合并进权重得 M_{t+1} → 对照分支（非 LLM 搜索；没有 SFT 基线）→ E_t（固定任务：为原始基座在 Seal-Tools 上写 LoRA；工具调用领域单独跑时换成代码领域）→ 留出评测。
- 配置：`lora_code/configs/base.yaml`、`lora_code/configs/domains/<domain>.yaml`。
- 改造者（2026-11-21，主搜索和 E_t 相同）：`Editor(..., tools=tools.editor_chat_tools(ws, cfg), mock_policy=tools.editor_mock_policy(policy, cfg), **editor_kwargs(cfg, server_args=common.main_server_args(cfg)))`；run_python 的 api = `Workspace.explore_api(tag)`（返回 `(ToolAPI, cleanup)`：begin_program(tag) / end_program；没有 evaluate、raw_lora，`editor.explore.gpu_tools` 为 false 时也没有 acts / train_loss / regmean_preserve，为 true 时记在单独的探索预算 `explore.gpu_seconds` 上；从不动本代的 `ws.api.spent` / `gpu_measured`）。`common.main_server_args(cfg)` = `apply_context_tokens(server.main_args, cfg)`，用于所有按 main_args 启动的服务器（main 副本；weights 模式下候选槽的临时服务器也用，保证打分服务器上下文一致）；theta0_args 不变。mock 探索代码 `print(api.budget())`。
- LoRA 存成 PEFT 格式（`adapter_config.json` + `adapter_model.safetensors`），目标模块名要能被 vLLM 0.24.0 的 Qwen3.5 LoRA 加载；唯一的训练（edit 模式第 0 代的 raw LoRA）和激活统计用 transformers + peft（只能用本代 `train` 角色的题）。
- mock：不加载真实权重；LoRA artifact 是一个小 JSON（含由程序参数算出的 gain / damage），登记到 mock registry 的是按领域的 dict：目标领域 = 父模型该领域 + gain，其余领域 = 父模型 − damage（2026-11-19），合并后的 M_{g+1} 继承这个 dict。

## 5. mock 模式（core agent 提供 `eval/rsicore/mock_server.py`）

- OpenAI 兼容：`/v1/models`、`/v1/chat/completions`、`/v1/completions`（支持 `prompt_logprobs` 的确定性假值）、`/v1/load_lora_adapter`、`/v1/unload_lora_adapter`。
- 答题：复用 `eval/tests/oracle_server.py` 的索引思路，对每个请求以概率 `quality` 返回标准答案、否则返回错误答案（用 `hash(seed, 题目) < quality` 决定，确定性；hash 里**不含**模型名，即所有模型对同一题用同一个随机数，同一批题上质量高的模型答对的题包含质量低的模型答对的题，分数随质量单调变化，2026-11-19）。贪心请求（temperature 为 0 / 缺省）不看请求 seed；temperature > 0 的请求再把请求的 `seed` 和 `temperature` 放进 hash，所以不同 seed 的重复评测会不同，同一 (seed, temperature, 题目) 仍确定（2026-11-19）。
- `quality` 从 mock registry（JSON 文件：`{served_name 或 lora 名: value}`）读取；没登记的名字用默认 0.7。value 可以是 float（所有请求同一个质量），也可以是按领域 / 任务的 dict，例如 `{"default": 0.6, "math": 0.7, "knowledge": 0.55, "dapo_math": 0.72}`，键为领域（math / code / knowledge / instruction_following / tool_calling，即 tasks.yaml 的 `category`）、run_eval 任务名或 "default"；查找顺序 任务名 > 领域 > default > 各值平均；也可包成 `{"quality": value, ...}`。假 prompt_logprobs / KL 用 `mock_quality.scalar(value)`（五个领域解析后的平均；只有 float / default 时就是它）。解析函数在 `eval/rsicore/mock_quality.py`（`resolve`、`scalar`、`as_map`、`unwrap`、`CATEGORY_DOMAIN`，只依赖标准库）。
- 改造者对话：请求里带 `"metadata": {"mock_editor": "<paper>"}` 或 served_name 以 `editor` 结尾时，返回由论文侧注册的 mock 程序（core 提供注册机制：论文在 mock 模式下把 `mock_policy` 交给 `Editor`，`Editor` 直接调用，不经过服务）。
- 要求：`./run_rsi.sh --paper A --domain math --gens 2 --mock` 和 `./run_rsi.sh --paper B --domain math --gens 2 --mock` 在本机几分钟内跑完，产出完整的运行目录；第二次运行同一个 run-id 应全部跳过。
