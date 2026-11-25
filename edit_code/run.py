"""Paper A orchestration: iterative self-quantization at a FIXED size, judged by KL to the original model only
(2026-11-25).

    python -I edit_code/run.py --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 \
        --run-id A_math_s42 [--budget-gib 11.5] [--mock] [--set search.threads=4]

Design: the editor's weights are frozen (loop.editor=frozen: the original BF16 model M_0 writes every generation;
loop.editor=current is the unfrozen ablation where the incumbent writes). What is iterated is the quantized model:
every candidate quantizes the ORIGINAL weights of M_0 to the same budget, and generation g tries to beat the
incumbent (lowest-KL quantization accepted so far).

The only number (2026-11-25): the KL set = kl_eval.n (256) items of generation 0's draw of the training pool
simple_<d>_pool (seeded); M_0 writes ONE thinking-mode answer per item in generation 0 and its top-k prompt logprobs
are cached (baseline/kl_gate). A config's evaluation = the per-trace token KL(M_0 || config) on the answer positions
of these fixed traces, averaged over the traces (rsicore.kl; the quantized model reads them, nothing is sampled).
There is no screen, no accuracy gate, no held-out test and no classic-baseline sweep.

Generation 0 (baseline/): draw; the KL set (M_0's traces + logprobs); calibration texts (prompt + M_0's own response,
kl.self_generated); layer cache + diagnostics of M_0 (run_dir/prep/, built once); budget = budget.gib or the packed
size of budget.reference (uniform W4 GPTQ); Q_0 = the reference config, its KL recorded (reference row). The incumbent
of generation 1 is M_0 (loop.start bf16, KL 0, over the budget) or Q_0 (loop.start q0).
Generation g = 1..gens (gen_XX/): inner search (search.threads x search.steps, 8 x 1) -> every built candidate gets ONE
KL evaluation (gen_XX/gate.json, visible to later steps / generations) -> the in-budget candidate with the LOWEST KL is
accepted when its KL is below the incumbent's (first quantization under loop.start bf16: the lowest KL, whatever it
is), else the incumbent stays -> the accepted model is exported. The loop stops after loop.patience generations
without a new incumbent. Every step is a rsicore.state.RunState key; rerunning the same run-id resumes.
"""
from __future__ import annotations

import random
import shutil
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rsicore import config as C  # noqa: E402
from rsicore import draws as D  # noqa: E402
from rsicore import records as R  # noqa: E402
from rsicore import search as S  # noqa: E402
from rsicore.chat_tools import (RUN_PYTHON_DOC, ChatTools, SearchHistory, chat_tools_kwargs,  # noqa: E402
                                with_mock_exploration)
from rsicore.editor import Editor, apply_context_tokens, editor_kwargs, editor_settings  # noqa: E402
from rsicore.sandbox import run_program  # noqa: E402
from rsicore.servers import VLLMServer  # noqa: E402
from rsicore.state import RunState  # noqa: E402

from edit_code import mock as M  # noqa: E402
from edit_code.common import (candidate_args, candidate_groups, deep_merge, error_line, gpu_layout, kl_eval_cfg,  # noqa: E402
                              load_processed, program_note, read_json, read_jsonl, record_text, search_settings, tail,
                              with_kl_chunk_cap, write_json)
from edit_code.problem import QuantProblem, add_self_generated, kl_texts_for  # noqa: E402
from edit_code.prompts import archive_row, archive_text, history_row, make_render_context, system_prompt  # noqa: E402
from edit_code.quant import schema, sizes  # noqa: E402
from edit_code.tools import make_tool_api  # noqa: E402

MOCK_EXPLORE_CODE = "result = api.profile_summary()"     # the mock editor's run_python call (editor.mock_tool_rounds)


def server_args(cfg: dict, args_key: str) -> list[str]:
    """vLLM args of a real server: vllm.<args_key>. The editor's server (editor_args) gets
    --max-model-len = editor.context_tokens (it overrides a --max-model-len written in editor_args; null keeps
    editor_args as written). Candidate servers (candidate_args) keep their own --max-model-len unless
    vllm.candidate_max_model_len is set (2026-11-21: 32768; edit_code.common.candidate_args)."""
    if args_key == "candidate_args":
        return candidate_args(cfg)
    if args_key != "editor_args":
        raise ValueError(f"unknown vLLM args key {args_key!r}")
    return with_kl_chunk_cap(cfg, apply_context_tokens(list((cfg.get("vllm") or {}).get("editor_args") or []), cfg))



class _Background:
    """fn() in a daemon thread; join() waits and re-raises its exception (raise_error=False: only waits)."""

    def __init__(self, fn):
        self.error = None

        def run():
            try:
                fn()
            except BaseException as e:   # noqa: BLE001 - re-raised in join()
                self.error = e
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def join(self, raise_error: bool = True):
        self.thread.join()
        if raise_error and self.error is not None:
            err, self.error = self.error, None
            raise err


class LazyServer:
    """Starts a VLLMServer on first use; close() stops it. With sleep_mode, sleep() offloads its weights to CPU
    RAM (the GPU can serve something else) and get() wakes it again."""

    def __init__(self, **kw):
        self.kw = kw
        self.srv = None

    def get(self) -> VLLMServer:
        if self.srv is None:
            self.srv = VLLMServer(**self.kw).start()
        elif self.srv.sleeping:
            self.srv.wake_up()
        return self.srv

    def sleep(self):
        if self.srv is not None and self.kw.get("sleep_mode"):
            self.srv.sleep()

    @property
    def name(self) -> str:
        return self.kw["served_name"]

    def close(self):
        if self.srv is not None:
            self.srv.stop()
            self.srv = None


def start_lazy(servers: list) -> list:
    """Start several LazyServers at the same time (replicas on different GPUs); returns their VLLMServers."""
    from rsicore.servers import start_all
    for s in servers:
        if s.srv is None:
            s.srv = VLLMServer(**s.kw)
    start_all([s.srv for s in servers])
    return [s.srv for s in servers]


class EditorPhases:
    """rsicore.search.inner_search `phases` (2026-11-20, gpu_layout.editor_sleep): the editor proposes, then sleeps
    (weights -> CPU RAM) while every candidate of the step is built and evaluated, its GPU lent to the problem as
    one more candidate slot; editor_phase() takes the slot back and wakes the editor."""

    def __init__(self, ed: "LazyServer", prob, gpus: list[int], enabled: bool, log=print):
        self.ed, self.prob, self.gpus, self.enabled, self.log = ed, prob, list(gpus), bool(enabled), log
        self.slot = None

    def editor_phase(self):
        if self.slot is not None:
            self.prob.remove_slot(self.slot)
            self.slot = None
        self.ed.get()                             # wakes it if it sleeps

    def eval_phase(self) -> int:
        if self.enabled and self.slot is None and self.ed.srv is not None:
            self.ed.sleep()
            self.slot = self.prob.add_slot(self.gpus)
            if not getattr(self, "_logged", False):
                self.log(f"[search] editor asleep while candidates are evaluated; its GPUs {self.gpus} serve as "
                         f"candidate slot {self.slot} ({self.prob.n_slots} slots)")
                self._logged = True
        return self.prob.n_slots


def effective_config(rc, args) -> dict:
    """rc.paper_cfg (+ mock_overrides in --mock runs, which win over --set and the profiles)."""
    cfg = dict(rc.paper_cfg)   # hardware profile already merged by rsicore.config (before --set)
    if rc.mock:
        cfg = deep_merge(cfg, cfg.get("mock_overrides") or {})
    if getattr(args, "budget_gib", None):
        cfg = deep_merge(cfg, {"budget": {"gib": float(args.budget_gib)}})
    return cfg


# KL results read per candidate from gen_XX/gate.json (one entry per candidate; rsicore.chat_tools reads the same file
# for search_history)
GATE_INFO = ("reason", "metric", "kl", "reference_kl", "gain", "kl_quantiles", "bytes_gib")


def gate_outcomes(d: Path) -> dict:
    """cid -> {"passed", + GATE_INFO present} from <d>/gate.json."""
    out = {}
    for sub in (read_json(Path(d) / "gate.json", None) or {}).get("submissions") or []:
        info = sub.get("info") if isinstance(sub.get("info"), dict) else {}
        out[str(sub.get("cid"))] = dict({k: info[k] for k in GATE_INFO if k in info}, passed=bool(sub.get("passed")))
    return out


class GateLog:
    """<search dir>/gate.json: {"submissions": [...]}, one entry per candidate (sorted by cid), written as each KL
    evaluation finishes (thread-safe), so later steps' search_history rows already show the results."""

    def __init__(self, path: Path):
        self.path, self.lock = Path(path), threading.Lock()

    def add(self, cid: str, passed: bool, info: dict):
        with self.lock:
            log = read_json(self.path, None) or {"submissions": []}
            subs = [x for x in log.get("submissions") or [] if x.get("cid") != cid]
            subs.append({"cid": cid, "passed": bool(passed), "info": info, "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
            write_json(self.path, {"submissions": sorted(subs, key=lambda x: str(x["cid"]))})


def kl_candidate(prob, art: dict, cid: str, log: GateLog) -> dict:
    """ONE KL evaluation of a built candidate on the fixed KL set -> prob.kl_stats; recorded in `log`. A config over
    the budget is refused without evaluation (QuantProblem.build already refuses those); the incumbent's own config
    is no change (gain 0, not evaluated again)."""
    if art["bytes_gib"] > prob.budget_gib + 1e-9:
        g = {"passed": False, "reason": f"size {art['bytes_gib']:.3f} GiB > budget {prob.budget_gib:.3f}: refused "
             "without evaluation", "bytes_gib": art["bytes_gib"]}
        log.add(cid, False, g)
        return g
    inc = prob.incumbent or {}
    if inc.get("config_hash") and art["config_hash"] == inc["config_hash"]:
        k = prob.incumbent_kl
        g = {"passed": False, "reason": "identical to the incumbent (no change), not evaluated", "metric": "kl",
             "kl": k, "reference_kl": k, "gain": 0.0, "bytes_gib": art["bytes_gib"]}
        log.add(cid, False, g)
        return g
    r = prob.evaluate(art, cid)
    g = prob.kl_stats(art, r)
    log.add(cid, g["passed"], g)
    return g


def kl_feedback(prob, art: dict, g: dict) -> dict:
    """The editor's feedback of an evaluated candidate: size, its KL result and where its quantization error
    concentrates (group_error, free)."""
    fb = {"bytes_gib": art["bytes_gib"], "budget_gib": art.get("budget_gib"),
          "linear_avg_bits": art.get("linear_avg_bits"), "kl_eval": g}
    r = prob._evals.get(art["config_hash"]) or {}
    top = (r.get("feedback") or {}).get("error_top_groups")
    if top is None:
        try:
            top = prob.group_error(prob.norm_of(art), top=5)["top"]
        except Exception:  # noqa: BLE001 - informative only
            top = None
    if top is not None:
        fb["error_top_groups"] = top
    return fb


def _history_on(cfg: dict) -> bool:
    """The editor has the search_history tool (editor.tool_rounds > 0 and editor.history_tool)."""
    es = editor_settings(cfg)
    return es["tool_rounds"] > 0 and bool(es["history_tool"])


class Runner:
    def __init__(self, rc, cfg: dict):
        self.rc, self.cfg = rc, cfg
        self.st = RunState(rc.run_dir)
        self.run_dir = Path(rc.run_dir)
        self.datasets = [D.train_dataset(rc)]       # simple_<d>_pool: the KL set and the calibration texts
        self.kl_cfg = kl_eval_cfg(cfg)
        self.registry = self.run_dir / "mock_registry.json"
        roles, unused = gpu_layout(rc, cfg)
        self.gpu_editor, self.gpu_cand = roles["editor"], roles["candidate"]
        self.cand_slots = candidate_groups(rc, cfg)   # one candidate server per slot, evaluated concurrently
        lay = cfg.get("gpu_layout") or {}
        # the editor's GPUs can be used for other work only when they are not shared with the candidates
        own = not (set(self.gpu_editor) & set(self.gpu_cand))
        self.editor_gpus_own = own
        self.editor_sleep = bool(lay.get("editor_sleep", True)) and own
        self.m0_replicas = bool(lay.get("m0_replicas", True)) and own
        self.prep_all_gpus = bool(lay.get("prep_all_gpus", True)) and own
        # every GPU group of the run: the editor's + one per candidate slot
        self.all_groups = ([list(self.gpu_editor)] if own else []) + [list(s) for s in self.cand_slots]
        print(f"[edit_code] hardware={cfg.get('hardware', 'b300')} gpus={rc.gpus} "
              f"editor={self.gpu_editor} candidate servers={self.cand_slots} "
              f"({len(self.cand_slots)} candidate(s) evaluated at a time)"
              + (f" UNUSED {unused} (gpu_layout.candidate_group_size)" if unused else ""), flush=True)
        self.editor_mode = (cfg.get("loop") or {}).get("editor", "frozen")
        if self.editor_mode not in ("frozen", "current"):
            raise ValueError(f"loop.editor must be frozen or current, got {self.editor_mode!r}")
        # loop.start: bf16 = generation 1 starts from the original BF16 model M_0 (KL 0, over the budget; the first
        # accepted config is the lowest-KL in-budget candidate); q0 = from the uniform reference quantization Q_0
        # (budget.reference). Q_0's KL is recorded in generation 0 either way (reference row)
        self.start = (cfg.get("loop") or {}).get("start", "bf16")
        if self.start not in ("bf16", "q0"):
            raise ValueError(f"loop.start must be bf16 or q0, got {self.start!r}")
        self.prep_dir = self.run_dir / "prep"
        self.logf = self.run_dir / "logs" / "edit_run.log"
        self.logf.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utils
    def log(self, *a):
        msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(msg, flush=True)
        with open(self.logf, "a") as f:
            f.write(msg + "\n")

    def gen_dir(self, g: int) -> Path:
        return self.run_dir / "baseline" if g == 0 else R.gen_dir(self.rc, g)

    def incumbent(self, g: int) -> dict:
        """Incumbent after generation g (g = 0: M_0 under loop.start bf16, else Q_0)."""
        inc = read_json(self.gen_dir(g) / "incumbent.json")
        if not inc:
            raise RuntimeError(f"no incumbent recorded for generation {g}")
        return inc

    def q0_record(self) -> dict:
        """Q_0, the uniform reference quantization (budget.reference): baseline/q0/record.json."""
        rec = read_json(self.run_dir / "baseline" / "q0" / "record.json")
        if not rec:
            raise RuntimeError("no Q_0 record (baseline/q0/record.json)")
        return rec

    def original_record(self) -> dict:
        """The incumbent of generation 1 under loop.start bf16: the original BF16 model M_0 (no config, over the
        budget, KL 0 to itself)."""
        bud = read_json(self.run_dir / "baseline" / "budget.json") or {}
        return {"gen": 0, "cid": "m0", "accepted": True, "original": True, "config": None, "config_hash": None,
                "bytes_gib": bud.get("bf16_gib"), "linear_avg_bits": 16.0, "kl": 0.0,
                "model_dir": None if self.rc.mock else str(self.rc.model_path)}

    def budget_gib(self) -> float:
        return float(read_json(self.run_dir / "baseline" / "budget.json")["gib"])

    def source_state(self) -> dict | None:
        return M.fresh_state(self.cfg["mock"]["base_quality"]) if self.rc.mock else None

    def _server(self, name: str, model_dir, quality: float | None, gpus: list[int], args_key: str,
                sleep_mode: bool = False) -> LazyServer:
        if self.rc.mock:
            M.register(self.registry, name, quality)
            extra = []
            path = "mock-" + name
        else:
            extra = server_args(self.cfg, args_key)
            path = model_dir
        return LazyServer(model_path=path, served_name=name, gpu_ids=list(gpus), mock=self.rc.mock,
                          mock_registry=self.registry, seed=self.rc.seed, log_dir=self.run_dir / "logs", extra_args=extra,
                          sleep_mode=sleep_mode)

    def m0_server(self, gpus: list[int] | None = None) -> LazyServer:
        """The original BF16 model M_0 on the editor GPU (editor in frozen mode, and the reference); gpus = a
        replica elsewhere (candidate GPUs, args candidate_args)."""
        on_editor = gpus is None or list(gpus) == list(self.gpu_editor)
        return self._server(f"{self.rc.run_id}-m0", Path(self.rc.model_path),
                            self.cfg["mock"]["base_quality"] if self.rc.mock else None,
                            self.gpu_editor if gpus is None else gpus, "editor_args" if on_editor else "candidate_args",
                            sleep_mode=self.editor_sleep and on_editor)

    def m0_replicas_for(self, m0: LazyServer) -> list[LazyServer]:
        """M_0 on every GPU group of the run (data parallelism for M_0's batch jobs, gpu_layout.m0_replicas):
        [m0 (editor GPU)] + one replica per candidate slot. Close the extra ones when done."""
        if not self.m0_replicas:
            return [m0]
        return [m0] + [self.m0_server(gpus=g) for g in self.cand_slots]

    def model_server(self, model_dir: Path, name: str, gpus: list[int], args_key: str) -> LazyServer:
        """An exported quantized model (fake-quant checkpoint, or mock_model.json in mock mode)."""
        q = M.load_state(Path(model_dir) / "mock_model.json")["quality"] if self.rc.mock else None
        return self._server(name, Path(model_dir), q, gpus, args_key)

    def kl_texts(self, m0: LazyServer, reps: list | None = None) -> list[tuple[str, str]]:
        """(domain, text) calibration texts of generation 0 (GPTQ / AWQ calibration of the layer cache and the
        profile table's contexts; the block keeps its name `kl`). kl.self_generated: prompt + M_0's own greedy
        response (cached in run_dir/kl_selfgen/gen_00.jsonl; M_0 is started only for responses not cached yet), else
        the prompt alone. reps: M_0's replicas already made by the caller (who closes them); else they are made here
        and the extra ones closed on return."""
        texts = kl_texts_for(self.rc, 0, self.datasets, self.cfg)
        if not self.cfg["kl"].get("self_generated"):
            return texts
        extra = []

        def endpoints():                    # only called when some response is not cached yet
            rs = reps if reps is not None else self.m0_replicas_for(m0)
            if reps is None:
                extra.extend(rs[1:])
            return [s.endpoint for s in start_lazy(rs)]
        try:
            return add_self_generated(texts, endpoints, m0.name, self.cfg,
                                      self.run_dir / "kl_selfgen" / "gen_00.jsonl")
        finally:
            for s in extra:                  # the candidate GPUs are needed for candidates next
                s.close()

    def kl_set_ids(self) -> dict:
        """{dataset: ids} of the KL set: kl_eval.n items of generation 0's draw (the screen role of the training
        pool, which excludes the eval / test items and the calibration texts), seeded by the run's seed."""
        out = {}
        for ds in self.datasets:
            pool = sorted(D.ids(self.rc, 0, ds, "screen"))
            n = min(int(self.kl_cfg["n"]), len(pool))
            out[ds] = sorted(random.Random(f"{self.rc.seed}:kl_eval:{self.rc.domain}").sample(pool, n))
        return out

    def kl_reference(self, m0: LazyServer, reps: list, kdir: Path) -> int:
        """The KL set: M_0 answers each item of kl_set_ids() once (thinking-mode sampling with the run's seed, up to
        kl_eval.max_tokens answer tokens; cached in kdir/answers.jsonl) and its top-k prompt logprobs on these traces
        are cached (kdir/ref.jsonl). kdir/traces.json = {"items": [{ds, id, prompt, text}], "kl_eval", "seed"}."""
        from rsicore.kl import reference_logprobs
        kc = self.kl_cfg
        items = []
        for ds, pick in self.kl_set_ids().items():
            recs = load_processed(ds, pick)
            items += [{"ds": ds, "id": i, "prompt": record_text(recs[i], max_chars=10 ** 9)} for i in pick if i in recs]
        eps = [s.endpoint for s in start_lazy(reps)]
        texts = add_self_generated([(it["ds"], it["prompt"]) for it in items], eps, m0.name, self.cfg,
                                   kdir / "answers.jsonl", max_tokens=int(kc["max_tokens"]),
                                   sampling={"temperature": kc.get("temperature"), "top_p": kc.get("top_p"),
                                             "top_k": kc.get("sample_top_k"), "seed": self.rc.seed})
        for it, (_d, t) in zip(items, texts):
            it["text"] = t
        write_json(kdir / "traces.json", {"items": items, "kl_eval": kc, "seed": self.rc.seed})
        reference_logprobs(eps, m0.name, items, kdir / "ref.jsonl", top_k=int(kc["top_k"]))
        return len(items)

    def problem(self, *, tag: str, gen: int, out_dir: Path, budget: float, kl_texts=None,
                incumbent: dict | None = None) -> QuantProblem:
        """Every problem quantizes the ORIGINAL model; the layer cache in run_dir/prep is shared."""
        return QuantProblem(self.rc, self.cfg, tag=tag, gen=gen, out_dir=out_dir, prep_dir=self.prep_dir,
                            source_dir=Path(self.rc.model_path) if not self.rc.mock else None,
                            source_state=self.source_state(), budget_gib=budget, datasets=self.datasets,
                            kl_texts=kl_texts, incumbent=incumbent, log=self.log,
                            prep_gpus=[g[0] for g in self.all_groups] if self.prep_all_gpus else None)

    def export(self, prob: QuantProblem, art: dict, dst: Path):
        """Write the quantized model (fake-quant BF16 checkpoint; mock: its mock state)."""
        if (dst / "fake_quant.json").exists():
            return
        if self.rc.mock:
            norm = prob.norm_of(art)
            stt = prob.mock_state(norm)
            M.save_state(dst / "mock_model.json", stt)
            write_json(dst / "fake_quant.json", {"config": norm, "mock": True, "bytes_gib": art["bytes_gib"]})
        else:  # pragma: no cover - GPU path
            prob.prepare()
            # the accepted candidate's evaluated checkpoint is the same export: move it instead of rewriting
            if not prob.adopt_checkpoint(art, dst):
                prob.materialize(art, dst)

    def _export_and_eval(self, prob: QuantProblem, art: dict, mdir: Path, cid: str) -> dict:
        """Q_0: export once, then its KL on every GPU group (every candidate slot plus the free editor GPU)."""
        self.export(prob, art, mdir)
        prob.exported[art["config_hash"]] = mdir
        lent = prob.add_slot(self.gpu_editor) if self.editor_gpus_own else None
        try:
            return prob.evaluate(art, cid, origin="reference")
        finally:
            if lent is not None:
                prob.remove_slot(lent)

    def incumbent_record(self, g: int, cid: str, art: dict, r: dict, model_dir: Path) -> dict:
        fb = r.get("feedback") or {}
        return {"gen": g, "cid": cid, "accepted": True, "config": art["config"], "config_hash": art["config_hash"],
                "bytes_gib": art["bytes_gib"], "linear_avg_bits": art.get("linear_avg_bits"), "kl": r.get("kl"),
                "kl_quantiles": fb.get("kl_quantiles"), "kl_records": fb.get("records"), "model_dir": str(model_dir)}

    # ------------------------------------------------------------------ search plumbing
    def search_fns(self, prob: QuantProblem, gdir: Path, ed: LazyServer, *, archive: str):
        rc, cfg = self.rc, self.cfg
        ss = search_settings(cfg["search"])
        gate_log = GateLog(gdir / "gate.json")
        doc_api = make_tool_api(prob, cfg)
        es = editor_settings(cfg)
        history = self.search_history(prob.gen) if _history_on(cfg) else None
        sys_prompt = system_prompt(prob, doc_api.describe(), cfg, history=history is not None)
        ekw = editor_kwargs(cfg, server_args=server_args(cfg, "editor_args"))
        tools = self.chat_tools(prob, history) if es["tool_rounds"] > 0 else None
        policy = None
        if rc.mock:                          # the mock editor explores only when the editor has tools
            policy = with_mock_exploration(M.mock_policy, es["mock_tool_rounds"] if tools else 0, MOCK_EXPLORE_CODE)

        def editor_factory(th):
            srv = None if rc.mock else ed.get()
            return Editor(srv.endpoint if srv else "http://mock.invalid/v1", ed.name, system_prompt=sys_prompt,
                          seed=rc.seed + 1000 * prob.gen + th, transcript=gdir / "editor" / f"thread_{th}.jsonl",
                          mock_policy=policy, tools=tools, **ekw)

        def build_fn(program, cid):
            api = make_tool_api(prob, cfg)
            res = run_program(program, api, timeout_s=cfg["sandbox"]["timeout_s"], mem_gb=cfg["sandbox"]["mem_gb"],
                              workdir=gdir / "sandbox" / cid)
            if not res.ok:
                return None, f"program failed: {res.error}\nstderr tail:\n{tail(res.stderr, 1500)}"
            result = res.result
            if isinstance(result, dict) and isinstance(result.get("config"), dict) and "default" not in result:
                note = result.get("note", result["config"].get("note"))      # also accept {"config": ..., "note": ...}
                result = dict(result["config"], **({"note": note} if note is not None else {}))
            try:
                art = prob.build(result, cid)
            except schema.ConfigError as e:
                return None, str(e)
            except (ValueError, TypeError) as e:
                return None, f"invalid config: {e}"
            note = program_note(result)              # its own account, for search_history / the archive
            if note:
                art["note"] = note
            art["cost"] = {"tool_calls": res.info.get("tool_calls"), "program_s": round(res.seconds, 2)}
            art["stdout_tail"] = tail(res.stdout, 600)
            return art, None

        def screen_fn(art, cid):
            """The candidate's ONE evaluation: its KL on the fixed KL set; score = -KL (rsicore.search ranks the
            highest score first, i.e. the lowest KL)."""
            g = kl_candidate(prob, art, cid, gate_log)
            return (None if g.get("kl") is None else -float(g["kl"])), kl_feedback(prob, art, g)

        render = make_render_context(prob, cfg, archive=archive, search=ss, history_tool=history is not None)
        return dict(editor_factory=editor_factory, render_context=render, build_fn=build_fn, screen_fn=screen_fn)

    def search_history(self, g: int) -> SearchHistory:
        """Records for the search_history chat tool (editor.history_tool, default on): the search of generation g
        sees every earlier generation's candidates (gen_01 .. gen_{g-1})."""
        past = {t: d for t in range(1, int(g)) if ((d := R.gen_dir(self.rc, t)) / "candidates.jsonl").exists()}

        def gen_fn(t, d):
            acc = read_json(d / "accepted.json") or {}
            out = {"accepted": acc.get("cid") if acc.get("accepted") else False,
                   "incumbent_kl": acc.get("incumbent_kl")}
            if acc.get("first_quantization"):     # loop.start bf16: that round's incumbent was the BF16 original
                out["first_quantization"] = ("the incumbent was the BF16 original (KL 0): the lowest KL within the "
                                             "budget was accepted")
            if not acc.get("accepted") and acc.get("reason"):
                out["reason"] = str(acc["reason"])[:200]
            return out
        return SearchHistory(past, row_fn=history_row, gen_fn=gen_fn,
                             key_fn=lambda r: (r.get("artifact") or {}).get("config_hash"))

    def chat_tools(self, prob: QuantProblem, history: SearchHistory | None = None) -> ChatTools:
        """The editor's chat tools (editor.tool_rounds > 0): run_python with the exploration API
        (make_tool_api(explore=True): group_error on cached settings only unless editor.explore.gpu_tools),
        read_record and (history given) search_history. Each run_python call gets a fresh API (thread-safe: the
        editor threads propose at the same time)."""
        cfg = self.cfg

        def explore_api(tag, ctx):
            return make_tool_api(prob, cfg, explore=True)
        names = explore_api("doc~x", {}).names()
        doc = RUN_PYTHON_DOC + f" Tools on `api` here: {', '.join(names)}."
        if "group_error" in names and not editor_settings(cfg)["explore"].get("gpu_tools"):
            doc += (" group_error here only accepts settings whose error is cached (those of hessian_table); "
                    "other settings raise an error.")
        return ChatTools(explore_api, run_python_doc=doc, history=history, **chat_tools_kwargs(cfg))

    def run_search(self, prob, gen, gdir, ed, *, archive) -> list:
        """The search ends with the editor asleep and its GPU lent to `prob` as a candidate slot (editor_sleep):
        the caller closes the editor; the KL evaluations / exports then use every slot."""
        fns = self.search_fns(prob, gdir, ed, archive=archive)
        phases = EditorPhases(ed, prob, self.gpu_editor, self.editor_sleep, log=self.log)
        ss = search_settings(self.cfg["search"])
        # candidates evaluated at once = proposals per step; each gets ceil(slots / that) GPUs
        prob.kl_parallel = max(1, min(int(ss["threads"]), int(ss["max_candidates"] or ss["threads"])))
        return S.inner_search(self.rc, gen, threads=ss["threads"], steps=ss["steps"],
                              max_candidates=ss["max_candidates"], out_dir=gdir, parallel=max(2, prob.n_slots),
                              phases=phases, **fns)

    # ------------------------------------------------------------------ baseline (generation 0)
    def baseline(self):
        rc, cfg, st = self.rc, self.cfg, self.st
        bdir = self.run_dir / "baseline"
        bdir.mkdir(exist_ok=True)
        D.ensure_gate(rc)                    # the eval / test sets are excluded from the pool draw
        with st.step("baseline/draw") as todo:
            if todo:
                D.draw_gen(rc, 0)
        if rc.mock:
            M.save_state(bdir / "mock_model.json", self.source_state())
        m0 = self.m0_server()
        reps = self.m0_replicas_for(m0)          # M_0 on every GPU: nothing else runs in generation 0 yet
        try:
            with st.step("baseline/kl_gate") as todo:   # the KL set: M_0's traces + its logprobs on them
                if todo:
                    n = self.kl_reference(m0, reps, bdir / "kl_gate")
                    self.log(f"[baseline] KL set: {n} reference traces written by M_0, its logprobs cached")
            kl0 = self.kl_texts(m0, reps=reps)   # calibration texts (prompt + M_0 response with kl.self_generated)
        finally:
            for s in reps:
                s.close()
        # layer cache + diagnostics of M_0 (once), budget, Q_0
        prob = self.problem(tag="q0", gen=0, out_dir=bdir / "q0", budget=float("inf"), kl_texts=kl0)
        with st.step("baseline/prep") as todo:
            prob.prepare()
        with st.step("baseline/budget") as todo:
            if todo:
                write_json(bdir / "budget.json", self.resolve_budget(prob))
        budget = self.budget_gib()
        prob.budget_gib = budget
        with st.step("baseline/q0") as todo:
            if todo:
                ref = cfg["budget"]["reference"]
                art = prob.build(ref, "q0")
                r = self._export_and_eval(prob, art, bdir / "q0" / "model", "q0")
                inc = self.incumbent_record(0, "q0", art, r, bdir / "q0" / "model")
                write_json(bdir / "q0" / "record.json", inc)        # the uniform reference row (both starts)
                if self.start == "q0":
                    write_json(bdir / "incumbent.json", inc)
                self.log(f"[baseline] budget {budget:.3f} GiB; Q_0 (reference) KL {inc['kl']}")
        if self.start == "bf16":
            with st.step("baseline/start_bf16") as todo:
                if todo:
                    inc = self.original_record()
                    write_json(bdir / "incumbent.json", inc)
                    self.log(f"[baseline] loop.start bf16: the incumbent of generation 1 is the original model "
                             f"({inc['bytes_gib']} GiB > budget {budget:.3f} GiB; KL 0)")
        R.update_summary(rc, 0, self.gen_summary(0))

    def resolve_budget(self, prob: QuantProblem) -> dict:
        bc = self.cfg["budget"]
        ref = bc.get("reference")
        bf16 = sizes.bf16_bytes(prob.spec) / 2 ** 30
        out = {"bf16_gib": round(bf16, 4), "reference": ref}
        if ref:
            rb = prob.size(ref)
            out["reference_gib"] = round(rb["gib"], 4)
            out["reference_linear_avg_bits"] = round(rb["linear_avg_bits"], 3)
        if bc.get("gib"):
            out["gib"], out["source"] = float(bc["gib"]), "budget.gib"
        elif ref:
            out["gib"], out["source"] = float(rb["gib"]), "budget.reference"   # exact: Q_0 fits it
        else:
            raise ValueError("budget: set budget.gib or budget.reference")
        if ref and rb["gib"] > out["gib"] + 1e-9:   # unrounded: reference_gib is rounded (W4 rounds up)
            raise ValueError(f"budget {out['gib']:.3f} GiB is smaller than budget.reference ({out['reference_gib']:.3f} "
                             "GiB), the uniform reference Q_0 built at the budget: lower budget.reference as well")
        out["fraction_of_bf16"] = round(out["gib"] / bf16, 4)
        return out

    # ------------------------------------------------------------------ editor
    def editor_server(self, g: int, m0: LazyServer) -> LazyServer:
        """frozen: M_0; current: the incumbent Q_{g-1}."""
        if self.editor_mode == "frozen":
            return m0
        inc = self.incumbent(g - 1)
        if inc.get("original"):          # loop.start bf16, nothing accepted yet: the incumbent is M_0 itself
            return m0
        return self.model_server(Path(inc["model_dir"]), f"{self.rc.run_id}-q{inc['gen']:02d}-editor",
                                 self.gpu_editor, "editor_args")

    # ------------------------------------------------------------------ generation g
    def archive(self, g: int, *, read_tools: bool = True) -> str:
        """Experience channel: per earlier round a header (new incumbent or why it stayed, the lowest KLs) and ONE
        row per candidate, failures included (prompts.archive_row: config in one line, size, KL, error, note).
        read_tools (search_history / read_record reach earlier rounds): full configs and programs are read on
        demand; else the archive.top_candidates best configs of each round are quoted. archive.max_chars (default
        30000): the oldest rounds' rows are dropped first to fit."""
        acfg = self.cfg["archive"]
        if not acfg.get("enabled", True):
            return ""
        k = int(acfg.get("top_candidates", 3))
        ents = []
        for t in range(1, g):
            gd = R.gen_dir(self.rc, t)
            acc = read_json(gd / "accepted.json") or {}
            cands = S.ranked(S.load_candidates(self.rc, t).values())     # lowest KL first (score = -KL)
            gates = gate_outcomes(gd)
            ents.append({"gen": t, "accepted": bool(acc.get("accepted")), "cid": acc.get("cid"),
                         "first_quantization": bool(acc.get("first_quantization")),
                         "reason": acc.get("reason"), "incumbent_kl": acc.get("incumbent_kl"), "n": len(cands),
                         "best": [(c.cid, -c.score) for c in cands if c.score is not None][:3],
                         "rows": [archive_row(c, gates.get(c.cid)) for c in sorted(cands, key=lambda c: c.cid)],
                         "configs": [(c.cid, (c.artifact or {}).get("config"))
                                     for c in cands if c.score is not None and c.artifact][:k]})
        return archive_text(ents, read_tools=read_tools, max_chars=int(acfg.get("max_chars", 30000)))

    def accept_lowest_kl(self, prob, g: int, gd: Path, ranked: list, inc: dict):
        """End of a generation: every built candidate has ONE entry in gen_XX/gate.json (a built candidate missing
        one, e.g. after a crash, is evaluated now; a failed one gets passed false, 'not evaluated'). Among the
        in-budget evaluated ones (not the incumbent's own config) the LOWEST KL is accepted when it is below the
        incumbent's KL (ties: the earlier cid). First quantization (loop.start bf16, the incumbent is still M_0 with
        KL 0): the lowest KL is accepted whatever it is. Otherwise the incumbent stays."""
        st, p = self.st, f"gen{g:02d}"
        log = GateLog(gd / "gate.json")
        outs = gate_outcomes(gd)
        for c in sorted(ranked, key=lambda c: c.cid):
            if c.cid in outs:
                continue
            if c.artifact is not None and c.error is None:
                kl_candidate(prob, c.artifact, c.cid, log)
            else:
                log.add(c.cid, False, {"reason": "not evaluated: " + (error_line(c.error) or "no artifact")})
        outs = gate_outcomes(gd)
        first = bool(inc.get("original"))
        inc_kl = 0.0 if first else inc.get("kl")
        ok = [c for c in ranked if outs.get(c.cid, {}).get("passed") and c.artifact
              and outs[c.cid].get("kl") is not None and c.artifact.get("config_hash") != inc.get("config_hash")]
        best = min(ok, key=lambda c: (float(outs[c.cid]["kl"]), c.cid)) if ok else None
        base = {"incumbent_kl": inc_kl, **({"first_quantization": True} if first else {})}
        if best is None or (not first and inc_kl is not None and float(outs[best.cid]["kl"]) >= float(inc_kl)):
            why = ("no candidate was built and evaluated within the budget" if best is None else
                   f"lowest KL {outs[best.cid]['kl']:.6f} ({best.cid}) is not below the incumbent's {inc_kl:.6f}")
            write_json(gd / "accepted.json", dict(base, accepted=False, reason=why,
                                                  best=None if best is None else best.cid))
            write_json(gd / "incumbent.json", dict(inc, accepted=False))
            st.mark(f"{p}/accepted", {"accepted": False, "reason": why})
            return
        o = outs[best.cid]
        r = prob.evaluate(best.artifact, best.cid)          # cached
        new = self.incumbent_record(g, best.cid, best.artifact, r, gd / "model")
        write_json(gd / "accepted.json", dict(base, accepted=True, cid=best.cid, config=best.artifact["config"],
                                              config_hash=best.artifact["config_hash"],
                                              bytes_gib=best.artifact["bytes_gib"], budget_gib=prob.budget_gib,
                                              kl=o["kl"], gain=None if inc_kl is None else round(inc_kl - o["kl"], 6),
                                              rule=("first quantization: the lowest KL among the in-budget candidates"
                                                    if first else "the lowest KL of the round, below the incumbent's")))
        write_json(gd / "incumbent.json", new)
        st.mark(f"{p}/accepted", {"accepted": True, "cid": best.cid, "kl": o["kl"]})
        self.log(f"[gen {g}] {sum(1 for x in outs.values() if x.get('kl') is not None)} of {len(outs)} candidates "
                 f"evaluated; accepted {best.cid} (KL {o['kl']}, incumbent {inc_kl})")

    def generation(self, g: int) -> bool:
        """Returns False when the loop stops (patience exhausted)."""
        rc, cfg, st = self.rc, self.cfg, self.st
        budget = self.budget_gib()
        gd = R.gen_dir(rc, g)
        p = f"gen{g:02d}"
        inc = self.incumbent(g - 1)
        m0 = self.m0_server()
        ed = None
        try:
            prob = self.problem(tag=f"g{g:02d}", gen=g, out_dir=gd, budget=budget, incumbent=inc)
            prob.prepare()
            ed = self.editor_server(g, m0)
            if ed is not m0:
                m0.close()
            if st.done(f"{p}/search"):
                ranked = S.ranked(S.load_candidates(rc, g).values())
            else:
                arch = self.archive(g, read_tools=_history_on(cfg))
                ranked = self.run_search(prob, g, gd, ed, archive=arch)
                st.mark(f"{p}/search", {"candidates": len(ranked)})
        finally:
            if ed is not None:
                ed.close()
            m0.close()
        with st.step(f"{p}/gate") as todo:
            if todo:
                self.accept_lowest_kl(prob, g, gd, ranked, inc)
        accepted = bool((read_json(gd / "accepted.json") or {}).get("accepted"))
        new = self.incumbent(g)
        if accepted:
            with st.step(f"{p}/export") as todo:
                if todo:
                    art = {"config_hash": new["config_hash"], "bytes_gib": new["bytes_gib"]}
                    self.export(prob, art, gd / "model")
                    shutil.rmtree(gd / "ckpt", ignore_errors=True)
            self.log(f"[gen {g}] new incumbent {new['cid']}: KL {new['kl']}")
        else:
            shutil.rmtree(gd / "ckpt", ignore_errors=True)   # up to one 52 GB checkpoint per slot
            self.log(f"[gen {g}] incumbent stays ({inc['cid']} from gen {inc['gen']}): "
                     f"{(st.get(f'{p}/accepted') or {}).get('reason')}")
        R.update_summary(rc, g, self.gen_summary(g))
        st.mark(f"{p}/done", {"accepted": accepted})
        stall = 0
        for t in range(g, 0, -1):
            if (st.get(f"gen{t:02d}/done") or {}).get("accepted"):
                break
            stall += 1
        patience = int((cfg.get("loop") or {}).get("patience", 2))
        if stall >= patience:
            st.mark("stopped", {"gen": g, "reason": f"no new incumbent for {stall} generation(s) (patience {patience})"})
            self.log(f"[gen {g}] stop: no new incumbent for {stall} generation(s)")
            return False
        return True

    # ------------------------------------------------------------------ summary
    def gen_summary(self, g: int) -> dict:
        bdir = self.run_dir / "baseline"
        d = self.gen_dir(g)
        out = {"gen": g}
        inc = read_json(d / "incumbent.json") or {}
        q0kl = (read_json(bdir / "q0" / "record.json") or {}).get("kl")
        if inc:
            out["incumbent"] = {"gen": inc["gen"], "cid": inc["cid"], "bytes_gib": inc["bytes_gib"],
                                "linear_avg_bits": inc.get("linear_avg_bits")}
            if inc.get("original"):      # loop.start bf16, nothing accepted yet: no quantization within the budget
                out["incumbent"]["original"] = True
            else:
                out["kl"] = inc.get("kl")
                out["kl_quantiles"] = inc.get("kl_quantiles")
                if inc.get("kl") is not None and q0kl:
                    out["kl_vs_q0"] = round(inc["kl"] / q0kl, 4)      # < 1: closer to M_0 than uniform GPTQ
        if g == 0:
            out["q0_kl"] = q0kl
            out["budget"] = read_json(bdir / "budget.json")
            out["kl_set"] = len((read_json(bdir / "kl_gate" / "traces.json") or {}).get("items") or [])
        else:
            out["accepted"] = bool((read_json(d / "accepted.json") or {}).get("accepted"))
            outs = gate_outcomes(d)
            out["candidate_kl"] = {cid: o.get("kl") for cid, o in sorted(outs.items())}
            cands = read_jsonl(d / "candidates.jsonl")
            out["candidates"] = {"n": len(cands), "errors": sum(1 for c in cands if c.get("error"))}
        return out

    def final(self) -> dict:
        s = read_json(self.run_dir / "summary.json", {}) or {}
        gens = s.get("gens") or {}
        done = sorted(int(k) for k in gens)
        last = gens.get(str(done[-1])) if done else {}
        g0 = gens.get("0") or {}
        return {"budget_gib": (g0.get("budget") or {}).get("gib"), "editor": self.editor_mode,
                "generations_run": max(done) if done else 0,
                "accepted_generations": [g for g in done if g > 0 and (gens[str(g)] or {}).get("accepted")],
                "kl": {"q0": g0.get("q0_kl"), "final": (last or {}).get("kl")},
                "kl_vs_q0": (last or {}).get("kl_vs_q0"), "final_incumbent": (last or {}).get("incumbent")}

    # ------------------------------------------------------------------ main loop
    def run(self):
        rc = self.rc
        self.log(f"[run] {rc.run_id} paper A domain={rc.domain} gens={rc.gens} editor={self.editor_mode} "
                 f"mock={rc.mock} kl_set={self.kl_cfg['n']} items of {self.datasets}")
        write_json(self.run_dir / "edit_config.json", self.cfg)
        self.baseline()
        for g in range(1, rc.gens + 1):
            stop = self.st.get("stopped")
            if stop:
                self.log(f"[run] stopped at gen {stop.get('gen')}: {stop.get('reason')}")
                break
            if self.st.done(f"gen{g:02d}/done"):
                continue
            self.log(f"[gen {g}] budget {self.budget_gib():.3f} GiB, editor = "
                     + ("M_0 (frozen)" if self.editor_mode == "frozen" else f"incumbent from gen {self.incumbent(g - 1)['gen']}"))
            if not self.generation(g):
                break
        s = read_json(self.run_dir / "summary.json", {}) or {}
        s["final"] = self.final()
        write_json(self.run_dir / "summary.json", s)
        f = s["final"]
        self.log(f"[run] done; KL to M_0: Q_0 {f['kl']['q0']} | final {f['kl']['final']} (ratio {f['kl_vs_q0']})")


def main(argv=None):
    ap = C.build_arg_parser("A")
    ap.add_argument("--budget-gib", type=float, default=None,
                    help="fixed packed size in GiB for every generation (default: config budget)")
    args = ap.parse_args(argv)
    from rsicore.servers import install_exit_handlers; install_exit_handlers()  # kill -> stop vLLM servers (2026-11-25)
    rc = C.load_run_config("A", args)
    cfg = effective_config(rc, args)
    Runner(rc, cfg).run()


if __name__ == "__main__":
    main()
