"""Paper A orchestration: iterative self-quantization at a FIXED size (2026-11-19).

    python -I edit_code/run.py --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 \
        --run-id A_math_s42 [--budget-gib 11.5] [--mock] [--set search.max_evaluate=12]

Design: the editor's weights are frozen (loop.editor=frozen: the original BF16 model M_0 writes every
generation; loop.editor=current is the unfrozen ablation where the incumbent writes). What is iterated
is the quantized model: every candidate quantizes the ORIGINAL weights of M_0 to the same budget, and
generation g tries to beat the incumbent Q_{g-1} (best accepted quantization so far).

Generation 0 (baseline/): fixed gate split; M_0 scored on the gate; noise band (sampled screen runs);
held-out evaluation of M_0; layer cache + diagnostics of M_0 (run_dir/prep/, built once, reused by every
generation); budget = budget.gib or the packed size of budget.reference (uniform W4 GPTQ by default);
Q_0 = the reference config, scored on the gate (it is the first incumbent) and on the held-out sets.
Scoring decoding (2026-11-21): every screen / full / gate scoring call (candidates, the incumbent, Q_0, M_0's
baseline gate) uses rsicore.config.stage_decoding(cfg, stage, seed) from the paper config's `decoding` block (null =
tasks.yaml greedy; objective simple: thinking 0.6 / 0.95 / top_k 20, 16384 tokens, screen n 4, gate n 4;
null seed = the run's --seed).
Generation g = 1..gens (gen_XX/): draw -> M_0's KL reference logprobs (M_0 is not scored on the items; skipped with the
self-generated KL texts when KL is unused, Runner.kl_used: objective simple, 2026-11-22) ->
incumbent scored on the same items (the reference of every candidate; shown to the editor) -> inner search (threads x steps, <= max_evaluate evaluations;
2026-11-21 search.charge_screens false: automatic screens are free and api.evaluate has its own pool, see
edit_code.common.search_settings; candidates from the shared cache; one candidate server per candidate-GPU group,
so the threads of a step are built and evaluated at the same time) -> optional baselines -> gate (<= 3 submissions of
candidates whose score beats the incumbent's: fits the budget AND paired gain over the incumbent on the
gate set) -> new incumbent exported + held-out evaluation, or the incumbent stays.
Gate per candidate (2026-11-21, objective simple; search.screen_candidates false + gate_every_candidate true +
program_evaluate_calls 0): the incumbent answers the generation's screen items once (decoding.screen n 4; shared;
summary in the context, full data via screen_results / screen_output); candidates are not screened and have no
api.evaluate; each built candidate gets ONE gate evaluation in screen_fn (gate_candidate: paired vs the incumbent's
gate records, recorded in gen_XX/gate.json as it finishes, so later steps and generations see the statistics); the
end of the generation accepts the passing candidate with the largest gain (Runner.accept_best_gate). The loop stops after
loop.patience generations without a new incumbent. Every step is a rsicore.state.RunState key; rerunning
the same run-id resumes (completed steps are skipped).
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import traceback
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
from rsicore.evaluate import score, score_split  # noqa: E402
from rsicore.gate import fixed_noise_band, measure_noise_band, noise_ids  # noqa: E402
from rsicore.heldout import HELDOUT, heldout_domains, run_heldout  # noqa: E402
from rsicore.sandbox import run_program  # noqa: E402
from rsicore.servers import VLLMServer  # noqa: E402
from rsicore.state import RunState  # noqa: E402

from edit_code import mock as M  # noqa: E402
from edit_code.baselines import run_baselines  # noqa: E402
from edit_code.common import (candidate_args, candidate_groups, deep_merge, error_line, gpu_layout,  # noqa: E402
                              program_note, read_json, read_jsonl, search_settings, tail, write_json)
from edit_code.problem import EvalBudgetExhausted, QuantProblem, add_self_generated, kl_texts_for  # noqa: E402
from edit_code.prompts import archive_row, archive_text, history_row, make_render_context, system_prompt  # noqa: E402
from edit_code.quant import schema, sizes  # noqa: E402
from edit_code.tools import make_tool_api  # noqa: E402

ALL_DOMAINS = ["math", "code", "knowledge", "instruction_following", "tool_calling"]
MOCK_EXPLORE_CODE = "result = api.profile_summary()"     # the mock editor's run_python call (editor.mock_tool_rounds)


def server_args(cfg: dict, args_key: str) -> list[str]:
    """vLLM args of a real server: vllm.<args_key>. The editor's server (editor_args) gets
    --max-model-len = editor.context_tokens (it overrides a --max-model-len written in editor_args; null keeps
    editor_args as written). Candidate servers (candidate_args) keep their own --max-model-len unless
    vllm.candidate_max_model_len is set (2026-11-21, objective simple: 32768; edit_code.common.candidate_args)."""
    if args_key == "candidate_args":
        return candidate_args(cfg)
    args = list((cfg.get("vllm") or {}).get(args_key) or [])
    return apply_context_tokens(args, cfg) if args_key == "editor_args" else args


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
    def sleeping(self) -> bool:
        return bool(self.srv is not None and self.srv.sleeping)

    @property
    def name(self) -> str:
        return self.kw["served_name"]

    def close(self):
        if self.srv is not None:
            self.srv.stop()
            self.srv = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


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
    """rc.paper_cfg (+ mock_overrides in --mock runs, which win over --set and the profiles; then
    mock_objective_overrides[<objective>], 2026-11-21: e.g. objective simple keeps its k threads x 1 step layout
    with free screens in mock runs)."""
    cfg = dict(rc.paper_cfg)   # hardware profile already merged by rsicore.config (before --set)
    if rc.mock:
        cfg = deep_merge(cfg, cfg.get("mock_overrides") or {})
        cfg = deep_merge(cfg, (cfg.get("mock_objective_overrides") or {}).get(cfg.get("objective")) or {})
    if getattr(args, "budget_gib", None):
        cfg = deep_merge(cfg, {"budget": {"gib": float(args.budget_gib)}})
    return cfg


# gate statistics kept per candidate in gen_XX/gate.json (gate per candidate, 2026-11-21; the same keys as
# rsicore.chat_tools.GATE_STATS, so search_history rows show them)
GATE_INFO = ("reason", "accuracy", "reference_accuracy", "gain", "ci", "broken", "fixed", "broken_ids", "fixed_ids",
             "bytes_gib")


def gate_outcomes(d: Path) -> dict:
    """cid -> {"passed", + GATE_INFO present} from <d>/gate.json."""
    out = {}
    for sub in (read_json(Path(d) / "gate.json", None) or {}).get("submissions") or []:
        info = sub.get("info") if isinstance(sub.get("info"), dict) else {}
        out[str(sub.get("cid"))] = dict({k: info[k] for k in GATE_INFO if k in info}, passed=bool(sub.get("passed")))
    return out


class GateLog:
    """<search dir>/gate.json of the gate-per-candidate flow (search.gate_every_candidate, 2026-11-21): the
    submissions-list shape of rsicore.search.submit_to_gate, one entry per candidate (sorted by cid), written as each
    gate evaluation finishes (thread-safe), so later steps' search_history rows already show the gate results."""

    def __init__(self, path: Path):
        self.path, self.lock = Path(path), threading.Lock()

    def add(self, cid: str, passed: bool, info: dict):
        with self.lock:
            log = read_json(self.path, None) or {"submissions": []}
            subs = [x for x in log.get("submissions") or [] if x.get("cid") != cid]
            subs.append({"cid": cid, "passed": bool(passed), "info": info, "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
            write_json(self.path, {"submissions": sorted(subs, key=lambda x: str(x["cid"]))})


def gate_candidate(prob, art: dict, cid: str, log: GateLog) -> dict:
    """ONE gate evaluation of a built candidate (decoding.gate, paired per item with the incumbent's gate records,
    which every candidate of the generation shares; same seed) -> prob.gate_stats; recorded in `log`. A config
    over the budget is refused without evaluation (QuantProblem.build already refuses those)."""
    if art["bytes_gib"] > prob.budget_gib + 1e-9:
        g = {"passed": False, "reason": f"size {art['bytes_gib']:.3f} GiB > budget {prob.budget_gib:.3f}: refused "
             "without evaluation", "bytes_gib": art["bytes_gib"]}
        log.add(cid, False, g)
        return g
    inc = prob.incumbent or {}
    if inc.get("config_hash") and art["config_hash"] == inc["config_hash"]:
        # (2026-11-22, as Paper B's empty 'keep' LoRA): the incumbent's own config is no change, gain exactly 0;
        # re-sampling it would cost a full gate run and could show a misleading noisy PASSED row to later sessions
        s = gate_mean(inc.get("gate_scores"))
        g = {"passed": False, "reason": "identical to the incumbent (no change): paired gain +0.0000, not evaluated",
             "accuracy": s, "reference_accuracy": s, "gain": 0.0, "ci": None, "broken": 0, "fixed": 0,
             "broken_ids": [], "fixed_ids": [], "bytes_gib": art["bytes_gib"]}
        log.add(cid, False, g)
        return g
    r = prob.evaluate(art, "gate", cid, charge=False, origin="gate")
    g = prob.gate_stats(art, r)
    log.add(cid, g["passed"], dict(g, probes=r["feedback"]["probes"]))
    return g


def gate_feedback(prob, art: dict, g: dict) -> dict:
    """The editor's feedback of a gated, unscreened candidate: size, gate statistics and per-domain gate probes
    (no item content, no record paths) and where its quantization error concentrates (group_error, free)."""
    r = prob._evals.get((art["config_hash"], "gate")) or {}
    probes = {d: {k: v for k, v in p.items() if k != "records"}
              for d, p in ((r.get("feedback") or {}).get("probes") or {}).items()}
    fb = {"stage": "gate", "bytes_gib": art["bytes_gib"], "budget_gib": art.get("budget_gib"),
          "linear_avg_bits": art.get("linear_avg_bits"), "gate": g, "probes": probes}
    try:
        fb["error_top_groups"] = prob.group_error(prob.norm_of(art), top=5)["top"]
    except Exception:  # noqa: BLE001 - informative only
        pass
    return fb


def gate_mean(scores: dict | None) -> float | None:
    vals = [v for v in (scores or {}).values() if v is not None]
    return round(sum(vals) / len(vals), 5) if vals else None


class Runner:
    def __init__(self, rc, cfg: dict):
        self.rc, self.cfg = rc, cfg
        self.st = RunState(rc.run_dir)
        self.run_dir = Path(rc.run_dir)
        doms = ALL_DOMAINS if rc.domain == "all" else [rc.domain]
        # data roles (2026-11-21, paper config data_roles, rsicore.draws.roles): self.datasets = the TRAIN role
        # (screen / full / signal items, noise band, feedback examples), self.gate_datasets = the GATE role
        # (baseline gate of M_0 / Q_0, the incumbent's gate records, candidate gate scoring, gate.json). Defaults:
        # the domain's training set for both; objective eval_overfit -> its eval set for both. --domain all: every
        # domain's training set for both (unchanged).
        self.datasets = [ds for d in doms for ds in D.DOMAIN_TRAIN[d]]
        self.gate_datasets = list(self.datasets)
        kd = cfg["kl"].get("domains", "all")
        self.kl_datasets = [ds for d in (ALL_DOMAINS if kd == "all" else doms) for ds in D.DOMAIN_TRAIN[d]]
        if rc.domain != "all":
            tds, gds = D.train_dataset(rc), D.gate_dataset(rc)
            self.datasets, self.gate_datasets = [tds], [gds]
            if D.is_eval_set(tds):                  # an eval-set train role: KL / calibration texts are its prompts
                self.kl_datasets = [tds]            # (the kl role: a fixed seeded subset, rsicore.draws.ids)
            elif kd != "all":                       # scope single: the train role's own texts
                self.kl_datasets = [tds]
        self.registry = self.run_dir / "mock_registry.json"
        roles, unused = gpu_layout(rc, cfg)
        self.gpu_editor, self.gpu_cand = roles["editor"], roles["candidate"]
        self.cand_slots = candidate_groups(rc, cfg)   # one candidate server per slot, evaluated concurrently
        lay = cfg.get("gpu_layout") or {}
        # the editor's GPUs can be used for other work only when they are not shared with the candidates
        own = not (set(self.gpu_editor) & set(self.gpu_cand))
        self.editor_sleep = bool(lay.get("editor_sleep", True)) and own
        self.m0_replicas = bool(lay.get("m0_replicas", True)) and own
        self.parallel_heldout = bool(lay.get("parallel_heldout", True))
        self.heldout_enabled = bool(cfg["heldout"].get("enabled", True))
        self.prep_all_gpus = bool(lay.get("prep_all_gpus", True)) and own
        # every GPU group of the run: the editor's + one per candidate slot
        self.all_groups = ([list(self.gpu_editor)] if own else []) + [list(s) for s in self.cand_slots]
        print(f"[edit_code] hardware={cfg.get('hardware', 'b300')} scope={cfg.get('scope', 'all')} gpus={rc.gpus} "
              f"editor={self.gpu_editor} candidate servers={self.cand_slots} "
              f"({len(self.cand_slots)} candidate(s) evaluated at a time)"
              + (f" UNUSED {unused} (gpu_layout.candidate_group_size)" if unused else ""), flush=True)
        self.editor_mode = (cfg.get("loop") or {}).get("editor", "frozen")
        if self.editor_mode not in ("frozen", "current"):
            raise ValueError(f"loop.editor must be frozen or current, got {self.editor_mode!r}")
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
        """Incumbent after generation g (g = 0: Q_0)."""
        inc = read_json(self.gen_dir(g) / "incumbent.json")
        if not inc:
            raise RuntimeError(f"no incumbent recorded for generation {g}")
        return inc

    def budget_gib(self) -> float:
        return float(read_json(self.run_dir / "baseline" / "budget.json")["gib"])

    def source_state(self) -> dict | None:
        return M.fresh_state(self.cfg["mock"]["base_quality"]) if self.rc.mock else None

    def _server(self, name: str, model_dir, quality: float | None, gpus: list[int], args_key: str,
                tasks: list[str] | None = None, sleep_mode: bool = False) -> LazyServer:
        if self.rc.mock:
            M.register(self.registry, name, quality)
            extra = ["--tasks", ",".join(tasks)] if tasks else []
            path = "mock-" + name
        else:
            extra = server_args(self.cfg, args_key)
            path = model_dir
        return LazyServer(model_path=path, served_name=name, gpu_ids=list(gpus), mock=self.rc.mock,
                          mock_registry=self.registry, seed=self.rc.seed, log_dir=self.run_dir / "logs", extra_args=extra,
                          sleep_mode=sleep_mode)

    def m0_server(self, tasks: list[str] | None = None, gpus: list[int] | None = None) -> LazyServer:
        """The original BF16 model M_0 on the editor GPU (editor in frozen mode, and the reference); gpus = a
        replica elsewhere (candidate GPUs, args candidate_args)."""
        on_editor = gpus is None or list(gpus) == list(self.gpu_editor)
        return self._server(f"{self.rc.run_id}-m0", Path(self.rc.model_path),
                            self.cfg["mock"]["base_quality"] if self.rc.mock else None,
                            self.gpu_editor if gpus is None else gpus, "editor_args" if on_editor else "candidate_args",
                            tasks, sleep_mode=self.editor_sleep and on_editor)

    def m0_replicas_for(self, m0: LazyServer, tasks: list[str] | None = None) -> list[LazyServer]:
        """M_0 on every GPU group of the run (data parallelism for M_0's batch jobs, gpu_layout.m0_replicas):
        [m0 (editor GPU)] + one replica per candidate slot. Close the extra ones when done."""
        if not self.m0_replicas:
            return [m0]
        return [m0] + [self.m0_server(tasks, gpus=g) for g in self.cand_slots]

    def heldout_on(self, servers: list, out_dir: Path, *, lora=None) -> dict:
        """Held-out evaluation with the tasks spread over the given replicas (started together)."""
        cfg = self.cfg
        srvs = start_lazy(servers)
        return run_heldout([s.endpoint for s in srvs], servers[0].name, out_dir, domains=heldout_domains(cfg),
                           mock=self.rc.mock, lora=lora, limit=cfg["heldout"].get("limit"),
                           decoding=cfg["heldout"].get("decoding"))

    def model_heldout(self, model_dir: Path, name: str, out_dir: Path) -> dict:
        """Held-out evaluation of an exported model: one server per free GPU group (gpu_layout.parallel_heldout;
        call it when the editor GPU is free) or just the first candidate slot."""
        groups = self.all_groups if self.parallel_heldout else [self.cand_slots[0]]
        servers = [self.model_server(model_dir, name, g, "candidate_args") for g in groups]
        try:
            return self.heldout_on(servers, out_dir)
        finally:
            for s in servers:
                s.close()

    def model_server(self, model_dir: Path, name: str, gpus: list[int], args_key: str) -> LazyServer:
        """An exported quantized model (fake-quant checkpoint, or mock_model.json in mock mode)."""
        q = M.load_state(Path(model_dir) / "mock_model.json")["quality"] if self.rc.mock else None
        return self._server(name, Path(model_dir), q, gpus, args_key)

    def kl_texts(self, gen: int, m0: LazyServer) -> list[tuple[str, str]]:
        """(domain, text) KL / calibration texts of generation `gen`. kl.self_generated: prompt + M_0's own greedy
        response (cached in run_dir/kl_selfgen/gen_XX.jsonl; M_0 is started only for responses not cached yet),
        else the prompt alone."""
        texts = kl_texts_for(self.rc, gen, self.kl_datasets, self.cfg)
        if not self.cfg["kl"].get("self_generated"):
            return texts
        extra = []

        def endpoints():                    # only called when some response is not cached yet
            reps = self.m0_replicas_for(m0)
            extra.extend(reps[1:])
            return [s.endpoint for s in start_lazy(reps)]
        try:
            return add_self_generated(texts, endpoints, m0.name, self.cfg,
                                      self.run_dir / "kl_selfgen" / f"gen_{gen:02d}.jsonl")
        finally:
            for s in extra:                  # the candidate GPUs are needed for candidates next
                s.close()

    def kl_used(self) -> bool:
        """Whether a search generation needs its KL texts with M_0's responses and M_0's reference logprobs
        (2026-11-22): the score uses KL (search.kl_weight > 0), E_t is on (held-out KL), or baselines run (they may
        calibrate / score KL on them). The generation-0 texts (calibration of the layer cache) are always built."""
        cfg = self.cfg
        return (float(cfg["search"].get("kl_weight", 0.0) or 0.0) > 0 or bool(cfg["e_task"].get("enabled", False))
                or bool(cfg["baselines"].get("methods") or []))

    def data_for(self, gen: int) -> dict:
        return {ds: {"screen": D.ids(self.rc, gen, ds, "screen"), "signal": D.ids(self.rc, gen, ds, "signal")}
                for ds in self.datasets}

    def problem(self, *, tag: str, gen: int, out_dir: Path, budget: float, data: dict, kl_texts, refs_dir: Path,
                eval_budget: int, show_examples: int, incumbent: dict | None = None, kl_heldout=()) -> QuantProblem:
        """Every problem quantizes the ORIGINAL model; the layer cache in run_dir/prep is shared."""
        return QuantProblem(self.rc, self.cfg, tag=tag, gen=gen, out_dir=out_dir, prep_dir=self.prep_dir,
                            source_dir=Path(self.rc.model_path) if not self.rc.mock else None,
                            source_state=self.source_state(), budget_gib=budget, datasets=self.datasets, data=data,
                            gate_datasets=self.gate_datasets,
                            kl_texts=kl_texts, kl_heldout=kl_heldout, refs_dir=refs_dir,
                            noise=read_json(self.run_dir / "baseline" / "noise.json", {}),
                            gate_baseline=read_json(self.run_dir / "baseline" / "gate.json", {}),
                            eval_budget=eval_budget, show_examples=show_examples, incumbent=incumbent, log=self.log,
                            prep_gpus=[g[0] for g in self.all_groups] if self.prep_all_gpus else None)

    def export(self, prob: QuantProblem, art: dict, dst: Path):
        """Write the quantized model (fake-quant BF16 checkpoint; mock: its mock state)."""
        if (dst / "fake_quant.json").exists():
            return
        if self.rc.mock:
            norm = prob.norm_of(art)
            stt = M.apply_config(prob.spec, prob.source_state, norm, self.cfg["mock"].get("lam", 1.2))
            M.save_state(dst / "mock_model.json", stt)
            write_json(dst / "fake_quant.json", {"config": norm, "mock": True, "bytes_gib": art["bytes_gib"]})
        else:  # pragma: no cover - GPU path
            prob.prepare()
            prob.materialize(art, dst)

    def incumbent_record(self, g: int, cid: str, art: dict, gate_fb: dict, model_dir: Path, *, accepted: bool) -> dict:
        probes = gate_fb.get("probes") or {}
        gate = {p["dataset"]: {"score": p.get("score"), "records": p.get("records")} for p in probes.values()}
        return {"gen": g, "cid": cid, "accepted": accepted, "config": art["config"], "config_hash": art["config_hash"],
                "bytes_gib": art["bytes_gib"], "linear_avg_bits": art.get("linear_avg_bits"), "gate": gate,
                "gate_scores": {ds: v["score"] for ds, v in gate.items()}, "model_dir": str(model_dir)}

    # ------------------------------------------------------------------ search plumbing
    def search_fns(self, prob: QuantProblem, gdir: Path, ed: LazyServer, *, task: str, archive: str, scfg: dict):
        rc, cfg = self.rc, self.cfg
        ss = search_settings(scfg)
        per_prog, free, ev_tool = ss["per_program"], not ss["charge_screens"], ss["evaluate_tool"]
        gate_log = GateLog(gdir / "gate.json") if ss["gate_every_candidate"] else None
        doc_api = make_tool_api(prob, cfg, cid="doc", per_program_evaluate=per_prog, free_repeats=free,
                                evaluate_tool=ev_tool)
        es = editor_settings(cfg)
        history = self.search_history(prob.gen, task) if es["tool_rounds"] > 0 and es.get("history_tool", True) else None
        sys_prompt = system_prompt(prob, doc_api.describe(), cfg, history=history is not None, search=ss)
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
            if not free and prob.evals_left <= 0:      # free screens: programs always run (2026-11-21)
                return None, "evaluate budget of this generation is exhausted; program not run"
            api = make_tool_api(prob, cfg, cid=cid, per_program_evaluate=per_prog, free_repeats=free,
                                evaluate_tool=ev_tool)
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
            note = program_note(result)              # (2026-11-22) its own account, for search_history / the archive
            if note:
                art["note"] = note
            art["cost"] = {"tool_calls": res.info.get("tool_calls"), "tool_evaluate": api.spent.get("evaluate", 0),
                           "program_s": round(res.seconds, 2)}
            art["stdout_tail"] = tail(res.stdout, 600)
            return art, None

        def screen_fn(art, cid):
            """screen (search.screen_candidates) and / or ONE gate evaluation (search.gate_every_candidate, 2026-11-21:
            an unscreened candidate's score is its paired gate gain over the incumbent)."""
            sc, fb = None, None
            if ss["screen_candidates"]:
                try:                                # search.charge_screens false: the automatic screen is free
                    r = prob.evaluate(art, "screen", cid, charge=not free)
                except EvalBudgetExhausted as e:
                    return None, {"error": str(e)}
                sc, fb = r["score"], r["feedback"]
            if gate_log is None:
                return sc, fb
            g = gate_candidate(prob, art, cid, gate_log)
            return (g.get("gain"), gate_feedback(prob, art, g)) if fb is None else (sc, dict(fb, gate=g))

        def full_fn(art, cid):
            r = prob.evaluate(art, "full", cid)
            return r["score"], r["feedback"]

        render = make_render_context(prob, cfg, task=task, archive=archive, search=ss,
                                     history_tool=history is not None)
        return dict(editor_factory=editor_factory, render_context=render, build_fn=build_fn, screen_fn=screen_fn,
                    full_fn=full_fn)

    def search_history(self, g: int, task: str) -> SearchHistory:
        """Records for the search_history chat tool (editor.history_tool, default on; 2026-11-21): the main
        search of generation g sees every earlier generation's candidates (gen_01 .. gen_{g-1}); the E_t
        session (task "e", a fresh session without archive) sees none."""
        past = {} if task != "main" else {t: R.gen_dir(self.rc, t) for t in range(1, int(g))
                                          if (R.gen_dir(self.rc, t) / "candidates.jsonl").exists()}

        def gen_fn(t, d):
            acc = read_json(d / "accepted.json") or {}
            out = {"accepted": acc.get("cid") if acc.get("accepted") else False}
            if not acc.get("accepted") and acc.get("gate_reason"):
                out["gate_reason"] = str(acc["gate_reason"])[:200]
            rs = read_json(d / "reference_screen.json")       # gate per candidate: the incumbent's screen run
            if rs:
                out["reference_screen"] = {k: rs.get(k) for k in ("accuracy", "correct_counts", "truncation_rate")}
            return out
        return SearchHistory(past, row_fn=history_row, gen_fn=gen_fn,
                             key_fn=lambda r: (r.get("artifact") or {}).get("config_hash"))

    def chat_tools(self, prob: QuantProblem, history: SearchHistory | None = None) -> ChatTools:
        """The editor's chat tools (editor.tool_rounds > 0): run_python with the exploration API
        (make_tool_api(explore=True): no evaluate; group_error on cached settings only unless
        editor.explore.gpu_tools), read_record and (history given) search_history. Each run_python call gets a
        fresh API (thread-safe: the editor threads propose at the same time); exploration never spends the
        generation's evaluate budget. editor.explore.persistent (2026-11-21, simple): rsicore keeps one session per
        proposal and calls explore_api once per proposal (the API is stateless apart from its call log, so that is
        safe); with search.screen_candidates false it also carries screen_results / screen_output."""
        cfg = self.cfg

        def explore_api(tag, ctx):
            return make_tool_api(prob, cfg, cid=tag, per_program_evaluate=0, explore=True)
        names = explore_api("doc~x", {}).names()
        doc = RUN_PYTHON_DOC + f" Tools on `api` here: {', '.join(names)}."
        if "group_error" in names and not editor_settings(cfg)["explore"].get("gpu_tools"):
            doc += (" group_error here only accepts settings whose error is cached (those of hessian_table); "
                    "other settings raise an error.")
        return ChatTools(explore_api, run_python_doc=doc, history=history, **chat_tools_kwargs(cfg))

    def run_search(self, prob, gen, gdir, ed, *, task, archive, scfg) -> list:
        """The search ends with the editor asleep and its GPU lent to `prob` as a candidate slot (editor_sleep):
        the caller closes the editor; the gate / exports then use every slot."""
        fns = self.search_fns(prob, gdir, ed, task=task, archive=archive, scfg=scfg)
        phases = EditorPhases(ed, prob, self.gpu_editor, self.editor_sleep, log=self.log)
        ss = search_settings(scfg)
        return S.inner_search(self.rc, gen, threads=ss["threads"], steps=ss["steps"],
                              max_candidates=ss["max_candidates"], top_full=ss["top_full"],
                              rerun_seeds=ss["rerun_seeds"], out_dir=gdir, parallel=max(2, prob.n_slots),
                              phases=phases, **fns)

    def shown_items(self, prob: QuantProblem) -> dict:
        shown = {}
        for r in read_jsonl(prob.out_dir / "evals.jsonl"):
            for x in (r.get("feedback") or {}).get("regressions") or []:
                shown.setdefault(x["dataset"], set()).add(x["id"])
        return {k: sorted(v) for k, v in shown.items()}

    # ------------------------------------------------------------------ baseline (generation 0)
    def baseline(self):
        rc, cfg, st = self.rc, self.cfg, self.st
        bdir = self.run_dir / "baseline"
        bdir.mkdir(exist_ok=True)
        D.ensure_gate(rc)
        with st.step("baseline/draw") as todo:
            if todo:
                D.draw_gen(rc, 0)
        if rc.mock:
            M.save_state(bdir / "mock_model.json", self.source_state())
        m0 = self.m0_server()
        reps = self.m0_replicas_for(m0)          # M_0 on every GPU: nothing else runs in generation 0 yet
        try:
            with st.step("baseline/gate") as todo:
                if todo:
                    eps = [s.endpoint for s in start_lazy(reps)]
                    res = {}
                    dec = C.stage_decoding(cfg, "gate", rc.seed)   # the candidates' gate decoding (2026-11-21)
                    for ds in self.gate_datasets:      # the gate role
                        r = score_split(eps, m0.name, ds, pool="pool", ids=D.gate_ids(rc, ds),
                                        out_dir=bdir / "gate", mock=rc.mock, decoding=dec)
                        res[ds] = _acc(r.records)
                    write_json(bdir / "gate.json", res)
                    self.log("[baseline] M_0 gate scores", res)
            with st.step("baseline/noise") as todo:
                if todo:
                    write_json(bdir / "noise.json", self.noise_band(m0, bdir))
            with st.step("baseline/heldout") as todo:
                if todo and self.heldout_enabled:   # heldout.enabled false (objective simple): no held-out report
                    h = self.heldout_on(reps, bdir / "heldout")
                    write_json(bdir / "heldout.json", h)
            # kl_texts creates its own candidate replicas; release the baseline replicas first.
            for replica in reps[1:]:
                replica.close()
            kl0 = self.kl_texts(0, m0)       # calibration / KL texts (prompt + M_0 response with kl.self_generated)
        finally:
            for s in reps:
                s.close()
        # layer cache + diagnostics of M_0 (once), budget, Q_0
        prob = self.problem(tag="q0", gen=0, out_dir=bdir / "q0", budget=float("inf"), data=self.data_for(0),
                            kl_texts=kl0, refs_dir=bdir / "q0" / "refs",
                            eval_budget=0, show_examples=0)
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
                r = prob.evaluate(art, "gate", "q0", charge=False, origin="reference")
                self.export(prob, art, bdir / "q0" / "model")
                inc = self.incumbent_record(0, "q0", art, r["feedback"], bdir / "q0" / "model", accepted=True)
                write_json(bdir / "incumbent.json", inc)
                self.log(f"[baseline] budget {budget:.3f} GiB; Q_0 (reference) gate scores {inc['gate_scores']}")
        with st.step("baseline/q0_heldout") as todo:
            if todo and self.heldout_enabled:
                h = self.model_heldout(bdir / "q0" / "model", f"{rc.run_id}-q0", bdir / "q0" / "heldout")
                write_json(bdir / "q0" / "heldout.json", h)
        if cfg["e_task"].get("enabled", False):
            self.run_e_task(0)
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
                             "GiB), which is the starting incumbent Q_0: lower budget.reference as well")
        out["fraction_of_bf16"] = round(out["gib"] / bf16, 4)
        return out

    def noise_band(self, ed: LazyServer, bdir: Path) -> dict:
        """{dataset: band} (max - min of repeated sampled screen runs of M_0); details in noise_detail.json.
        noise.fixed gives the band directly (no re-runs for those datasets); noise.n scores a subset of the
        generation-0 screen items (null = all of them)."""
        rc, ncfg = self.rc, self.cfg["noise"]
        detail = {}
        for ds in self.datasets:
            fixed = fixed_noise_band(ncfg.get("fixed"), ds)
            if fixed is not None:
                detail[ds] = {"band": fixed, "source": "fixed (noise.fixed)", "task": ds}
                continue
            srv = ed.get()
            ids = noise_ids(D.ids(rc, 0, ds, "screen"), ncfg.get("n"), rc.seed)
            detail[ds] = measure_noise_band(srv.endpoint, ed.name, ds, ids, bdir / "noise" / ds,
                                            repeats=int(ncfg.get("repeats", 3)),
                                            temperature=float(ncfg.get("temperature", 0.0)),
                                            top_p=float(ncfg.get("top_p", 1.0)), base_seed=rc.seed,
                                            mock=rc.mock)   # source: measured | greedy (temperature 0: band 0)
        write_json(bdir / "noise_detail.json", detail)
        out = {ds: round(v["band"], 4) for ds, v in detail.items()}
        self.log("[baseline] noise band", out)
        return out

    # ------------------------------------------------------------------ editor
    def editor_server(self, g: int, m0: LazyServer) -> LazyServer:
        """frozen: M_0 (the same server as the references); current: the incumbent Q_{g-1}."""
        if self.editor_mode == "frozen":
            return m0
        inc = self.incumbent(g - 1)
        return self.model_server(Path(inc["model_dir"]), f"{self.rc.run_id}-q{inc['gen']:02d}-editor",
                                 self.gpu_editor, "editor_args")

    # ------------------------------------------------------------------ E_t (optional)
    def run_e_task(self, g: int):
        """Fresh session (no archive, generation-0 items) of generation g's editor on the fixed-size task,
        starting from Q_0. Informative only with loop.editor=current (a frozen editor does not change)."""
        rc, cfg, st = self.rc, self.cfg, self.st
        key = f"e_t/{g:02d}"
        if st.done(key):
            return
        ec = cfg["e_task"]
        out = self.gen_dir(g) / "e_t"
        m0 = self.m0_server()
        texts = self.kl_texts(0, m0)                # the generation-0 texts (same cache as the baseline)
        # E_t's own shape; max_candidates / program_evaluate_total follow from it (2026-11-21)
        scfg = {k: v for k, v in dict(cfg["search"], threads=ec["threads"], steps=ec["steps"],
                                      max_evaluate=ec["evaluate"], top_full=ec.get("top_full", 1)).items()
                if k not in ("max_candidates", "program_evaluate_total")}
        prob = self.problem(tag=f"e{g:02d}", gen=g, out_dir=out, budget=float(ec.get("budget_gib") or self.budget_gib()),
                            data=self.data_for(0), kl_texts=texts[0::2], kl_heldout=texts[1::2],
                            refs_dir=self.run_dir / "e_task" / "refs", eval_budget=search_settings(scfg)["eval_budget"],
                            show_examples=0, incumbent=self.incumbent(0))
        prob.prepare()
        ed = self.editor_server(g + 1, m0)          # frozen: M_0; current: the incumbent after generation g
        try:
            srv = m0.get()
            prob.compute_refs(srv.endpoint, m0.name)
            prob.evaluate_incumbent()               # Q_0 on the generation-0 items: the candidates' reference
            if ed is not m0:
                m0.close()
            ranked = self.run_search(prob, g, out, ed, task="e", archive="", scfg=scfg)
        finally:
            ed.close()
            m0.close()
        best = next((c for c in ranked if c.score is not None and c.artifact), None)
        res = {"gen": g, "editor": ed.name, "budget_gib": prob.budget_gib, "n_candidates": len(ranked),
               "n_errors": sum(1 for c in ranked if c.error), "best": None}
        if best is not None:
            hk = prob.heldout_kl(best.artifact)
            with prob.serve(best.artifact, f"{rc.run_id}-e{g:02d}-best") as srv:
                h = run_heldout(srv.endpoint, srv.served_name, out / "heldout",
                                domains=heldout_domains(cfg), mock=rc.mock,
                                limit=ec.get("heldout_limit"), decoding=cfg["heldout"].get("decoding"))
            res["best"] = {"cid": best.cid, "score": best.score, "bytes_gib": best.artifact["bytes_gib"],
                           "config": best.artifact["config"], "heldout_kl": hk, "heldout": h,
                           "heldout_by_domain": _by_domain(h)}
        write_json(out / "result.json", res)
        st.mark(key, {"best": (res["best"] or {}).get("cid")})
        self.log(f"[E_{g}] best {(res['best'] or {}).get('cid')} score {(res['best'] or {}).get('score')}")

    # ------------------------------------------------------------------ generation g
    def archive(self, g: int, *, read_tools: bool = True) -> str:
        """Experience channel (2026-11-22, structured as Paper B's archive): per earlier round a header (new incumbent
        or why it stayed, the largest gate gains, the incumbent's screen summary from gen_XX/reference_screen.json) and
        ONE row per candidate, failures included (prompts.archive_row: config in one line, size, gate result, error,
        note). read_tools (search_history / read_record reach earlier rounds): full configs and programs are read on
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
            cands = S.ranked(S.load_candidates(self.rc, t).values())
            gates = gate_outcomes(gd)
            ss = search_settings(self.cfg["search"])
            gate_every = ss["gate_every_candidate"] and not ss["screen_candidates"]   # scores = paired gate gains
            ents.append({"gen": t, "accepted": bool(acc.get("accepted")), "cid": acc.get("cid"),
                         "gate_reason": acc.get("gate_reason"), "n": len(cands), "gate_every": gate_every,
                         "best": [(c.cid, c.score) for c in cands if c.score is not None][:3],
                         "incumbent_score": (read_json(gd / "incumbent_scores.json") or {}),
                         "reference_screen": read_json(gd / "reference_screen.json"),
                         "rows": [archive_row(c, gates.get(c.cid)) for c in sorted(cands, key=lambda c: c.cid)],
                         "configs": [(c.cid, (c.artifact or {}).get("config"))
                                     for c in cands if c.score is not None and c.artifact][:k]})
        return archive_text(ents, read_tools=read_tools, max_chars=int(acfg.get("max_chars", 30000)))

    def accept_best_gate(self, prob, g: int, gd: Path, ranked: list, inc: dict, budget: float):
        """End of a gate-per-candidate generation (search.gate_every_candidate, 2026-11-21): every candidate has ONE
        gate entry in gen_XX/gate.json (a built candidate missing one, e.g. after a crash, is gated now; a failed one
        gets passed false, 'not evaluated'); among the passing ones (size, gate.min_delta, CI if required; not the
        incumbent's own config) the LARGEST paired gain is accepted (ties: higher screen score, then the earlier
        cid); none passes -> the incumbent stays."""
        st, p = self.st, f"gen{g:02d}"
        log = GateLog(gd / "gate.json")
        outs = gate_outcomes(gd)
        for c in sorted(ranked, key=lambda c: c.cid):
            if c.cid in outs:
                continue
            if c.artifact is not None and c.error is None:
                gate_candidate(prob, c.artifact, c.cid, log)
            else:
                log.add(c.cid, False, {"reason": "not evaluated: " + (error_line(c.error) or "no artifact")})
        outs = gate_outcomes(gd)
        ok = [c for c in ranked if outs.get(c.cid, {}).get("passed") and c.artifact
              and outs[c.cid].get("gain") is not None and float(outs[c.cid]["gain"]) > 0   # strictly better (min_delta 0)
              and c.artifact.get("config_hash") != inc["config_hash"]]
        def screen_score(c):                    # an unscreened candidate's screen slot holds its gate feedback
            sc = c.screen or {}
            return None if (sc.get("feedback") or {}).get("stage") == "gate" else sc.get("score")
        acc = min(ok, key=lambda c: (-outs[c.cid]["gain"], -S._sc(screen_score(c)), c.cid)) if ok else None
        if acc is None:
            reasons = [f"{cid}: {o.get('reason')}" for cid, o in sorted(outs.items())] or ["no candidate"]
            write_json(gd / "accepted.json", {"accepted": False, "gate_reason": reasons})
            write_json(gd / "incumbent.json", dict(inc, accepted=False))
            st.mark(f"{p}/accepted", {"accepted": False, "reasons": reasons})
            return
        o = outs[acc.cid]
        gfb = prob.evaluate(acc.artifact, "gate", acc.cid, charge=False, origin="gate")["feedback"]   # cached
        new = self.incumbent_record(g, acc.cid, acc.artifact, gfb, gd / "model", accepted=True)
        write_json(gd / "accepted.json", {"accepted": True, "cid": acc.cid, "config": acc.artifact["config"],
                                          "config_hash": acc.artifact["config_hash"],
                                          "bytes_gib": acc.artifact["bytes_gib"], "budget_gib": budget,
                                          "score": acc.score, "gain": o.get("gain"), "ci": o.get("ci"),
                                          "gate_reason": o.get("reason"),
                                          "rule": "largest passing gate gain of the generation"})
        write_json(gd / "incumbent.json", new)
        st.mark(f"{p}/accepted", {"accepted": True, "cid": acc.cid, "gain": o.get("gain")})
        self.log(f"[gen {g}] gate per candidate: {sum(1 for x in outs.values() if x.get('passed'))} of {len(outs)} "
                 f"passed; accepted {acc.cid} (gain {o.get('gain')})")

    def generation(self, g: int) -> bool:
        """Returns False when the loop stops (patience exhausted)."""
        rc, cfg, st = self.rc, self.cfg, self.st
        budget = self.budget_gib()
        gd = R.gen_dir(rc, g)
        p = f"gen{g:02d}"
        inc = self.incumbent(g - 1)
        with st.step(f"{p}/draw") as todo:
            if todo:
                D.draw_gen(rc, g)
        m0 = self.m0_server(tasks=self.datasets)
        ed = None
        try:
            # (2026-11-22) KL unused (objective simple: kl_weight 0, no E_t, no baselines): the prompts only, without
            # M_0's self-generated responses and reference logprobs (Paper B does no KL work under simple either)
            kl_used = self.kl_used()
            prob = self.problem(tag=f"g{g:02d}", gen=g, out_dir=gd, budget=budget, data=self.data_for(g),
                                kl_texts=(self.kl_texts(g, m0) if kl_used
                                          else kl_texts_for(self.rc, g, self.kl_datasets, self.cfg)),
                                refs_dir=gd / "refs",
                                eval_budget=search_settings(cfg["search"])["eval_budget"],
                                show_examples=cfg["feedback"].get("examples", 2), incumbent=inc)
            prob.prepare()
            with st.step(f"{p}/refs") as todo:
                if todo and kl_used:                # M_0 = the KL reference only (not scored on the items)
                    srv = m0.get()
                    prob.compute_refs(srv.endpoint, m0.name)
            # the incumbent on this generation's items = the reference of every candidate (uncharged; cached)
            prob.evaluate_incumbent()
            write_json(gd / "incumbent_scores.json", prob.incumbent_scores)
            if prob.screen_tools:                   # the latest model's screen run, for later rounds (archive / history)
                write_json(gd / "reference_screen.json", prob.screen_summary(examples=0))
            ed = self.editor_server(g, m0)
            if ed is not m0:
                m0.close()
            if st.done(f"{p}/search"):
                ranked = S.ranked(S.load_candidates(rc, g).values())
            else:
                es = editor_settings(cfg)
                arch = self.archive(g, read_tools=es["tool_rounds"] > 0 and es.get("history_tool", True))
                ranked = self.run_search(prob, g, gd, ed, task="main", archive=arch, scfg=cfg["search"])
                st.mark(f"{p}/search", {"candidates": len(ranked), "evaluations": prob.evals_used})
        finally:
            if ed is not None:
                ed.close()
            m0.close()
        methods = cfg["baselines"].get("methods") or []
        with st.step(f"{p}/baselines") as todo:
            if todo and methods:
                out = run_baselines(prob, cfg, methods, gd / "baselines",
                                    stage_full=cfg["baselines"].get("full", False), log=self.log)
                write_json(gd / "baselines" / "summary.json", out)
        with st.step(f"{p}/gate") as todo:
            if todo and search_settings(cfg["search"])["gate_every_candidate"]:
                self.accept_best_gate(prob, g, gd, ranked, inc, budget)
            elif todo:
                # only candidates that beat the incumbent on this generation's items are submitted
                def beats(c):
                    if c.score is None or not c.artifact or c.artifact.get("config_hash") == inc["config_hash"]:
                        return False
                    ref = prob.incumbent_scores.get("full" if c.full else "screen")
                    return ref is None or c.score > ref
                subs = [c for c in ranked if beats(c)]

                def gate_fn(c):
                    r = prob.evaluate(c.artifact, "gate", c.cid, charge=False, origin="gate")
                    gi = r["feedback"]["gate"]
                    return gi["passed"], {"reason": gi["reason"], "bytes_gib": c.artifact["bytes_gib"],
                                          "probes": r["feedback"]["probes"]}
                acc = S.submit_to_gate(rc, g, subs, gate_fn, max_submissions=cfg["gate"]["max_submissions"],
                                       parallel=prob.n_slots)   # the submissions are gate-evaluated at once
                gl = read_json(gd / "gate.json", {"submissions": []}) or {"submissions": []}
                if acc is None:
                    reasons = [s["info"].get("reason") for s in gl["submissions"]] or \
                        ["no candidate beat the incumbent on this generation's items"]
                    write_json(gd / "accepted.json", {"accepted": False, "gate_reason": reasons})
                    write_json(gd / "incumbent.json", dict(inc, accepted=False))
                    st.mark(f"{p}/accepted", {"accepted": False, "reasons": reasons})
                else:
                    reason = next(s["info"]["reason"] for s in gl["submissions"] if s["cid"] == acc.cid)
                    gfb = prob.evaluate(acc.artifact, "gate", acc.cid, charge=False, origin="gate")["feedback"]
                    new = self.incumbent_record(g, acc.cid, acc.artifact, gfb, gd / "model", accepted=True)
                    write_json(gd / "accepted.json", {"accepted": True, "cid": acc.cid, "config": acc.artifact["config"],
                                                      "config_hash": acc.artifact["config_hash"],
                                                      "bytes_gib": acc.artifact["bytes_gib"], "budget_gib": budget,
                                                      "score": acc.score, "gate_reason": reason})
                    write_json(gd / "incumbent.json", new)
                    st.mark(f"{p}/accepted", {"accepted": True, "cid": acc.cid})
        with st.step(f"{p}/mark_shown") as todo:
            if todo:
                D.mark_shown(rc, g, self.shown_items(prob))
        accepted = bool((read_json(gd / "accepted.json") or {}).get("accepted"))
        new = self.incumbent(g)
        if accepted:
            with st.step(f"{p}/export") as todo:
                if todo:
                    art = {"config_hash": new["config_hash"], "bytes_gib": new["bytes_gib"]}
                    self.export(prob, art, gd / "model")
                    shutil.rmtree(gd / "ckpt", ignore_errors=True)
            with st.step(f"{p}/heldout") as todo:
                if todo and self.heldout_enabled:   # the editor is closed: every GPU group is free
                    h = self.model_heldout(gd / "model", f"{rc.run_id}-q{g:02d}", gd / "heldout")
                    write_json(gd / "heldout.json", h)
            self.log(f"[gen {g}] new incumbent {new['cid']}: gate {new['gate_scores']}")
        else:
            self.log(f"[gen {g}] incumbent stays ({inc['cid']} from gen {inc['gen']}): "
                     f"{(st.get(f'{p}/accepted') or {}).get('reasons')}")
        if cfg["e_task"].get("enabled", False):
            self.run_e_task(g)
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
    def heldout_of(self, g: int) -> dict | None:
        """Held-out results of the incumbent after generation g."""
        inc = read_json(self.gen_dir(g) / "incumbent.json") or {}
        if not inc:
            return None
        src = self.run_dir / "baseline" / "q0" if inc.get("gen", 0) == 0 else R.gen_dir(self.rc, inc["gen"])
        return read_json(src / "heldout.json")

    def gen_summary(self, g: int) -> dict:
        bdir = self.run_dir / "baseline"
        d = self.gen_dir(g)
        out = {"gen": g}
        inc = read_json(d / "incumbent.json") or {}
        q0 = read_json(bdir / "incumbent.json") or {}
        m0_gate = read_json(bdir / "gate.json") or {}
        if inc:
            out["incumbent"] = {"gen": inc["gen"], "cid": inc["cid"], "bytes_gib": inc["bytes_gib"],
                                "linear_avg_bits": inc.get("linear_avg_bits")}
            out["gate_scores"] = inc["gate_scores"]
            out["gate_mean"] = gate_mean(inc["gate_scores"])
            q0m, m0m = gate_mean(q0.get("gate_scores")), gate_mean(m0_gate)
            if out["gate_mean"] is not None and q0m is not None:
                out["gain_vs_q0"] = round(out["gate_mean"] - q0m, 5)
                if m0m is not None and m0m > q0m:
                    out["gap_closed"] = round((out["gate_mean"] - q0m) / (m0m - q0m), 4)
        h = self.heldout_of(g)
        if h:
            out["heldout"] = {t: v.get("accuracy") for t, v in h.items()}
            out["heldout_by_domain"] = _by_domain(h)
            b = _by_domain(read_json(bdir / "heldout.json", {}) or {})
            out["heldout_ratio_vs_m0"] = {k: (None if not b.get(k) or v is None else round(v / b[k], 4))
                                          for k, v in out["heldout_by_domain"].items()}
        e = read_json(d / "e_t" / "result.json")
        if e and e.get("best"):
            out["E"] = {"score": e["best"]["score"], "heldout_kl": e["best"]["heldout_kl"],
                        "heldout_by_domain": e["best"]["heldout_by_domain"], "bytes_gib": e["best"]["bytes_gib"]}
        if g == 0:
            out["m0_gate_scores"] = m0_gate
            out["m0_heldout_by_domain"] = _by_domain(read_json(bdir / "heldout.json", {}) or {})
            out["noise_band"] = read_json(bdir / "noise.json")
            out["budget"] = read_json(bdir / "budget.json")
        else:
            out["accepted"] = bool((read_json(d / "accepted.json") or {}).get("accepted"))
            out["incumbent_scores_this_gen"] = read_json(d / "incumbent_scores.json")
            out["gate"] = read_json(d / "gate.json")
            out["baselines"] = read_json(d / "baselines" / "summary.json")
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
                "gate_mean": {"m0": gate_mean(g0.get("m0_gate_scores")), "q0": g0.get("gate_mean"),
                              "final": (last or {}).get("gate_mean")},
                "gain_vs_q0": (last or {}).get("gain_vs_q0"), "gap_closed": (last or {}).get("gap_closed"),
                "final_incumbent": (last or {}).get("incumbent")}

    # ------------------------------------------------------------------ main loop
    def run(self):
        rc = self.rc
        self.log(f"[run] {rc.run_id} paper A domain={rc.domain} gens={rc.gens} editor={self.editor_mode} "
                 f"mock={rc.mock} scope={self.cfg.get('scope', 'all')} datasets={self.datasets} "
                 f"gate_datasets={self.gate_datasets} kl_datasets={self.kl_datasets} "
                 f"heldout={heldout_domains(self.cfg)}")
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
        self.log(f"[run] done; gate mean M_0 {f['gate_mean']['m0']} | Q_0 {f['gate_mean']['q0']} | "
                 f"final {f['gate_mean']['final']} (gap closed {f['gap_closed']})")


def _acc(records: list[dict]):
    per = {}
    for r in records:
        if r.get("score") is not None:
            per.setdefault(r["id"], []).append(float(r["score"]))
    return round(sum(sum(v) / len(v) for v in per.values()) / len(per), 4) if per else None


def _by_domain(h: dict) -> dict:
    out = {}
    known = {t for tasks in HELDOUT.values() for t, _ in tasks}
    extra = [(t, "pool") for t in h if t not in known]          # objective eval_overfit: the eval set itself
    for dom, tasks in {**HELDOUT, **({"eval_target": extra} if extra else {})}.items():
        vals = [h[t]["accuracy"] for t, _ in tasks if t in h and h[t].get("accuracy") is not None]
        out[dom] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def main(argv=None):
    ap = C.build_arg_parser("A")
    ap.add_argument("--budget-gib", type=float, default=None,
                    help="fixed packed size in GiB for every generation (default: config budget)")
    args = ap.parse_args(argv)
    rc = C.load_run_config("A", args)
    cfg = effective_config(rc, args)
    from rsicore.evaluate import set_max_tokens
    set_max_tokens(cfg.get("max_tokens"))                # per-task max generation length for every run_eval call
    try:
        Runner(rc, cfg).run()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
