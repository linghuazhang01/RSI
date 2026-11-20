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
Generation g = 1..gens (gen_XX/): draw -> M_0's KL reference logprobs (M_0 is not scored on the items) ->
incumbent scored on the same items (the reference of every candidate; shown to the editor) -> inner search (threads x steps, <= max_evaluate evaluations;
candidates from the shared cache; one candidate server per candidate-GPU group, so the threads of a step
are built and evaluated at the same time) -> optional baselines -> gate (<= 3 submissions of
candidates whose score beats the incumbent's: fits the budget AND paired gain over the incumbent on the
gate set) -> new incumbent exported + held-out evaluation, or the incumbent stays. The loop stops after
loop.patience generations without a new incumbent. Every step is a rsicore.state.RunState key; rerunning
the same run-id resumes (completed steps are skipped).
"""
from __future__ import annotations

import json
import shutil
import sys
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
from rsicore.editor import Editor  # noqa: E402
from rsicore.evaluate import score  # noqa: E402
from rsicore.gate import fixed_noise_band, measure_noise_band, noise_ids  # noqa: E402
from rsicore.heldout import HELDOUT, run_heldout  # noqa: E402
from rsicore.sandbox import run_program  # noqa: E402
from rsicore.servers import VLLMServer  # noqa: E402
from rsicore.state import RunState  # noqa: E402

from edit_code import mock as M  # noqa: E402
from edit_code.baselines import run_baselines  # noqa: E402
from edit_code.common import (candidate_groups, deep_merge, gpu_layout, read_json, read_jsonl, tail,  # noqa: E402
                              write_json)
from edit_code.problem import EvalBudgetExhausted, QuantProblem, add_self_generated, kl_texts_for  # noqa: E402
from edit_code.prompts import archive_text, make_render_context, system_prompt  # noqa: E402
from edit_code.quant import schema, sizes  # noqa: E402
from edit_code.tools import make_tool_api  # noqa: E402

ALL_DOMAINS = ["math", "code", "knowledge", "instruction_following", "tool_calling"]


class LazyServer:
    """Starts a VLLMServer on first use; close() stops it."""

    def __init__(self, **kw):
        self.kw = kw
        self.srv = None

    def get(self) -> VLLMServer:
        if self.srv is None:
            self.srv = VLLMServer(**self.kw).start()
        return self.srv

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


def effective_config(rc, args) -> dict:
    cfg = dict(rc.paper_cfg)   # hardware profile already merged by rsicore.config (before --set)
    if rc.mock:
        cfg = deep_merge(cfg, cfg.get("mock_overrides") or {})
    if getattr(args, "budget_gib", None):
        cfg = deep_merge(cfg, {"budget": {"gib": float(args.budget_gib)}})
    return cfg


def gate_mean(scores: dict | None) -> float | None:
    vals = [v for v in (scores or {}).values() if v is not None]
    return round(sum(vals) / len(vals), 5) if vals else None


class Runner:
    def __init__(self, rc, cfg: dict):
        self.rc, self.cfg = rc, cfg
        self.st = RunState(rc.run_dir)
        self.run_dir = Path(rc.run_dir)
        doms = ALL_DOMAINS if rc.domain == "all" else [rc.domain]
        self.datasets = [ds for d in doms for ds in D.DOMAIN_TRAIN[d]]
        kd = cfg["kl"].get("domains", "all")
        self.kl_datasets = [ds for d in (ALL_DOMAINS if kd == "all" else doms) for ds in D.DOMAIN_TRAIN[d]]
        self.registry = self.run_dir / "mock_registry.json"
        roles, unused = gpu_layout(rc, cfg)
        self.gpu_editor, self.gpu_cand = roles["editor"], roles["candidate"]
        self.cand_slots = candidate_groups(rc, cfg)   # one candidate server per slot, evaluated concurrently
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
                tasks: list[str] | None = None) -> LazyServer:
        if self.rc.mock:
            M.register(self.registry, name, quality)
            extra = ["--tasks", ",".join(tasks)] if tasks else []
            path = "mock-" + name
        else:
            extra = list(self.cfg["vllm"].get(args_key, []))
            path = model_dir
        return LazyServer(model_path=path, served_name=name, gpu_ids=list(gpus), mock=self.rc.mock,
                          mock_registry=self.registry, seed=self.rc.seed, log_dir=self.run_dir / "logs", extra_args=extra)

    def m0_server(self, tasks: list[str] | None = None) -> LazyServer:
        """The original BF16 model M_0 on the editor GPU (editor in frozen mode, and the reference)."""
        return self._server(f"{self.rc.run_id}-m0", Path(self.rc.model_path),
                            self.cfg["mock"]["base_quality"] if self.rc.mock else None, self.gpu_editor, "editor_args", tasks)

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
        return add_self_generated(texts, lambda: m0.get().endpoint, m0.name, self.cfg,
                                  self.run_dir / "kl_selfgen" / f"gen_{gen:02d}.jsonl")

    def data_for(self, gen: int) -> dict:
        return {ds: {"screen": D.ids(self.rc, gen, ds, "screen"), "signal": D.ids(self.rc, gen, ds, "signal")}
                for ds in self.datasets}

    def problem(self, *, tag: str, gen: int, out_dir: Path, budget: float, data: dict, kl_texts, refs_dir: Path,
                eval_budget: int, show_examples: int, incumbent: dict | None = None, kl_heldout=()) -> QuantProblem:
        """Every problem quantizes the ORIGINAL model; the layer cache in run_dir/prep is shared."""
        return QuantProblem(self.rc, self.cfg, tag=tag, gen=gen, out_dir=out_dir, prep_dir=self.prep_dir,
                            source_dir=Path(self.rc.model_path) if not self.rc.mock else None,
                            source_state=self.source_state(), budget_gib=budget, datasets=self.datasets, data=data,
                            kl_texts=kl_texts, kl_heldout=kl_heldout, refs_dir=refs_dir,
                            noise=read_json(self.run_dir / "baseline" / "noise.json", {}),
                            gate_baseline=read_json(self.run_dir / "baseline" / "gate.json", {}),
                            eval_budget=eval_budget, show_examples=show_examples, incumbent=incumbent, log=self.log)

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
        per_prog = int(scfg.get("program_evaluate_calls", 2))
        doc_api = make_tool_api(prob, cfg, cid="doc", per_program_evaluate=per_prog)
        sys_prompt = system_prompt(prob, doc_api.describe(), cfg)
        ecfg = cfg["editor"]

        def editor_factory(th):
            srv = None if rc.mock else ed.get()
            return Editor(srv.endpoint if srv else "http://mock.invalid/v1", ed.name, system_prompt=sys_prompt,
                          seed=rc.seed + 1000 * prob.gen + th, transcript=gdir / "editor" / f"thread_{th}.jsonl",
                          temperature=ecfg["temperature"], top_p=ecfg["top_p"], max_tokens=ecfg["max_tokens"],
                          mock_policy=M.mock_policy if rc.mock else None, extra_body=ecfg.get("extra_body"))

        def build_fn(program, cid):
            if prob.evals_left <= 0:
                return None, "evaluate budget of this generation is exhausted; program not run"
            api = make_tool_api(prob, cfg, cid=cid, per_program_evaluate=per_prog)
            res = run_program(program, api, timeout_s=cfg["sandbox"]["timeout_s"], mem_gb=cfg["sandbox"]["mem_gb"],
                              workdir=gdir / "sandbox" / cid)
            if not res.ok:
                return None, f"program failed: {res.error}\nstderr tail:\n{tail(res.stderr, 1500)}"
            try:
                art = prob.build(res.result, cid)
            except schema.ConfigError as e:
                return None, str(e)
            except (ValueError, TypeError) as e:
                return None, f"invalid config: {e}"
            art["cost"] = {"tool_calls": res.info.get("tool_calls"), "tool_evaluate": api.spent.get("evaluate", 0),
                           "program_s": round(res.seconds, 2)}
            art["stdout_tail"] = tail(res.stdout, 600)
            return art, None

        def screen_fn(art, cid):
            try:
                r = prob.evaluate(art, "screen", cid)
            except EvalBudgetExhausted as e:
                return None, {"error": str(e)}
            return r["score"], r["feedback"]

        def full_fn(art, cid):
            r = prob.evaluate(art, "full", cid)
            return r["score"], r["feedback"]

        render = make_render_context(prob, cfg, task=task, archive=archive)
        return dict(editor_factory=editor_factory, render_context=render, build_fn=build_fn, screen_fn=screen_fn,
                    full_fn=full_fn)

    def run_search(self, prob, gen, gdir, ed, *, task, archive, scfg) -> list:
        fns = self.search_fns(prob, gdir, ed, task=task, archive=archive, scfg=scfg)
        return S.inner_search(self.rc, gen, threads=scfg["threads"], steps=scfg["steps"],
                              max_candidates=scfg.get("max_candidates", scfg["max_evaluate"]),
                              top_full=scfg.get("top_full", 3), rerun_seeds=scfg.get("rerun_seeds", 0),
                              out_dir=gdir, parallel=len(prob.slots), **fns)

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
        try:
            with st.step("baseline/gate") as todo:
                if todo:
                    srv = m0.get()
                    res = {}
                    for ds in self.datasets:
                        r = score(srv.endpoint, m0.name, ds, pool="pool", ids=D.gate_ids(rc, ds),
                                  out_dir=bdir / "gate", mock=rc.mock)
                        res[ds] = _acc(r.records)
                    write_json(bdir / "gate.json", res)
                    self.log("[baseline] M_0 gate scores", res)
            with st.step("baseline/noise") as todo:
                if todo:
                    write_json(bdir / "noise.json", self.noise_band(m0, bdir))
            with st.step("baseline/heldout") as todo:
                if todo:
                    srv = m0.get()
                    h = run_heldout(srv.endpoint, m0.name, bdir / "heldout",
                                    domains=cfg["heldout"].get("domains", "all"), mock=rc.mock,
                                    limit=cfg["heldout"].get("limit"), decoding=cfg["heldout"].get("decoding"))
                    write_json(bdir / "heldout.json", h)
            kl0 = self.kl_texts(0, m0)       # calibration / KL texts (prompt + M_0 response with kl.self_generated)
        finally:
            m0.close()
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
            if todo:
                srv = self.model_server(bdir / "q0" / "model", f"{rc.run_id}-q0", self.cand_slots[0], "candidate_args")
                with srv:
                    s = srv.get()
                    h = run_heldout(s.endpoint, srv.name, bdir / "q0" / "heldout",
                                    domains=cfg["heldout"].get("domains", "all"), mock=rc.mock,
                                    limit=cfg["heldout"].get("limit"), decoding=cfg["heldout"].get("decoding"))
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
        prob = self.problem(tag=f"e{g:02d}", gen=g, out_dir=out, budget=float(ec.get("budget_gib") or self.budget_gib()),
                            data=self.data_for(0), kl_texts=texts[0::2], kl_heldout=texts[1::2],
                            refs_dir=self.run_dir / "e_task" / "refs", eval_budget=ec["evaluate"], show_examples=0,
                            incumbent=self.incumbent(0))
        prob.prepare()
        ed = self.editor_server(g + 1, m0)          # frozen: M_0; current: the incumbent after generation g
        try:
            srv = m0.get()
            prob.compute_refs(srv.endpoint, m0.name)
            prob.evaluate_incumbent()               # Q_0 on the generation-0 items: the candidates' reference
            if ed is not m0:
                m0.close()
            scfg = dict(cfg["search"], threads=ec["threads"], steps=ec["steps"], max_evaluate=ec["evaluate"],
                        top_full=ec.get("top_full", 1))
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
                                domains=cfg["heldout"].get("domains", "all"), mock=rc.mock,
                                limit=ec.get("heldout_limit"), decoding=cfg["heldout"].get("decoding"))
            res["best"] = {"cid": best.cid, "score": best.score, "bytes_gib": best.artifact["bytes_gib"],
                           "config": best.artifact["config"], "heldout_kl": hk, "heldout": h,
                           "heldout_by_domain": _by_domain(h)}
        write_json(out / "result.json", res)
        st.mark(key, {"best": (res["best"] or {}).get("cid")})
        self.log(f"[E_{g}] best {(res['best'] or {}).get('cid')} score {(res['best'] or {}).get('score')}")

    # ------------------------------------------------------------------ generation g
    def archive(self, g: int) -> str:
        """Experience channel: per earlier round, whether a new incumbent was accepted and its top candidates."""
        acfg = self.cfg["archive"]
        if not acfg.get("enabled", True):
            return ""
        k = int(acfg.get("top_candidates", 3))
        ents = []
        for t in range(1, g):
            gd = R.gen_dir(self.rc, t)
            acc = read_json(gd / "accepted.json") or {}
            cands = S.ranked(S.load_candidates(self.rc, t).values())
            top = [{"cid": c.cid, "score": c.score, "bytes_gib": (c.artifact or {}).get("bytes_gib"),
                    "config": (c.artifact or {}).get("config")} for c in cands if c.score is not None and c.artifact][:k]
            ents.append({"gen": t, "accepted": bool(acc.get("accepted")), "gate_reason": acc.get("gate_reason"),
                         "incumbent_score": (read_json(gd / "incumbent_scores.json") or {}), "top": top})
        return archive_text(ents)

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
            prob = self.problem(tag=f"g{g:02d}", gen=g, out_dir=gd, budget=budget, data=self.data_for(g),
                                kl_texts=self.kl_texts(g, m0), refs_dir=gd / "refs",
                                eval_budget=cfg["search"]["max_evaluate"],
                                show_examples=cfg["feedback"].get("examples", 2), incumbent=inc)
            prob.prepare()
            with st.step(f"{p}/refs") as todo:
                if todo:                            # M_0 = the KL reference only (not scored on the items)
                    srv = m0.get()
                    prob.compute_refs(srv.endpoint, m0.name)
            # the incumbent on this generation's items = the reference of every candidate (uncharged; cached)
            prob.evaluate_incumbent()
            write_json(gd / "incumbent_scores.json", prob.incumbent_scores)
            ed = self.editor_server(g, m0)
            if ed is not m0:
                m0.close()
            if st.done(f"{p}/search"):
                ranked = S.ranked(S.load_candidates(rc, g).values())
            else:
                ranked = self.run_search(prob, g, gd, ed, task="main", archive=self.archive(g), scfg=cfg["search"])
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
            if todo:
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
                acc = S.submit_to_gate(rc, g, subs, gate_fn, max_submissions=cfg["gate"]["max_submissions"])
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
                if todo:
                    srv = self.model_server(gd / "model", f"{rc.run_id}-q{g:02d}", self.cand_slots[0], "candidate_args")
                    with srv:
                        s = srv.get()
                        h = run_heldout(s.endpoint, srv.name, gd / "heldout",
                                        domains=cfg["heldout"].get("domains", "all"), mock=rc.mock,
                                        limit=cfg["heldout"].get("limit"), decoding=cfg["heldout"].get("decoding"))
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
                 f"kl_datasets={self.kl_datasets} heldout={self.cfg['heldout'].get('domains', 'all')}")
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
    for dom, tasks in HELDOUT.items():
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
