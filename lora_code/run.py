#!/usr/bin/env python
"""Paper B orchestration: recursive self-written LoRA (plan/PLAN.md §4, RSI_LOOP_DESIGN §3).

  python -I lora_code/run.py --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id B_math_s42 [--mock]

Generation g (editor M_g -> successor M_{g+1}; the target domain is --domain in every generation):
  draw (target=domain) -> train data -> diagnostics -> SFT baseline (+ screen/full/gate) -> inner search
  (editor = M_g on gpu0 with multi-LoRA; candidate LoRAs built on gpu1, hot-loaded for evaluation)
  -> gate (target >= SFT - eps; retention >= gen-0 - delta; learned targets >= post-learning - delta;
  sandbox checks) -> accept (fallback: SFT LoRA if it passes, else keep M_g) -> merge into weights
  (gen_XX/model) -> non-LLM branch (random search) + global-scale / checkpoint frontiers -> E_t for each
  branch -> held-out evaluation of M_{g+1} -> summary (ΔE_train, ΔE_write, ΔE_nonLLM with paired CIs).
Baseline first: M_0 on the fixed gate (noise band) and held-out; E-ref (SFT LoRA for θ0 on the E task).
Every step is recorded in state.json (rsicore.state.RunState); rerunning the same run-id resumes.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "eval")]

from lora_code.common import (DOMAIN_DATASET, DOMAINS, deep_merge, gold_target, load_records,  # noqa: E402
                              prompt_text, read_json, render_messages, retention_domains, role_gpus, write_json)


# ============================================================================ servers
class Servers:
    """One live server per slot ('main' / 'theta0' on their gpu_roles GPUs); reused while the model is the same."""

    def __init__(self, rc, cfg):
        self.rc, self.cfg = rc, cfg
        self.live = {}

    def get(self, slot: str, model_path, served: str):
        from rsicore.servers import VLLMServer
        cur = self.live.get(slot)
        if cur is not None and cur[0] == (str(model_path), served) and cur[1].alive():
            return cur[1]
        if cur is not None:
            cur[1].stop()
        sc = self.cfg.get("server", {})
        gpu = role_gpus(self.rc, self.cfg, "main" if slot == "main" else "theta0")
        if self.rc.mock:   # mock_server.py takes none of vLLM's flags
            extra = ["--tasks", ",".join(sc["mock_tasks"])] if sc.get("mock_tasks") else []
        else:
            extra = list(sc.get("main_args" if slot == "main" else "theta0_args", []) or [])
        srv = VLLMServer(model_path, served, gpu, enable_lora=True, max_loras=int(sc.get("max_loras", 8)),
                         max_lora_rank=int(self.cfg.get("lora", {}).get("rank_cap", 64)), extra_args=extra,
                         mock=self.rc.mock, mock_registry=self.rc.run_dir / "mock_registry.json", seed=self.rc.seed,
                         log_dir=self.rc.run_dir / "logs")
        srv.start()
        self.live[slot] = ((str(model_path), served), srv)
        return srv

    def stop(self, slot=None):
        for k in ([slot] if slot else list(self.live)):
            v = self.live.pop(k, None)
            if v is not None:
                v[1].stop()


# ============================================================================ helpers
def limit_ids(ids, n):
    ids = sorted(ids)
    return ids[: int(n)] if n else ids


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


class Run:
    def __init__(self, rc, cfg):
        from rsicore.state import RunState
        self.rc, self.cfg = rc, cfg
        self.st = RunState(rc.run_dir)
        self.servers = Servers(rc, cfg)
        self.registry = rc.run_dir / "mock_registry.json"
        self.target = rc.domain
        self.tds = DOMAIN_DATASET[rc.domain]
        retention = cfg.get("retention", {}).get("domains")
        self.retention_domains = retention_domains(self.target) if retention is None else list(retention)
        if any(d not in DOMAINS or d == self.target for d in self.retention_domains):
            raise ValueError("retention.domains must contain only non-target domains")
        self.datasets = [DOMAIN_DATASET[d] for d in DOMAINS
                         if d == self.target or d in self.retention_domains]
        self.mcfg = cfg.get("mock", {})
        self._stores = {}
        self._hf = {}

    # ------------------------------------------------------------------ models
    def served(self, g: int) -> str:
        return f"m{g}"

    def model_path(self, g: int) -> Path:
        if g == 0:
            return Path(self.rc.model_path)
        info = read_json(self.gdir(g - 1) / "model.json")
        if info is None:
            raise RuntimeError(f"generation {g - 1} has no model.json (run it first)")
        return Path(info["path"])

    def quality_value(self, name: str):
        """Raw mock registry value of `name` (float or per-domain dict; 0.7 if absent); None outside mock."""
        if not self.rc.mock:
            return None
        from rsicore.mock_quality import unwrap
        v = unwrap((read_json(self.registry, {}) or {}).get(name))
        return 0.7 if v is None else v

    def quality(self, name: str) -> float | None:
        """Scalar mock quality (mean over the five domains) of `name`; None outside mock."""
        if not self.rc.mock:
            return None
        from rsicore.mock_quality import scalar
        return scalar(self.quality_value(name))

    def quality_of(self, name: str, domain: str) -> float | None:
        if not self.rc.mock:
            return None
        from rsicore.mock_quality import resolve
        return resolve(self.quality_value(name), domain=domain)

    def set_quality(self, name: str, q) -> None:
        reg = read_json(self.registry, {}) or {}
        reg[name] = {k: round(float(v), 6) for k, v in q.items()} if isinstance(q, dict) else round(float(q), 6)
        write_json(self.registry, reg)

    def store(self, path: Path):
        if self.rc.mock:
            from lora_code.mock import MockStore
            return self._stores.setdefault("mock", MockStore())
        key = str(path)
        if key not in self._stores:
            from lora_code.primitives.weights import WeightStore
            dev = self.train_device() if self.cfg.get("train", {}).get("store_on_gpu", True) else "cpu"
            self._stores[key] = WeightStore(path, device=dev, cache_dir=self.rc.run_dir / "cache")
        return self._stores[key]

    def hf(self, path: Path):
        if self.rc.mock:
            return None
        key = str(path)
        if key not in self._hf:
            from lora_code.primitives.train import HFModel
            tc = self.cfg.get("train", {})
            self._hf[key] = HFModel(path, self.store(path), device=self.train_device(),
                                    devices=[f"cuda:{g}" for g in role_gpus(self.rc, self.cfg, "train")],
                                    micro_batch=tc.get("micro_batch"),
                                    dtype=tc.get("dtype", "bfloat16"), max_len=int(tc.get("max_len", 4096)),
                                    seed=self.rc.seed)
        return self._hf[key]

    def train_device(self) -> str:
        """First GPU of the train role: weight store, diagnostics, merge, HF model inputs."""
        return f"cuda:{role_gpus(self.rc, self.cfg, 'train')[0]}"

    def release_hf(self):
        for h in self._hf.values():
            h.release()
        self._hf.clear()

    def gdir(self, g: int) -> Path:
        from rsicore.records import gen_dir
        return gen_dir(self.rc, g)

    def load_lora(self, path: Path):
        if self.rc.mock:
            from lora_code.mock import MockLoRA
            return MockLoRA.from_json(read_json(path))
        from lora_code.primitives.lora import LoRA
        return LoRA.load_raw(path)

    def save_lora(self, lo, path_noext: Path) -> Path:
        if self.rc.mock:
            p = path_noext.with_suffix(".json")
            write_json(p, lo.to_json())
        else:
            p = path_noext.with_suffix(".safetensors")
            lo.save_raw(p)
        return p

    def lora_file(self, path_noext: Path) -> Path | None:
        for ext in (".json", ".safetensors"):
            p = path_noext.with_suffix(ext)
            if p.exists():
                return p
        return None

    def stack(self, g: int) -> list:
        out = []
        for k in range(g):
            p = self.lora_file(self.gdir(k) / "accepted_lora")
            if p is not None:
                out.append((k, self.load_lora(p)))
        return out

    # ------------------------------------------------------------------ ids / data
    def level_ids(self, g: int) -> dict:
        from rsicore import draws
        ec = self.cfg.get("eval", {})
        ids = {"screen": {}, "full": {}, "gate": {}}
        for ds in self.datasets:
            ids["screen"][ds] = limit_ids(draws.ids(self.rc, g, ds, "screen"), ec.get("screen_limit"))
            ids["full"][ds] = limit_ids(draws.ids(self.rc, g, ds, "signal"), ec.get("full_limit"))
            ids["gate"][ds] = limit_ids(draws.gate_ids(self.rc, ds), ec.get("gate_limit"))
        return ids

    def gate_level(self) -> dict:
        from rsicore import draws
        lim = self.cfg.get("eval", {}).get("gate_limit")
        return {ds: limit_ids(draws.gate_ids(self.rc, ds), lim) for ds in self.datasets}

    def kl_texts(self, g: int) -> list[str]:
        from rsicore import draws
        n = self.cfg.get("eval", {}).get("kl_limit")
        texts = []
        for d in self.retention_domains:
            ds = DOMAIN_DATASET[d]
            ids = limit_ids(draws.ids(self.rc, g, ds, "kl"), n)
            recs = load_records(ds, ids)
            texts += [prompt_text(ds, recs[i]) for i in ids if i in recs]
        return texts

    def build_train_data(self, dataset, ids, out_path: Path, *, endpoint=None, served=None, mode="gold"):
        """{id: {messages, target, source}} for train ids. mode 'self_correct_then_gold': sample the
        current model on the train items and keep a correct sample (rejection sampling); else the gold
        response in the evaluator's format."""
        got = read_json(out_path)
        if got is not None:
            return got
        recs = load_records(dataset, ids)
        selfs = {}
        tc = self.cfg.get("train_data", {})
        if mode == "self_correct_then_gold" and endpoint:
            from rsicore.evaluate import score
            n = int(tc.get("samples", 4))
            er = score(endpoint, served, dataset, pool="pool", ids=list(ids), out_dir=out_path.parent / "train_samples",
                       extra=["--n", str(n), "--temperature", str(tc.get("temperature", 0.7))], mock=self.rc.mock)
            for r in er.records:
                outs = r.get("raw_output")
                outs = outs if isinstance(outs, list) else [outs]
                for o, s in zip(outs, r.get("sample_scores") or []):
                    if s is not None and s >= 1 and o:
                        selfs[r["id"]] = o
                        break
        ex = {}
        for i in ids:
            it = recs.get(i)
            if it is None:
                continue
            try:
                msgs = render_messages(dataset, it)
            except Exception:  # noqa: BLE001
                continue
            tgt, src = selfs.get(i), "self"
            if tgt is None:
                tgt, src = gold_target(dataset, it), "gold"
            ex[i] = {"messages": msgs, "target": tgt, "source": src if tgt else None}
        for i in ids:  # every train id stays allowed, even without a target
            ex.setdefault(i, {"messages": [], "target": None, "source": None})
        write_json(out_path, ex)
        return ex

    # ------------------------------------------------------------------ evaluator / workspace factories
    def evaluator(self, g, out_dir, srv, served, ids, kl_texts, store):
        from lora_code.evaluator import Evaluator
        return Evaluator(self.rc, self.cfg, out_dir=out_dir, endpoint=srv.endpoint, served=served, server=srv,
                         ids=ids, target_dataset=self.tds, kl_texts=kl_texts, store=store, mock=self.rc.mock,
                         registry=self.registry, parent_quality=self.quality_value(served),
                         max_loaded=int(self.cfg.get("server", {}).get("max_loras", 8)) - 2)

    def workspace(self, out_dir, ev, store, hf, train, budget, prefix, stack=(), diag=None, ret_texts=()):
        from lora_code.primitives.train import TrainData
        from lora_code.tools import Workspace
        return Workspace(mock=self.rc.mock, store=store, train=TrainData(train), cfg=self.cfg, out_dir=out_dir,
                         evaluator=ev, hf=hf, stack=stack, retention_texts=ret_texts, budget=budget,
                         name_prefix=prefix, diagnostics=diag, seed=self.rc.seed)

    # ================================================================== baseline
    def baseline(self):
        from rsicore import draws
        from rsicore.gate import measure_noise_band
        bdir = self.rc.run_dir / "baseline"
        draws.ensure_gate(self.rc)
        if self.rc.mock and not (read_json(self.registry, {}) or {}).get(self.served(0)):
            self.set_quality(self.served(0), float(self.mcfg.get("base_quality", 0.6)))
            self.set_quality("theta0", float(self.mcfg.get("base_quality", 0.6)))
        key = "baseline/gate"
        if not self.st.done(key):
            srv = self.servers.get("main", self.model_path(0), self.served(0))
            from rsicore.evaluate import score
            bc = self.cfg.get("baseline", {})
            reps = int(bc.get("noise_repeats", 3))
            gate = self.gate_level()
            res = {}
            for ds, ids in gate.items():
                # gate score: greedy, like every candidate's gate evaluation
                er = score(srv.endpoint, self.served(0), ds, pool="pool", ids=ids, out_dir=bdir / "gate" / ds,
                           mock=self.rc.mock)
                recs = [{"id": x.get("id"), "score": x.get("score")} for x in er.records]
                sc = [x["score"] for x in recs if x["score"] is not None]
                m = mean_or_none(sc)
                se = (sum((x - m) ** 2 for x in sc) / max(len(sc) - 1, 1) / max(len(sc), 1)) ** 0.5 if sc else None
                # noise band: spread of `reps` sampled re-runs (seeds seed, seed+1, ...); the gate margin is
                # max(band, 2 SE) because the band of a few re-runs underestimates item-sampling noise
                nb = measure_noise_band(srv.endpoint, self.served(0), ds, ids, bdir / "noise" / ds, repeats=reps,
                                        temperature=float(bc.get("noise_temperature", 0.6)), base_seed=self.rc.seed,
                                        mock=self.rc.mock)
                res[ds] = {"score": m, "n": len(recs), "n_scored": len(sc), "noise_band_repeats": nb["band"],
                           "noise_scores": nb["scores"], "noise_seeds": nb["seeds"], "noise_se": nb["se"],
                           "se": se, "noise_band": max(nb["band"], 2 * se if se else 0.0), "records": recs}
            write_json(bdir / "gate_scores.json", res)
            self.st.mark(key, {ds: {k: v[k] for k in ("score", "noise_band")} for ds, v in res.items()})
        if self.cfg.get("heldout", {}).get("enabled", True) and not self.st.done("baseline/heldout"):
            from rsicore.heldout import run_heldout
            srv = self.servers.get("main", self.model_path(0), self.served(0))
            h = run_heldout(srv.endpoint, self.served(0), bdir / "heldout",
                            domains=self.cfg.get("heldout", {}).get("domains", "all"), mock=self.rc.mock,
                            limit=self.cfg.get("heldout", {}).get("limit"))
            write_json(bdir / "heldout.json", h)
            self.st.mark("baseline/heldout", {"tasks": len(h)})
        if self.cfg.get("e_t", {}).get("enabled", True) and not self.st.done("e_ref"):
            self.e_ref()

    def gen0_gate(self) -> dict:
        return read_json(self.rc.run_dir / "baseline" / "gate_scores.json", {})

    # ================================================================== E task
    def theta0(self):
        srv = self.servers.get("theta0", self.rc.model_path, "theta0")
        st = self.store(Path(self.rc.model_path))
        return {"endpoint": srv.endpoint, "served": "theta0", "server": srv, "store": st,
                "hf": self.hf(Path(self.rc.model_path)), "quality": self.quality_value("theta0")}

    def e_setup(self):
        from lora_code.etask import e_items, e_task_for
        task = e_task_for(self.target, self.cfg)
        items = e_items(self.rc, task, self.cfg)
        tdata = self.build_train_data(task["dataset"], items["train"], self.rc.run_dir / "e_task" / "train_data.json",
                                      mode="gold")
        return task, items, tdata

    def e_ref(self):
        from lora_code import baselines
        from lora_code.etask import e_test
        from lora_code.evaluator import Evaluator
        task, items, tdata = self.e_setup()
        out = self.rc.run_dir / "e_ref"
        th = self.theta0()
        ev = Evaluator(self.rc, self.cfg, out_dir=out, endpoint=th["endpoint"], served="theta0", server=th["server"],
                       ids={"screen": {task["dataset"]: items["signal"]}}, target_dataset=task["dataset"], kl_texts=[],
                       store=th["store"], mock=self.rc.mock, registry=self.registry, parent_quality=th["quality"])
        gb = float(self.cfg.get("e_t", {}).get("gpu_seconds", 3600))
        ws = self.workspace(out, ev, th["store"], th["hf"], tdata, {"evaluate": 1e9, "gpu_seconds": gb}, "eref")
        lo, _, info = baselines.train_sft(ws, baselines.recipe(self.cfg), gb, seed=self.rc.seed)
        name = f"E-{self.rc.run_id}-ref"
        ev.register(lo, name)
        ev.ensure_loaded(name)
        base = e_test(th["endpoint"], "theta0", task, out / "base", lora=None, mock=self.rc.mock,
                      limit=self.cfg.get("e_t", {}).get("test_limit"))
        ref = e_test(th["endpoint"], "theta0", task, out / "sft", lora=name, mock=self.rc.mock,
                     limit=self.cfg.get("e_t", {}).get("test_limit"))
        write_json(out / "result.json", {"task": task, "theta0": base, "sft_ref": ref, "train": info})
        self.st.mark("e_ref", {"theta0": base["score"], "sft_ref": ref["score"], "task": task["key"]})
        self.release_hf()

    # ================================================================== one generation
    def generation(self, g: int):
        from rsicore import draws
        rc, cfg, st = self.rc, self.cfg, self.st
        gd = self.gdir(g)
        P = f"gen{g:02d}"
        parent, served = self.model_path(g), self.served(g)
        draws.draw_gen(rc, g, target=self.target)
        ids = self.level_ids(g)
        train_ids = draws.ids(rc, g, self.tds, "train")
        st.mark(f"{P}/draw", {"train": len(train_ids)}) if not st.done(f"{P}/draw") else None
        store = self.store(parent)
        stack = self.stack(g)

        def main_srv():
            return self.servers.get("main", parent, served)

        # ---- train data
        tmode = cfg.get("train_data", {}).get("mode", "self_correct_then_gold")
        if not st.done(f"{P}/train_data"):
            srv = main_srv() if tmode != "gold" else None
            ex = self.build_train_data(self.tds, train_ids, gd / "train_data.json", mode=tmode,
                                       endpoint=srv.endpoint if srv else None, served=served)
            st.mark(f"{P}/train_data", {"n": len(ex), "with_target": sum(1 for e in ex.values() if e["target"]),
                                        "self": sum(1 for e in ex.values() if e["source"] == "self")})
        tdata = read_json(gd / "train_data.json")
        kl_texts = self.kl_texts(g)

        # ---- diagnostics
        from lora_code import diagnostics as D
        if self.rc.mock:
            from lora_code.mock import mock_diagnostics
            diag = D.load_or_compute(gd / "diagnostics.json", lambda: mock_diagnostics(store, stack, rc.seed))
        else:
            def real_diag():
                hf = self.hf(parent)
                tt = ["\n\n".join(m["content"] for m in e["messages"]) for e in list(tdata.values())[:64] if e["messages"]]
                return D.compute(store, stack, base_store=self.store(Path(rc.model_path)), hf=hf, target_texts=tt,
                                 retention_texts=kl_texts, svd_k=int(cfg.get("diagnostics", {}).get("svd_k", 8)),
                                 device=self.train_device())
            diag = D.load_or_compute(gd / "diagnostics.json", real_diag)
        st.mark(f"{P}/diagnostics", {"rows": diag["n_modules"]}) if not st.done(f"{P}/diagnostics") else None

        budget = {"evaluate": int(cfg.get("search", {}).get("max_evaluate", 32)),
                  "gpu_seconds": float(cfg.get("budget", {}).get("gpu_seconds", 7200))}
        state = {}

        def ctx_objects():
            if "ws" in state:
                return state["ev"], state["ws"]
            srv = main_srv()
            ev = self.evaluator(g, gd, srv, served, ids, kl_texts, store)
            ws = self.workspace(gd, ev, store, self.hf(parent), tdata, budget, f"g{g:02d}", stack=stack, diag=diag,
                                ret_texts=kl_texts)
            ws.build_api()
            sft = read_json(gd / "baselines" / "sft.json")
            if sft:
                ev.sft_name = sft["name"]
            state.update(ev=ev, ws=ws, srv=srv)
            return ev, ws

        # ---- SFT baseline
        from lora_code import baselines as BL
        if not st.done(f"{P}/sft"):
            ev, ws = ctx_objects()
            bws = self.workspace(gd, ev, store, self.hf(parent), tdata, {"evaluate": 1e9, "gpu_seconds": budget["gpu_seconds"]},
                                 f"g{g:02d}", stack=stack, ret_texts=kl_texts)
            lo, ck, info = BL.train_sft(bws, BL.recipe(cfg), budget["gpu_seconds"], seed=rc.seed)
            self.save_lora(lo, gd / "baselines" / "sft_lora")
            for f, c in ck:
                self.save_lora(c, gd / "baselines" / f"sft_ckpt_{int(round(f * 100)):03d}")
            name = bws.lora_name(lo)
            ok, chk = bws.check(lo)
            pts = {lvl: BL.point(bws, lo, "sft", lvl) for lvl in ("screen", "full")}
            gate = ev.run(name, "gate") if ok else None
            write_json(gd / "baselines" / "sft.json", {"name": name, "train": info, "check_ok": ok,
                                                      "check_errors": chk.get("errors", [])[:10], "points": pts,
                                                      "gate": {ds: r["score"] for ds, r in (gate or {}).items()},
                                                      "ckpt_fracs": [f for f, _ in ck]})
            ev.sft_name = name
            st.mark(f"{P}/sft", {"name": name, "screen": pts["screen"]["score"], "check_ok": ok})
        sft = read_json(gd / "baselines" / "sft.json")

        # ---- inner search
        ranked = None
        if not st.done(f"{P}/search"):
            ranked = self.search(g, gd, ctx_objects, diag, sft, budget)
            st.mark(f"{P}/search", {"candidates": len(ranked), "built": sum(1 for c in ranked if c.artifact),
                                    "best": ranked[0].cid if ranked else None})
        # ---- gate
        if not st.done(f"{P}/gate"):
            from rsicore.search import load_candidates, ranked as rank_fn, submit_to_gate
            if ranked is None:
                ranked = rank_fn(load_candidates(rc, g).values())
            ev, ws = ctx_objects()
            learned = (st.get("learned") or {}).get("scores", {})

            def gate_fn(c):
                lo = ws.load_candidate(c.artifact["cid"])
                return self.gate_check(ev, ws, lo, sft, learned)

            acc = submit_to_gate(rc, g, ranked, gate_fn, max_submissions=int(cfg.get("gate", {}).get("max_submissions", 3)))
            if acc is not None:
                lo = ws.load_candidate(acc.cid)
                rec = {"source": "llm", "cid": acc.cid, "name": ws.lora_name(lo) if lo.modules() else None}
            else:
                lo = self.load_lora(self.lora_file(gd / "baselines" / "sft_lora"))
                ok, info = self.gate_check(ev, ws, lo, sft, learned) if sft.get("check_ok") else (False, {"reason": "sft failed checks"})
                if ok:
                    rec = {"source": "sft_fallback", "cid": None, "name": sft["name"], "gate": info}
                else:
                    lo, rec = None, {"source": "keep", "cid": None, "name": None,
                                     "reason": "no candidate and no SFT LoRA passed the gate", "sft_gate": info}
            if lo is not None and lo.modules():
                self.save_lora(lo, gd / "accepted_lora")
                g_res = ev.run(rec["name"], "gate")
                rec["gate_scores"] = {ds: r["score"] for ds, r in g_res.items()}
                sc = dict(learned)
                if rec["gate_scores"].get(self.tds) is not None:
                    sc[self.tds] = rec["gate_scores"][self.tds]
                st.mark("learned", {"scores": sc})
            write_json(gd / "accepted.json", rec)
            st.mark(f"{P}/gate", {"source": rec["source"], "cid": rec.get("cid")})
        acc = read_json(gd / "accepted.json")

        # ---- merge -> M_{g+1}
        if not st.done(f"{P}/merge"):
            lf = self.lora_file(gd / "accepted_lora")
            if lf is None:
                write_json(gd / "model.json", {"path": str(parent), "source": "parent (no accepted LoRA)"})
                if rc.mock:
                    self.set_quality(self.served(g + 1), self.quality_value(served))
                info = {"merged": False}
            elif rc.mock:   # successor inherits the per-domain qualities of M_g + accepted LoRA
                lo = self.load_lora(lf)
                q = lo.quality(self.quality_value(served), salt=acc["name"], target=self.target)
                self.set_quality(self.served(g + 1), q)
                write_json(gd / "model.json", {"path": str(parent), "source": "mock merge", "quality": q,
                                               "lora": acc["name"]})
                info = {"merged": True, "quality": q}
            else:
                from lora_code.primitives.lora import merge_into_model
                lo = self.load_lora(lf)
                rep = merge_into_model(lo, parent, gd / "model", store=store,
                                       device=self.train_device() if cfg.get("merge", {}).get("gpu", True) else "cpu")
                write_json(gd / "model.json", {"path": str(gd / "model"), "source": "merged", "lora": acc["name"],
                                               "max_realised_rel_err": rep["max_realised_rel_err"]})
                info = {"merged": True, "max_realised_rel_err": rep["max_realised_rel_err"]}
            st.mark(f"{P}/merge", info)

        # ---- non-LLM branch + frontiers
        if not st.done(f"{P}/nonllm"):
            ev, ws = ctx_objects()
            nc = cfg.get("nonllm", {})
            bws = self.workspace(gd, ev, store, self.hf(parent), tdata, {"evaluate": 1e9, "gpu_seconds": budget["gpu_seconds"]},
                                 f"g{g:02d}", stack=stack, ret_texts=kl_texts)
            out = {}
            if nc.get("frontiers", True):
                fin = self.load_lora(self.lora_file(gd / "baselines" / "sft_lora"))
                ck = [(f, self.load_lora(self.lora_file(gd / "baselines" / f"sft_ckpt_{int(round(f * 100)):03d}")))
                      for f in sft.get("ckpt_fracs", [])]
                out["frontiers"] = BL.frontiers(bws, fin, ck, nc.get("scales", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]))
            best, log = BL.random_search(bws, BL.recipe(cfg), evals=int(nc.get("evaluate", budget["evaluate"])),
                                         gpu_budget_s=budget["gpu_seconds"], n_configs=int(nc.get("n_configs", 4)),
                                         scales=nc.get("search_scales", [0.5, 0.75, 1.0, 1.25]), seed=rc.seed)
            out["random_search"] = log
            name = None
            if best is not None and best.modules():
                self.save_lora(best, gd / "baselines" / "nonllm_lora")
                name = bws.lora_name(best)
            out["name"] = name
            write_json(gd / "baselines" / "nonllm.json", out)
            st.mark(f"{P}/nonllm", {"name": name, "best": (log.get("best") or {}).get("score")})
        nonllm = read_json(gd / "baselines" / "nonllm.json")

        # ---- E_t per branch
        if cfg.get("e_t", {}).get("enabled", True):
            self.release_hf()
            branches = {"parent": None, "sft": sft.get("name"), "llm": acc.get("name") if acc.get("source") == "llm" else None,
                        "nonllm": nonllm.get("name")}
            if acc.get("source") == "sft_fallback":
                branches["llm"] = sft.get("name")
            for b in cfg.get("e_t", {}).get("branches", ["parent", "sft", "llm", "nonllm"]):
                key = f"{P}/e_t/{b}"
                if st.done(key):
                    continue
                ev, ws = ctx_objects()
                lname = branches.get(b)
                if b != "parent" and lname is None:
                    st.mark(key, {"skipped": f"no {b} LoRA"})
                    continue
                if lname is not None:
                    ev.ensure_loaded(lname)
                res = self.e_branch(g, b, ev.endpoint, lname or served,
                                    self.quality(lname) if lname else self.quality(served))
                st.mark(key, {"score": res["score"], "best_cid": res["best_cid"]})

        # ---- held-out evaluation of M_{g+1}
        if cfg.get("heldout", {}).get("enabled", True) and not st.done(f"{P}/heldout"):
            from rsicore.heldout import run_heldout
            nxt = self.model_path(g + 1)
            srv = self.servers.get("main", nxt, self.served(g + 1))
            state.clear()
            h = run_heldout(srv.endpoint, self.served(g + 1), gd / "heldout",
                            domains=cfg.get("heldout", {}).get("domains", "all"), mock=rc.mock,
                            limit=cfg.get("heldout", {}).get("limit"))
            write_json(gd / "heldout.json", h)
            st.mark(f"{P}/heldout", {"tasks": len(h)})

        # ---- shown items, summary
        if not st.done(f"{P}/summary"):
            ev = state.get("ev")
            shown = {}
            for p in [gd / "shown.json"]:
                shown = read_json(p, {}) or {}
            if ev is not None:
                for ds, s in ev.shown.items():
                    shown[ds] = sorted(set(shown.get(ds, [])) | s)
            shown.update(self.shown_from_feedback(gd))
            if shown:
                write_json(gd / "shown.json", shown)
                draws.mark_shown(rc, g, shown)
            info = self.summarize(g, gd)
            from rsicore.records import update_summary
            update_summary(rc, g, info)
            st.mark(f"{P}/summary", {"accepted": info.get("accepted", {}).get("source")})

    def shown_from_feedback(self, gd: Path) -> dict:
        """Items whose content appeared in feedback files (errors / regressions), recomputed from disk so
        a resumed run still marks them."""
        shown = {}
        for p in sorted((gd / "feedback").glob("*.json")):
            fb = (read_json(p, {}) or {}).get("screen") or {}
            fb = fb.get("feedback") or {}
            for e in ((fb.get("target") or {}).get("errors") or []) + (fb.get("regressions") or []):
                if e.get("dataset") and e.get("id"):
                    shown.setdefault(e["dataset"], set()).add(e["id"])
        return {k: sorted(v) for k, v in shown.items()}

    # ------------------------------------------------------------------ search
    def search(self, g, gd, ctx_objects, diag, sft, budget):
        from rsicore.editor import Editor
        from rsicore.search import inner_search

        from lora_code import diagnostics as D
        from lora_code import mock as M
        from lora_code.prompts import ContextRenderer, system_prompt, task_text
        from lora_code.tools import program_build
        rc, cfg = self.rc, self.cfg
        ev, ws = ctx_objects()
        sc = cfg.get("search", {})
        sp = (sft.get("points") or {}).get("screen") or {}
        sftinfo = {"rank": (sft.get("train") or {}).get("rank"), "target_gain": sp.get("target_gain"),
                   "worst_retention": sp.get("worst_retention_delta")}
        ctx = ContextRenderer(task_text=task_text(gen=g, target=self.target, dataset=self.tds,
                                                  retention=self.retention_domains, n_train=len(ws.train.ids()),
                                                  sft_info=sftinfo),
                              diag_text=D.render(diag), archive_text=self.archive(g) if sc.get("archive", True) else "")
        sysmsg = system_prompt(cfg, budget=budget)
        policy = M.make_mock_policy(rc.seed, skill=self.quality(self.served(g)) or 0.7, tag=f"g{g}") if rc.mock else None

        def editor_factory(th):
            return Editor(ev.endpoint, self.served(g), system_prompt=sysmsg, seed=rc.seed + 1000 * g + th,
                          transcript=gd / "editor" / f"thread_{th}.jsonl", mock_policy=policy,
                          temperature=float(cfg.get("editor", {}).get("temperature", 0.6)),
                          max_tokens=int(cfg.get("editor", {}).get("max_tokens", 8192)))

        sb = cfg.get("sandbox", {})

        def build(program, cid):
            return program_build(ws, program, cid, timeout_s=int(sb.get("timeout_s", 1800)),
                                 mem_gb=int(sb.get("mem_gb", 16)), workdir=gd / "work" / cid)

        def screen(art, cid):
            return ws.screen_candidate(art["cid"])   # charged on the workspace's ToolAPI (identical LoRA: free)

        def full(art, cid):
            return ws.evaluate_lora(ws.load_candidate(art["cid"]), "full", charge=False)

        return inner_search(rc, g, editor_factory=editor_factory, render_context=ctx, build_fn=build, screen_fn=screen,
                            full_fn=full, threads=int(sc.get("threads", 4)), steps=int(sc.get("steps", 8)),
                            max_candidates=int(sc.get("max_evaluate", 32)), top_full=int(sc.get("top_full", 3)),
                            rerun_seeds=int(sc.get("rerun_seeds", 0)))

    def archive(self, g: int) -> str:
        """Experience channel: best programs and outcomes of earlier generations (RSI_LOOP_DESIGN §1.1)."""
        if g == 0:
            return ""
        from rsicore.search import load_candidates, ranked
        parts = ["Archive of earlier generations:"]
        for k in range(g):
            cs = ranked(load_candidates(self.rc, k).values())
            acc = read_json(self.gdir(k) / "accepted.json", {}) or {}
            parts.append(f"- generation {k}: accepted={acc.get('source')} {acc.get('cid') or ''}; "
                         f"{len(cs)} candidates; best scores " +
                         ", ".join(f"{c.cid}={c.score:.4f}" for c in cs[:3] if c.score is not None))
            if cs and cs[0].program:
                parts.append(f"  best program of generation {k}:\n```python\n{cs[0].program[:3000]}```")
        return "\n".join(parts)

    # ------------------------------------------------------------------ gate
    def gate_check(self, ev, ws, lo, sft, learned) -> tuple[bool, dict]:
        gcfg = self.cfg.get("gate", {})
        eps, delta = float(gcfg.get("epsilon", 0.01)), float(gcfg.get("delta", 0.02))
        use_nb = bool(gcfg.get("use_noise_band", True))
        ok, rep = ws.check(lo)
        info = {"check_ok": ok, "reasons": []}
        if not ok:
            info["reasons"].append("sandbox checks failed")
            return False, info
        name = ws.lora_name(lo) if lo.modules() else None
        if name is not None and not ev.known(name):
            ev.register(lo, name)
        res = ev.run(name, "gate")
        g0 = self.gen0_gate()
        sft_gate = sft.get("gate") or {}
        scores = {ds: r["score"] for ds, r in res.items()}
        info["scores"] = scores
        t, s = scores.get(self.tds), sft_gate.get(self.tds)
        if t is None:
            info["reasons"].append("target gate score unavailable")
        elif s is not None and t < s - eps:
            info["reasons"].append(f"target {t:.4f} < SFT {s:.4f} - eps {eps}")
        info["pending"] = []
        for ds in self.datasets:
            if ds == self.tds:
                continue
            c, b = scores.get(ds), (g0.get(ds) or {}).get("score")
            if c is None or b is None:
                info["pending"].append(ds)
                continue
            d = max(delta, float((g0.get(ds) or {}).get("noise_band") or 0)) if use_nb else delta
            if c < b - d:
                info["reasons"].append(f"retention {ds} {c:.4f} < gen-0 {b:.4f} - {d:.4f}")
        for ds, v in (learned or {}).items():
            c = scores.get(ds)
            if c is not None and v is not None and c < v - delta:
                info["reasons"].append(f"learned {ds} {c:.4f} < post-learning {v:.4f} - delta {delta}")
        return not info["reasons"], info

    # ------------------------------------------------------------------ E branch
    def e_branch(self, g, branch, endpoint, served_name, skill):
        from lora_code.etask import run_etask
        task, items, tdata = self.e_setup()
        th = self.theta0()
        out = self.gdir(g) / "e_t" / branch

        def make_ws(out_dir, ev, budget, prefix):
            return self.workspace(out_dir, ev, th["store"], th["hf"], tdata, budget, prefix)

        return run_etask(self.rc, self.cfg, out_dir=out, branch=branch, editor_endpoint=endpoint,
                         editor_served=served_name, editor_skill=skill, theta0=th, task=task, items=items,
                         make_workspace=make_ws)

    # ------------------------------------------------------------------ summary
    def summarize(self, g: int, gd: Path) -> dict:
        from rsicore.evaluate import paired_delta
        acc = read_json(gd / "accepted.json", {}) or {}
        gate = read_json(gd / "gate.json", {}) or {}
        sft = read_json(gd / "baselines" / "sft.json", {}) or {}
        non = read_json(gd / "baselines" / "nonllm.json", {}) or {}
        info = {"gen": g, "editor": self.served(g), "successor": self.served(g + 1),
                "accepted": {k: acc.get(k) for k in ("source", "cid", "name", "gate_scores")},
                "gate_submissions": [{"cid": s["cid"], "passed": s["passed"], "reasons": (s.get("info") or {}).get("reasons")}
                                     for s in gate.get("submissions", [])],
                "sft": {"name": sft.get("name"), "gate": sft.get("gate"), "screen": (sft.get("points") or {}).get("screen")},
                "nonllm": {"name": non.get("name"), "best": (non.get("random_search") or {}).get("best")},
                "model": read_json(gd / "model.json"), "heldout": read_json(gd / "heldout.json")}
        from rsicore.search import load_candidates
        cands = list(load_candidates(self.rc, g).values())
        strat = {}
        for c in cands:
            k = (c.artifact or {}).get("strategy") or ("error" if c.error else "none")
            strat[k] = strat.get(k, 0) + 1
        info["search"] = {"candidates": len(cands), "built": sum(1 for c in cands if c.artifact),
                          "failed": sum(1 for c in cands if c.error), "strategies": strat,
                          "best_screen": max((c.screen or {}).get("score") or -1e9 for c in cands) if cands else None}
        E = {}
        for b in ("parent", "sft", "llm", "nonllm"):
            r = read_json(gd / "e_t" / b / "result.json")
            if r:
                E[b] = r
        eref = read_json(self.rc.run_dir / "e_ref" / "result.json", {}) or {}
        ref_score = (eref.get("sft_ref") or {}).get("score")
        info["E"] = {b: {"score": r["score"], "vs_sft_ref": (r["score"] - ref_score) if (r["score"] is not None and ref_score is not None) else None,
                         "per_task": r["per_task"], "best_cid": r["best_cid"]} for b, r in E.items()}
        info["E_ref"] = {"theta0": (eref.get("theta0") or {}).get("score"), "sft_ref": ref_score}

        def d(a, b):
            if a in E and b in E:
                return paired_delta(E[a]["records"], E[b]["records"], seed=self.rc.seed)
            return None
        info["dE_train"] = d("parent", "sft")      # E(SFT) - E(Parent)
        info["dE_write"] = d("sft", "llm")         # E(LLM-written) - E(SFT)
        info["dE_nonLLM"] = d("sft", "nonllm")     # E(Non-LLM) - E(SFT)
        return info

    # ------------------------------------------------------------------ main
    def run(self):
        t0 = time.time()
        try:
            self.baseline()
            for g in range(self.rc.gens):
                self.generation(g)
        finally:
            self.servers.stop()
            self.release_hf()
        print(f"[lora_code] run {self.rc.run_id}: {self.rc.gens} generation(s) done in {time.time() - t0:.0f}s "
              f"-> {self.rc.run_dir}", flush=True)


def main(argv=None):
    from rsicore.config import build_arg_parser, load_run_config
    ap = build_arg_parser("B")
    args = ap.parse_args(argv)
    rc = load_run_config("B", args)
    cfg = rc.paper_cfg   # hardware profile already merged by rsicore.config (before --set)
    hw = cfg.get("hardware", "b300")
    if rc.mock:
        cfg = deep_merge(cfg, cfg.get("mock_overrides", {}) or {})
    roles = {r: role_gpus(rc, cfg, r) for r in ("main", "theta0", "train")}   # fail fast: enough --gpus?
    print(f"[lora_code] hardware={hw} gpu roles {roles}", flush=True)
    try:
        Run(rc, cfg).run()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
