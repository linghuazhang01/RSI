#!/usr/bin/env python
"""Paper B orchestration: recursive self-written LoRA (plan/PLAN.md §4, RSI_LOOP_DESIGN §3).

  python -I lora_code/run.py --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id B_math_s42 [--mock]

Generation g (editor M_g -> successor M_{g+1}; the target domain is --domain in every generation):
  draw (target=domain) -> train data (sampled answers only when the raw LoRA is trained; else gold)
  -> diagnostics -> [method.mode=edit, generation 0 only] raw LoRA (fixed `raw` recipe, size raw.n_train /
     steps / rank: the ONLY training of a run; there is no SFT baseline, 2026-11-20)
  -> inner search (editor = M_g on the main GPU with multi-LoRA; candidates built on the train GPU,
     hot-loaded for evaluation; no training / gradient tools: edit = edit RAW, scratch = from zero)
  -> gate (target vs the reference: edit raw - eps (gen 0) / parent + min_gain (later), scratch parent +
     min_gain; retention >= gen-0 - delta; learned targets >= post-learning - delta; sandbox checks)
  -> accept (fallback: edit raw in gen 0, else keep) -> merge into weights (gen_XX/model; edit: the LoRA
     replaces L_g) -> non-LLM branch (by mode) -> E_t for each branch -> held-out evaluation of M_{g+1}
  -> summary (ΔE_write = E(LLM) - E(reference branch), ΔE_nonLLM, with paired CIs).
Baseline first: M_0 on the fixed gate (noise band) and held-out; E-ref (θ0 on the E test; mode=edit: the
E-task raw LoRA for θ0, trained once).
Every step is recorded in state.json (rsicore.state.RunState); rerunning the same run-id resumes.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "eval")]

from lora_code.common import (DOMAIN_DATASET, DOMAINS, deep_merge, gate_policy, gold_target,  # noqa: E402
                              gpu_layout, load_records, method_mode, prompt_text, read_json, render_messages,
                              retention_domains, role_gpus, write_json)


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
        self.mode = method_mode(cfg)
        self.gate_ref, self.gate_fallback = gate_policy(cfg)
        self._stores = {}
        self._hf = {}

    # ------------------------------------------------------------------ models
    @property
    def chain(self) -> bool:
        """mode=edit: one LoRA, trained once (generation 0) and edited by the model in every generation.
        M_g = M_0 + L_g served as an adapter on the M_0 server (no merge); candidates replace L_g."""
        return self.mode == "edit"

    def served(self, g: int) -> str:
        return "m0" if self.chain else f"m{g}"

    def current_lora(self, g: int):
        """mode=edit: (LoRA, name, file) of L_g (M_g = M_0 + L_g), from generation g-1's model.json; None if
        g == 0 or no LoRA has been accepted yet."""
        if not self.chain or g == 0:
            return None
        info = read_json(self.gdir(g - 1) / "model.json") or {}
        if not info.get("lora_file"):
            return None
        f = Path(info["lora_file"])
        return self.load_lora(f), info["lora"], f

    def editor_name(self, g: int) -> str:
        """Model name of M_g on the main server: the current LoRA's adapter name in mode=edit."""
        cur = self.current_lora(g)
        return cur[1] if cur else self.served(g)

    def model_path(self, g: int) -> Path:
        if g == 0 or self.chain:
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

    def raw_file(self, gd: Path, rj: dict) -> Path:
        """File of the raw LoRA described by raw.json `rj` (its "file", else the raw LoRA of gd)."""
        if rj.get("file"):
            return Path(rj["file"])
        return self.lora_file(gd / "baselines" / "raw_lora")

    def stack(self, g: int) -> list:
        if self.chain:   # accepted LoRAs replace each other; nothing is merged
            return []
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

    def train_data_mode(self, g: int) -> str:
        """Answers are sampled (train_data.mode) only when something is trained: the raw LoRA of mode=edit in
        generation 0. Otherwise 'gold', no sampling: the train items only feed activation statistics and
        api.train_example."""
        if self.chain and g == 0:
            return self.cfg.get("train_data", {}).get("mode", "self_correct")
        return "gold"

    def build_train_data(self, dataset, ids, out_path: Path, *, endpoint=None, served=None, mode="gold"):
        """{id: {messages, target, source}} for train ids. Sampling modes sample the current model on the train
        items (train_data: 4 samples at 0.7 / 0.8 / top_k 20, the run seed) and keep the first correct sample
        (rejection sampling). 'self_correct' (default since 2026-11-20): an item with no correct sample gets no
        target and is skipped by training; 'self_correct_then_gold': it gets the gold response instead;
        'gold': gold responses only, no sampling. messages = the evaluator's own prompt messages (the same
        chat-format request the model is scored with)."""
        got = read_json(out_path)
        if got is not None:
            return got
        recs = load_records(dataset, ids)
        selfs = {}
        tc = self.cfg.get("train_data", {})
        if mode in ("self_correct", "self_correct_then_gold") and endpoint:
            from rsicore.evaluate import score
            n = int(tc.get("samples", 4))
            er = score(endpoint, served, dataset, pool="pool", ids=list(ids), out_dir=out_path.parent / "train_samples",
                       extra=["--n", str(n)], temperature=float(tc.get("temperature", 0.7)),
                       top_p=float(tc.get("top_p", 0.8)), top_k=tc.get("top_k", 20), seed=self.rc.seed,
                       mock=self.rc.mock)
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
            if tgt is None and mode in ("gold", "self_correct_then_gold"):
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

    def workspace(self, out_dir, ev, store, hf, train, budget, prefix, stack=(), diag=None, ret_texts=(),
                  raw=None, raw_info=None):
        from lora_code.primitives.train import TrainData
        from lora_code.tools import Workspace
        return Workspace(mock=self.rc.mock, store=store, train=TrainData(train), cfg=self.cfg, out_dir=out_dir,
                         evaluator=ev, hf=hf, stack=stack, retention_texts=ret_texts, budget=budget,
                         name_prefix=prefix, diagnostics=diag, seed=self.rc.seed, mode=self.mode, raw=raw,
                         raw_info=raw_info)

    # ================================================================== baseline
    def baseline(self):
        from rsicore import draws
        from rsicore.gate import fixed_noise_band, measure_noise_band, noise_ids
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
                fixed = fixed_noise_band(bc.get("noise_fixed"), ds)
                if fixed is not None:
                    # baseline.noise_fixed: the configured band is used as is, no sampled re-runs
                    res[ds] = {"score": m, "n": len(recs), "n_scored": len(sc), "se": se, "noise_band": fixed,
                               "noise_source": "fixed (baseline.noise_fixed)", "records": recs}
                    continue
                # noise band: spread of `reps` sampled re-runs (seeds seed, seed+1, ...), 0 with greedy decoding
                # (the default: nothing is re-run); the gate margin is max(band, 2 SE) because the band of a few
                # re-runs underestimates item-sampling noise
                nb = measure_noise_band(srv.endpoint, self.served(0), ds, noise_ids(ids, bc.get("noise_n"), self.rc.seed),
                                        bdir / "noise" / ds, repeats=reps,
                                        temperature=float(bc.get("noise_temperature", 0.0)),
                                        top_p=float(bc.get("noise_top_p", 1.0)), base_seed=self.rc.seed,
                                        mock=self.rc.mock)
                res[ds] = {"score": m, "n": len(recs), "n_scored": len(sc), "noise_band_repeats": nb["band"],
                           "noise_scores": nb["scores"], "noise_seeds": nb["seeds"], "noise_se": nb["se"],
                           "noise_items": nb["n"],
                           "se": se, "noise_band": max(nb["band"], 2 * se if se else 0.0),
                           "noise_source": nb.get("source", "measured") + " (gate margin = max(band, 2 SE))",
                           "records": recs}
            write_json(bdir / "gate_scores.json", res)
            self.st.mark(key, {ds: {k: v[k] for k in ("score", "noise_band")} for ds, v in res.items()})
        if self.cfg.get("heldout", {}).get("enabled", True) and not self.st.done("baseline/heldout"):
            from rsicore.heldout import run_heldout
            srv = self.servers.get("main", self.model_path(0), self.served(0))
            h = run_heldout(srv.endpoint, self.served(0), bdir / "heldout",
                            domains=self.cfg.get("heldout", {}).get("domains", "all"), mock=self.rc.mock,
                            limit=self.cfg.get("heldout", {}).get("limit"), decoding=self.cfg.get("heldout", {}).get("decoding"))
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
        lim = self.cfg.get("e_t", {}).get("test_limit")
        dec = (self.cfg.get("heldout") or {}).get("decoding")
        base = e_test(th["endpoint"], "theta0", task, out / "base", lora=None, mock=self.rc.mock, limit=lim, decoding=dec)
        res = {"task": task, "theta0": base}
        if self.mode == "edit":   # the E-task editor edits a raw LoRA for θ0 made with the same raw recipe
            rrec, n, rgb = baselines.raw_recipe(self.cfg)
            gb = float(self.cfg.get("e_t", {}).get("gpu_seconds", rgb))
            ws = self.workspace(out, ev, th["store"], th["hf"], tdata, {"evaluate": 1e9, "gpu_seconds": gb}, "eref")
            rids = baselines.subset_ids(tdata.keys(), n, self.rc.seed)
            rlo, rinfo = baselines.train_lora(ws, rrec, gb, seed=self.rc.seed, ids=rids)
            rinfo.update({"n_train": len(rids), "rank": int(rrec["rank"])})
            rname = f"E-{self.rc.run_id}-raw"
            self.save_lora(rlo, out / "raw_lora")
            ev.register(rlo, rname)
            ev.ensure_loaded(rname)
            rt = e_test(th["endpoint"], "theta0", task, out / "raw", lora=rname, mock=self.rc.mock, limit=lim,
                        decoding=dec)
            res["raw"] = {"name": rname, "train": rinfo, "test": rt}
        write_json(out / "result.json", res)
        self.st.mark("e_ref", {"theta0": base["score"], "task": task["key"],
                               "raw_ref": ((res.get("raw") or {}).get("test") or {}).get("score")})
        self.release_hf()

    def e_raw(self):
        """mode=edit: (raw LoRA for θ0 on the E task, its name, info) from e_ref; None otherwise."""
        if self.mode != "edit":
            return None
        out = self.rc.run_dir / "e_ref"
        r = (read_json(out / "result.json", {}) or {}).get("raw")
        if not r:
            return None
        f = self.lora_file(out / "raw_lora")
        return (self.load_lora(f), r["name"], r["train"]) if f else None

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
        cur = self.current_lora(g)                        # mode=edit, g >= 1: (L_g, name, file) or None
        edit_later = self.chain and g > 0                 # mode=edit after generation 0: no training at all
        diag_stack = [(g - 1, cur[0])] if cur else stack  # diagnostics show the current LoRA

        def main_srv():
            return self.servers.get("main", parent, served)

        # ---- train data (sampled answers only when the raw LoRA is trained, see train_data_mode)
        tmode = self.train_data_mode(g)
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
            diag = D.load_or_compute(gd / "diagnostics.json", lambda: mock_diagnostics(store, diag_stack, rc.seed))
        else:
            def real_diag():
                hf = self.hf(parent)
                tt = ["\n\n".join(m["content"] for m in e["messages"]) for e in list(tdata.values())[:64] if e["messages"]]
                return D.compute(store, diag_stack, base_store=self.store(Path(rc.model_path)), hf=hf, target_texts=tt,
                                 retention_texts=kl_texts, svd_k=int(cfg.get("diagnostics", {}).get("svd_k", 8)),
                                 device=self.train_device())
            diag = D.load_or_compute(gd / "diagnostics.json", real_diag)
        st.mark(f"{P}/diagnostics", {"rows": diag["n_modules"]}) if not st.done(f"{P}/diagnostics") else None

        budget = {"evaluate": int(cfg.get("search", {}).get("max_evaluate", 32)),
                  "gpu_seconds": float(cfg.get("budget", {}).get("gpu_seconds", 7200))}
        state = {}

        def get_ev():
            """Evaluator on the main server; knows the raw LoRA once it exists (also after a resume)."""
            if "ev" not in state:
                srv = main_srv()
                ev = self.evaluator(g, gd, srv, served, ids, kl_texts, store)
                if cur:   # mode=edit: parent / editor = M_0 + L_g
                    ev.set_parent(cur[0], cur[1])
                state.update(ev=ev, srv=srv)
            ev = state["ev"]
            info = read_json(gd / "baselines" / "raw.json")
            if not edit_later and info and info.get("name") and ev.raw_name != info["name"]:
                # (edit_later: RAW = the current LoRA = the parent, no separate vs_raw)
                f = self.raw_file(gd, info)
                if f is not None and not ev.known(info["name"]):
                    ev.register(self.load_lora(f), info["name"])
                ev.raw_name = info["name"]
            return ev

        def ctx_objects():
            """(evaluator, editor workspace). In mode=edit the workspace holds the raw LoRA (step raw first)."""
            ev = get_ev()
            if "ws" in state:
                return ev, state["ws"]
            raw, raw_info = None, None
            if self.mode == "edit":
                rj = read_json(gd / "baselines" / "raw.json")
                if rj is None:
                    raise RuntimeError(f"{P}: mode=edit needs the raw LoRA (step {P}/raw) before the workspace")
                raw = self.load_lora(self.raw_file(gd, rj))
                raw_info = dict(rj["train"], source=rj.get("source", "trained"))
            ws = self.workspace(gd, ev, store, self.hf(parent), tdata, budget, f"g{g:02d}", stack=stack, diag=diag,
                                ret_texts=kl_texts, raw=raw, raw_info=raw_info)
            ws.build_api()
            state["ws"] = ws
            return ev, ws

        from lora_code import baselines as BL

        # ---- raw LoRA (mode=edit): the LoRA the editor edits. Generation 0: trained with the `raw` recipe;
        # later generations: the current LoRA L_g (or generation 0's raw LoRA while none was accepted).
        raw = None
        if self.chain and edit_later and not st.done(f"{P}/raw"):
            r0 = read_json(self.gdir(0) / "baselines" / "raw.json") or {}
            if cur:
                rj = {"name": cur[1], "file": str(cur[2]), "source": "current", "check_ok": True,
                      "train": dict(r0.get("train") or {}, trained_in_generation=0, edited_generations=g)}
            else:
                rj = dict(r0, source="generation-0 raw (none accepted yet)",
                          file=str(self.raw_file(self.gdir(0), r0)))
            write_json(gd / "baselines" / "raw.json", rj)
            st.mark(f"{P}/raw", {"name": rj["name"], "source": rj["source"]})
        if self.chain:
            if not st.done(f"{P}/raw"):
                rrec, n_raw, rgb = BL.raw_recipe(cfg)
                ev = get_ev()
                bws = self.workspace(gd, ev, store, self.hf(parent), tdata, {"evaluate": 1e9, "gpu_seconds": rgb},
                                     f"g{g:02d}", stack=stack, ret_texts=kl_texts)
                rids = BL.subset_ids(bws.train.ids(), n_raw, rc.seed)
                lo, info = BL.train_lora(bws, rrec, rgb, seed=rc.seed, ids=rids)
                info.update({"n_train": len(rids), "rank": int(rrec["rank"])})
                self.save_lora(lo, gd / "baselines" / "raw_lora")
                ok, chk = bws.check(lo)
                pts = {lvl: BL.point(bws, lo, "raw", lvl) for lvl in ("screen", "full")}
                gate = ev.run(bws.lora_name(lo), "gate") if ok else None
                rj = {"name": bws.lora_name(lo), "train": info, "check_ok": ok,
                      "check_errors": chk.get("errors", [])[:10], "points": pts,
                      "gate": {ds: r["score"] for ds, r in (gate or {}).items()}}
                rj["source"] = "trained"
                rj["file"] = str(self.raw_file(gd, rj))
                write_json(gd / "baselines" / "raw.json", rj)
                st.mark(f"{P}/raw", {"name": rj["name"], "n_train": rj["train"].get("n_train"),
                                     "screen": ((rj.get("points") or {}).get("screen") or {}).get("score")})
            raw = read_json(gd / "baselines" / "raw.json")

        # ---- inner search
        ranked = None
        if not st.done(f"{P}/search"):
            ranked = self.search(g, gd, ctx_objects, diag, budget, raw)
            st.mark(f"{P}/search", {"candidates": len(ranked), "built": sum(1 for c in ranked if c.artifact),
                                    "best": ranked[0].cid if ranked else None})
        # ---- gate
        if not st.done(f"{P}/gate"):
            from rsicore.search import load_candidates, ranked as rank_fn, submit_to_gate
            if ranked is None:
                ranked = rank_fn(load_candidates(rc, g).values())
            ev, ws = ctx_objects()
            learned = (st.get("learned") or {}).get("scores", {})
            ref = self.gate_reference(gd, ev, raw, kind="parent" if edit_later else None)
            fallback = "keep" if edit_later else self.gate_fallback

            def gate_fn(c):
                lo = ws.load_candidate(c.artifact["cid"])
                return self.gate_check(ev, ws, lo, ref, learned)

            # reference = the current model: submit only candidates that beat it on the signal items (score =
            # target gain vs the current model - penalties > 0), like Paper A (2026-11-20). Reference = raw
            # (mode=edit, generation 0): every ranked candidate, the gate checks target >= raw - epsilon.
            subs = [c for c in ranked if c.score is not None and c.score > 0] if ref["kind"] == "parent" else ranked
            acc = submit_to_gate(rc, g, subs, gate_fn, max_submissions=int(cfg.get("gate", {}).get("max_submissions", 3)))
            if acc is not None:
                lo = ws.load_candidate(acc.cid)
                rec = {"source": "llm", "cid": acc.cid, "name": ws.lora_name(lo) if lo.modules() else None,
                       "gate_reference": ref["kind"]}
            elif fallback == "raw":
                lo = self.load_lora(self.raw_file(gd, raw))
                ok, info = (self.gate_check(ev, ws, lo, ref, learned) if raw.get("check_ok")
                            else (False, {"reasons": ["raw LoRA failed checks"]}))
                if ok:
                    rec = {"source": "raw_fallback", "cid": None, "name": raw["name"], "gate": info,
                           "gate_reference": ref["kind"]}
                else:
                    lo, rec = None, {"source": "keep", "cid": None, "name": None, "gate_reference": ref["kind"],
                                     "reason": "no candidate and no raw LoRA passed the gate",
                                     "fallback_gate": info}
            else:
                lo, rec = None, {"source": "keep", "cid": None, "name": None, "gate_reference": ref["kind"],
                                 "reason": "no candidate passed the gate (fallback: keep)"}
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
        if self.chain and not st.done(f"{P}/merge"):
            lf = self.lora_file(gd / "accepted_lora")
            if lf is not None:     # accepted: L_{g+1} = the accepted LoRA (it replaces L_g)
                info = {"path": str(parent), "lora": acc["name"], "lora_file": str(lf), "source": acc["source"]}
            elif cur:              # nothing accepted: L_{g+1} = L_g
                info = {"path": str(parent), "lora": cur[1], "lora_file": str(cur[2]), "source": "keep (current LoRA)"}
            else:
                info = {"path": str(parent), "lora": None, "lora_file": None, "source": "keep (original model)"}
            if rc.mock:
                info["quality"] = self.quality_value(info["lora"] or served)
            write_json(gd / "model.json", info)
            st.mark(f"{P}/merge", {"merged": False, "lora": info["lora"], "source": info["source"]})
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

        # ---- non-LLM branch
        if not st.done(f"{P}/nonllm"):
            ev, ws = ctx_objects()
            nc = cfg.get("nonllm", {})
            bws = self.workspace(gd, ev, store, self.hf(parent), tdata, {"evaluate": 1e9, "gpu_seconds": budget["gpu_seconds"]},
                                 f"g{g:02d}", stack=stack, ret_texts=kl_texts)
            out = {}
            n_ev = int(nc.get("evaluate", budget["evaluate"]))
            if self.mode == "edit":
                rlo = self.load_lora(self.raw_file(gd, raw))
                best, log = BL.random_edit_search(bws, rlo, evals=n_ev, seed=rc.seed,
                                                  scales=nc.get("search_scales", [0.5, 0.75, 1.0, 1.25]),
                                                  use_regmean=bool(kl_texts))
            else:
                best, log = BL.random_construct_search(bws, evals=n_ev, seed=rc.seed)
            out["kind"] = {"edit": "random_edit_search", "scratch": "random_construct_search"}[self.mode]
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
            branches = {"parent": cur[1] if cur else None, "llm": acc.get("name") if acc.get("source") == "llm" else None,
                        "nonllm": nonllm.get("name"), "raw": (raw or {}).get("name")}
            if acc.get("source") == "raw_fallback":
                branches["llm"] = acc.get("name")
            blist = list(cfg.get("e_t", {}).get("branches", ["parent", "llm", "nonllm"]))
            if raw and "raw" not in blist:   # mode=edit: M_t ⊕ raw is the reference branch of ΔE_write
                blist.insert(1 if "parent" in blist else 0, "raw")
            for b in blist:
                key = f"{P}/e_t/{b}"
                if st.done(key):
                    continue
                ev = get_ev()
                lname = branches.get(b)
                if b != "parent" and lname is None:
                    st.mark(key, {"skipped": f"no {b} LoRA"})
                    continue
                if lname is not None:
                    if b == "parent" and cur and not ev.known(lname):
                        ev.register(cur[0], lname)
                    if not ev.known(lname):   # resumed run: register the branch LoRA from disk
                        f = (self.raw_file(gd, raw) if b == "raw" else
                             self.lora_file(gd / {"llm": "accepted_lora", "nonllm": "baselines/nonllm_lora"}.get(b, "accepted_lora")))
                        ev.register(self.load_lora(f), lname)
                    ev.ensure_loaded(lname)
                res = self.e_branch(g, b, ev.endpoint, lname or served,
                                    self.quality(lname) if lname else self.quality(served))
                st.mark(key, {"score": res["score"], "best_cid": res["best_cid"]})

        # ---- held-out evaluation of M_{g+1}
        if cfg.get("heldout", {}).get("enabled", True) and not st.done(f"{P}/heldout"):
            from rsicore.heldout import run_heldout
            if self.chain:   # M_{g+1} = M_0 + L_{g+1}: the same M_0 server, the adapter as the model
                ev = get_ev()
                nxt = self.current_lora(g + 1)
                if nxt:
                    if not ev.known(nxt[1]):
                        ev.register(nxt[0], nxt[1])
                    ev.ensure_loaded(nxt[1])
                h = run_heldout(ev.endpoint, served, gd / "heldout", lora=nxt[1] if nxt else None,
                                domains=cfg.get("heldout", {}).get("domains", "all"), mock=rc.mock,
                                limit=cfg.get("heldout", {}).get("limit"), decoding=cfg.get("heldout", {}).get("decoding"))
            else:
                nxt = self.model_path(g + 1)
                srv = self.servers.get("main", nxt, self.served(g + 1))
                state.clear()
                h = run_heldout(srv.endpoint, self.served(g + 1), gd / "heldout",
                                domains=cfg.get("heldout", {}).get("domains", "all"), mock=rc.mock,
                                limit=cfg.get("heldout", {}).get("limit"), decoding=cfg.get("heldout", {}).get("decoding"))
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

    def stall(self, g: int) -> int:
        """Consecutive generations up to g whose gate kept the current model (accepted.json source 'keep')."""
        n = 0
        for k in range(g, -1, -1):
            acc = read_json(self.gdir(k) / "accepted.json") or {}
            if acc.get("source") != "keep":
                break
            n += 1
        return n

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
    def search(self, g, gd, ctx_objects, diag, budget, raw=None):
        from rsicore.editor import Editor
        from rsicore.search import inner_search

        from lora_code import diagnostics as D
        from lora_code import mock as M
        from lora_code.prompts import ContextRenderer, system_prompt, task_text
        from lora_code.tools import program_build
        rc, cfg = self.rc, self.cfg
        ev, ws = ctx_objects()
        sc = cfg.get("search", {})
        rawinfo = None
        if raw:
            rp = (raw.get("points") or {}).get("screen") or {}
            tr = raw.get("train") or {}
            rawinfo = {"n_train": tr.get("n_train"), "rank": tr.get("rank"), "steps": tr.get("steps") or tr.get("steps_done"),
                       "target_gain": rp.get("target_gain"), "worst_retention": rp.get("worst_retention_delta"),
                       "source": raw.get("source"), "gen": g}
        ctx = ContextRenderer(task_text=task_text(gen=g, target=self.target, dataset=self.tds,
                                                  retention=self.retention_domains, n_train=len(ws.train.ids()),
                                                  raw_info=rawinfo),
                              diag_text=D.render(diag), archive_text=self.archive(g) if sc.get("archive", True) else "")
        has_ret = bool(ws.retention_texts)
        sysmsg = system_prompt(cfg, budget=budget, mode=self.mode, retention=has_ret)
        editor = self.editor_name(g)   # M_g: in mode=edit the current LoRA's adapter on the M_0 server (pinned)
        policy = (M.make_mock_policy(rc.seed, skill=self.quality(editor) or 0.7, tag=f"g{g}", mode=self.mode,
                                     retention=has_ret) if rc.mock else None)

        def editor_factory(th):
            return Editor(ev.endpoint, editor, system_prompt=sysmsg, seed=rc.seed + 1000 * g + th,
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
    def gate_reference(self, gd, ev, raw, kind: str | None = None) -> dict:
        """Target reference of the gate by gate.reference (auto: edit raw, scratch parent; mode=edit after
        generation 0: parent = M_0 + the current LoRA): {"kind", "scores": {dataset: gate score}}.
        The parent's gate scores are computed once per generation."""
        kind = kind or self.gate_ref
        if kind == "raw":
            return {"kind": "raw", "scores": dict((raw or {}).get("gate") or {})}
        pg = read_json(gd / "parent_gate.json")
        if pg is None:
            pg = {ds: r["score"] for ds, r in ev.run(None, "gate").items()}
            write_json(gd / "parent_gate.json", pg)
        return {"kind": "parent", "scores": pg}

    def gate_check(self, ev, ws, lo, ref, learned) -> tuple[bool, dict]:
        """ref = gate_reference(...). Target: raw -> >= ref - epsilon; parent -> paired per item on the gate set
        (2026-11-20, like Paper A): mean paired gain >= min_gain and, with gate.require_significant, the 95% CI
        lower bound of the paired gain > 0 (bootstrap over items, rsicore.evaluate.paired_delta)."""
        gcfg = self.cfg.get("gate", {})
        eps, delta = float(gcfg.get("epsilon", 0.01)), float(gcfg.get("delta", 0.02))
        min_gain = float(gcfg.get("min_gain", 0.01))
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
        scores = {ds: r["score"] for ds, r in res.items()}
        info["scores"] = scores
        info["reference"] = ref["kind"]
        t, s = scores.get(self.tds), (ref.get("scores") or {}).get(self.tds)
        if t is None:
            info["reasons"].append("target gate score unavailable")
        elif ref["kind"] == "parent":
            from rsicore.evaluate import paired_delta
            par = (ev.run(None, "gate").get(self.tds) or {}).get("records") or []
            cand = (res.get(self.tds) or {}).get("records") or []
            d = paired_delta([r for r in par if r.get("score") is not None],
                             [r for r in cand if r.get("score") is not None], seed=self.rc.seed)
            info["paired_gain"], info["paired_ci"] = d["delta"], d["ci"]
            if s is None or d["delta"] is None:
                info["reasons"].append("parent target gate score unavailable")
            elif d["delta"] < min_gain:
                info["reasons"].append(f"paired gain {d['delta']:+.4f} vs parent ({s:.4f}) < min_gain {min_gain}")
            elif bool(gcfg.get("require_significant", True)) and not (d["ci"][0] is not None and d["ci"][0] > 0):
                info["reasons"].append(f"paired gain {d['delta']:+.4f} not significant "
                                       f"(95% CI [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}])")
        elif s is not None and t < s - eps:
            info["reasons"].append(f"target {t:.4f} < {ref['kind']} {s:.4f} - eps {eps}")
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

        er = self.e_raw()   # mode=edit: the E-task raw LoRA for θ0 (from e_ref)

        def make_ws(out_dir, ev, budget, prefix):
            return self.workspace(out_dir, ev, th["store"], th["hf"], tdata, budget, prefix,
                                  raw=er[0] if er else None, raw_info=er[2] if er else None)

        return run_etask(self.rc, self.cfg, out_dir=out, branch=branch, editor_endpoint=endpoint,
                         editor_served=served_name, editor_skill=skill, theta0=th, task=task, items=items,
                         make_workspace=make_ws, mode=self.mode, raw=(er[0], er[1]) if er else None)

    # ------------------------------------------------------------------ summary
    def summarize(self, g: int, gd: Path) -> dict:
        from rsicore.evaluate import paired_delta
        acc = read_json(gd / "accepted.json", {}) or {}
        gate = read_json(gd / "gate.json", {}) or {}
        non = read_json(gd / "baselines" / "nonllm.json", {}) or {}
        raw = read_json(gd / "baselines" / "raw.json", {}) or {}
        info = {"gen": g, "editor": self.editor_name(g), "successor": (read_json(gd / "model.json") or {}).get("lora")
                if self.chain else self.served(g + 1), "method": self.mode,
                "scope": self.cfg.get("scope", "all"), "retention_domains": list(self.retention_domains),
                "gate_reference": acc.get("gate_reference") or self.gate_ref,
                "gate_fallback": "keep" if (self.chain and g > 0) else self.gate_fallback,
                "accepted": {k: acc.get(k) for k in ("source", "cid", "name", "gate_scores")},
                "gate_submissions": [{"cid": s["cid"], "passed": s["passed"], "reasons": (s.get("info") or {}).get("reasons")}
                                     for s in gate.get("submissions", [])],
                "nonllm": {"name": non.get("name"), "kind": non.get("kind"),
                           "best": (non.get("random_search") or {}).get("best")},
                "raw": ({"name": raw.get("name"), "source": raw.get("source"), "gate": raw.get("gate"),
                         "n_train": (raw.get("train") or {}).get("n_train"),
                         "screen": (raw.get("points") or {}).get("screen")} if raw else None),
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
        for b in ("parent", "raw", "llm", "nonllm"):
            r = read_json(gd / "e_t" / b / "result.json")
            if r:
                E[b] = r
        eref = read_json(self.rc.run_dir / "e_ref" / "result.json", {}) or {}
        th0 = (eref.get("theta0") or {}).get("score")
        info["E"] = {b: {"score": r["score"], "vs_theta0": (r["score"] - th0) if (r["score"] is not None and th0 is not None) else None,
                         "per_task": r["per_task"], "best_cid": r["best_cid"]} for b, r in E.items()}
        info["E_ref"] = {"theta0": th0, "raw_ref": (((eref.get("raw") or {}).get("test")) or {}).get("score")}

        def d(a, b):
            if a in E and b in E:
                return paired_delta(E[a]["records"], E[b]["records"], seed=self.rc.seed)
            return None
        # reference branch of ΔE_write by mode: edit = M_t ⊕ raw (falls back to parent when there is no raw
        # branch), scratch = parent (no training anywhere in the lineage)
        refb = "raw" if (self.mode == "edit" and "raw" in E) else "parent"
        info["dE_reference_branch"] = refb
        info["dE_write"] = d(refb, "llm")          # E(LLM-written) - E(reference branch)
        info["dE_nonLLM"] = d(refb, "nonllm")      # E(Non-LLM) - E(reference branch)
        return info

    # ------------------------------------------------------------------ main
    def run(self):
        t0 = time.time()
        try:
            self.baseline()
            patience = (self.cfg.get("loop") or {}).get("patience")
            for g in range(self.rc.gens):
                stop = self.st.get("stopped")
                if stop:
                    print(f"[lora_code] stopped at generation {stop.get('gen')}: {stop.get('reason')}", flush=True)
                    break
                self.generation(g)
                stall = self.stall(g)
                if patience and stall >= int(patience):
                    self.st.mark("stopped", {"gen": g, "reason": f"nothing accepted for {stall} generation(s) "
                                                                 f"(loop.patience {patience})"})
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
    from rsicore.evaluate import set_max_tokens
    set_max_tokens(cfg.get("max_tokens"))                # per-task max generation length for every run_eval call
    try:
        roles, unused = gpu_layout(rc, cfg)                                      # fail fast: enough --gpus?
    except ValueError as e:
        print(f"[lora_code] {e}", flush=True)
        return 1
    mode, (gref, gfb) = method_mode(cfg), gate_policy(cfg)                      # fail fast: valid method / gate
    if not (cfg.get("e_t") or {}).get("enabled", True):
        roles = {k: v for k, v in roles.items() if k != "theta0"}               # theta0 server never starts
    print(f"[lora_code] hardware={hw} scope={cfg.get('scope', 'all')} gpus={rc.gpus} roles {roles} method={mode} "
          f"gate reference={gref} fallback={gfb}"
          + (f" (UNUSED {unused}: tensor parallelism needs 1/2/4/8 GPUs per role)" if unused else ""), flush=True)
    try:
        Run(rc, cfg).run()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
