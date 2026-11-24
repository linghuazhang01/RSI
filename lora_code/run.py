#!/usr/bin/env python
"""Paper B orchestration: recursive self-written LoRA (plan/PLAN.md §4, RSI_LOOP_DESIGN §3).

  python -I lora_code/run.py --domain math --gens 5 --model qwen3.8-27b --seed 42 --gpus 0,1 --run-id B_math_s42 [--mock]

Generation g (editor M_g -> successor M_{g+1}; the target domain is --domain in every generation):
  draw (target=domain) -> train data (sampled answers only when the raw LoRA is trained; else gold)
  -> diagnostics -> [method.mode=edit, generation 0 only] raw LoRA (fixed `raw` recipe, size raw.n_train /
     steps / rank: the ONLY training of a run; there is no SFT baseline, 2026-11-20)
  -> inner search (editor = M_g on the main GPU with multi-LoRA; candidates built on the train GPU,
     hot-loaded for evaluation; no training / gradient tools: edit = edit RAW, scratch = from zero)
  -> gate (2026-11-23, the only flow: every candidate ONE gate evaluation during the search, paired vs the current
     model + min_gain; learned targets >= post-learning - delta; sandbox checks)
  -> accept the passing candidate with the largest paired gain (fallback: edit raw in gen 0 if it passes the same
     gate, else keep) -> merge into weights (gen_XX/model; edit: the LoRA
     replaces L_g) -> non-LLM branch (by mode) -> held-out evaluation of M_{g+1} -> summary.
Data (2026-11-22, the one flow): train pool simple_<d>_pool (a fresh screen draw per generation), gate
simple_<d>_eval, held-out simple_<d>_test (rsicore.draws). Baseline first: M_0 on the fixed gate and held-out.
Every step is recorded in state.json (rsicore.state.RunState); rerunning the same run-id resumes.
"""
from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "eval")]

from lora_code.common import (deep_merge, gate_fallback, gold_target,  # noqa: E402
                              adapter_rank_budget, gpu_layout, load_records, main_groups, main_server_args,
                              merge_mode, method_mode, prompt_text, read_json, render_messages,
                              role_gpus, server_lora_rank, tools_on_replicas, write_json)


# ============================================================================ servers
class Servers:
    """Live servers per slot ('main'); reused while the model is the same. 'main' = one replica per
    main GPU group (main_groups, 2026-11-20: data parallelism; every replica holds the same LoRAs): get() returns
    the first, replicas() all of them."""

    def __init__(self, rc, cfg, weights: bool = False):
        self.rc, self.cfg = rc, cfg
        self.live = {}
        # method.mode weights (2026-11-21): the first main group serves the current model (the editor, with vLLM
        # sleep mode); the other main groups are candidate slots (WeightSlots), one exported candidate each
        self.weights = bool(weights)

    @property
    def main_gpu_groups(self) -> list:
        groups = main_groups(self.rc, self.cfg)
        return groups[:1] if self.weights else groups

    @property
    def slot_groups(self) -> list:
        return main_groups(self.rc, self.cfg)[1:] if self.weights else []

    def get(self, slot: str, model_path, served: str):
        return self.replicas(slot, model_path, served)[0]

    def replicas(self, slot: str = "main", model_path=None, served: str | None = None) -> list:
        """All replicas of `slot` (started for model_path / served if given and different or not running)."""
        from rsicore.servers import VLLMServer, start_all
        cur = self.live.get(slot)
        if model_path is None:
            if cur is None:
                raise RuntimeError(f"no {slot} server running")
            return cur[1]
        if cur is not None and cur[0] == (str(model_path), served) and all(s.alive() for s in cur[1]):
            return cur[1]
        if cur is not None:
            for s in cur[1]:
                s.stop()
        sc = self.cfg.get("server", {})
        if slot != "main":
            raise ValueError(f"unknown server slot {slot!r} (Paper B has only 'main')")
        if self.rc.mock:   # mock_server.py takes none of vLLM's flags
            extra = ["--tasks", ",".join(sc["mock_tasks"])] if sc.get("mock_tasks") else []
        else:
            extra = main_server_args(self.cfg)
        srvs = [VLLMServer(model_path, served, g, enable_lora=True, max_loras=int(sc.get("max_loras", 8)),
                           max_lora_rank=server_lora_rank(self.cfg), extra_args=extra,
                           mock=self.rc.mock, mock_registry=self.rc.run_dir / "mock_registry.json", seed=self.rc.seed,
                           log_dir=self.rc.run_dir / "logs", sleep_mode=self.weights)
                for g in self.main_gpu_groups]
        start_all(srvs)
        self.live[slot] = ((str(model_path), served), srvs)
        return srvs

    def extra_main(self, model_path, served: str) -> list:
        """Temporary servers of model_path on the candidate slots (weights mode; e.g. held-out, generation-0 gate
        scoring on every GPU); the caller stops them. Empty outside weights mode."""
        from rsicore.servers import VLLMServer, start_all
        if not self.slot_groups:
            return []
        extra = [] if self.rc.mock else main_server_args(self.cfg)
        srvs = [VLLMServer(model_path, served, g, extra_args=extra, mock=self.rc.mock,
                           mock_registry=self.rc.run_dir / "mock_registry.json", seed=self.rc.seed,
                           log_dir=self.rc.run_dir / "logs") for g in self.slot_groups]
        start_all(srvs)
        return srvs

    def stop(self, slot=None):
        for k in ([slot] if slot else list(self.live)):
            v = self.live.pop(k, None)
            if v is not None:
                for s in v[1]:
                    s.stop()


class WeightSlots:
    """method.mode weights (2026-11-21): a candidate weight edit cannot be hot-loaded like an adapter, so each one is
    exported (parent model + edit: only the shards with an edited tensor are rewritten, the rest hard-linked) and
    served on a free candidate slot (one GPU group), scored, then the server and the export are removed. Slots =
    the main groups after the first; while the editor sleeps its group is lent as one more (lend / reclaim)."""

    def __init__(self, run):
        self.run = run
        self.q: queue.Queue = queue.Queue()
        self.groups = None                          # the slot GPU groups, filled on first use (lazy)
        self._init_lock = threading.Lock()
        self.lent = None
        self.export_lock = threading.Lock()
        self.parent, self.store = None, None

    def _init(self):
        with self._init_lock:
            if self.groups is None:
                self.groups = [list(g) for g in self.run.servers.slot_groups]
                for g in self.groups:
                    self.q.put(g)

    @property
    def n(self) -> int:
        self._init()
        return len(self.groups) + (1 if self.lent is not None else 0)

    def set_parent(self, parent_dir, store) -> None:
        self.parent, self.store = Path(parent_dir), store

    def lend(self, gpus) -> None:
        self._init()
        if self.lent is None:
            self.lent = list(gpus)
            self.q.put(self.lent)

    def reclaim(self) -> None:
        """Take the lent group back once it is free (waits for a candidate running on it)."""
        if self.lent is None:
            return
        held = []
        while True:
            g = self.q.get()
            if g is self.lent:
                break
            held.append(g)
        for g in held:
            self.q.put(g)
        self.lent = None

    def score(self, name: str, path, ids_by_ds: dict, dirs_by_ds: dict,
              decoding: dict | None = None) -> dict:  # pragma: no cover - GPU path
        """decoding: the level's rsicore decoding dict (the same as the parent's; None = tasks.yaml greedy)."""
        from rsicore.evaluate import score_split
        from rsicore.servers import VLLMServer

        from lora_code.primitives.wedit import WeightEdit, edit_into_model
        if self.n == 0:
            raise RuntimeError("no candidate slot for a weight edit (needs a free main GPU group)")
        run = self.run
        g = self.q.get()
        mdir = run.rc.run_dir / "cand_models" / name
        try:
            with self.export_lock:              # one export at a time (train GPU + disk)
                edit_into_model(WeightEdit.load_raw(path), self.parent, mdir, store=self.store,
                                base_store=run.store(Path(run.rc.model_path)), device=run.train_device())
            srv = VLLMServer(mdir, name, g, extra_args=main_server_args(run.cfg),
                             seed=run.rc.seed, log_dir=run.rc.run_dir / "logs")
            srv.start()
            try:
                return {ds: score_split([srv.endpoint], name, ds, pool="pool", ids=ids, out_dir=dirs_by_ds[ds],
                                        decoding=decoding)
                        for ds, ids in ids_by_ds.items()}
            finally:
                srv.stop()
        finally:
            shutil.rmtree(mdir, ignore_errors=True)
            self.q.put(g)


class _WeightPhases:
    """inner_search phases in weights mode: the editor (current model) proposes, then sleeps (weights -> CPU RAM)
    while the step's candidates are exported and evaluated at the same time on every slot, its GPU group lent as one
    more; editor_phase() takes the group back and wakes it."""

    def __init__(self, run):
        self.run = run

    def editor_phase(self):
        self.run.wslots.reclaim()
        for s in self.run.servers.replicas("main"):
            s.wake_up()

    def eval_phase(self) -> int:
        ed = self.run.servers.replicas("main")[0]
        ed.sleep()
        self.run.wslots.lend(ed.gpu_ids)
        return max(1, self.run.wslots.n)


class _ReplicaPhases:
    """inner_search phases with several main replicas (2026-11-20): every thread proposes at once (thread k on
    replica k mod n), then the candidates are built and evaluated one at a time, each scoring job split over all
    replicas (the editor workspace serialises its tool calls). n > 1 (search.parallel_builds, 2026-11-21): up to n
    candidates (at most the number of replicas) are built and gate-evaluated at once; tool calls stay serialised by
    the workspace lock except tensor_compute."""

    def __init__(self, n: int = 1):
        self.n = max(1, int(n))

    def editor_phase(self):
        pass

    def eval_phase(self) -> int:
        return self.n


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
        self._gate_lock = threading.Lock()       # gen_XX/gate.json while candidates are gated at once (2026-11-21)
        self.servers = Servers(rc, cfg, weights=method_mode(cfg) == "weights")
        self.wslots = WeightSlots(self) if method_mode(cfg) == "weights" else None
        self.registry = rc.run_dir / "mock_registry.json"
        self.target = rc.domain
        # datasets (rsicore.draws, 2026-11-22: fixed): tds = simple_<d>_pool, what the editor works from (train ids /
        # train_example / acts / train_loss, the raw LoRA's data, the screen / full items); gds = simple_<d>_eval,
        # what accepts a candidate (baseline gate scores, the paired gate vs the reference). Only the target domain
        # is evaluated.
        from rsicore import draws as _dr
        self.tds = _dr.train_dataset(rc)
        self.gds = _dr.gate_dataset(rc)
        self.mcfg = cfg.get("mock", {})
        self.mode = method_mode(cfg)
        self.merge_mode = "adapter" if self.mode == "edit" else merge_mode(cfg)
        self.budget_rank = adapter_rank_budget(cfg)
        self.gate_fallback = gate_fallback(cfg)
        self._stores = {}
        self._hf = {}

    # ------------------------------------------------------------------ models
    @property
    def chain(self) -> bool:
        """mode=edit: one LoRA, trained once (generation 0) and edited by the model in every generation.
        M_g = M_0 + L_g served as an adapter on the M_0 server (no merge); candidates replace L_g."""
        return self.mode == "edit"

    @property
    def adapter(self) -> bool:
        """scratch + merge.mode adapter (2026-11-22): M_g = M_0 + C_g, C_g the cumulative LoRA (every accepted delta
        rank-concatenated), served as an adapter on the M_0 server; never merged. Candidates are deltas on M_g
        (Evaluator.set_base composes them). Unlike chain, nothing is trained and candidates do not replace C_g."""
        return self.mode == "scratch" and self.merge_mode == "adapter"

    @property
    def on_m0(self) -> bool:
        """M_g is served as an adapter on the one M_0 server (mode=edit, or scratch + merge.mode adapter)."""
        return self.chain or self.adapter

    def served(self, g: int) -> str:
        return "m0" if self.on_m0 else f"m{g}"

    def current_lora(self, g: int):
        """(LoRA, name, file) of the adapter that makes M_g on M_0, from generation g-1's model.json: mode=edit L_g,
        merge.mode adapter the cumulative C_g. None if g == 0 or no LoRA has been accepted yet (M_g = M_0)."""
        if not self.on_m0 or g == 0:
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
        if g == 0 or self.on_m0:
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

    def store(self, path: Path, device: str | None = None):
        """Weight store of `path` on the train GPU, or (device, 2026-11-22: a main replica's GPU with
        gpu_layout.tools_on_replicas) on that GPU; all share the run's SVD cache."""
        if self.rc.mock:
            from lora_code.mock import MockStore
            return self._stores.setdefault("mock", MockStore())
        key = str(path) if device is None else f"{path}@{device}"
        if key not in self._stores:
            from lora_code.primitives.weights import WeightStore
            dev = device or (self.train_device() if self.cfg.get("train", {}).get("store_on_gpu", True) else "cpu")
            self._stores[key] = WeightStore(path, device=dev, cache_dir=self.rc.run_dir / "cache")
        return self._stores[key]

    def hf(self, path: Path, device: str | None = None):
        """HF model of `path` on the train GPU(s), or (device, 2026-11-22) a whole copy on that one GPU."""
        if self.rc.mock:
            return None
        key = str(path) if device is None else f"{path}@{device}"
        if key not in self._hf:
            from lora_code.primitives.train import HFModel
            tc = self.cfg.get("train", {})
            self._hf[key] = HFModel(path, self.store(path, device), device=device or self.train_device(),
                                    devices=([device] if device else
                                             [f"cuda:{g}" for g in role_gpus(self.rc, self.cfg, "train")]),
                                    micro_batch=tc.get("micro_batch"),
                                    dtype=tc.get("dtype", "bfloat16"), max_len=int(tc.get("max_len", 4096)),
                                    seed=self.rc.seed)
            self._hf[key].base_store = self.store(Path(self.rc.model_path))   # weight edits' toward_base ops
        return self._hf[key]

    def train_device(self) -> str:
        """First GPU of the train role: weight store, diagnostics, merge, HF model inputs."""
        return f"cuda:{role_gpus(self.rc, self.cfg, 'train')[0]}"

    def release_hf(self, devices_only: bool = False):
        """Free the HF models (devices_only, 2026-11-22: only the copies on the main replicas' GPUs, keys
        '<path>@<device>') and the replica GPUs' weight stores (their cached fp32 weights; a workspace still holding
        one reloads from disk on use)."""
        for k in [k for k in self._hf if not devices_only or "@" in k]:
            self._hf.pop(k).release()
        for k in [k for k in self._stores if "@" in k]:
            st = self._stores.pop(k)
            lock = getattr(st, "_lru_lock", None)
            if lock is not None:
                with lock:
                    st._lru.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def gdir(self, g: int) -> Path:
        from rsicore.records import gen_dir
        return gen_dir(self.rc, g)

    def load_lora(self, path: Path):
        """A saved LoRA, or in weights mode a saved weight edit (accepted / non-LLM candidates)."""
        if self.mode == "weights":
            if self.rc.mock:
                from lora_code.mock import MockWeightEdit
                return MockWeightEdit.from_json(read_json(path))
            from lora_code.primitives.wedit import WeightEdit
            return WeightEdit.load_raw(path)
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
        if self.chain or self.mode == "weights":   # edit: LoRAs replace each other; weights: dense edits, no adapters
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
        ds = self.tds
        ids = {"screen": {ds: limit_ids(draws.ids(self.rc, g, ds, "screen"), ec.get("screen_limit"))},
               "full": {ds: limit_ids(draws.ids(self.rc, g, ds, "signal"), ec.get("full_limit"))}}
        ids["gate"] = self.gate_level()             # the gate dataset differs from the train one
        return ids

    def gate_level(self) -> dict:
        from rsicore import draws
        lim = self.cfg.get("eval", {}).get("gate_limit")
        return {self.gds: limit_ids(draws.gate_ids(self.rc, self.gds), lim)}

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
            from rsicore.evaluate import score_split
            n = int(tc.get("samples", 4))
            eps = endpoint if isinstance(endpoint, (list, tuple)) else [endpoint]
            er = score_split(eps, served, dataset, pool="pool", ids=list(ids), out_dir=out_path.parent / "train_samples",
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
    def decodings(self) -> dict:
        """{level: rsicore decoding dict | None} of the scoring stages (paper config `decoding`, 2026-11-21; null
        seeds = the run's --seed); None = tasks.yaml greedy."""
        from rsicore.config import stage_decoding
        return {lv: stage_decoding(self.cfg, lv, self.rc.seed) for lv in ("screen", "full", "gate")}

    def evaluator(self, g, out_dir, srv, served, ids, store):
        from lora_code.evaluator import Evaluator
        return Evaluator(self.rc, self.cfg, out_dir=out_dir, endpoint=srv.endpoint, served=served, server=srv,
                         decoding=self.decodings(),
                         replicas=self.servers.replicas("main"),
                         ids=ids, target_dataset=self.tds, gate_dataset=self.gds, store=store,
                         mock=self.rc.mock,
                         registry=self.registry,
                         parent_quality=self.quality_value(self.editor_name(g) if self.adapter else served),
                         max_loaded=int(self.cfg.get("server", {}).get("max_loras", 8)) - 2,
                         weights_scorer=self.wslots.score if (self.wslots is not None and not self.rc.mock) else None)

    def workspace(self, out_dir, ev, store, hf, train, budget, prefix, stack=(), diag=None,
                  raw=None, raw_info=None, base_lora=None):
        from lora_code.primitives.train import TrainData
        from lora_code.tools import Workspace
        return Workspace(mock=self.rc.mock, store=store, train=TrainData(train), cfg=self.cfg, out_dir=out_dir,
                         evaluator=ev, hf=hf, stack=stack, budget=budget,
                         name_prefix=prefix, diagnostics=diag, seed=self.rc.seed, mode=self.mode, raw=raw,
                         raw_info=raw_info, base_store=self.store(Path(self.rc.model_path)), base_lora=base_lora)

    # ================================================================== baseline
    def baseline(self):
        from rsicore import draws
        bdir = self.rc.run_dir / "baseline"
        draws.ensure_gate(self.rc)
        if self.rc.mock and not (read_json(self.registry, {}) or {}).get(self.served(0)):
            self.set_quality(self.served(0), float(self.mcfg.get("base_quality", 0.6)))
        key = "baseline/gate"
        if not self.st.done(key):
            self.servers.get("main", self.model_path(0), self.served(0))
            tmp = self.servers.extra_main(self.model_path(0), self.served(0))   # weights mode: the slots too
            eps = [s.endpoint for s in self.servers.replicas("main") + tmp]
            from rsicore.evaluate import score_split
            gate = self.gate_level()
            res = {}
            for ds, ids in gate.items():
                # gate score: the gate decoding (greedy unless decoding.gate is set), like every candidate's gate
                er = score_split(eps, self.served(0), ds, pool="pool", ids=ids, out_dir=bdir / "gate" / ds,
                                 mock=self.rc.mock, decoding=self.decodings()["gate"])
                recs = [{"id": x.get("id"), "score": x.get("score")} for x in er.records]
                sc = [x["score"] for x in recs if x["score"] is not None]
                m = mean_or_none(sc)
                se = (sum((x - m) ** 2 for x in sc) / max(len(sc) - 1, 1) / max(len(sc), 1)) ** 0.5 if sc else None
                # M_0's gate score and its standard error (reported; there is no noise band, 2026-11-23)
                res[ds] = {"score": m, "n": len(recs), "n_scored": len(sc), "se": se, "records": recs}
            write_json(bdir / "gate_scores.json", res)
            for s in tmp:
                s.stop()
            self.st.mark(key, {ds: {"score": v["score"]} for ds, v in res.items()})
        if self.cfg.get("heldout", {}).get("enabled", True) and not self.st.done("baseline/heldout"):
            from rsicore.heldout import heldout_domains, run_heldout
            self.servers.get("main", self.model_path(0), self.served(0))
            tmp = self.servers.extra_main(self.model_path(0), self.served(0))   # weights mode: the slots too
            try:
                h = run_heldout([s.endpoint for s in self.servers.replicas("main") + tmp], self.served(0), bdir / "heldout",
                                domains=heldout_domains(self.cfg, self.rc.domain), mock=self.rc.mock,
                                limit=self.cfg.get("heldout", {}).get("limit"),
                                decoding=self.cfg.get("heldout", {}).get("decoding"))
            finally:
                for s in tmp:
                    s.stop()
            write_json(bdir / "heldout.json", h)
            self.st.mark("baseline/heldout", {"tasks": len(h)})

    # ================================================================== one generation
    def generation(self, g: int):
        from rsicore import draws

        from lora_code.tools import search_settings
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
        cur = self.current_lora(g)                        # on M_0, g >= 1: (L_g / C_g, name, file) or None
        edit_later = self.chain and g > 0                 # mode=edit after generation 0: no training at all
        # diagnostics show mode=edit's current LoRA; scratch (merged or adapter) the per-generation accepted deltas
        diag_stack = [(g - 1, cur[0])] if (cur and self.chain) else stack
        base = cur[0] if (cur and self.adapter) else None   # merge.mode adapter: C_g (current model = M_0 + C_g)
        if base is not None and not rc.mock:              # W_g = W_0 + B_C A_C for the diagnostics (fp32 overlay)
            from lora_code.primitives.weights import OverlayStore
            store = OverlayStore(store, base)

        def main_srv():
            return self.servers.get("main", parent, served)

        # ---- train data (sampled answers only when the raw LoRA is trained, see train_data_mode)
        tmode = self.train_data_mode(g)
        tdata = None
        if not st.done(f"{P}/train_data"):
            srv = main_srv() if tmode != "gold" else None
            ex = self.build_train_data(self.tds, train_ids, gd / "train_data.json", mode=tmode,
                                       endpoint=[s.endpoint for s in self.servers.replicas("main")] if srv else None,
                                       served=served)
            st.mark(f"{P}/train_data", {"n": len(ex), "with_target": sum(1 for e in ex.values() if e["target"]),
                                        "self": sum(1 for e in ex.values() if e["source"] == "self")})
            tdata = {k: ex[k] for k in sorted(ex)}         # = the file's (sorted) order, not read back (~72k items)
        if tdata is None:
            tdata = read_json(gd / "train_data.json")

        # ---- diagnostics
        from lora_code import diagnostics as D
        if self.rc.mock:
            from lora_code.mock import mock_diagnostics
            diag = D.load_or_compute(gd / "diagnostics.json", lambda: mock_diagnostics(store, diag_stack, rc.seed))
        else:
            def real_diag():
                from contextlib import nullcontext
                hf = self.hf(parent)
                tt = ["\n\n".join(m["content"] for m in e["messages"]) for e in list(tdata.values())[:64] if e["messages"]]
                with (hf.applied(base) if base is not None else nullcontext()):   # adapter mode: HF model = M_g
                    return D.compute(store, diag_stack, base_store=self.store(Path(rc.model_path)), hf=hf,
                                     target_texts=tt,
                                     svd_k=int(cfg.get("diagnostics", {}).get("svd_k", 8)), device=self.train_device())
            diag = D.load_or_compute(gd / "diagnostics.json", real_diag)
        st.mark(f"{P}/diagnostics", {"rows": diag["n_modules"]}) if not st.done(f"{P}/diagnostics") else None

        # the programs' budget: GPU-seconds of acts / train_loss (there is no evaluate budget, 2026-11-23)
        budget = {"gpu_seconds": float(cfg.get("budget", {}).get("gpu_seconds", 7200))}
        state = {}

        def get_ev():
            """Evaluator on the main server; knows the raw LoRA once it exists (also after a resume)."""
            if "ev" not in state:
                srv = main_srv()
                ev = self.evaluator(g, gd, srv, served, ids, store)
                if cur:   # parent / editor = M_0 + L_g (mode=edit) or M_0 + C_g (merge.mode adapter)
                    ev.set_parent(cur[0], cur[1])
                if self.adapter:   # candidates are deltas on M_g: served as C_g (+) delta
                    ev.set_base(base, self.budget_rank)
                if self.wslots is not None:   # weights mode: the slots export edits of M_g (also on a resume that
                    self.wslots.set_parent(parent, self.store(parent))   # skips the search, e.g. into nonllm)
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
            ws = self.workspace(gd, ev, self.store(parent), self.hf(parent), tdata, budget, f"g{g:02d}", stack=stack,
                                diag=diag, raw=raw, raw_info=raw_info, base_lora=base)
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
                bws = self.workspace(gd, ev, store, self.hf(parent), tdata, {"gpu_seconds": rgb},
                                     f"g{g:02d}", stack=stack)
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
            from rsicore.search import load_candidates, ranked as rank_fn
            if ranked is None:
                ranked = rank_fn(load_candidates(rc, g).values())
            ev, ws = ctx_objects()
            learned = (st.get("learned") or {}).get("scores", {})
            ref = self.gate_reference(gd, ev, g)
            fallback = "keep" if edit_later else self.gate_fallback
            # every candidate was gate-evaluated during the search (the only flow): the largest passing paired gain
            acc = self.accept_best_gate(gd, ranked)
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
                if rec["gate_scores"].get(self.gds) is not None:
                    sc[self.gds] = rec["gate_scores"][self.gds]
                st.mark("learned", {"scores": sc})
            write_json(gd / "accepted.json", rec)
            st.mark(f"{P}/gate", {"source": rec["source"], "cid": rec.get("cid")})
        acc = read_json(gd / "accepted.json")

        # (2026-11-22) the search is over: free the HF copies on the main replicas' GPUs (tools_on_replicas) before
        # the merge and the held-out evaluation restart the replicas with M_{g+1}
        self.release_hf(devices_only=True)

        # ---- merge -> M_{g+1}
        if self.adapter and not st.done(f"{P}/merge"):
            # merge.mode adapter: nothing is merged. C_{g+1} = C_g (+) accepted delta, rebuilt by the same deterministic
            # composition the evaluator served it with (Evaluator.compose: same fp32 inputs, same budget), i.e. the
            # adapter the gate scored, kept under the accepted candidate's adapter name; rejected -> C_g stays.
            lf = self.lora_file(gd / "accepted_lora")
            if lf is not None:
                from lora_code.primitives.lora import compose_cumulative
                C = compose_cumulative(base, self.load_lora(lf), self.budget_rank)
                self.save_lora(C, gd / "cumulative_lora")
                info = {"path": str(parent), "merge_mode": "adapter", "lora": acc["name"],
                        "lora_file": str(self.lora_file(gd / "cumulative_lora")), "source": "adapter (C_g + accepted)",
                        "components": [k for k, _ in stack] + [g], "max_rank": int(C.max_rank()),
                        "modules": len(C.modules()), "recompressed": C.meta.get("recompressed")}
            elif cur:
                info = {"path": str(parent), "merge_mode": "adapter", "lora": cur[1], "lora_file": str(cur[2]),
                        "source": "keep (current adapter)", "components": [k for k, _ in stack]}
            else:
                info = {"path": str(parent), "merge_mode": "adapter", "lora": None, "lora_file": None,
                        "source": "keep (original model)", "components": []}
            if rc.mock:
                info["quality"] = self.quality_value(info["lora"] or served)
            write_json(gd / "model.json", info)
            st.mark(f"{P}/merge", {"merged": False, "merge_mode": "adapter", "lora": info["lora"],
                                   "source": info["source"]})
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
                lo = self.load_lora(lf)
                dev = self.train_device() if cfg.get("merge", {}).get("gpu", True) else "cpu"
                if getattr(lo, "kind", "lora") == "weights":   # weights mode: write the edited weights
                    from lora_code.primitives.wedit import edit_into_model
                    rep = edit_into_model(lo, parent, gd / "model", store=store,
                                          base_store=self.store(Path(rc.model_path)), device=dev)
                    rels = [v.get("rel_change", 0.0) for v in (rep.get("modules") or {}).values()]
                    stat = {"max_rel_change": max(rels) if rels else 0.0}
                else:
                    from lora_code.primitives.lora import merge_into_model
                    rep = merge_into_model(lo, parent, gd / "model", store=store, device=dev)
                    stat = {"max_realised_rel_err": rep["max_realised_rel_err"]}
                write_json(gd / "model.json", {"path": str(gd / "model"), "source": "merged", "lora": acc["name"], **stat})
                info = {"merged": True, **stat}
            st.mark(f"{P}/merge", info)

        # ---- non-LLM branch (random-construction control). nonllm.enabled false = never run it;
        # nonllm.skip_if_rejected (default true, 2026-11-21) = not run in a generation whose gate kept the current
        # model (accepted.json source 'keep'): there is no accepted LoRA to compare it with. Absent keys = enabled
        # true / skip_if_rejected true; base.yaml only documents them, so run configs locked earlier still resume.
        nc = cfg.get("nonllm") or {}
        skip = ("nonllm.enabled false" if not nc.get("enabled", True) else
                "no candidate passed the gate (nonllm.skip_if_rejected)"
                if nc.get("skip_if_rejected", True) and acc.get("source") == "keep" else None)
        if skip and not st.done(f"{P}/nonllm"):
            write_json(gd / "baselines" / "nonllm.json", {"kind": None, "name": None, "skipped": skip})
            st.mark(f"{P}/nonllm", {"name": None, "skipped": skip})
            print(f"[lora_code] gen {g}: non-LLM branch skipped: {skip}", flush=True)
        if not st.done(f"{P}/nonllm"):
            ev, ws = ctx_objects()
            bws = self.workspace(gd, ev, self.store(parent), self.hf(parent), tdata,
                                 {"gpu_seconds": budget["gpu_seconds"]},
                                 f"g{g:02d}", stack=stack, base_lora=base)
            out = {}
            n_ev = int(nc.get("evaluate", search_settings(cfg)["max_candidates"]))   # absent: the editor's count
            if self.mode == "edit":
                rlo = self.load_lora(self.raw_file(gd, raw))
                best, log = BL.random_edit_search(bws, rlo, evals=n_ev, seed=rc.seed,
                                                  scales=nc.get("search_scales", [0.5, 0.75, 1.0, 1.25]))
            elif self.mode == "weights":
                best, log = BL.random_weight_search(bws, evals=n_ev, seed=rc.seed)
            else:
                best, log = BL.random_construct_search(bws, evals=n_ev, seed=rc.seed)
            out["kind"] = {"edit": "random_edit_search", "scratch": "random_construct_search",
                           "weights": "random_weight_search"}[self.mode]
            out["random_search"] = log
            name = None
            if best is not None and best.modules():
                self.save_lora(best, gd / "baselines" / "nonllm_lora")
                name = bws.lora_name(best)
            out["name"] = name
            write_json(gd / "baselines" / "nonllm.json", out)
            st.mark(f"{P}/nonllm", {"name": name, "best": (log.get("best") or {}).get("score")})

        # ---- held-out evaluation of M_{g+1}
        # (2026-11-21) gate kept the current model -> M_{g+1} = M_g, already evaluated (previous generation, or the
        # baseline for generation 0): reuse that result instead of scoring the same model again.
        prev_h = (self.gdir(g - 1) / "heldout.json") if g > 0 else (rc.run_dir / "baseline" / "heldout.json")
        if (cfg.get("heldout", {}).get("enabled", True) and not st.done(f"{P}/heldout")
                and acc.get("source") == "keep" and prev_h.exists()):
            write_json(gd / "heldout.json", read_json(prev_h))
            st.mark(f"{P}/heldout", {"tasks": len(read_json(prev_h) or {}), "reused_from": str(prev_h.relative_to(rc.run_dir))})
            print(f"[lora_code] gen {g}: gate kept the current model; held-out reused from {prev_h.relative_to(rc.run_dir)}",
                  flush=True)
        if cfg.get("heldout", {}).get("enabled", True) and not st.done(f"{P}/heldout"):
            from rsicore.heldout import heldout_domains, run_heldout
            if self.on_m0:   # M_{g+1} = M_0 + L_{g+1} / C_{g+1}: the same M_0 server, the adapter as the model
                ev = get_ev()
                nxt = self.current_lora(g + 1)
                if nxt:
                    if not ev.known(nxt[1]):
                        ev.register(nxt[0], nxt[1], full=True)
                    ev.ensure_loaded(nxt[1])
                h = run_heldout(ev.endpoints, served, gd / "heldout", lora=nxt[1] if nxt else None,
                                domains=heldout_domains(cfg, rc.domain), mock=rc.mock,
                                limit=cfg.get("heldout", {}).get("limit"), decoding=cfg.get("heldout", {}).get("decoding"))
            else:
                nxt = self.model_path(g + 1)
                if self.wslots is not None:          # weights mode: take the editor's group back first
                    self.wslots.reclaim()
                self.servers.get("main", nxt, self.served(g + 1))
                state.clear()
                tmp = self.servers.extra_main(nxt, self.served(g + 1))   # weights mode: the slots too
                try:
                    h = run_heldout([s.endpoint for s in self.servers.replicas("main") + tmp], self.served(g + 1),
                                    gd / "heldout", domains=heldout_domains(cfg, rc.domain), mock=rc.mock,
                                    limit=cfg.get("heldout", {}).get("limit"), decoding=cfg.get("heldout", {}).get("decoding"))
                finally:
                    for s in tmp:
                        s.stop()
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
            if shown:                               # a record of the items shown in feedback (never re-drawn:
                write_json(gd / "shown.json", shown)   # every screen comes from the clean pool)
            info = self.summarize(g, gd)
            from rsicore.records import update_summary
            update_summary(rc, g, info)
            st.mark(f"{P}/summary", {"accepted": info.get("accepted", {}).get("source")})
        # (2026-11-22) the next generation works on M_{g+1}: free this generation's HF models (else each generation
        # left one 52 GB model on the train GPU)
        self.release_hf()

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
        """Items whose content appeared in feedback files (target errors), recomputed from disk so a resumed run
        still records them."""
        shown = {}
        for p in sorted((gd / "feedback").glob("*.json")):
            fb = (read_json(p, {}) or {}).get("screen") or {}
            fb = fb.get("feedback") or {}
            for e in (fb.get("target") or {}).get("errors") or []:
                if e.get("dataset") and e.get("id"):
                    shown.setdefault(e["dataset"], set()).add(e["id"])
        return {k: sorted(v) for k, v in shown.items()}

    # ------------------------------------------------------------------ search
    def eval_parallelism(self, ev) -> int:
        """Candidates built / gate-evaluated at once outside mode=weights (2026-11-21): the number of main replicas when
        search.parallel_builds is on and there are several, else 1 (one after another, as before)."""
        from lora_code.tools import search_settings
        n_rep = len(ev.endpoints)
        return n_rep if (self.mode != "weights" and n_rep > 1 and search_settings(self.cfg)["parallel_builds"]) else 1

    def search(self, g, gd, ctx_objects, diag, budget, raw=None):
        from rsicore.editor import Editor, editor_kwargs, editor_settings
        from rsicore.search import inner_search

        from lora_code import diagnostics as D
        from lora_code import mock as M
        from lora_code.prompts import ContextRenderer, system_prompt, task_text
        from lora_code.prompts import screen_text
        from lora_code.tools import (editor_chat_tools, editor_mock_policy, paper_tools_enabled, program_build,
                                     screen_tools_enabled, search_settings, tensor_compute_enabled)
        rc, cfg = self.rc, self.cfg
        ev, ws = ctx_objects()
        if tools_on_replicas(cfg) and not ws.devs:
            # (2026-11-22) only for the search: editor session k runs its tools on replica k mod n's GPU (an HF copy
            # of the current model and a weight store there), in the order of the editor endpoints; the copies load
            # now, one after another, outside any budget (gate / nonllm / a resumed generation never load them)
            parent = self.model_path(g)
            ws.set_devices([{"server": srv,
                             "hf": None if rc.mock else self.hf(parent, device=f"cuda:{srv.gpu_ids[0]}"),
                             "store": None if rc.mock else self.store(parent, device=f"cuda:{srv.gpu_ids[0]}")}
                            for srv in ev.replicas])
            ws.warm_devices()
        sc = cfg.get("search", {})
        ss = search_settings(cfg)
        # one step (8 sessions x 1 step): no session sees this generation's other candidates, so
        # each is told its index and to use search_history / the archive to differ from earlier generations
        hist_tool = int(editor_settings(cfg).get("tool_rounds") or 0) > 0 and editor_settings(cfg).get("history_tool", True)
        rawinfo = None
        if raw:
            rp = (raw.get("points") or {}).get("screen") or {}
            tr = raw.get("train") or {}
            rawinfo = {"n_train": tr.get("n_train"), "rank": tr.get("rank"), "steps": tr.get("steps") or tr.get("steps_done"),
                       "target_gain": rp.get("target_gain"), "source": raw.get("source"), "gen": g}
        ctx = ContextRenderer(task_text=task_text(gen=g, target=self.target, dataset=self.tds,
                                                  n_train=len(ws.train.ids()), raw_info=rawinfo,
                                                  gate_dataset=self.gds),
                              diag_text=D.render(diag, adapter=self.adapter),
                              archive_text=(self.archive(g, read_tools=bool(hist_tool),
                                                         max_chars=int(sc.get("archive_chars", 30000)))
                                            if sc.get("archive", True) else ""),
                              budget_fn=lambda: ws.budget_status(), timeout_s=int(cfg.get("sandbox", {}).get("timeout_s", 1800)),
                              sessions=ss["threads"] if ss["steps"] == 1 and ss["threads"] > 1 else None,
                              history_tool=bool(hist_tool))
        # the gate rule as configured (2026-11-21): its item count, and whether the raw LoRA is the fallback (mode=edit,
        # generation 0)
        gate_n = len(((getattr(ev, "ids", None) or {}).get("gate") or {}).get(self.gds) or []) or None
        raw_fb = not (self.chain and g > 0) and self.gate_fallback == "raw"
        sysmsg = system_prompt(cfg, budget=budget, mode=self.mode, gate_n=gate_n, raw_fallback=raw_fb)
        editor = self.editor_name(g)   # M_g: in mode=edit the current LoRA's adapter on the M_0 server (pinned)
        # mock: with the tensor tool on, thread 1 of a scratch search uses api.tensor_compute (a T-handle vector into
        # rank1 + LoRA factors), so a mock run exercises it end to end (paper + screen tools on: thread 1 uses the
        # paper-style recipe instead)
        policy = (M.make_mock_policy(rc.seed, skill=self.quality(editor) or 0.7, tag=f"g{g}", mode=self.mode,
                                     tensor_compute=self.mode == "scratch" and tensor_compute_enabled(cfg),
                                     paper=self.mode in ("scratch", "weights") and paper_tools_enabled(cfg)
                                     and screen_tools_enabled(cfg))
                  if rc.mock else None)
        # 2026-11-21: Qwen3 thinking + the tool loop (editor.tool_rounds; 0 = one call per candidate) with run_python
        # on the workspace's exploration api (no raw_lora / GPU tools, no budget spent) and read_record
        # search_history (2026-11-21): every candidate of the earlier generations' searches, besides this step's
        past = {k: self.gdir(k) for k in range(g) if (self.gdir(k) / "candidates.jsonl").exists()}
        policy, tools = editor_mock_policy(policy, cfg), editor_chat_tools(ws, cfg, past=past)
        ekw = editor_kwargs(cfg, server_args=main_server_args(cfg))

        def editor_factory(th):
            return Editor(ev.endpoints[th % len(ev.endpoints)], editor, system_prompt=sysmsg, seed=rc.seed + 1000 * g + th,
                          transcript=gd / "editor" / f"thread_{th}.jsonl", mock_policy=policy, tools=tools, **ekw)

        sb = cfg.get("sandbox", {})

        def build(program, cid):
            return program_build(ws, program, cid, timeout_s=int(sb.get("timeout_s", 1800)),
                                 mem_gb=int(sb.get("mem_gb", 16)), workdir=gd / "work" / cid)

        # gate per candidate (2026-11-21; the only flow since 2026-11-23): every built candidate gets ONE gate
        # evaluation (decoding.gate) paired vs the current model, whose gate records are computed once per generation
        # (gate_reference); the result is recorded in gen_XX/gate.json at once, so later steps / generations see it.
        # It is inner_search's screen_fn: score = the paired gain.
        gref = self.gate_reference(gd, ev, g)
        learned = (self.st.get("learned") or {}).get("scores", {})

        def screen(art, cid):
            ok, info = self.gate_check(ev, ws, ws.load_candidate(art["cid"]), gref, learned)
            fb = {"gate": self.gate_feedback(ok, info, self.gds)}
            self.record_gate(gd, cid, fb["gate"])
            return fb["gate"].get("gain"), fb

        n_rep = len(ev.endpoints)
        if self.mode == "weights":
            # candidates are scored on the slots while the editor sleeps: the current model's references first (full:
            # the non-LLM branch's pick)
            parent = self.model_path(g)
            self.wslots.set_parent(parent, self.store(parent))
            for lv in ("screen", "full", "gate"):
                ev.run(None, lv)
            phases, parallel = _WeightPhases(self), 2
        else:
            # several replicas: all threads propose at once, then the builds + gate evaluations run one at a time, or
            # (search.parallel_builds, 2026-11-21) up to one per replica at once
            phases, parallel = (_ReplicaPhases(self.eval_parallelism(ev)), 2) if n_rep > 1 else (None, 1)
        if screen_tools_enabled(cfg):
            # the current model's own answers on the screen items (once per generation): a summary in every step's
            # context, the details through api.screen_results / screen_output / acts(source='own')
            sd = self.reference_screen(gd, ev)
            ws.screen = sd
            ctx.screen_text = screen_text(sd["summary"])
        return inner_search(rc, g, editor_factory=editor_factory, render_context=ctx, build_fn=build, screen_fn=screen,
                            threads=ss["threads"], steps=ss["steps"], max_candidates=ss["max_candidates"],
                            parallel=parallel, phases=phases)

    def archive(self, g: int, *, read_tools: bool = True, max_chars: int = 30000) -> str:
        """Experience channel (RSI_LOOP_DESIGN §1.1; structured 2026-11-22): per earlier generation a header (accepted,
        best gains, the then-current model's own screen answers) and one row per candidate, failures included
        (prompts.archive_row: what it is, gate result, error, note). read_tools (read_record / search_history reach
        earlier generations): full programs are read on demand; else the best program of each generation is quoted
        (3000 chars) as before. max_chars: the rows of the oldest generations are dropped first to fit."""
        if g == 0:
            return ""
        from rsicore.chat_tools import _gate_outcomes
        from rsicore.search import load_candidates, ranked

        from lora_code.prompts import ARCHIVE_HEAD, ARCHIVE_READ, archive_row
        blocks = []
        for k in range(g):
            cs = ranked(load_candidates(self.rc, k).values())
            acc = read_json(self.gdir(k) / "accepted.json", {}) or {}
            gate_every = (read_json(self.gdir(k) / "gate.json", {}) or {}).get("rule", "").startswith("gate_every")
            gates = _gate_outcomes(self.gdir(k))
            head = [f"- generation {k}: accepted={acc.get('source')} {acc.get('cid') or ''}; "
                    f"{len(cs)} candidates; best " + ("paired gate gains " if gate_every else "scores ") +
                    ", ".join(f"{c.cid}={c.score:.4f}" for c in cs[:3] if c.score is not None)]
            rs = (read_json(self.gdir(k) / "reference_screen.json", {}) or {}).get("summary") or {}
            if rs.get("n_items"):                   # the then-current model's own screen answers (2026-11-21)
                head.append(f"  current model on that generation's screen: accuracy {rs.get('accuracy')} over "
                            f"{rs['n_items']} items x {rs.get('samples_per_item')} samples, correct-count "
                            f"distribution {rs.get('correct_counts')}, truncation {rs.get('truncation_rate')}")
            rows = ["  " + json.dumps(archive_row(c, gates.get(c.cid)), ensure_ascii=False, default=str)
                    for c in sorted(cs, key=lambda c: c.cid)]
            if not read_tools and cs and cs[0].program:
                rows.append(f"  best program of generation {k}:\n```python\n{cs[0].program[:3000]}```")
            blocks.append((k, head, rows))
        tail = [ARCHIVE_READ] if read_tools else []

        def text(drop: int) -> str:
            out = [ARCHIVE_HEAD]
            for i, (k, head, rows) in enumerate(blocks):
                out += head + (rows if i >= drop else
                               [f"  ({len(rows)} rows omitted for length" +
                                (f"; search_history(gen={k}) lists them)" if read_tools else ")")])
            return "\n".join(out + tail)
        drop = 0
        while drop < len(blocks) - 1 and len(text(drop)) > max_chars:
            drop += 1
        return text(drop)

    # ------------------------------------------------------------------ reference screen (2026-11-21)
    def reference_screen(self, gd, ev) -> dict:
        """The current model's own answers on this generation's screen items (decoding.screen: thinking, n 4), scored ONCE per generation (the evaluator's parent cache, keyed by the decoding) and shared by
        every session: {"records": {ds: [run_eval records with every sample's text and thinking]}, "items": {ds: {id:
        {prompt, reference, messages}}}, "summary": screen_view.summary}. gen_XX/reference_screen.json keeps the
        summary for later generations (search_history, archive)."""
        from rsicore import screen_view as SV
        ev.run(None, "screen")
        recs = ev.full_records(None, "screen")
        items = {}
        for ds, rs in recs.items():
            items[ds] = {}
            for i, it in load_records(ds, [r["id"] for r in rs]).items():
                try:
                    msgs = render_messages(ds, it)
                except Exception:  # noqa: BLE001
                    msgs = []
                items[ds][i] = {"prompt": prompt_text(ds, it), "messages": msgs,
                                "reference": it.get("reference") if it.get("reference") is not None else gold_target(ds, it)}
        summ = SV.summary(recs, items, examples=3)
        if read_json(gd / "reference_screen.json") is None:
            write_json(gd / "reference_screen.json", {"summary": summ, "decoding": ev.level_decoding("screen")})
        return {"records": recs, "items": items, "summary": summ}

    @staticmethod
    def gate_feedback(ok: bool, info: dict, gds: str) -> dict:
        """The editor-visible gate result of one candidate (gate per candidate, 2026-11-21): statistics and item ids
        only (keys as rsicore.chat_tools.GATE_STATS reads them from gen_XX/gate.json)."""
        sc = info.get("scores") or {}
        out = {"passed": bool(ok), "accuracy": sc.get(gds), "reference_accuracy": info.get("reference_score"),
               "gain": info.get("paired_gain"), "ci": info.get("paired_ci"), "n": info.get("gate_n"),
               "broken": info.get("broken"), "fixed": info.get("fixed"), "broken_ids": info.get("broken_ids"),
               "fixed_ids": info.get("fixed_ids"), "reasons": list(info.get("reasons") or [])}
        return {k: v for k, v in out.items() if v is not None}

    def record_gate(self, gd, cid: str, fb: dict) -> None:
        """Append / replace one candidate's gate result in gen_XX/gate.json during the search (submissions shape)."""
        with self._gate_lock:
            p = gd / "gate.json"
            log = read_json(p) or {"submissions": []}
            log["submissions"] = [x for x in log["submissions"] if x.get("cid") != cid]
            log["submissions"].append({"cid": cid, "passed": bool(fb.get("passed")), "info": fb,
                                       "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
            log["submissions"].sort(key=lambda x: str(x.get("cid")))
            log["rule"] = "gate_every_candidate: the passing candidate with the largest paired gain is accepted"
            write_json(p, log)

    def accept_best_gate(self, gd, cands):
        """End of a generation: among the candidates whose gate result passed (min_gain, CI with require_significant,
        learned rule), the largest paired gain wins (ties: the earlier cid); None when none passed. Marks it in
        gen_XX/gate.json ("accepted")."""
        by = {c.cid: c for c in cands}
        log = read_json(gd / "gate.json") or {"submissions": []}
        ok = [x for x in log["submissions"] if x.get("passed") and x["cid"] in by
              and (x.get("info") or {}).get("gain") is not None
              and float(x["info"]["gain"]) > 0]       # a candidate that changes nothing is never accepted (min_gain 0)

        best = sorted(ok, key=lambda x: (-float(x["info"]["gain"]), x["cid"]))[0] if ok else None
        log["accepted"] = best["cid"] if best else None
        for x in log["submissions"]:
            x["accepted"] = best is not None and x["cid"] == best["cid"]
        write_json(gd / "gate.json", log)
        return by[best["cid"]] if best else None

    # ------------------------------------------------------------------ gate
    def gate_reference(self, gd, ev, g: int) -> dict:
        """Reference of the gate: always the current model (2026-11-23; mode=edit after generation 0: M_0 + the
        current LoRA): {"kind": "parent", "scores": {dataset: gate score}}. The parent's gate scores are computed once
        per generation, or (2026-11-24) taken from an earlier scoring of the same model (parent_gate_source), so the
        gate set is not sampled again for a model that was already scored on it."""
        pg = read_json(gd / "parent_gate.json")
        if pg is None:
            src = self.parent_gate_source(g)
            if src is not None and ev.adopt_parent("gate", src[1], layout=src[0]):
                print(f"[lora_code] gen {g}: current model's gate records reused from "
                      f"{src[1].relative_to(self.rc.run_dir)}", flush=True)
            pg = {ds: r["score"] for ds, r in ev.run(None, "gate").items()}
            write_json(gd / "parent_gate.json", pg)
        return {"kind": "parent", "scores": pg}

    def parent_gate_source(self, g: int) -> tuple[str, Path] | None:
        """(layout, dir) of an earlier gate scoring of M_g, the same model on the same gate items with the same
        decoding (2026-11-24), for Evaluator.adopt_parent; None = score it:
        - g = 0: M_g = M_0, scored by the baseline (baseline/gate);
        - generation g-1 kept its model (no LoRA accepted): M_g = M_{g-1}, its parent's records;
        - generation g-1 accepted a LoRA and M_g is served exactly as the gate scored it: mode=edit (M_0 + the
          accepted LoRA) and merge.mode adapter (C_g (+) delta, rebuilt by the same composition): that candidate's
          records. A merged model (merge.mode merge, weights mode) is new weights files whose numerics differ from
          the adapter the gate scored, so it is scored again."""
        if g == 0:
            return "baseline", self.rc.run_dir / "baseline" / "gate"
        prev = read_json(self.gdir(g - 1) / "accepted.json")
        if not prev:
            return None
        if prev.get("source") == "keep" or not prev.get("name"):
            return "cache", self.gdir(g - 1) / "evals" / "__parent__"
        if self.on_m0:
            return "cache", self.gdir(g - 1) / "evals" / str(prev["name"])
        return None

    def gate_check(self, ev, ws, lo, ref, learned) -> tuple[bool, dict]:
        """ref = gate_reference(...): paired per item on the gate set vs the current model (2026-11-20, like Paper A):
        mean paired gain >= min_gain and, with gate.require_significant, the 95% CI lower bound of the paired gain > 0
        (bootstrap over items, rsicore.evaluate.paired_delta)."""
        gcfg = self.cfg.get("gate", {})
        delta = float(gcfg.get("delta", 0.02))
        min_gain = float(gcfg.get("min_gain", 0.01))
        with ws._lock:                               # gate submissions run at once; the checks read the weight store
            ok, rep = ws.check(lo)
        info = {"check_ok": ok, "reasons": []}
        if not ok:
            info["reasons"].append("sandbox checks failed")
            return False, info
        name = ws.lora_name(lo) if lo.modules() else None
        if name is not None and not ev.known(name):
            ev.register(lo, name)
        res = ev.run(name, "gate")
        scores = {ds: r["score"] for ds, r in res.items()}
        info["scores"] = scores
        info["reference"] = ref["kind"]
        info["reference_score"] = (ref.get("scores") or {}).get(self.gds)
        gds = self.gds                               # the gate dataset (simple_<d>_eval)
        t, s = scores.get(gds), (ref.get("scores") or {}).get(gds)
        if t is None:
            info["reasons"].append("target gate score unavailable")
        else:
            from rsicore.evaluate import paired_delta
            par = (ev.run(None, "gate").get(gds) or {}).get("records") or []
            cand = (res.get(gds) or {}).get("records") or []
            d = paired_delta([r for r in par if r.get("score") is not None],
                             [r for r in cand if r.get("score") is not None], seed=self.rc.seed)
            info["paired_gain"], info["paired_ci"] = d["delta"], d["ci"]
            # item flips vs the parent (ids and counts only; a record is right when its score > 0.5, 2026-11-21)
            pm = {r["id"]: r["score"] for r in par if r.get("score") is not None}
            cm = {r["id"]: r["score"] for r in cand if r.get("score") is not None}
            both = sorted(set(pm) & set(cm))
            broken = [i for i in both if pm[i] > 0.5 >= cm[i]]
            fixed = [i for i in both if cm[i] > 0.5 >= pm[i]]
            info.update(gate_n=len(both), broken=len(broken), fixed=len(fixed), broken_ids=broken[:50],
                        fixed_ids=fixed[:50])
            if s is None or d["delta"] is None:
                info["reasons"].append("parent target gate score unavailable")
            elif d["delta"] < min_gain or (min_gain <= 0 and d["delta"] <= 0):   # threshold 0: strictly positive
                info["reasons"].append(f"paired gain {d['delta']:+.4f} vs parent ({s:.4f}) < min_gain {min_gain}")
            elif bool(gcfg.get("require_significant", True)) and not (d["ci"][0] is not None and d["ci"][0] > 0):
                info["reasons"].append(f"paired gain {d['delta']:+.4f} not significant "
                                       f"(95% CI [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}])")
        for ds, v in (learned or {}).items():
            c = scores.get(ds)
            if c is not None and v is not None and c < v - delta:
                info["reasons"].append(f"learned {ds} {c:.4f} < post-learning {v:.4f} - delta {delta}")
        return not info["reasons"], info

    # ------------------------------------------------------------------ summary
    def summarize(self, g: int, gd: Path) -> dict:
        acc = read_json(gd / "accepted.json", {}) or {}
        gate = read_json(gd / "gate.json", {}) or {}
        non = read_json(gd / "baselines" / "nonllm.json", {}) or {}
        raw = read_json(gd / "baselines" / "raw.json", {}) or {}
        info = {"gen": g, "editor": self.editor_name(g), "successor": (read_json(gd / "model.json") or {}).get("lora")
                if self.on_m0 else self.served(g + 1), "method": self.mode, "merge_mode": self.merge_mode,
                "data_roles": {"train": self.tds, "gate": self.gds},
                "gate_reference": acc.get("gate_reference") or "parent",
                "gate_fallback": "keep" if (self.chain and g > 0) else self.gate_fallback,
                "accepted": {k: acc.get(k) for k in ("source", "cid", "name", "gate_scores")},
                "gate_submissions": [{"cid": s["cid"], "passed": s["passed"], "reasons": (s.get("info") or {}).get("reasons")}
                                     for s in gate.get("submissions", [])],
                "nonllm": {"name": non.get("name"), "kind": non.get("kind"),
                           "best": (non.get("random_search") or {}).get("best"), "skipped": non.get("skipped")},
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
                          "best_screen": max((s for s in ((c.screen or {}).get("score") for c in cands) if s is not None),
                                             default=None)}
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


def context_shortfall(cfg: dict, prompt_tokens: int = 4096) -> str | None:
    """A configured scoring max_tokens (decoding.screen / full / gate, heldout.decoding) + prompt_tokens that does not
    fit the --max-model-len of the servers that score it (2026-11-21): the main replicas / weights slots
    (main_server_args). None when everything fits / nothing is set."""
    from rsicore.config import stage_decoding
    from rsicore.servers import max_model_len
    need = [(lv, (stage_decoding(cfg, lv) or {}).get("max_tokens")) for lv in ("screen", "full", "gate")]
    hd = (cfg.get("heldout") or {}).get("decoding") or {}
    need.append(("heldout", hd.get("max_tokens") if isinstance(hd, dict) else None))
    n = max_model_len(main_server_args(cfg))
    for stage, mt in need:
        if mt and n and int(mt) + prompt_tokens > n:
            return (f"{stage} decoding max_tokens {mt} + ~{prompt_tokens} prompt tokens > main --max-model-len "
                    f"{n}: raise it (editor.context_tokens) or lower max_tokens")
    return None


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
    mode, gfb = method_mode(cfg), gate_fallback(cfg)                            # fail fast: valid method / gate
    mmode = "adapter" if mode == "edit" else merge_mode(cfg)                     # fail fast: valid merge.mode
    lrank = server_lora_rank(cfg)                                                # fail fast: adapter rank <= 512
    short = context_shortfall(cfg) if not rc.mock else None                     # fail fast: prompt + max_tokens fit?
    if short:
        print(f"[lora_code] {short}", flush=True)
        return 1
    print(f"[lora_code] hardware={hw} gpus={rc.gpus} roles {roles} method={mode} "
          f"gate reference=parent fallback={gfb} merge={mmode} max_lora_rank={lrank}"
          + (f" (UNUSED {unused}: tensor parallelism needs 1/2/4/8 GPUs per role)" if unused else ""), flush=True)
    try:
        Run(rc, cfg).run()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
