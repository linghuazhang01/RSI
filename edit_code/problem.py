"""One quantization search problem = (source model, size budget, evaluation data, references).

Paper A (fixed size, 2026-11-19): every problem quantizes the ORIGINAL BF16 model M_0 to the same
budget; generation g tries to beat the incumbent (best accepted quantization so far). It owns: the model spec, the layer
cache + diagnostics (prepare), config validation / sizing, candidate build (artifact), lazy
fake-quant materialization, candidate serving, and staged evaluation (screen / full / gate) with
structured L1 feedback (RSI_LOOP_DESIGN §2.2).

References (2026-11-20): every evaluation is compared with the INCUMBENT on the same items (it is what a
candidate must beat): the gate pairs a candidate's gate records with the incumbent's per item (2026-11-23: the only
evaluation a search candidate gets, run.gate_candidate; the classic baselines get the same gate, paired with Q_0,
run.Runner.run_baselines). Screen / full evaluations score the incumbent itself on the generation's items (the
editor reads them); their per-group Hessian-weighted error is kept (group_error.json). M_0 is not scored on the
items, and there is no KL term (2026-11-23).

Candidate serving (2026-11-19): the candidate GPUs are split into groups (common.candidate_groups); each
group is one slot of a pool. serve() takes a free slot, exports the candidate's fake-quant checkpoint into
that slot's directory on the slot's own GPU (2026-11-23: exports of different slots run in parallel, one layer
cache per slot), starts vLLM
on the slot's GPUs, and gives the slot back afterwards. So with N groups, N candidates are evaluated at
the same time (rsicore.search.inner_search(parallel=N)). evaluate() is thread-safe: an identical config
being evaluated by another thread is waited for (evaluated once).
"""
from __future__ import annotations

import math
import queue
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from . import mock as M
from .common import (append_jsonl, candidate_args, candidate_groups, config_hash, free_cuda, load_processed,
                     read_json, read_jsonl, record_text, role_gpus, scoring_concurrency, tail, write_json)
from .diagnostics import bytes_table, load_tables, save_tables, summarize_profile
from .quant import schema, sizes
from .quant.spec import ModelSpec, spec_from_checkpoint, synthetic_spec

DATASET_DOMAIN = {"dapo_math": "math", "taco_verified": "code", "triviaqa_train": "knowledge",
                  "ifeval_train": "instruction_following", "sealtool_train": "tool_calling"}
# simple_<domain>_{train,pool,eval,test} (the run's data, 2026-11-21) belong to their domain
DATASET_DOMAIN.update({f"simple_{d}_{r}": d for d in ("math", "code", "knowledge", "instruction_following",
                                                       "tool_calling") for r in ("train", "pool", "eval", "test")})


def _mean_per_id(records: list[dict]) -> tuple[float | None, dict, int]:
    per, pending = {}, 0
    for r in records:
        if r.get("score") is None:
            pending += 1
            continue
        per.setdefault(r["id"], []).append(float(r["score"]))
    per = {k: sum(v) / len(v) for k, v in per.items()}
    return (sum(per.values()) / len(per) if per else None), per, pending


class QuantProblem:
    def __init__(self, rc, cfg: dict, *, tag: str, gen: int, out_dir: Path, prep_dir: Path,
                 source_dir: Path | None, source_state: dict | None, budget_gib: float,
                 datasets: list[str], data: dict, kl_texts: list[tuple[str, str]] | None = None,
                 gate_baseline: dict | None = None, show_examples: int = 2, incumbent: dict | None = None, log=print,
                 prep_gpus: list[int] | None = None, gate_datasets: list[str] | None = None):
        from rsicore import evaluate as _ev  # noqa: F401  (fail early if rsicore is missing)
        self.rc, self.cfg, self.tag, self.gen = rc, cfg, tag, gen
        self.mock = bool(rc.mock)
        self.out_dir, self.prep_dir = Path(out_dir), Path(prep_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = Path(source_dir) if source_dir else None
        self.source_state = source_state
        self.budget_gib = float(budget_gib)
        # data roles (rsicore.draws): datasets = the train role simple_<d>_pool (screen / full items, feedback,
        # compare), gate_datasets = the gate role simple_<d>_eval (gate stage; default: the same datasets)
        self.datasets = list(datasets)
        self.gate_datasets = list(gate_datasets) if gate_datasets else list(self.datasets)
        self.data = data                                  # {dataset: {"screen": ids, "signal": ids}}
        self.kl_texts = list(kl_texts or [])              # [(domain, text)]: calibration texts (generation 0 prep)
        self.gate_baseline = gate_baseline or {}
        # incumbent = best accepted quantization so far: {"gen", "cid", "config", "gate": {ds: {"score", "records"}}}
        self.incumbent = incumbent
        self.incumbent_scores: dict = {}                  # {"screen": x, "full": y} on this problem's items
        self.show_examples = int(show_examples)
        self.log = log
        self.space = dict(schema.DEFAULT_ACTION_SPACE, **(cfg.get("action_space") or {}))
        self.registry = Path(rc.run_dir) / "mock_registry.json"
        self.spec: ModelSpec | None = None
        self.tables: dict | None = None
        self.cache = None
        self.shown: dict[str, set] = {}
        self._lock = threading.RLock()
        # (2026-11-23) exports run in parallel, one layer cache per candidate slot GPU (_slot_cache); this lock now
        # only serialises herr_module's GPU use of self.cache
        self._export_lock = threading.Lock()
        self._slot_caches: dict = {}               # slot index -> (QuantCache on the slot's first GPU, its lock)
        self.exported: dict = {}                   # config hash -> a finished fake-quant export (served by serve_all)
        self._inflight: dict = {}                  # (hash, stage) -> threading.Event
        self.slots = candidate_groups(rc, cfg)     # [[GPU ids]] one candidate server per slot
        self._removed: set = set()                 # slot indices taken back (add_slot / remove_slot)
        # (2026-11-24) candidates gated at the same time (run.Runner.run_search sets it to the threads of a step);
        # None = one slot per candidate gate. See gate_slots().
        self.gate_parallel: int | None = None
        self.prep_gpus = list(prep_gpus) if prep_gpus else None   # generation-0 prep on these GPUs (one worker each)
        from rsicore.draws import is_eval_set
        # an eval-set train role (simple_<d>_pool is one): the editor may see every item, so feedback examples
        # also show the reference answer
        self._eval_mode = any(is_eval_set(ds) for ds in self.datasets)
        self._pool: queue.Queue = queue.Queue()
        for i in range(len(self.slots)):
            self._pool.put(i)
        # decoding of each scoring stage (2026-11-21, paper config `decoding`, rsicore.config.stage_decoding; None =
        # tasks.yaml greedy). The same dict (seed included) for the incumbent and every candidate. It is part of
        # every evaluation record ("decoding", absent when None) and a cached record is reused only under the
        # same decoding.
        from rsicore.config import stage_decoding
        self.decoding = {st: stage_decoding(cfg, st, getattr(rc, "seed", None)) for st in ("screen", "full", "gate")}
        self._evals = {}
        for r in read_jsonl(self.out_dir / "evals.jsonl"):
            if r.get("decoding") == self._dec_key(r["stage"]):
                self._evals[(r["hash"], r["stage"])] = r
        self._herr_lock = threading.Lock()
        self._herr: dict | None = None             # {"module|q|b|gs": [h_err, tr(W H W^T), tokens]} (persisted)

    # ------------------------------------------------------------------ bookkeeping
    @property
    def n_slots(self) -> int:
        """Candidate slots in use (a lent editor GPU counts while it is lent)."""
        return len(self.slots) - len(self._removed)

    def add_slot(self, gpus) -> int:
        """Lend GPUs to the candidate-server pool (e.g. the sleeping editor's); returns the slot index."""
        gpus = list(gpus)
        with self._lock:
            if gpus in self.slots and self.slots.index(gpus) in self._removed:
                i = self.slots.index(gpus)
                self._removed.discard(i)
            else:
                self.slots.append(gpus)
                i = len(self.slots) - 1
        self._pool.put(i)
        return i

    def remove_slot(self, i: int) -> None:
        """Take a lent slot back once it is free (waits for a running evaluation on it to finish)."""
        held = []
        while True:
            j = self._pool.get()
            if j == i:
                break
            held.append(j)
        for j in held:
            self._pool.put(j)
        with self._lock:
            self._removed.add(i)

    def gate_slots(self) -> int:
        """Slots ONE candidate gate may use (2026-11-24, gpu_layout.gate_split, default on): ceil(n_slots /
        gate_parallel), e.g. 8 slots and 4 proposals per step -> 2 GPUs per candidate (1 x 8 -> 8, 8 x 1 -> 1 as
        before). The gate items are split over the replicas (score_split; same decoding and seed per item)."""
        if not self.gate_parallel or not (self.cfg.get("gpu_layout") or {}).get("gate_split", True):
            return 1
        return max(1, math.ceil(self.n_slots / max(1, int(self.gate_parallel))))

    def _dec_key(self, stage: str) -> str | None:
        """The stage's decoding as a canonical string (None = tasks.yaml decoding), stored in evaluation records."""
        import json
        dec = self.decoding.get(stage)
        return None if dec is None else json.dumps(dec, sort_keys=True)

    @property
    def domains(self) -> list[str]:
        return [DATASET_DOMAIN.get(d, d) for d in self.datasets]

    # ------------------------------------------------------------------ preparation
    def prepare(self):
        """Spec, diagnostics and (real mode) the layer cache for the source model; cached in prep_dir."""
        sp = self.prep_dir / "spec.json"
        if sp.exists():
            self.spec = ModelSpec.from_json(read_json(sp))
        elif self.mock:
            self.spec = synthetic_spec(self.cfg.get("mock", {}).get("spec"))
        else:
            self.spec = spec_from_checkpoint(self.source_dir, count_visual=self.cfg["bytes"].get("count_visual", True))
        write_json(sp, self.spec.to_json())
        dcfg = self.cfg["diagnostics"]
        self.tables = load_tables(self.prep_dir / "diagnostics")
        if self.tables is None:
            if self.mock:
                prof = M.profile_table(self.spec, self.source_state, dcfg["profile_bits"], dcfg["profile_group_size"])
                act = M.activation_table(self.spec)
                meta = {"mock": True}
            else:
                prof, act, meta = self._prepare_real()
            self.tables = save_tables(self.prep_dir / "diagnostics", prof, act,
                                      bytes_table(self.spec, [b for b in self.space["bits"] if b < 16],
                                                  self.space["group_sizes"]), meta)
        if not self.mock and self.cache is None:
            self.cache = self._open_cache()
        if self.tables.get("hessian") is None:     # built after the layer cache (it reads cached entries)
            self.tables["hessian"] = self.hessian_table()
            write_json(self.prep_dir / "diagnostics" / "hessian_table.json", self.tables["hessian"])
        return self

    # ------------------------------------------------------------------ Hessian-weighted error
    def _herr_store(self) -> dict:
        if self._herr is None:
            self._herr = read_json(self.prep_dir / "diagnostics" / "herr_cache.json", {}) or {}
        return self._herr

    def herr_module(self, module: str, setting: dict, cache=None, lock=None) -> tuple[float, float, int]:
        """(tr((W - Wq) H (W - Wq)^T), tr(W H W^T), calibration tokens) of one module under a resolved
        setting; H = the module's input Hessian from the calibration pass (sum over tokens). Memoised and
        persisted (prep/diagnostics/herr_cache.json). bits 16 -> 0 error. cache / lock: a worker's own layer
        cache on its GPU (hessian_table over several GPUs); default self.cache under the export lock."""
        b, q, gs = int(setting["bits"]), setting.get("quantizer", "rtn"), int(setting.get("group_size") or 128)
        key = f"{module}|{q}|{b}|{gs}"
        with self._herr_lock:
            got = self._herr_store().get(key)
        if got is not None:
            return tuple(got)
        if b >= 16:
            return (0.0, 1.0, 1)
        if self.mock:
            m = self.spec.module(module)
            d = M.setting_damage(self.spec, f"{m.layer}.{m.group}", setting) * (0.9 + 0.2 * M._h("herr", module))
            val = [d, 1.0, 1]
        else:  # pragma: no cover - GPU path
            import torch
            from .quant import quantizers as Q
            from .quant.hessian import load_hessian
            m = self.spec.module(module)
            c = cache or self.cache
            with (lock or self._export_lock):      # one GPU user at a time per cache
                H, n, _ = load_hessian(self.prep_dir / "hessians", m.layer, m.input_key, device=c.device)
                W = c.weight(module).to(c.device).float()
                Wq = c.dequant(module, {"bits": b, "quantizer": q, "group_size": gs},
                               dtype=torch.float32).to(c.device)
                err = Q.h_error(W, Wq, H)
                ref = float(((W @ H.float()) * W).sum())
                del H, W, Wq
            val = [err, ref, int(n)]
        with self._herr_lock:
            self._herr_store()[key] = val
            self._herr_dirty = True
        return tuple(val)

    def _herr_flush(self) -> None:
        with self._herr_lock:
            if getattr(self, "_herr_dirty", False):
                write_json(self.prep_dir / "diagnostics" / "herr_cache.json", self._herr_store())
                self._herr_dirty = False

    def herr_missing(self, config_or_norm) -> list[str]:
        """Module settings ("module|quantizer|bits|gs") of a config whose Hessian-weighted error is not in the
        herr cache yet (group_error would compute them: lazy quantization, GPU). bits 16 is never missing.
        Used by the editor's exploration API (tools.make_tool_api(explore=True))."""
        norm = self.normalize(config_or_norm) if "default" in config_or_norm else config_or_norm
        groups = self.spec.groups()
        out = []
        with self._herr_lock:
            store = self._herr_store()
            for key, s in norm["groups"].items():
                b, q, gs = int(s["bits"]), s.get("quantizer", "rtn"), int(s.get("group_size") or 128)
                if b >= 16:
                    continue
                out.extend(k for k in (f"{m.name}|{q}|{b}|{gs}" for m in groups[key]) if k not in store)
        return out

    def group_error(self, config_or_norm, top: int | None = None) -> dict:
        """Hessian-weighted output error of every linear module group under a config (no forward pass):
        {"groups": {key: {"setting", "herr_per_token", "herr_rel", "share"}}, "total_per_token", "top"}.
        herr_per_token = tr(dW H dW^T) / calibration tokens (summed over the group's modules); herr_rel = that
        / tr(W H W^T) (scale-free); share = fraction of the config's total. Embedding / lm_head not included."""
        norm = self.normalize(config_or_norm) if "default" in config_or_norm else config_or_norm
        groups = self.spec.groups()
        out, total = {}, 0.0
        for key, s in norm["groups"].items():
            err = ref = 0.0
            tok = 1
            for m in groups[key]:
                e, r, n = self.herr_module(m.name, s)
                err, ref, tok = err + e, ref + r, max(1, n)
            pt = err / tok
            out[key] = {"setting": s, "herr_per_token": pt, "herr_rel": err / ref if ref > 0 else 0.0}
            total += pt
        for v in out.values():
            v["share"] = v["herr_per_token"] / total if total > 0 else 0.0
            v["herr_per_token"] = round(v["herr_per_token"], 8)
            v["herr_rel"] = round(v["herr_rel"], 8)
            v["share"] = round(v["share"], 5)
        self._herr_flush()
        ranked = sorted(out, key=lambda k: -out[k]["share"])
        n = len(ranked) if top is None else int(top)
        return {"total_per_token": round(total, 8), "top": [(k, out[k]["share"]) for k in ranked[:n]],
                "groups": out}

    def hessian_table(self) -> list[dict]:
        """One row per (group key, quantizer, bits) at diagnostics.profile_group_size: the Hessian-weighted
        error if ONLY that group is quantized (herr_per_token, herr_rel). Uses every calibration token and
        the cached quantized weights; no forward pass. Quantizers: diagnostics.hessian_quantizers."""
        d = self.cfg["diagnostics"]
        gs = int(d.get("profile_group_size", 128))
        qs = [q for q in d.get("hessian_quantizers", ["rtn", "gptq", "awq"]) if q in self.space["quantizers"]]
        bits = [b for b in d.get("hessian_bits") or d.get("profile_bits", [2, 3, 4]) if b < 16]
        gp, t0 = self.spec.group_params(), time.time()
        groups = list(self.spec.groups().items())

        def rows_for(items, cache=None, lock=None):
            out = []
            for key, mods in items:
                layer, g = key.split(".", 1)
                for q in qs:
                    for b in bits:
                        s = {"bits": b, "group_size": gs, "quantizer": q}
                        err = ref = 0.0
                        tok = 1
                        for m in mods:
                            e, r, n = self.herr_module(m.name, s, cache=cache, lock=lock)
                            err, ref, tok = err + e, ref + r, max(1, n)
                        out.append({"key": key, "layer": int(layer), "group": g, "quantizer": q, "bits": b,
                                    "group_size": gs, "herr_per_token": round(err / tok, 8),
                                    "herr_rel": round(err / ref, 8) if ref > 0 else 0.0, "params": gp[key]})
            return out
        devs = [f"cuda:{g}" for g in self.prep_gpus] if (self.prep_gpus and not self.mock) else []
        if len(devs) > 1:                          # one worker (own layer cache) per GPU, groups round-robin
            from concurrent.futures import ThreadPoolExecutor
            shares = [groups[i::len(devs)] for i in range(len(devs))]
            caches = [self._open_cache(device=d) for d in devs]
            with ThreadPoolExecutor(max_workers=len(devs)) as ex:
                parts = list(ex.map(lambda i: rows_for(shares[i], caches[i], threading.Lock()), range(len(devs))))
            order = {k: i for i, (k, _) in enumerate(groups)}
            rows = sorted((r for p in parts for r in p), key=lambda r: order[r["key"]])
        else:
            rows = rows_for(groups)
        self._herr_flush()
        self.log(f"[prep:{self.tag}] hessian table: {len(rows)} rows in {time.time() - t0:.0f}s")
        return rows

    def _device(self) -> str:
        g = role_gpus(self.rc, self.cfg, "candidate")[0]
        dev = self.cfg["cache"].get("device", "cuda")
        return f"cuda:{g}" if dev == "cuda" else dev

    def _open_cache(self, device: str | None = None):
        from .quant.cache import QuantCache
        return QuantCache(self.prep_dir / "cache", self.spec, self.source_dir, hessian_dir=self.prep_dir / "hessians",
                          device=device or self._device(), damp=self.cfg["cache"].get("gptq_damp", 0.01),
                          lazy=self.cfg["cache"].get("lazy", True), log=self.log)

    def _prepare_real(self):  # GPU path on the cluster; exercised on CPU by tests/test_quant.py
        import torch
        from .diagnostics import activation_table_real, profile_rows, warn_slow_kernels
        from .quant.hessian import collect_stats, load_hf_model, tokenize_texts
        dcfg, ccfg = self.cfg["diagnostics"], self.cfg["calibration"]
        dev = self._device()
        t0 = time.time()
        texts = [t for _, t in self.kl_texts][: ccfg.get("max_texts", 256)]
        if self.prep_gpus and len(self.prep_gpus) > 1:
            return self._prepare_real_parallel(texts, [f"cuda:{g}" for g in self.prep_gpus])
        self.log(f"[prep:{self.tag}] loading {self.source_dir} on {dev}")
        model = load_hf_model(self.source_dir, dev)
        batches = tokenize_texts(self.source_dir, texts, ccfg.get("max_len", 2048))
        stats = collect_stats(model, self.spec, batches, self.prep_dir / "hessians",
                              layers_per_pass=ccfg.get("layers_per_pass", 16), device=dev, log=self.log)
        warn_slow_kernels(self.log)
        ptexts = texts[: dcfg["profile_contexts"]]
        prof = profile_rows(model, self.source_dir, self.spec, ptexts, dcfg, rows_dir=self.prep_dir / "profile_rows",
                            log=self.log)
        act = activation_table_real(self.spec, stats)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cache = self._open_cache()
        self.cache.warm(self.cfg["cache"].get("warm", []))
        free_cuda()
        return prof, act, {"seconds": time.time() - t0, "contexts": len(batches), "profile_contexts": len(ptexts),
                           "cache_disk_gib": self.cache.disk_bytes() / 2 ** 30}

    def _prepare_real_parallel(self, texts: list[str], devs: list[str]):  # pragma: no cover - GPU path
        """Generation-0 prep on several GPUs (gpu_layout.prep_all_gpus, 2026-11-20), one worker + one HF model per
        GPU: (1) calibration Hessians, texts round-robin per worker, summed afterwards (H and the token count
        are sums over tokens, absmax a max: identical to one pass over all texts); (2) the profile table, group
        keys round-robin; (3) the cache warm-up, modules round-robin (the hessian table follows in prepare)."""
        import multiprocessing as mp
        import torch
        from concurrent.futures import ThreadPoolExecutor
        from safetensors.torch import save_file
        from .diagnostics import activation_table_real, profile_meta, profile_worker
        from .quant.hessian import activation_summary, collect_stats, hessian_path, load_hessian, load_hf_model, \
            tokenize_texts
        dcfg, ccfg = self.cfg["diagnostics"], self.cfg["calibration"]
        t0, k = time.time(), len(devs)
        stats, hdir = {}, self.prep_dir / "hessians"
        keys = sorted({(m.layer, m.input_key) for m in self.spec.modules})
        # (1) calibration, threads (one HF model per GPU). Skipped when every summed Hessian is already on disk
        # (2026-11-21: a restarted prep no longer recomputes them).
        if all(hessian_path(hdir, l, key).exists() for (l, key) in keys):
            self.log(f"[prep:{self.tag}] calibration: all {len(keys)} Hessians already in {hdir}, skipped")
        else:
            self.log(f"[prep:{self.tag}] {k} workers on {devs}: loading {self.source_dir}")
            # Avoid concurrent Transformers lazy imports during HF model construction.
            models = [load_hf_model(self.source_dir, d) for d in devs]
            parts = [texts[i::k] for i in range(k)]

            def calib(i):
                b = tokenize_texts(self.source_dir, parts[i], ccfg.get("max_len", 2048))
                collect_stats(models[i], self.spec, b, self.prep_dir / "hessians_parts" / f"w{i}",
                              layers_per_pass=ccfg.get("layers_per_pass", 16), device=devs[i], log=self.log,
                              return_stats=False)     # (2026-11-23) the parts' stats were read back and discarded
            with ThreadPoolExecutor(max_workers=k) as ex:
                list(ex.map(calib, range(k)))
            del models, calib
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            def reduce(lk):                            # sum the workers' Hessians (always in worker order 0..k-1)
                layer, key = lk
                dst = hessian_path(hdir, layer, key)
                if dst.exists():
                    return
                H = absmax = None
                n = 0
                for i in range(k):
                    h, ni, a = load_hessian(self.prep_dir / "hessians_parts" / f"w{i}", layer, key)
                    H = h if H is None else H + h
                    absmax = a if absmax is None else torch.maximum(absmax, a)
                    n += ni
                dst.parent.mkdir(parents=True, exist_ok=True)
                save_file({"H": H.contiguous(), "absmax": absmax.contiguous(),
                           "n": torch.tensor([n], dtype=torch.int64)}, str(dst))
                stats[f"{layer}.{key}"] = activation_summary(H, n, absmax)   # no second read of the summed file
            # (2026-11-23) several keys at once (the reads dominate; the per-key sum order is unchanged, so the summed
            # Hessians are bitwise the same as before)
            with ThreadPoolExecutor(max_workers=min(k, 8)) as ex:
                list(ex.map(reduce, keys))
            shutil.rmtree(self.prep_dir / "hessians_parts", ignore_errors=True)
            self.log(f"[prep:{self.tag}] calibration done ({len(keys)} Hessians, {time.time() - t0:.0f}s)")
        for (layer, key) in keys:                      # Hessians summed by an earlier (interrupted) prep
            if f"{layer}.{key}" not in stats:
                H, n, absmax = load_hessian(hdir, layer, key)
                stats[f"{layer}.{key}"] = activation_summary(H, n, absmax)
        # (2) profile table: one PROCESS per GPU (2026-11-21; the threads shared one GIL and the GPUs sat ~90% idle),
        # group keys round-robin, contexts batched, each finished group saved to prep/profile_rows/<key>.json
        group_keys = self.spec.group_keys()
        share = [group_keys[i::k] for i in range(k)]
        ptexts = texts[: dcfg["profile_contexts"]]
        rows_dir = self.prep_dir / "profile_rows"
        rows_dir.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=profile_worker, name=f"profile-w{i}",
                             args=(str(self.source_dir), devs[i], str(self.prep_dir / "spec.json"), ptexts,
                                   dict(dcfg), str(rows_dir), share[i], self.tag))
                 for i in range(k)]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        bad = [(pr.name, pr.exitcode) for pr in procs if pr.exitcode != 0]
        if bad:
            raise RuntimeError(f"profile workers failed: {bad} (finished groups are kept in {rows_dir}; rerun to resume)")
        meta = profile_meta(ptexts, dcfg)
        prow, missing = [], []
        for key in group_keys:
            rec = read_json(rows_dir / f"{key}.json") or {}
            if rec.get("meta") != meta:
                missing.append(key)
            else:
                prow += rec["rows"]
        if missing:
            raise RuntimeError(f"profile table incomplete: {len(missing)} groups missing, e.g. {missing[:3]}")
        self.log(f"[prep:{self.tag}] profile table: {len(group_keys)} groups on {k} processes, {time.time() - t1:.0f}s")
        order = {g: i for i, g in enumerate(group_keys)}
        prof_rows = sorted(prow, key=lambda r: (order[r["key"]], r["bits"]))
        act = activation_table_real(self.spec, stats)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mods = [m.name for m in self.spec.modules]
        caches = [self._open_cache(device=d) for d in devs]
        warm = self.cfg["cache"].get("warm", [])
        with ThreadPoolExecutor(max_workers=k) as ex:
            list(ex.map(lambda i: caches[i].warm(warm, modules=mods[i::k]), range(k)))
        self.cache = self._open_cache()
        free_cuda()
        return prof_rows, act, {"seconds": time.time() - t0, "contexts": len(texts), "workers": devs,
                                "profile_contexts": dcfg["profile_contexts"],
                                "cache_disk_gib": self.cache.disk_bytes() / 2 ** 30}

    # ------------------------------------------------------------------ configs
    def normalize(self, config) -> dict:
        return schema.normalize(config, self.spec, self.space)

    def size(self, config_or_norm) -> dict:
        norm = self.normalize(config_or_norm) if "default" in config_or_norm else config_or_norm
        return sizes.config_bytes(norm, self.spec)

    def parent_config(self) -> dict | None:
        if self.mock:
            st = self.source_state or {}
            if not st.get("bits"):
                return None
            return {"bits_by_group": st["bits"], "embed": st.get("embed"), "lm_head": st.get("lm_head")}
        fq = read_json(self.source_dir / "fake_quant.json") if self.source_dir else None
        return schema.compress(fq["config"], self.spec) if fq else None

    def build(self, config, cid: str, *, check_budget: bool = True) -> dict:
        """Validate + size a config -> artifact dict (raises schema.ConfigError / ValueError). check_budget False:
        no size check (the uniform AWQ baseline, whose channel scales overshoot the budget by a few MiB)."""
        norm = self.normalize(config)
        b = sizes.config_bytes(norm, self.spec)
        if check_budget and b["gib"] > self.budget_gib + 1e-9:
            raise ValueError(f"config is {b['gib']:.3f} GiB > budget {self.budget_gib:.3f} GiB "
                             f"(linear avg {b['linear_avg_bits']:.2f} bits; by group GiB: "
                             + ", ".join(f"{g}={v:.2f}" for g, v in b["by_group_gib"].items()) + ")")
        h = config_hash(norm)
        write_json(self.out_dir / "configs" / f"{h}.json", norm)
        art = {"config_hash": h, "config": schema.compress(norm, self.spec), "bytes_gib": round(b["gib"], 4),
               "budget_gib": self.budget_gib, "linear_avg_bits": round(b["linear_avg_bits"], 3), "tag": self.tag}
        if self.mock:
            st = M.apply_config(self.spec, self.source_state, norm, self.cfg["mock"].get("lam", 1.2))
            art["mock_quality"] = round(st["quality"], 6)
        return art

    def norm_of(self, art: dict) -> dict:
        return read_json(self.out_dir / "configs" / f"{art['config_hash']}.json")

    # ------------------------------------------------------------------ candidate model
    def materialize(self, art: dict, dst: Path | None = None, slot: int = 0) -> Path | None:
        """Write the fake-quant checkpoint (real mode) on the first GPU of candidate slot `slot` (free at this
        point: its previous server has stopped). Default dst: out_dir/ckpt/slot<k>/<hash>, one checkpoint
        per slot on disk (candidates.keep_checkpoints <= 1). Returns its path."""
        if self.mock:
            return None
        from .quant.export import export_fake_quant
        sdir = self.out_dir / "ckpt" / f"slot{slot}"
        dst = Path(dst) if dst else sdir / art["config_hash"]
        if (dst / "fake_quant.json").exists():
            return dst
        if dst.parent == sdir and self.cfg["candidates"].get("keep_checkpoints", 1) <= 1:
            for p in sdir.glob("*"):  # bounded disk: one candidate checkpoint per slot
                if p != dst:
                    shutil.rmtree(p, ignore_errors=True)
        cache, lock = self._slot_cache(slot)
        with lock:              # (2026-11-23) per slot: the exports of different slots run at the same time
            export_fake_quant(self.source_dir, dst, self.norm_of(art), self.spec, cache,
                              meta={"tag": self.tag, "gen": self.gen, "budget_gib": self.budget_gib}, log=self.log)
            free_cuda()   # the candidate vLLM server starts on this GPU next (80 GB cards: every GB counts)
        return dst

    def _slot_cache(self, slot: int):
        """(QuantCache, lock) of candidate slot `slot`: the layer cache on the slot's first GPU (cache.device cuda),
        else on cache.device. Every slot cache reads the same on-disk entries; a missing entry is computed once
        (QuantCache._entry_lock), so the exported tensors are the same as with one shared cache."""
        with self._lock:
            got = self._slot_caches.get(slot)
            if got is None:
                dev = self._device()
                if str(dev).startswith("cuda"):
                    dev = f"cuda:{self.slots[slot][0]}"
                got = self._slot_caches[slot] = (self._open_cache(device=dev), threading.Lock())
            return got

    def adopt_checkpoint(self, art: dict, dst: Path) -> bool:
        """Move a finished candidate checkpoint of this config (out_dir/ckpt/slot<k>/<hash>, written for its gate
        by this problem) to dst instead of exporting it again (2026-11-23). The same export_fake_quant output: same
        source, normalized config and meta. False when there is none (a slot reused by a later candidate)."""
        import json
        import os
        if self.mock:
            return False
        want = self.norm_of(art)
        for d in sorted((self.out_dir / "ckpt").glob(f"slot*/{art['config_hash']}")):
            fq = d / "fake_quant.json"
            if not fq.exists():
                continue
            info = json.loads(fq.read_text())
            if info.get("config") != want or info.get("gen") != self.gen or info.get("tag") != self.tag \
                    or str(info.get("source")) != str(self.source_dir):
                continue
            dst = Path(dst)
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(d, dst)
            self.log(f"[export] reused the gate checkpoint {d} -> {dst}")
            return True
        return False

    @contextmanager
    def serve(self, art: dict, name: str):
        """Serve one candidate on a free slot of the candidate-server pool (blocks until one is free)."""
        from rsicore.servers import VLLMServer
        slot = self._pool.get()
        try:
            gpus = self.slots[slot]
            logs = Path(self.rc.run_dir) / "logs"
            if self.mock:
                M.register(self.registry, name, art["mock_quality"])
                # no --tasks: the mock server's default index (the training sets; simple_* prompts resolve to
                # them); the index is cached
                with VLLMServer("mock-" + name, name, gpus, mock=True, mock_registry=self.registry,
                                seed=self.rc.seed, log_dir=logs) as srv:
                    yield srv
            else:  # pragma: no cover - GPU path
                ckpt = self.materialize(art, slot=slot)
                with VLLMServer(ckpt, name, gpus, seed=self.rc.seed, log_dir=logs,
                                extra_args=candidate_args(self.cfg)) as srv:
                    yield srv
        finally:
            self._pool.put(slot)

    @contextmanager
    def serve_all(self, art: dict, name: str, most: int | None = None, keep: bool = False):
        """Serve one config on EVERY free slot of the pool at once (2026-11-22, like Paper B's parent screen over all
        main replicas): blocks for the first slot, then takes every other slot that is free right now (at most
        `most` slots in all; None = no limit), so it never waits on a running evaluation. Real mode: the checkpoint
        is written once (the incumbent's exported model when it is there, else a finished export registered in
        self.exported, e.g. Q_0's baseline/q0/model, else a shared copy under out_dir/ckpt/shared/, removed
        afterwards; keep=True, a candidate's gate (2026-11-24): its usual checkpoint out_dir/ckpt/slot<k>/<hash> on
        the first slot, kept for adopt_checkpoint) and every slot serves it; the servers start together.
        Yields the list of servers."""
        from rsicore.servers import VLLMServer, start_all
        slots = [self._pool.get()]
        while most is None or len(slots) < most:
            try:
                slots.append(self._pool.get_nowait())
            except queue.Empty:
                break
        tmp = None
        try:
            logs = Path(self.rc.run_dir) / "logs"
            if self.mock:
                M.register(self.registry, name, art["mock_quality"])
                srvs = [VLLMServer("mock-" + name, name, self.slots[s], mock=True, mock_registry=self.registry,
                                   seed=self.rc.seed, log_dir=logs) for s in slots]
            else:  # pragma: no cover - GPU path
                inc = self.incumbent or {}
                ckpt = Path(inc["model_dir"]) if (inc.get("config_hash") == art["config_hash"] and inc.get("model_dir")
                                                  and (Path(inc["model_dir"]) / "fake_quant.json").exists()) else None
                ex = self.exported.get(art["config_hash"])
                if ckpt is None and ex and (Path(ex) / "fake_quant.json").exists():   # e.g. Q_0's exported model
                    ckpt = Path(ex)
                if ckpt is None and keep:
                    ckpt = self.materialize(art, slot=slots[0])
                elif ckpt is None:
                    tmp = self.out_dir / "ckpt" / "shared" / art["config_hash"]
                    ckpt = self.materialize(art, dst=tmp, slot=slots[0])
                srvs = [VLLMServer(ckpt, name, self.slots[s], seed=self.rc.seed, log_dir=logs,
                                   extra_args=candidate_args(self.cfg)) for s in slots]
            try:
                start_all(srvs)
                yield srvs
            finally:
                for s in srvs:
                    s.stop()
        finally:
            for s in slots:
                self._pool.put(s)
            if tmp is not None:          # also the export's staging dir when it failed (quant.export: <dst>.partial)
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.rmtree(tmp.with_name(tmp.name + ".partial"), ignore_errors=True)

    # ------------------------------------------------------------------ evaluation
    def stage_datasets(self, stage: str) -> list[str]:
        """Datasets scored at `stage`: the gate role for gate, the train role for screen / full."""
        return self.gate_datasets if stage == "gate" else self.datasets

    def _ids(self, ds: str, stage: str) -> list[str]:
        if stage == "gate":
            from rsicore.draws import gate_ids
            return gate_ids(self.rc, ds)
        return list(self.data[ds]["signal" if stage == "full" else "screen"])

    def evaluate(self, config_or_art, stage: str, cid: str, *, origin: str = "candidate") -> dict:
        """Evaluate a config at `stage` (screen | full | gate) with the stage's decoding (self.decoding). Cached by
        (config hash, stage). Returns {"score", "feedback", ...}. Search candidates only get the gate stage
        (run.gate_candidate); screen / full score the incumbent (origin "incumbent") and the baselines."""
        art = config_or_art if "config_hash" in config_or_art else self.build(config_or_art, cid)
        key = (art["config_hash"], stage)
        while True:
            with self._lock:
                if key in self._evals:
                    r = dict(self._evals[key])
                    r["cached"] = True
                    return r
                busy = self._inflight.get(key)
                if busy is None:   # this thread evaluates it
                    self._inflight[key] = threading.Event()
                    break
            busy.wait()            # another thread is evaluating the same config: use its result
        try:
            return self._evaluate(art, stage, cid, key, origin)
        finally:
            with self._lock:
                self._inflight.pop(key).set()

    def _evaluate(self, art: dict, stage: str, cid: str, key: tuple, origin: str) -> dict:
        t0 = time.time()
        from rsicore.evaluate import paired_delta, score, score_split
        name = f"{self.rc.run_id}-{self.tag}-{art['config_hash']}"
        edir = self.out_dir / "evals" / f"{cid}" / stage
        probes, regressions = {}, []
        # the same datasets AND items as an existing screen evaluation of this config (full = screen): reuse its
        # records instead of scoring them again. The gate (simple_<d>_eval, another dataset than the train role) is
        # always scored, and so is a stage with
        # another decoding than the screen's (2026-11-21: e.g. a gate with n 4 never reuses n 1 screen records).
        dsets = self.stage_datasets(stage)
        dec = self.decoding.get(stage)
        scr = self._evals.get((art["config_hash"], "screen")) if stage != "screen" else None
        reuse = None
        if scr is not None and dsets == self.datasets and dec == self.decoding.get("screen") \
                and all(sorted(self._ids(ds, stage)) == sorted(self._ids(ds, "screen")) for ds in dsets):
            src = {ds: self.records_path(scr["cid"], "screen", ds) for ds in dsets}
            if all(p.exists() for p in src.values()):
                reuse = {ds: read_jsonl(p) for ds, p in src.items()}
        from contextlib import nullcontext
        # (2026-11-22) the incumbent's screen / full run (once per generation, before the search; nothing else holds a
        # slot then) is split over every free candidate slot, like Paper B's parent screen over all main replicas.
        # (2026-11-23) so is Q_0's gate (origin "reference", generation 0, nothing else running): it was on ONE GPU
        # (2026-11-23) and so are the classic baselines' gates (run.Runner.run_baselines, after the generations)
        split = not reuse and (origin in ("reference", "baseline") or (origin == "incumbent" and stage != "gate"))
        # (2026-11-24) a candidate's gate on gate_slots() slots: e.g. 8 GPUs, 4 proposals per step -> 2 GPUs each
        share = self.gate_slots() if (origin == "gate" and not reuse and not split) else 1
        if reuse:
            ctx = nullcontext()
        elif split:
            ctx = self.serve_all(art, name)
        elif share > 1:
            ctx = self.serve_all(art, name, most=share, keep=True)
        else:
            ctx = self.serve(art, name)
        with ctx as srv:
            srvs = srv if isinstance(srv, list) else [srv]
            srv = srvs[0]
            for ds in dsets:
                ids = self._ids(ds, stage)
                if reuse:
                    edir.mkdir(parents=True, exist_ok=True)
                    (edir / f"{ds}.jsonl").write_text("".join(__import__("json").dumps(r, ensure_ascii=False) + "\n"
                                                              for r in reuse[ds]))
                    res = SimpleNamespace(records=reuse[ds])
                elif len(srvs) > 1:
                    res = score_split([s.endpoint for s in srvs], name, ds, pool="pool", ids=ids, out_dir=edir,
                                      mock=self.mock, decoding=dec, concurrency=scoring_concurrency(self.cfg))
                else:
                    res = score(srv.endpoint, name, ds, pool="pool", ids=ids, out_dir=edir, mock=self.mock,
                                decoding=dec, concurrency=scoring_concurrency(self.cfg))
                acc, per, pending = _mean_per_id(res.records)
                dom = DATASET_DOMAIN.get(ds, ds)
                p = {"dataset": ds, "score": None if acc is None else round(acc, 4), "n": len(per), "pending": pending}
                if acc is not None and per:
                    se = math.sqrt(max(acc * (1 - acc), 1e-9) / len(per))
                    p["ci"] = [round(max(0.0, acc - 1.96 * se), 4), round(min(1.0, acc + 1.96 * se), 4)]
                if stage == "gate":
                    base = self.gate_baseline.get(ds)
                    p["records"] = str(edir / f"{ds}.jsonl")
                    p["original"] = base
                    p["ratio_vs_original"] = None if (acc is None or not base) else round(acc / base, 4)
                    inc = ((self.incumbent or {}).get("gate") or {}).get(ds)
                    if inc and inc.get("records"):
                        d = paired_delta(read_jsonl(Path(inc["records"])), res.records, seed=self.rc.seed)
                        p["incumbent"] = inc.get("score")
                        p["vs_incumbent"] = None if d["delta"] is None else round(d["delta"], 5)
                        p["delta_ci"] = [None if x is None else round(x, 5) for x in d["ci"]]
                elif origin == "incumbent":           # the reference itself
                    p.update(incumbent=p["score"], vs_incumbent=0.0, retention=1.0 if acc else None)
                else:
                    pref = self._ref_records(ds, stage)
                    if pref is not None:
                        pacc, _, _ = _mean_per_id(pref)
                        d = paired_delta(pref, res.records, seed=self.rc.seed)
                        p["incumbent"] = None if pacc is None else round(pacc, 4)
                        p["vs_incumbent"] = None if d["delta"] is None else round(d["delta"], 4)
                        p["delta_ci"] = [None if x is None else round(x, 4) for x in d["ci"]]
                        if d["delta"] is not None:
                            lo, hi = d["ci"]
                            p["significant"] = bool(abs(d["delta"]) > 0.0 and (lo > 0 or hi < 0))
                        if self.show_examples and stage == "screen":
                            regressions += self._regressions(ds, pref, res.records)
                        if pacc:
                            p["retention"] = None if acc is None else round(acc / pacc, 4)
                probes[dom] = p
        fb = {"stage": stage, "bytes_gib": art["bytes_gib"], "budget_gib": art["budget_gib"],
              "linear_avg_bits": art["linear_avg_bits"], "probes": probes}
        if stage in ("screen", "full"):            # per-group Hessian-weighted error of this config
            ge = self.group_error(self.norm_of(art), top=5)
            write_json(edir / "group_error.json", ge)
            fb["error_top_groups"] = ge["top"]
        if regressions:
            fb["regressions"] = regressions[: self.show_examples * max(1, len(self.datasets))]
            with self._lock:
                for r in fb["regressions"]:
                    self.shown.setdefault(r["dataset"], set()).add(r["id"])
        if stage == "gate":
            passed, why = self._gate_decision(art, probes)
            fb["gate"] = {"passed": passed, "reason": why}
            sc = 1.0 if passed else 0.0
        else:
            rets = [p.get("retention") for p in probes.values() if p.get("retention") is not None]
            ret = sum(rets) / len(rets) if rets else None
            sc = None if ret is None else round(ret, 5)      # nothing scored (e.g. code without Docker): None
            fb["retention"] = None if ret is None else round(ret, 4)
        rec = {"hash": art["config_hash"], "stage": stage, "cid": cid, "origin": origin, "score": sc, "feedback": fb,
               "seconds": round(time.time() - t0, 1)}
        if self._dec_key(stage) is not None:
            rec["decoding"] = self._dec_key(stage)
        with self._lock:
            self._evals[key] = rec
            append_jsonl(self.out_dir / "evals.jsonl", rec)
        return dict(rec, feedback=dict(fb))

    def _gate_decision(self, art: dict, probes: dict) -> tuple[bool, str]:
        """Fixed-size gate: fits the budget AND beats the incumbent on the gate set (paired per item):
        mean paired delta over gate datasets >= gate.min_delta; with gate.require_significant, at least one
        dataset's 95% CI lower bound > 0 and no dataset's upper bound < 0."""
        g = self.cfg["gate"]
        if art["bytes_gib"] > self.budget_gib + 1e-9:
            return False, f"size {art['bytes_gib']:.3f} GiB > budget {self.budget_gib:.3f}"
        if not self.incumbent:
            return False, "no incumbent to compare with"
        deltas = {d: p for d, p in probes.items() if p.get("vs_incumbent") is not None}
        if not deltas:
            if g.get("allow_unscored", False):
                return True, "no scored gate items (allowed by gate.allow_unscored)"
            return False, "no scored gate items"
        mean = sum(p["vs_incumbent"] for p in deltas.values()) / len(deltas)
        desc = ", ".join(f"{d} {p['vs_incumbent']:+.4f} [{p['delta_ci'][0]:+.4f}, {p['delta_ci'][1]:+.4f}]"
                         for d, p in deltas.items())
        md = float(g.get("min_delta", 0.0))
        if mean < md or (md <= 0 and mean <= 0):          # threshold 0 (2026-11-21): a strictly positive gain
            return False, f"mean delta vs incumbent {mean:+.4f} < {g.get('min_delta', 0.0)} ({desc})"
        if g.get("require_significant", True):
            worse = [d for d, p in deltas.items() if p["delta_ci"][1] < 0]
            if worse:
                return False, f"significantly worse than the incumbent on {worse} ({desc})"
            if not any(p["delta_ci"][0] > 0 for p in deltas.values()):
                return False, f"no significant gain over the incumbent ({desc})"
        return True, f"beats the incumbent: mean delta {mean:+.4f} ({desc})"

    def _regressions(self, ds: str, ref: list[dict], cand: list[dict]) -> list[dict]:
        """Items the incumbent got right and the candidate got wrong (first show_examples, by id)."""
        pp = {r["id"]: r for r in ref if r.get("score") is not None}
        out = []
        recs = None
        for r in sorted(cand, key=lambda r: r["id"]):
            p = pp.get(r["id"])
            if p is None or r.get("score") is None or not (p["score"] > 0.5 >= r["score"]):
                continue
            if recs is None:
                recs = load_processed(ds)
            item = recs.get(r["id"], {})
            q = item.get("prompt") or (item.get("messages") or [{}])[-1].get("content", "")
            out.append({"dataset": ds, "id": r["id"], "prompt": tail(str(q)[:400], 400),
                        "incumbent_answer": str(p.get("parsed_answer"))[:200],
                        **({"reference": str(item.get("reference"))[:200]} if self._eval_mode else {}),
                        "candidate_answer": str(r.get("parsed_answer"))[:200],
                        "candidate_output_tail": tail(r.get("raw_output") or "", 300)})
            if len(out) >= self.show_examples:
                break
        return out

    # ------------------------------------------------------------------ references
    def records_path(self, cid: str, stage: str, ds: str) -> Path:
        """Records of candidate `cid` at `stage`. A candidate whose config was already evaluated under another
        cid (identical config hash) shares that cid's records."""
        p = self.out_dir / "evals" / f"{cid}" / stage / f"{ds}.jsonl"
        if p.exists() or cid == "incumbent":
            return p
        h = None
        for r in read_jsonl(self.out_dir / "candidates.jsonl"):
            if r.get("cid") == cid:
                h = (r.get("artifact") or {}).get("config_hash")
        rec = self._evals.get((h, stage)) if h else None
        return self.out_dir / "evals" / f"{rec['cid']}" / stage / f"{ds}.jsonl" if rec else p

    def _ref_records(self, ds: str, stage: str) -> list[dict] | None:
        """The incumbent's records on this problem's `stage` items (evaluate_incumbent first)."""
        p = self.records_path("incumbent", stage, ds)
        return read_jsonl(p) if p.exists() else None

    def evaluate_incumbent(self, stages=("screen", "full")) -> dict:
        """Score the incumbent on this problem's items (cid 'incumbent'): the editor's view of the latest model
        (screen_results / screen_output) and the reference of the baselines' retention, paired deltas and regression
        examples. -> {stage: score}. Sampled ONCE per problem
        (generation) with the stage's decoding and seed and cached (evals.jsonl, evals/incumbent/<stage>/): every
        candidate is paired with the same records (2026-11-21; full reuses the screen records when items and
        decoding are the same)."""
        if not self.incumbent:
            return {}
        art = self.build(self.incumbent["config"], "incumbent")
        self.incumbent_scores = {st: self.evaluate(art, st, "incumbent", origin="incumbent")["score"]
                                 for st in stages}
        return self.incumbent_scores

    # ------------------------------------------------------------------ reference screen + gate stats (simple, 2026-11-21)
    def screen_records(self, gen: int | None = None) -> dict:
        """{dataset: run_eval records} of the incumbent's screen run (the latest model's answers to this
        generation's screen items, decoding.screen; gen = an earlier generation of the main search: its own
        incumbent screen, gen_XX/evals/incumbent/screen/)."""
        if gen is None or int(gen) == int(self.gen):
            base = self.out_dir
        else:
            from rsicore.records import gen_dir
            base = gen_dir(self.rc, int(gen))
        # (2026-11-22) kept in memory once complete (the 8 sessions' screen_results / screen_output calls used to
        # re-read the large jsonl files on every call; Paper B keeps one shared copy): a run's records never change
        key = str(base)
        cache = self.__dict__.setdefault("_screen_cache", {})
        with self._lock:
            if key in cache:
                return dict(cache[key])
        out = {}
        for ds in self.datasets:
            p = base / "evals" / "incumbent" / "screen" / f"{ds}.jsonl"
            if p.exists():
                out[ds] = read_jsonl(p)
        if out and len(out) == len(self.datasets):
            with self._lock:
                cache[key] = out
        return dict(out)

    def screen_items(self, records_by_ds: dict) -> dict:
        """{dataset: {id: processed record}} for the ids in records_by_ds (prompt + gold; training-pool items)."""
        return {ds: load_processed(ds, [r["id"] for r in recs]) for ds, recs in records_by_ds.items()}

    def screen_summary(self, examples: int = 3) -> dict:
        """rsicore.screen_view.summary of the incumbent's screen run (accuracy avg@n, correct-count distribution,
        truncation rate, a few failing items with prompt / gold / answers)."""
        from rsicore import screen_view
        recs = self.screen_records()
        return screen_view.summary(recs, self.screen_items(recs), examples=examples) if recs else {"n_items": 0}

    def gate_stats(self, art: dict, r: dict) -> dict:
        """Statistics of one gate evaluation r (evaluate(..., 'gate')) for the editor / gate.json (no item content):
        passed, reason, accuracy, reference_accuracy (the incumbent's), gain (mean paired delta), ci (95% CI of the
        paired delta; {dataset: ci} with several gate datasets), broken / fixed counts (incumbent right ->
        candidate wrong / the reverse; per-item mean over the samples > 0.5 = right) and their ids, size."""
        fb = r["feedback"]
        probes = [p for p in (fb.get("probes") or {}).values() if isinstance(p, dict)]
        gi = fb.get("gate") or {}
        accs = [p["score"] for p in probes if p.get("score") is not None]
        refs = [p["incumbent"] for p in probes if p.get("incumbent") is not None]
        gains = [p["vs_incumbent"] for p in probes if p.get("vs_incumbent") is not None]
        cis = {p["dataset"]: p.get("delta_ci") for p in probes if p.get("delta_ci") is not None}
        broken, fixed = [], []
        for p in probes:
            inc = ((self.incumbent or {}).get("gate") or {}).get(p["dataset"]) or {}
            if not inc.get("records") or not p.get("records") or not Path(p["records"]).exists() \
                    or not Path(inc["records"]).exists():
                continue
            _, a, _ = _mean_per_id(read_jsonl(Path(p["records"])))
            _, b, _ = _mean_per_id(read_jsonl(Path(inc["records"])))
            ids = sorted(set(a) & set(b))
            broken += [i for i in ids if b[i] > 0.5 >= a[i]]
            fixed += [i for i in ids if a[i] > 0.5 >= b[i]]
        rd = lambda v, k=5: None if v is None else round(v, k)  # noqa: E731
        return {"passed": bool(gi.get("passed")), "reason": gi.get("reason"),
                "accuracy": rd(sum(accs) / len(accs), 4) if accs else None,
                "reference_accuracy": rd(sum(refs) / len(refs), 4) if refs else None,
                "gain": rd(sum(gains) / len(gains)) if gains else None,
                "ci": (next(iter(cis.values())) if len(cis) == 1 else cis) or None,
                "broken": len(broken), "fixed": len(fixed), "broken_ids": broken[:50], "fixed_ids": fixed[:50],
                "bytes_gib": art.get("bytes_gib")}


# ---------------------------------------------------------------------------------- data helpers
def kl_texts_for(rc, gen: int, datasets_all: list[str], cfg: dict) -> list[tuple[str, str]]:
    """(domain, text) calibration contexts from the draw's `kl` role (generation 0: GPTQ / AWQ calibration and the
    profile table's KL contexts; the block keeps its name `kl`)."""
    from rsicore.draws import ids
    out = []
    for ds in datasets_all:
        recs = load_processed(ds, ids(rc, gen, ds, "kl"))
        for i in sorted(recs):
            out.append((DATASET_DOMAIN.get(ds, ds), record_text(recs[i], max_chars=cfg["kl"].get("max_chars", 8000))))
    return out


def add_self_generated(texts: list[tuple[str, str]], endpoint, served_name: str, cfg: dict,
                       cache: Path) -> list[tuple[str, str]]:
    """prompt + response: append the reference model's own greedy response to each KL / calibration prompt
    (2504.04823: reasoning-style calibration text), so the calibration statistics and the profile table's KL also
    cover the answer tokens. endpoint: a URL or a list of URLs (replicas: the batches are spread over them, 2026-11-20), or a
    callable returning one (called only when some response is not cached yet, so a resume needs no server).
    Requests go in batches of kl.self_generated_batch prompts. Cached as jsonl
    {sha, response, finish_reason}; <|im_end|> is appended only to responses that finished (not cut by
    kl.self_generated_max_tokens)."""
    import hashlib
    import json
    import urllib.request
    kc = cfg["kl"]
    have = {r["sha"]: r for r in read_jsonl(cache)}
    shas = [hashlib.sha256(t.encode()).hexdigest() for _, t in texts]
    todo = [(sha, t) for sha, (_, t) in zip(shas, texts) if sha not in have]
    todo = list(dict(todo).items())                        # one request per distinct prompt
    if todo:
        eps = endpoint() if callable(endpoint) else endpoint
        urls = [e.rstrip("/") + "/completions" for e in (eps if isinstance(eps, (list, tuple)) else [eps])]
        bs = max(1, int(kc.get("self_generated_batch", 64)))
        chunks = [todo[b:b + bs] for b in range(0, len(todo), bs)]
        lock = threading.Lock()

        def worker(url):
            while True:
                with lock:
                    if not chunks:
                        return
                    chunk = chunks.pop(0)
                body = {"model": served_name, "prompt": [t for _, t in chunk], "temperature": 0.0,
                        "max_tokens": int(kc.get("self_generated_max_tokens", 2048))}
                req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=7200) as r:
                    resp = json.loads(r.read())
                with lock:
                    for ch in sorted(resp["choices"], key=lambda c: c.get("index", 0)):
                        sha = chunk[int(ch.get("index", 0))][0]
                        have[sha] = {"sha": sha, "response": ch.get("text") or "", "finish_reason": ch.get("finish_reason")}
                        append_jsonl(cache, have[sha])
        if len(urls) == 1:
            worker(urls[0])
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(urls)) as ex:
                list(ex.map(worker, urls))
    out = []
    for sha, (d, t) in zip(shas, texts):
        r = have[sha]
        out.append((d, t + r["response"] + ("<|im_end|>" if r.get("finish_reason") == "stop" else "")))
    return out
