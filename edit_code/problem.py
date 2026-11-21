"""One quantization search problem = (source model, size budget, evaluation data, references).

Paper A (fixed size, 2026-11-19): every problem quantizes the ORIGINAL BF16 model M_0 to the same
budget; generation g tries to beat the incumbent (best accepted quantization so far). It owns: the model spec, the layer
cache + diagnostics (prepare), config validation / sizing, candidate build (artifact), lazy
fake-quant materialization, candidate serving, and staged evaluation (screen / full / gate) with
structured L1 feedback (RSI_LOOP_DESIGN §2.2).

References (2026-11-20): every candidate is compared with the INCUMBENT on the same items (it is what a
candidate must beat): retention = candidate accuracy / incumbent accuracy, paired deltas and regression
examples (incumbent right, candidate wrong) are vs the incumbent. M_0 is no longer scored on the signal
items; it only provides the KL reference (prompt logprobs on the KL texts, compute_refs). Per candidate
the per-text KL (kl_per_text.json) and the per-group Hessian-weighted error (group_error.json) are kept.

Candidate serving (2026-11-19): the candidate GPUs are split into groups (common.candidate_groups); each
group is one slot of a pool. serve() takes a free slot, exports the candidate's fake-quant checkpoint into
that slot's directory on the slot's own GPU (exports are serialized: one shared layer cache), starts vLLM
on the slot's GPUs, and gives the slot back afterwards. So with N groups, N candidates are evaluated at
the same time (rsicore.search.inner_search(parallel=N)). evaluate() is thread-safe: an identical config
being evaluated by another thread is waited for (evaluated once), and charged screen evaluations reserve
their budget before they start.
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
from .common import (append_jsonl, candidate_groups, config_hash, free_cuda, load_processed, read_json, read_jsonl,
                     record_text, role_gpus, tail, write_json)
from .diagnostics import bytes_table, load_tables, save_tables, summarize_profile
from .quant import schema, sizes
from .quant.spec import ModelSpec, spec_from_checkpoint, synthetic_spec

DATASET_DOMAIN = {"dapo_math": "math", "taco_verified": "code", "triviaqa_train": "knowledge",
                  "ifeval_train": "instruction_following", "sealtool_train": "tool_calling"}


class EvalBudgetExhausted(RuntimeError):
    pass


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
                 datasets: list[str], data: dict, kl_texts: list[tuple[str, str]], kl_heldout: list[tuple[str, str]] = (),
                 refs_dir: Path, noise: dict | None = None, gate_baseline: dict | None = None,
                 eval_budget: int = 24, show_examples: int = 2, incumbent: dict | None = None, log=print,
                 prep_gpus: list[int] | None = None, gate_datasets: list[str] | None = None):
        from rsicore import evaluate as _ev  # noqa: F401  (fail early if rsicore is missing)
        self.rc, self.cfg, self.tag, self.gen = rc, cfg, tag, gen
        self.mock = bool(rc.mock)
        self.out_dir, self.prep_dir, self.refs_dir = Path(out_dir), Path(prep_dir), Path(refs_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = Path(source_dir) if source_dir else None
        self.source_state = source_state
        self.budget_gib = float(budget_gib)
        # data roles (2026-11-21, rsicore.draws.roles): datasets = the train role (screen / full items, feedback,
        # compare), gate_datasets = the gate role (gate stage; default: the same datasets)
        self.datasets = list(datasets)
        self.gate_datasets = list(gate_datasets) if gate_datasets else list(self.datasets)
        self.data = data                                  # {dataset: {"screen": ids, "signal": ids}}
        self.kl_texts = list(kl_texts)                    # [(domain, text)]
        self.kl_heldout = list(kl_heldout)
        self.noise = noise or {}
        self.gate_baseline = gate_baseline or {}
        # incumbent = best accepted quantization so far: {"gen", "cid", "config", "gate": {ds: {"score", "records"}}}
        self.incumbent = incumbent
        self.incumbent_scores: dict = {}                  # {"screen": x, "full": y} on this problem's items
        self.eval_budget = int(eval_budget)
        self.show_examples = int(show_examples)
        self.log = log
        self.space = dict(schema.DEFAULT_ACTION_SPACE, **(cfg.get("action_space") or {}))
        self.registry = Path(rc.run_dir) / "mock_registry.json"
        self.spec: ModelSpec | None = None
        self.tables: dict | None = None
        self.cache = None
        self.shown: dict[str, set] = {}
        self._lock = threading.RLock()
        self._export_lock = threading.Lock()       # one export at a time (shared layer cache)
        self._inflight: dict = {}                  # (hash, stage) -> threading.Event
        self._reserved = 0                         # charged screen evaluations running now
        self.slots = candidate_groups(rc, cfg)     # [[GPU ids]] one candidate server per slot
        self._removed: set = set()                 # slot indices taken back (add_slot / remove_slot)
        self.prep_gpus = list(prep_gpus) if prep_gpus else None   # generation-0 prep on these GPUs (one worker each)
        from rsicore.draws import is_eval_set
        # an eval-set train role (objective eval_overfit, or data_roles.train = an eval set): the editor may see
        # every item, so feedback examples also show the reference answer
        self._eval_mode = any(is_eval_set(ds) for ds in self.datasets)
        self._pool: queue.Queue = queue.Queue()
        for i in range(len(self.slots)):
            self._pool.put(i)
        self._evals = {}
        for r in read_jsonl(self.out_dir / "evals.jsonl"):
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

    @property
    def evals_used(self) -> int:
        return sum(1 for (h, st), r in self._evals.items() if st == "screen" and r.get("charged", True))

    @property
    def evals_left(self) -> int:
        return max(0, self.eval_budget - self.evals_used)

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
            with ThreadPoolExecutor(max_workers=k) as ex:
                models = list(ex.map(lambda d: load_hf_model(self.source_dir, d), devs))
            parts = [texts[i::k] for i in range(k)]

            def calib(i):
                b = tokenize_texts(self.source_dir, parts[i], ccfg.get("max_len", 2048))
                collect_stats(models[i], self.spec, b, self.prep_dir / "hessians_parts" / f"w{i}",
                              layers_per_pass=ccfg.get("layers_per_pass", 16), device=devs[i], log=self.log)
            with ThreadPoolExecutor(max_workers=k) as ex:
                list(ex.map(calib, range(k)))
            del models, calib
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            for (layer, key) in keys:                  # sum the workers' Hessians
                dst = hessian_path(hdir, layer, key)
                if not dst.exists():
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
            shutil.rmtree(self.prep_dir / "hessians_parts", ignore_errors=True)
            self.log(f"[prep:{self.tag}] calibration done ({len(keys)} Hessians, {time.time() - t0:.0f}s)")
        for (layer, key) in keys:
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

    def build(self, config, cid: str) -> dict:
        """Validate + size a config -> artifact dict (raises schema.ConfigError / ValueError)."""
        norm = self.normalize(config)
        b = sizes.config_bytes(norm, self.spec)
        if b["gib"] > self.budget_gib + 1e-9:
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
        with self._export_lock:
            dev = self.cache.device
            if str(dev).startswith("cuda"):
                self.cache.device = f"cuda:{self.slots[slot][0]}"
            try:
                export_fake_quant(self.source_dir, dst, self.norm_of(art), self.spec, self.cache,
                                  meta={"tag": self.tag, "gen": self.gen, "budget_gib": self.budget_gib}, log=self.log)
            finally:
                self.cache.device = dev
            free_cuda()   # the candidate vLLM server starts on this GPU next (80 GB cards: every GB counts)
        return dst

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
                # no --tasks: candidates are also scored on held-out tasks (E_t); the index is cached
                with VLLMServer("mock-" + name, name, gpus, mock=True, mock_registry=self.registry,
                                seed=self.rc.seed, log_dir=logs) as srv:
                    yield srv
            else:  # pragma: no cover - GPU path
                ckpt = self.materialize(art, slot=slot)
                with VLLMServer(ckpt, name, gpus, seed=self.rc.seed, log_dir=logs,
                                extra_args=self.cfg["vllm"].get("candidate_args", [])) as srv:
                    yield srv
        finally:
            self._pool.put(slot)

    # ------------------------------------------------------------------ evaluation
    def stage_datasets(self, stage: str) -> list[str]:
        """Datasets scored at `stage`: the gate role for gate, the train role for screen / full."""
        return self.gate_datasets if stage == "gate" else self.datasets

    def _ids(self, ds: str, stage: str) -> list[str]:
        if stage == "gate":
            from rsicore.draws import gate_ids
            return gate_ids(self.rc, ds)
        return list(self.data[ds]["signal" if stage == "full" else "screen"])

    def kl_eval_texts(self) -> list[tuple[str, str]]:
        """The KL texts scored for every candidate: kl.eval_texts of them, evenly spaced over the list (so every
        domain keeps its share); null = all. Calibration (prepare) always uses every text."""
        n = self.cfg["kl"].get("eval_texts")
        if n is None or int(n) >= len(self.kl_texts):
            return self.kl_texts
        step = len(self.kl_texts) / int(n)
        return [self.kl_texts[int(i * step)] for i in range(int(n))]

    def _kl(self, srv, art: dict, stage: str, heldout: bool = False) -> dict | None:
        texts = self.kl_heldout if heldout else self.kl_eval_texts()
        if not texts:
            return None
        if self.mock:
            st = M.apply_config(self.spec, self.source_state, self.norm_of(art), self.cfg["mock"].get("lam", 1.2))
            inc = st["kl_vs_parent"]
            out = {"overall": round(inc, 5)}
            for d in sorted({d for d, _ in texts}):
                out[d] = round(inc * M.DOMAIN_KL.get(d, 1.0), 5)
            out["vs_original"] = round(st["kl_vs_original"], 5)
            out["per_text"] = [round(inc * M.DOMAIN_KL.get(d, 1.0) * (0.5 + M._h("klt", art["config_hash"], i)), 6)
                               for i, (d, _) in enumerate(texts)]
            return out
        from rsicore.evaluate import token_kl  # pragma: no cover - GPU path
        ref = self.refs_dir / ("kl_ref_heldout.jsonl" if heldout else "kl_ref.jsonl")
        r = token_kl(ref, srv.endpoint, srv.served_name, [t for _, t in texts], top_k=self.cfg["kl"]["top_k"])
        out = {"overall": round(r["mean"], 5), "token_mean": round(r["token_mean"], 5),
               "per_text": [None if v is None else round(v, 6) for v in r["per_text"]]}
        by = {}
        for (d, _), v in zip(texts, r["per_text"]):
            by.setdefault(d, []).append(v)
        for d, v in by.items():
            out[d] = round(sum(v) / len(v), 5)
        return out

    def evaluate(self, config_or_art, stage: str, cid: str, *, charge: bool = True, origin: str = "candidate") -> dict:
        """Evaluate a config at `stage` (screen | full | gate). Cached by (config hash, stage).
        Returns {"score", "feedback", ...}. Screen evaluations count against the evaluate budget."""
        art = config_or_art if "config_hash" in config_or_art else self.build(config_or_art, cid)
        key = (art["config_hash"], stage)
        charged = bool(charge and stage == "screen")
        while True:
            with self._lock:
                if key in self._evals:
                    r = dict(self._evals[key])
                    r["cached"] = True
                    return r
                busy = self._inflight.get(key)
                if busy is None:   # this thread evaluates it
                    if charged and self.evals_left - self._reserved <= 0:
                        raise EvalBudgetExhausted(f"evaluate budget of this generation exhausted ({self.eval_budget})")
                    self._inflight[key] = threading.Event()
                    self._reserved += int(charged)
                    break
            busy.wait()            # another thread is evaluating the same config: use its result
        try:
            return self._evaluate(art, stage, cid, key, charged, origin)
        finally:
            with self._lock:
                self._reserved -= int(charged)
                self._inflight.pop(key).set()

    def _evaluate(self, art: dict, stage: str, cid: str, key: tuple, charged: bool, origin: str) -> dict:
        t0 = time.time()
        from rsicore.evaluate import paired_delta, score
        name = f"{self.rc.run_id}-{self.tag}-{art['config_hash']}"
        edir = self.out_dir / "evals" / f"{cid}" / stage
        probes, regressions, scored = {}, [], {}
        # the same datasets AND items as an existing screen evaluation of this config (objective eval_overfit: every
        # stage is the whole eval set): reuse its records instead of scoring them again. A gate on another dataset
        # than the train role (data_roles) is always scored.
        dsets = self.stage_datasets(stage)
        scr = self._evals.get((art["config_hash"], "screen")) if stage != "screen" else None
        reuse = None
        if scr is not None and dsets == self.datasets \
                and all(sorted(self._ids(ds, stage)) == sorted(self._ids(ds, "screen")) for ds in dsets):
            src = {ds: self.records_path(scr["cid"], "screen", ds) for ds in dsets}
            if all(p.exists() for p in src.values()):
                reuse = {ds: read_jsonl(p) for ds, p in src.items()}
        from contextlib import nullcontext
        with (nullcontext() if reuse else self.serve(art, name)) as srv:
            for ds in dsets:
                ids = self._ids(ds, stage)
                if reuse:
                    edir.mkdir(parents=True, exist_ok=True)
                    (edir / f"{ds}.jsonl").write_text("".join(__import__("json").dumps(r, ensure_ascii=False) + "\n"
                                                              for r in reuse[ds]))
                    res = SimpleNamespace(records=reuse[ds])
                else:
                    res = score(srv.endpoint, name, ds, pool="pool", ids=ids, out_dir=edir, mock=self.mock)
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
                        nb = self.noise.get(ds)
                        p["noise_band"] = nb
                        if d["delta"] is not None:
                            lo, hi = d["ci"]
                            p["significant"] = bool(abs(d["delta"]) > (nb or 0.0) and (lo > 0 or hi < 0))
                        if self.show_examples and stage == "screen":
                            regressions += self._regressions(ds, pref, res.records)
                        if pacc:
                            p["retention"] = None if acc is None else round(acc / pacc, 4)
                probes[dom] = p
                scored[ds] = acc
            klw = float(self.cfg["search"].get("kl_weight", 0.0))
            if reuse:
                kl = (scr.get("feedback") or {}).get("kl") if stage in ("screen", "full") else None
            else:   # kl_weight 0 (objective eval_overfit): the score does not use KL, so it is not computed
                kl = self._kl(srv, art, stage) if (stage in ("screen", "full") and klw > 0) else None
        if kl and "per_text" in kl:                # kept on disk, not in the editor's feedback
            kl = dict(kl)
            per_text = kl.pop("per_text")
            write_json(edir / "kl_per_text.json",
                       [{"i": i, "domain": d, "kl": v} for i, ((d, _), v) in enumerate(zip(self.kl_eval_texts(), per_text))])
        fb = {"stage": stage, "bytes_gib": art["bytes_gib"], "budget_gib": art["budget_gib"],
              "linear_avg_bits": art["linear_avg_bits"], "kl": kl, "probes": probes}
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
            klw = self.cfg["search"].get("kl_weight", 0.0)
            klv = (kl or {}).get("overall") or 0.0
            sc = None if ret is None else round(ret - klw * klv, 5)
            if sc is None:  # nothing scored (e.g. code tasks without Docker): rank by KL only
                sc = round(1.0 - klw * klv, 5) if kl else None
            fb["retention"] = None if ret is None else round(ret, 4)
        rec = {"hash": art["config_hash"], "stage": stage, "cid": cid, "origin": origin, "score": sc, "feedback": fb,
               "charged": charged, "seconds": round(time.time() - t0, 1)}
        with self._lock:
            self._evals[key] = rec
            append_jsonl(self.out_dir / "evals.jsonl", rec)
        out = dict(rec)
        out["feedback"] = dict(fb, budget_left={"evaluate": self.evals_left})
        return out

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
        if mean < float(g.get("min_delta", 0.0)):
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

    def compute_refs(self, endpoint: str, served_name: str) -> None:
        """M_0 (served as the editor) is only the KL reference: cache its prompt logprobs on the KL texts
        (real mode). It is not scored on the signal items (candidates are compared with the incumbent)."""
        if self.mock or float(self.cfg["search"].get("kl_weight", 0.0)) <= 0:   # no KL in the score: no reference
            return
        from rsicore.evaluate import reference_logprobs  # pragma: no cover - GPU path
        k = self.cfg["kl"]["top_k"]
        if self.kl_texts:
            reference_logprobs(endpoint, served_name, [t for _, t in self.kl_eval_texts()],
                               self.refs_dir / "kl_ref.jsonl", k)
        if self.kl_heldout:
            reference_logprobs(endpoint, served_name, [t for _, t in self.kl_heldout],
                               self.refs_dir / "kl_ref_heldout.jsonl", k)

    def evaluate_incumbent(self, stages=("screen", "full")) -> dict:
        """Score the incumbent on this problem's items (uncharged; cid 'incumbent'): the reference of every
        candidate's retention, paired deltas and regression examples. -> {stage: score}."""
        if not self.incumbent:
            return {}
        art = self.build(self.incumbent["config"], "incumbent")
        self.incumbent_scores = {st: self.evaluate(art, st, "incumbent", charge=False, origin="incumbent")["score"]
                                 for st in stages}
        return self.incumbent_scores

    # ------------------------------------------------------------------ analysis (editor tool)
    def compare(self, cid: str, other: str = "incumbent", stage: str = "screen", max_ids: int = 50) -> dict:
        """Item-level comparison of two evaluated candidates on the same items, WITHOUT item content (ids
        and statistics only, so the signal items are not used up): per dataset, accuracy of both, flips
        (other right -> cid wrong = broken, other wrong -> cid right = fixed) with their ids, flips by the
        reference's output length (tercile buckets), truncation rate and mean output length of both."""
        out = {"cid": cid, "other": other, "stage": stage, "datasets": {}}
        for ds in self.stage_datasets(stage):
            pa, pb = self.records_path(cid, stage, ds), self.records_path(other, stage, ds)
            if not pa.exists() or not pb.exists():
                missing = [c for c, q in ((cid, pa), (other, pb)) if not q.exists()]
                raise ValueError(f"no {stage} records for {missing} on {ds} (evaluated candidate ids only; "
                                 f"'incumbent' = the incumbent)")
            A = {r["id"]: r for r in read_jsonl(pa) if r.get("score") is not None}
            B = {r["id"]: r for r in read_jsonl(pb) if r.get("score") is not None}
            ids = sorted(set(A) & set(B))
            broken = [i for i in ids if B[i]["score"] > 0.5 >= A[i]["score"]]
            fixed = [i for i in ids if A[i]["score"] > 0.5 >= B[i]["score"]]

            def olen(r):
                o = r.get("raw_output")
                return len(o if isinstance(o, str) else " ".join(map(str, o or [])))

            def trunc(r):
                return any(f == "length" for f in (r.get("finish_reason") or []))
            lens = sorted(olen(B[i]) for i in ids)
            cut1, cut2 = (lens[len(lens) // 3], lens[2 * len(lens) // 3]) if lens else (0, 0)

            def bucket(i):
                n = olen(B[i])
                return "short" if n <= cut1 else ("medium" if n <= cut2 else "long")
            by = {k: {"n": 0, "broken": 0, "fixed": 0} for k in ("short", "medium", "long")}
            for i in ids:
                by[bucket(i)]["n"] += 1
            for i in broken:
                by[bucket(i)]["broken"] += 1
            for i in fixed:
                by[bucket(i)]["fixed"] += 1
            n = max(len(ids), 1)
            out["datasets"][ds] = {
                "n": len(ids), "acc": round(sum(A[i]["score"] for i in ids) / n, 4),
                "acc_other": round(sum(B[i]["score"] for i in ids) / n, 4),
                "broken": len(broken), "fixed": len(fixed), "net": len(fixed) - len(broken),
                "broken_ids": broken[:max_ids], "fixed_ids": fixed[:max_ids],
                "by_reference_length": by,
                "truncated": round(sum(trunc(A[i]) for i in ids) / n, 4),
                "truncated_other": round(sum(trunc(B[i]) for i in ids) / n, 4),
                "mean_output_chars": round(sum(olen(A[i]) for i in ids) / n, 1),
                "mean_output_chars_other": round(sum(olen(B[i]) for i in ids) / n, 1)}
        return out

    def heldout_kl(self, art: dict) -> dict | None:
        if not self.kl_heldout:
            return None
        with self.serve(art, f"{self.rc.run_id}-{self.tag}-{art['config_hash']}-hk") as srv:
            return self._kl(srv, art, "full", heldout=True)


# ---------------------------------------------------------------------------------- data helpers
def kl_texts_for(rc, gen: int, datasets_all: list[str], cfg: dict) -> list[tuple[str, str]]:
    """(domain, text) KL / calibration contexts from the draw's `kl` role."""
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
    (2504.04823: reasoning-style calibration text), so KL and the calibration statistics also cover the answer
    tokens. endpoint: a URL or a list of URLs (replicas: the batches are spread over them, 2026-11-20), or a
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
