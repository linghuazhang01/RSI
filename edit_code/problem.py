"""One quantization search problem = (source model, size budget, the fixed KL set, the incumbent).

Paper A (fixed size; KL only since 2026-11-25): every problem quantizes the ORIGINAL BF16 model M_0 to the same budget;
generation g tries to beat the incumbent (lowest-KL quantization accepted so far; loop.start bf16: until one is
accepted the incumbent is M_0 itself and the lowest-KL in-budget candidate is accepted). It owns: the model spec, the
layer cache + diagnostics (prepare), config validation / sizing, candidate build (artifact), lazy checkpoint
materialization, candidate serving, and the ONE evaluation a config gets: the per-trace token KL(M_0 || config) on
the fixed KL set (kl_eval.n items of generation 0's draw of the training pool, one thinking-mode answer each written
by M_0 in generation 0; the quantized model reads them, nothing is sampled; rsicore.kl). There is no screen, no
accuracy gate and no held-out test.

Checkpoint format (candidates.format, 2026-11-27; set by the inference profile in configs/base.yaml): dequant = the
fake-quant BF16 checkpoint (plain vLLM BF16); packed = the real GPTQ-format checkpoint served by vLLM's Humming kernels
(--quantization humming). Under action_space.packed (the packed and packed_dequant profiles) every config must be
writable as the packed checkpoint (schema.packed_problems) and the layer cache computes bf16 scales, so both formats of
one config give the same weights.

Candidate serving (2026-11-19): the candidate GPUs are split into groups (common.candidate_groups); each
group is one slot of a pool. serve() takes a free slot, exports the candidate's checkpoint into
that slot's directory on the slot's own GPU (exports of different slots run in parallel, one layer cache per slot),
starts vLLM on the slot's GPUs, and gives the slot back afterwards. So with N groups, N candidates are evaluated at
the same time (rsicore.search.inner_search(parallel=N)). evaluate() is thread-safe: an identical config
being evaluated by another thread is waited for (evaluated once).
"""
from __future__ import annotations

import contextlib
import json
import math
import queue
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import mock as M
from .common import (append_jsonl, candidate_args, candidate_groups, config_hash, free_cuda, kl_eval_cfg,
                     load_processed, read_json, read_jsonl, record_text, role_gpus, write_json, write_jsonl)
from .diagnostics import bytes_table, load_tables, save_tables
from .quant import schema, sizes
from .quant.spec import ModelSpec, spec_from_checkpoint, synthetic_spec

DATASET_DOMAIN = {"dapo_math": "math", "taco_verified": "code", "triviaqa_train": "knowledge",
                  "ifeval_train": "instruction_following", "sealtool_train": "tool_calling"}
# simple_<domain>_{train,pool,eval,test} (the run's data, 2026-11-21) belong to their domain
DATASET_DOMAIN.update({f"simple_{d}_{r}": d for d in ("math", "code", "knowledge", "instruction_following",
                                                       "tool_calling") for r in ("train", "pool", "eval", "test")})


def _setting(s: dict) -> tuple[int, str, int]:
    """(bits, quantizer, group_size) of a resolved module setting (defaults rtn, 128)."""
    return int(s["bits"]), s.get("quantizer", "rtn"), int(s.get("group_size") or 128)


def _herr_key(module: str, s: dict, tag: str = "") -> str:
    """The herr cache key of a module under a setting: "module|quantizer|bits|gs" (+ "|sbf16" for the bf16-scale
    cache of action_space.packed)."""
    b, q, gs = _setting(s)
    return f"{module}|{q}|{b}|{gs}{tag}"


def checkpoint_format(cfg: dict) -> str:
    """candidates.format: "dequant" (default) or "packed"."""
    from .quant.export import FORMATS
    fmt = str((cfg.get("candidates") or {}).get("format") or "dequant")
    if fmt not in FORMATS:
        raise ValueError(f"candidates.format must be one of {FORMATS}, got {fmt!r}")
    return fmt


PACKED_TP1 = ("inference=packed serves every candidate with tensor parallel 1: under --quantization humming vLLM 0.24.0 "
              "splits in_proj_ba across GPUs (it keeps it whole only for AWQ / GPTQ / INC configs) and loads a split "
              "shard's packed zero points at the wrong offset (2-bit gdn_ba at TP 2, 2/4-bit at TP 4, any at TP 8)")


def check_packed_run(cfg: dict, *, mock: bool, candidate_slots=None) -> str:
    """Before any server starts (run.Runner.__init__): the inference profile is consistent, packed candidates are
    served with TP 1, the current-mode editor ablation is not combined with packed checkpoints (a TP editor would
    serve the packed incumbent), and the vLLM environment has the humming kernels. Returns the checkpoint format."""
    space = dict(schema.DEFAULT_ACTION_SPACE, **(cfg.get("action_space") or {}))
    fmt = checkpoint_format(cfg)
    check_packed_space(space, fmt)
    if fmt == "packed":
        if int((cfg.get("gpu_layout") or {}).get("candidate_group_size", 1) or 1) > 1 or \
                any(len(s) > 1 for s in (candidate_slots or [])):
            raise ValueError(PACKED_TP1 + "; set gpu_layout.candidate_group_size: 1 (or use inference=packed_dequant)")
        if (cfg.get("loop") or {}).get("editor", "frozen") == "current":
            raise ValueError("loop.editor=current would serve the packed incumbent as the (tensor-parallel) editor; "
                             "use inference=packed_dequant for that ablation")
        if not mock:
            _humming_ok()
    return fmt


def check_packed_space(space: dict, fmt: str) -> None:
    """A packed checkpoint format needs the packed action space, and that space must allow only what the packed
    format can express (set together by the inference profiles of configs/base.yaml)."""
    if fmt == "packed" and not space.get("packed"):
        raise ValueError("candidates.format packed needs action_space.packed: true (use --set inference=packed)")
    if space.get("packed"):
        bad = [q for q in space["quantizers"] if q not in schema.PACKED_QUANTIZERS]
        bad += [f"{k}={v!r}" for k in ("embed", "lm_head") for v in space[k] if v != "bf16"]
        if bad:
            raise ValueError(f"action_space.packed allows only quantizers {list(schema.PACKED_QUANTIZERS)} and a BF16 "
                             f"embed / lm_head; remove {bad} (the inference=packed profile sets this)")


_HUMMING = {"ok": False}


def _humming_ok() -> None:
    """Once per process: the vLLM environment can import humming (candidates.format packed)."""
    if not _HUMMING["ok"]:
        from .quant.export import humming_check
        humming_check()
        _HUMMING["ok"] = True


class HerrDeadline(RuntimeError):
    """(2026-11-26) An exploration group_error ran out of its run_python call's time (tools.py turns it into a
    ToolError that says how many settings were computed and cached)."""


class QuantProblem:
    def __init__(self, rc, cfg: dict, *, tag: str, gen: int, out_dir: Path, prep_dir: Path,
                 source_dir: Path | None, source_state: dict | None, budget_gib: float,
                 datasets: list[str], kl_texts: list[tuple[str, str]] | None = None,
                 incumbent: dict | None = None, log=print, prep_gpus: list[int] | None = None):
        from rsicore import evaluate as _ev  # noqa: F401  (fail early if rsicore is missing)
        self.rc, self.cfg, self.tag, self.gen = rc, cfg, tag, gen
        self.mock = bool(rc.mock)
        self.out_dir, self.prep_dir = Path(out_dir), Path(prep_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = Path(source_dir) if source_dir else None
        self.source_state = source_state
        self.budget_gib = float(budget_gib)
        self.datasets = list(datasets)                    # the train role simple_<d>_pool (calibration texts)
        self.kl_texts = list(kl_texts or [])              # [(domain, text)]: calibration texts (generation 0 prep)
        # incumbent = lowest-KL accepted quantization so far: {"gen", "cid", "config", "kl", "kl_records", ...}
        self.incumbent = incumbent
        self.kl_cfg = kl_eval_cfg(cfg)
        self.log = log
        self.space = dict(schema.DEFAULT_ACTION_SPACE, **(cfg.get("action_space") or {}))
        self.fmt = checkpoint_format(cfg)          # candidate checkpoints: dequant | packed (2026-11-27)
        check_packed_space(self.space, self.fmt)
        self.registry = Path(rc.run_dir) / "mock_registry.json"
        self.spec: ModelSpec | None = None
        self.tables: dict | None = None
        self.cache = None
        self._lock = threading.RLock()
        # (2026-11-23) exports run in parallel, one layer cache per candidate slot GPU (_slot_cache); this lock now
        # only serialises herr_module's GPU use of self.cache
        self._export_lock = threading.Lock()
        self._slot_caches: dict = {}               # slot index -> (QuantCache on the slot's first GPU, its lock)
        self.exported: dict = {}                   # config hash -> a finished candidate export (served by serve_all)
        self._inflight: dict = {}                  # (hash, stage) -> threading.Event
        self.slots = candidate_groups(rc, cfg)     # [[GPU ids]] one candidate server per slot
        self._removed: set = set()                 # slot indices taken back (add_slot / remove_slot)
        # candidates evaluated at the same time (run.Runner.run_search sets it to the threads of a step); None = one
        # slot per candidate. See kl_slots().
        self.kl_parallel: int | None = None
        self.prep_gpus = list(prep_gpus) if prep_gpus else None   # generation-0 prep on these GPUs (one worker each)
        self._pool: queue.Queue = queue.Queue()
        for i in range(len(self.slots)):
            self._pool.put(i)
        self._kl_items: list | None = None
        self._evals = {r["hash"]: r for r in read_jsonl(self.out_dir / "evals.jsonl") if r.get("stage") == "kl"}
        self._herr_lock = threading.Lock()
        self._herr: dict | None = None             # {"module|q|b|gs": [h_err, tr(W H W^T), tokens]} (persisted)
        # (2026-11-26) GPUs of the editor's exploration GPU tools (run.Runner sets them for the search; [] = the layer
        # cache on the first candidate GPU): a pool, one layer cache + lock per GPU, see tool_device()
        self.tool_gpus: list[int] = []
        self._tool_pool: queue.Queue | None = None
        self._tool_caches: dict = {}
        self._herr_dirty = False

    # ------------------------------------------------------------------ format (2026-11-27)
    @property
    def packed_space(self) -> bool:
        """action_space.packed: configs must be writable as the packed checkpoint; the cache uses bf16 scales."""
        return bool((getattr(self, "space", None) or {}).get("packed"))

    @property
    def checkpoint_fmt(self) -> str:
        return getattr(self, "fmt", "dequant")

    @property
    def _herr_tag(self) -> str:
        return "|sbf16" if self.packed_space else ""

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

    def kl_slots(self) -> int:
        """Slots ONE candidate's KL evaluation may use (gpu_layout.gate_split, default on): ceil(n_slots /
        kl_parallel), e.g. 8 slots and 8 proposals per step -> 1 GPU each; the traces are spread over the replicas."""
        if not self.kl_parallel or not (self.cfg.get("gpu_layout") or {}).get("gate_split", True):
            return 1
        return max(1, math.ceil(self.n_slots / max(1, int(self.kl_parallel))))

    # ------------------------------------------------------------------ preparation
    def prepare(self):
        """Spec, diagnostics and (real mode) the layer cache for the source model; cached in prep_dir."""
        if self.spec is not None and self.tables and self.tables.get("hessian") is not None \
                and (self.mock or self.cache is not None):     # already prepared (run.export calls it again)
            return self
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
        if not self.mock and self.checkpoint_fmt == "packed":
            _humming_ok()
        if self.tables.get("hessian") is None:     # built after the layer cache (it reads cached entries)
            self.tables["hessian"] = self.hessian_table()
            write_json(self.prep_dir / "diagnostics" / "hessian_table.json", self.tables["hessian"])
        return self

    # ------------------------------------------------------------------ Hessian-weighted error
    def _herr_store(self) -> dict:
        if self._herr is None:
            self._herr = read_json(self.prep_dir / "diagnostics" / "herr_cache.json", {}) or {}
        return self._herr

    def herr_module(self, module: str, setting: dict, cache=None, lock=None,
                    deadline: float | None = None) -> tuple[float, float, int]:
        """(tr((W - Wq) H (W - Wq)^T), tr(W H W^T), calibration tokens) of one module under a resolved
        setting; H = the module's input Hessian from the calibration pass (sum over tokens). Memoised and
        persisted (prep/diagnostics/herr_cache.json). bits 16 -> 0 error. cache / lock: a worker's own layer
        cache on its GPU (hessian_table over several GPUs); default self.cache under the export lock."""
        key, (b, q, gs) = _herr_key(module, setting, self._herr_tag), _setting(setting)
        with self._herr_lock:
            got = self._herr_store().get(key)
        if got is not None:
            return tuple(got)
        if b >= 16:
            return (0.0, 1.0, 1)
        if deadline is not None and time.monotonic() >= deadline:   # (2026-11-26) an exploration call's time is up
            raise HerrDeadline(module)
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

    def _group_herr(self, mods, setting: dict, cache=None, lock=None,
                    deadline: float | None = None) -> tuple[float, float, int]:
        """herr_module summed over a group's modules: (error, tr(W H W^T), tokens of the last module, at least 1)."""
        err = ref = 0.0
        tok = 1
        for m in mods:
            e, r, n = self.herr_module(m.name, setting, cache=cache, lock=lock, deadline=deadline)
            err, ref, tok = err + e, ref + r, max(1, n)
        return err, ref, tok

    def _herr_flush(self) -> None:
        with self._herr_lock:
            if self._herr_dirty:
                write_json(self.prep_dir / "diagnostics" / "herr_cache.json", self._herr_store())
                self._herr_dirty = False

    def herr_missing(self, config_or_norm) -> list[str]:
        """Module settings ("module|quantizer|bits|gs") of a config whose Hessian-weighted error is not in the
        herr cache yet (group_error would compute them: lazy quantization, GPU). bits 16 is never missing.
        Used by the editor's exploration API (tools.make_tool_api(explore=True))."""
        norm = self._as_norm(config_or_norm)
        groups = self.spec.groups()
        out = []
        with self._herr_lock:
            store = self._herr_store()
            for key, s in norm["groups"].items():
                if _setting(s)[0] >= 16:
                    continue
                out.extend(k for k in (_herr_key(m.name, s, self._herr_tag) for m in groups[key]) if k not in store)
        return out

    @contextlib.contextmanager
    def tool_device(self, deadline: float | None = None):
        """(cache, lock) for ONE exploration GPU-tool call (2026-11-26): a free GPU of tool_gpus with its own layer
        cache (waits while every tool GPU is busy, so several editor sessions use different GPUs at once; until
        `deadline` (time.monotonic), then HerrDeadline); (None, None) without tool GPUs or in mock mode = self.cache
        under the export lock, as before. cache.device cpu: the caches stay on the CPU (the pool only limits
        concurrency), as for the slot caches."""
        if not self.tool_gpus or self.mock:
            yield None, None
            return
        with self._lock:
            if self._tool_pool is None:
                self._tool_pool = queue.Queue()
                for g in self.tool_gpus:
                    self._tool_pool.put(int(g))
        try:
            g = self._tool_pool.get(timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise HerrDeadline("every exploration tool GPU stayed busy") from None
        try:
            with self._lock:
                got = self._tool_caches.get(g)
                if got is None:
                    dev = self._device()
                    dev = f"cuda:{g}" if str(dev).startswith("cuda") else dev
                    got = self._tool_caches[g] = (self._open_cache(device=dev), threading.Lock())
            yield got
        finally:
            self._tool_pool.put(g)

    def group_error(self, config_or_norm, top: int | None = None, cache=None, lock=None,
                    deadline: float | None = None) -> dict:
        """Hessian-weighted output error of every linear module group under a config (no forward pass):
        {"groups": {key: {"setting", "herr_per_token", "herr_rel", "share"}}, "total_per_token", "top"}.
        herr_per_token = tr(dW H dW^T) / calibration tokens (summed over the group's modules); herr_rel = that
        / tr(W H W^T) (scale-free); share = fraction of the config's total. Embedding / lm_head not included."""
        norm = self._as_norm(config_or_norm)
        groups = self.spec.groups()
        out, total = {}, 0.0
        for key, s in norm["groups"].items():
            try:
                err, ref, tok = self._group_herr(groups[key], s, cache=cache, lock=lock, deadline=deadline)
            except HerrDeadline:
                self._herr_flush()                 # what was computed so far is kept: a retry continues from it
                raise
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
                        err, ref, tok = self._group_herr(mods, s, cache=cache, lock=lock)
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
        import torch
        from .quant.cache import QuantCache
        # action_space.packed (2026-11-27): bf16 scales, the ones vLLM's Humming kernels use (cache files *_sbf16)
        sd = torch.bfloat16 if self.packed_space else torch.float16
        return QuantCache(self.prep_dir / "cache", self.spec, self.source_dir, hessian_dir=self.prep_dir / "hessians",
                          device=device or self._device(), damp=self.cfg["cache"].get("gptq_damp", 0.01),
                          lazy=self.cfg["cache"].get("lazy", True), scale_dtype=sd, log=self.log)

    def _warm_variants(self) -> list[dict]:
        """cache.warm without the quantizers the action space does not allow (awq under action_space.packed)."""
        allowed = (getattr(self, "space", None) or schema.DEFAULT_ACTION_SPACE)["quantizers"]
        return [v for v in (self.cfg["cache"].get("warm") or []) if v.get("quantizer") in allowed]

    def _prepare_real(self):  # GPU path on the cluster; exercised on CPU by tests/test_quant.py
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
        free_cuda()
        self.cache = self._open_cache()
        self.cache.warm(self._warm_variants())
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
        from .diagnostics import activation_table_real, profile_meta, profile_worker, saved_profile_rows
        from .quant.hessian import activation_summary, collect_stats, hessian_path, load_hessian, load_hf_model, \
            read_stats, save_hessian, tokenize_texts
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
            # Transformers may retain model references after calibration; offload weights
            # explicitly before the profile processes load their own GPU copies.
            for model in models:
                model.to("cpu")
            del model, models, calib
            import gc
            gc.collect()
            free_cuda()
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
                save_hessian(dst, H, absmax, n)
                stats[f"{layer}.{key}"] = activation_summary(H, n, absmax)   # no second read of the summed file
            # (2026-11-23) several keys at once (the reads dominate; the per-key sum order is unchanged, so the summed
            # Hessians are bitwise the same as before)
            with ThreadPoolExecutor(max_workers=min(k, 8)) as ex:
                list(ex.map(reduce, keys))
            shutil.rmtree(self.prep_dir / "hessians_parts", ignore_errors=True)
            self.log(f"[prep:{self.tag}] calibration done ({len(keys)} Hessians, {time.time() - t0:.0f}s)")
        stats = read_stats(hdir, keys, have=stats)     # + Hessians summed by an earlier (interrupted) prep
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
            rows = saved_profile_rows(rows_dir, key, meta)
            if rows is None:
                missing.append(key)
            else:
                prow += rows
        if missing:
            raise RuntimeError(f"profile table incomplete: {len(missing)} groups missing, e.g. {missing[:3]}")
        self.log(f"[prep:{self.tag}] profile table: {len(group_keys)} groups on {k} processes, {time.time() - t1:.0f}s")
        order = {g: i for i, g in enumerate(group_keys)}
        prof_rows = sorted(prow, key=lambda r: (order[r["key"]], r["bits"]))
        act = activation_table_real(self.spec, stats)
        free_cuda()
        mods = [m.name for m in self.spec.modules]
        caches = [self._open_cache(device=d) for d in devs]
        warm = self._warm_variants()
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

    def _as_norm(self, config_or_norm) -> dict:
        """A normalized config (a raw config, recognised by its "default" key, is normalized first)."""
        return self.normalize(config_or_norm) if "default" in config_or_norm else config_or_norm

    def size(self, config_or_norm) -> dict:
        return sizes.config_bytes(self._as_norm(config_or_norm), self.spec)

    def mock_state(self, norm: dict) -> dict:
        """Mock mode: the model state of a normalized config (mock.apply_config with mock.lam)."""
        return M.apply_config(self.spec, self.source_state, norm, (self.cfg.get("mock") or {}).get("lam", 1.2))

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
            st = self.mock_state(norm)
            art["mock_quality"] = round(st["quality"], 6)
        return art

    def norm_of(self, art: dict) -> dict:
        return read_json(self.out_dir / "configs" / f"{art['config_hash']}.json")

    # ------------------------------------------------------------------ candidate model
    def has_checkpoint(self, d) -> bool:
        """d holds a finished export in this problem's format (candidates.format)."""
        from .quant.export import read_marker
        info = read_marker(d) if d else None
        return bool(info) and info.get("format", "dequant") == self.checkpoint_fmt

    def materialize(self, art: dict, dst: Path | None = None, slot: int = 0) -> Path | None:
        """Write the candidate checkpoint (candidates.format: dequant or packed) on the first GPU of candidate slot
        `slot` (free at this point: its previous server has stopped). Default dst: out_dir/ckpt/slot<k>/<hash>, one
        checkpoint per slot on disk (candidates.keep_checkpoints <= 1). Returns its path."""
        if self.mock:
            return None
        from .quant.export import export_checkpoint
        sdir = self.out_dir / "ckpt" / f"slot{slot}"
        dst = Path(dst) if dst else sdir / art["config_hash"]
        if self.has_checkpoint(dst):
            return dst
        if dst.parent == sdir and self.cfg["candidates"].get("keep_checkpoints", 1) <= 1:
            for p in sdir.glob("*"):  # bounded disk: one candidate checkpoint per slot
                if p != dst:
                    shutil.rmtree(p, ignore_errors=True)
        cache, lock = self._slot_cache(slot)
        with lock:              # (2026-11-23) per slot: the exports of different slots run at the same time
            export_checkpoint(self.checkpoint_fmt, self.source_dir, dst, self.norm_of(art), self.spec, cache,
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
        """Move a finished candidate checkpoint of this config (out_dir/ckpt/slot<k>/<hash>, written for its KL
        by this problem) to dst instead of exporting it again (2026-11-23). The same export_fake_quant output: same
        source, normalized config and meta. False when there is none (a slot reused by a later candidate)."""
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
                    or str(info.get("source")) != str(self.source_dir) or info.get("format", "dequant") != self.checkpoint_fmt:
                continue
            dst = Path(dst)
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(d, dst)
            self.log(f"[export] reused the evaluated checkpoint {d} -> {dst}")
            return True
        return False

    def _check_tp(self, gpus) -> None:
        if self.checkpoint_fmt == "packed" and len(gpus) > 1:
            raise RuntimeError(PACKED_TP1 + f" (slot GPUs {list(gpus)})")

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
                from .quant.export import server_args
                self._check_tp(gpus)
                ckpt = self.materialize(art, slot=slot)
                with VLLMServer(ckpt, name, gpus, seed=self.rc.seed, log_dir=logs,
                                extra_args=candidate_args(self.cfg) + server_args(ckpt)) as srv:
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
        afterwards; keep=True, a candidate's KL evaluation (2026-11-24): its usual checkpoint out_dir/ckpt/slot<k>/<hash> on
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
                from .quant.export import server_args
                for s in slots:
                    self._check_tp(self.slots[s])
                inc = self.incumbent or {}
                ckpt = Path(inc["model_dir"]) if (inc.get("config_hash") == art["config_hash"] and inc.get("model_dir")
                                                  and self.has_checkpoint(inc["model_dir"])) else None
                ex = self.exported.get(art["config_hash"])
                if ckpt is None and ex and self.has_checkpoint(ex):   # e.g. Q_0's exported model
                    ckpt = Path(ex)
                if ckpt is None and keep:
                    ckpt = self.materialize(art, slot=slots[0])
                elif ckpt is None:
                    tmp = self.out_dir / "ckpt" / "shared" / art["config_hash"]
                    ckpt = self.materialize(art, dst=tmp, slot=slots[0])
                srvs = [VLLMServer(ckpt, name, self.slots[s], seed=self.rc.seed, log_dir=logs,
                                   extra_args=candidate_args(self.cfg) + server_args(ckpt)) for s in slots]
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

    # ------------------------------------------------------------------ evaluation (KL only, 2026-11-25)
    def kl_items(self) -> list[dict]:
        """The fixed KL set [{ds, id, prompt, text}]: M_0's own answers to the kl_eval.n items of generation 0's
        draw (run_dir/baseline/kl_gate/traces.json, written once in generation 0)."""
        if self._kl_items is None:
            tr = read_json(self.rc.run_dir / "baseline" / "kl_gate" / "traces.json", None)
            if not tr:
                raise RuntimeError("no KL reference traces (baseline/kl_gate/traces.json); generation 0 writes them")
            self._kl_items = list(tr["items"])
        return self._kl_items

    def evaluate(self, config_or_art, cid: str, *, origin: str = "candidate") -> dict:
        """The ONE evaluation of a config: the per-trace token KL(M_0 || config) on the fixed KL set (nothing is
        sampled; the quantized model reads M_0's traces). Cached by config hash (evals.jsonl); an identical config
        being evaluated by another thread is waited for. -> {"hash", "cid", "origin", "kl", "feedback", ...}."""
        art = config_or_art if "config_hash" in config_or_art else self.build(config_or_art, cid)
        key = art["config_hash"]
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
            return self._evaluate(art, cid, origin)
        finally:
            with self._lock:
                self._inflight.pop(key).set()

    def _evaluate(self, art: dict, cid: str, origin: str) -> dict:
        from rsicore.kl import token_kl
        t0 = time.time()
        name = f"{self.rc.run_id}-{self.tag}-{art['config_hash']}"
        edir = self.out_dir / "evals" / f"{cid}"
        # Q_0 (origin "reference") runs while nothing else holds a slot: every free slot; a candidate gets
        # kl_slots() slots (e.g. 8 slots, 8 proposals per step -> 1 each)
        if origin == "reference":
            ctx = self.serve_all(art, name)
        elif self.kl_slots() > 1:
            ctx = self.serve_all(art, name, most=self.kl_slots(), keep=True)
        else:
            ctx = self.serve(art, name)
        items = self.kl_items()
        with ctx as srv:
            srvs = srv if isinstance(srv, list) else [srv]
            r = token_kl(self.rc.run_dir / "baseline" / "kl_gate" / "ref.jsonl", [s.endpoint for s in srvs], name,
                         [it["text"] for it in items], top_k=int(self.kl_cfg.get("top_k", 20)))
        recs = [{"id": it["id"], "ds": it["ds"], "kl": round(k, 6), "n_positions": n}
                for it, k, n in zip(items, r["per_text"], r["n_positions"])]
        edir.mkdir(parents=True, exist_ok=True)
        write_jsonl(edir / "kl.jsonl", recs)
        kl = round(float(r["mean"]), 6)
        fb = {"bytes_gib": art["bytes_gib"], "budget_gib": art["budget_gib"],
              "linear_avg_bits": art["linear_avg_bits"], "kl": kl, "token_mean_kl": round(float(r["token_mean"]), 6),
              "kl_quantiles": _quantiles([x["kl"] for x in recs]), "n_traces": r["n_texts"],
              "n_tokens": r["n_tokens"], "top_k": r["top_k"], "records": str(edir / "kl.jsonl")}
        try:
            fb["error_top_groups"] = self.group_error(self.norm_of(art), top=5)["top"]
        except Exception:  # noqa: BLE001 - informative only
            pass
        rec = {"hash": art["config_hash"], "stage": "kl", "cid": cid, "origin": origin, "kl": kl, "feedback": fb,
               "seconds": round(time.time() - t0, 1)}
        with self._lock:
            self._evals[art["config_hash"]] = rec
            append_jsonl(self.out_dir / "evals.jsonl", rec)
        return dict(rec, feedback=dict(fb))

    @property
    def incumbent_is_original(self) -> bool:
        """loop.start bf16 and nothing accepted yet (2026-11-25): the incumbent is the original BF16 model M_0."""
        return bool((self.incumbent or {}).get("original"))

    @property
    def incumbent_kl(self) -> float | None:
        """The incumbent's mean KL on the KL set (M_0 itself: 0)."""
        inc = self.incumbent or {}
        return 0.0 if inc.get("original") else inc.get("kl")

    def kl_stats(self, art: dict, r: dict) -> dict:
        """One evaluation r for the editor / gate.json: kl (mean over the traces), reference_kl (the incumbent's),
        gain = reference_kl - kl (positive = closer to M_0 than the incumbent), passed = fits the budget and scored
        (acceptance picks the lowest KL at the end of the round, run.Runner.accept_lowest_kl), quantiles, size."""
        fb = r.get("feedback") or {}
        kl, ref = r.get("kl"), self.incumbent_kl
        ok = kl is not None and art.get("bytes_gib", 0) <= self.budget_gib + 1e-9
        why = (f"KL {kl:.6f} (incumbent {ref if ref is None else round(ref, 6)})" if kl is not None
               else "no KL (nothing scored)")
        return {"passed": ok, "reason": why, "metric": "kl", "kl": kl, "reference_kl": ref,
                "gain": None if (kl is None or ref is None) else round(ref - kl, 6),
                "kl_quantiles": fb.get("kl_quantiles"), "bytes_gib": art.get("bytes_gib")}


def _quantiles(xs: list[float]) -> dict | None:
    """{p10, p50, p90, max} of the per-trace KL values."""
    if not xs:
        return None
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]  # noqa: E731
    return {"p10": round(q(0.1), 6), "p50": round(q(0.5), 6), "p90": round(q(0.9), 6), "max": round(s[-1], 6)}

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
                       cache: Path, *, sampling: dict | None = None, max_tokens: int | None = None,
                       batch: int | None = None) -> list[tuple[str, str]]:
    """prompt + response: append the reference model's own greedy response to each KL / calibration prompt
    (2504.04823: reasoning-style calibration text), so the calibration statistics and the profile table's KL also
    cover the answer tokens. endpoint: a URL or a list of URLs (replicas: the batches are spread over them, 2026-11-20), or a
    callable returning one (called only when some response is not cached yet, so a resume needs no server).
    Requests go in batches of kl.self_generated_batch prompts. Cached as jsonl
    {sha, response, finish_reason}; <|im_end|> is appended only to responses that finished (not cut by
    kl.self_generated_max_tokens). sampling (the KL set's reference traces): {temperature, top_p, top_k,
    seed} instead of greedy; max_tokens / batch override the kl block's."""
    import hashlib
    import urllib.request
    kc = cfg["kl"]
    have = {}
    if Path(cache).exists():                  # tolerant of a torn last line (crash mid-write): that prompt is redone
        for line in Path(cache).read_text().splitlines():
            try:
                r = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                r = None
            if r and r.get("sha"):
                have[r["sha"]] = r
        if Path(cache).stat().st_size and not Path(cache).read_bytes().endswith(b"\n"):
            with open(cache, "a") as f:
                f.write("\n")
    shas = [hashlib.sha256(t.encode()).hexdigest() for _, t in texts]
    todo = [(sha, t) for sha, (_, t) in zip(shas, texts) if sha not in have]
    todo = list(dict(todo).items())                        # one request per distinct prompt
    if todo:
        eps = endpoint() if callable(endpoint) else endpoint
        urls = [e.rstrip("/") + "/completions" for e in (eps if isinstance(eps, (list, tuple)) else [eps])]
        bs = max(1, int(batch or kc.get("self_generated_batch", 64)))
        chunks = [todo[b:b + bs] for b in range(0, len(todo), bs)]
        lock = threading.Lock()

        def worker(url):
            while True:
                with lock:
                    if not chunks:
                        return
                    chunk = chunks.pop(0)
                body = {"model": served_name, "prompt": [t for _, t in chunk], "temperature": 0.0,
                        "max_tokens": int(max_tokens or kc.get("self_generated_max_tokens", 2048))}
                for k_, v_ in (sampling or {}).items():   # sampled reference traces (seeded) instead of greedy
                    if v_ is not None:
                        body[k_] = v_
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
