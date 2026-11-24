"""Evaluation of LoRA candidates on the served model M_t (vLLM multi-LoRA, hot-loaded adapters).

Levels: `screen` (this generation's screen draw of the target domain), `full` (the signal draw; = screen in the
simple flow), `gate` (the fixed gate set of the target domain). Every result is cached on disk per (name, level), so
a resumed run never re-evaluates. Feedback follows RSI_LOOP_DESIGN §3.4 (target domain only).
datasets (2026-11-21): the screen / full levels hold the train dataset (`target_dataset`, simple_<d>_pool), the gate
level the gate dataset (`gate_dataset`, simple_<d>_eval; default the same); a level's target is the one it holds.
decoding (2026-11-21): `decoding` = {level: rsicore decoding dict | None} (rsicore.config.stage_decoding; None =
tasks.yaml greedy). The parent (the current model) and every candidate of a level are scored with the same dict,
seed included; a level's cache directory carries a tag of its decoding (evals/<name>/<level>@<tag>/, plain <level>/
when greedy), and a level shares the screen results only with equal items AND equal decoding. Each (model, level,
dataset) is scored once, also when several candidates are screened at once (a lock per key): the parent's screen /
gate results are sampled once per generation and shared by every candidate. With n > 1 a record's score is the mean
of its samples (avg@n); compact records keep the first sample's text, the mean length and the truncated fraction.
"""
from __future__ import annotations

import json
import random
import threading
from collections import OrderedDict
from pathlib import Path

from .common import load_records, read_json, stable_hash, write_json


def mean_ci(scores: list[float], seed: int = 42, n_boot: int = 500) -> tuple[float | None, list]:
    if not scores:
        return None, [None, None]
    m = sum(scores) / len(scores)
    rng = random.Random(seed)
    n = len(scores)
    bs = sorted(sum(scores[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return m, [bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]]


def _multi(rec: dict) -> bool:
    """A compact record scored with several samples (avg@n, 2026-11-21)."""
    return int(rec.get("ns") or 1) > 1


def _trunc(rec: dict) -> float:
    """Fraction of a compact record's samples cut at max_tokens."""
    if rec.get("trunc") is not None:
        return float(rec["trunc"])
    return float(rec.get("finish") == "length")


def _compact(r: dict) -> dict:
    """Compact record of one run_eval record; raw_output / finish_reason are lists when n > 1 (2026-11-21)."""
    outs, fins = r.get("raw_output"), r.get("finish_reason")
    out = {"id": r.get("id"), "score": r.get("score"), "out": _short(outs, 600),
           "finish": (fins or [None])[0] if isinstance(fins, list) else fins,
           "parsed": r.get("parsed_answer") is not None, "status": r.get("status")}
    if isinstance(outs, list) and len(outs) > 1:
        out["len"] = round(sum(len(str(o or "")) for o in outs) / len(outs))
        out["ns"] = len(outs)
        fl = fins if isinstance(fins, list) else [fins]
        out["trunc"] = round(sum(f == "length" for f in fl) / max(len(fl), 1), 4)
        if r.get("truncated_thinking"):
            out["truncated_thinking"] = r["truncated_thinking"]
    else:
        out["len"] = len(str((outs[0] if outs else "") if isinstance(outs, list) else (outs or "")))
    return out


def _short(s, n=400):
    if s is None:
        return None
    if isinstance(s, list):
        s = s[0] if s else ""
    s = str(s)
    return s if len(s) <= n else s[: n // 2] + " ... " + s[-n // 2:]


class Evaluator:
    def __init__(self, rc, cfg: dict, *, out_dir: Path, endpoint: str, served: str, server, ids: dict,
                 target_dataset: str, store, mock: bool, registry: Path | None,
                 parent_quality: float | dict | None = None, max_loaded: int = 6,
                 replicas: list | None = None, weights_scorer=None, gate_dataset: str | None = None,
                 decoding: dict | None = None):
        self.rc, self.cfg = rc, cfg
        self.decoding = {k: v for k, v in (decoding or {}).items() if v}   # level -> decoding dict (None = greedy)
        self._key_locks: dict[tuple, threading.Lock] = {}
        self._busy: dict[str, int] = {}        # adapters being scored right now (never evicted meanwhile)
        self.out_dir = Path(out_dir)
        self.endpoint, self.served, self.server = endpoint, served, server
        # replicas of the served model (2026-11-20, data parallelism): every LoRA is loaded on all of them and the
        # items of a scoring job are split over them (rsicore.evaluate.score_split)
        self.replicas = list(replicas) if replicas else [server]
        if self.replicas[0] is not server and server in self.replicas:
            self.replicas.remove(server)
            self.replicas.insert(0, server)
        self.endpoints = [s.endpoint for s in self.replicas]
        # method.mode weights (2026-11-21): candidates are weight edits, not adapters. Real runs score them through
        # weights_scorer(name, path, {dataset: ids}, {dataset: out_dir}) -> {dataset: EvalResult} (export the edited
        # model, serve it on a free GPU slot); mock runs score them like adapters (registry quality by name).
        self.weights_scorer = weights_scorer
        self._kinds: dict[str, str] = {}
        self.ids = ids                      # level -> {dataset: [ids]}
        self.target = target_dataset        # screen / full target (the train dataset)
        self.gate_target = gate_dataset or target_dataset   # gate target (the gate dataset)
        self.store = store
        self.mock = mock
        self.registry = registry
        self.parent_quality = parent_quality
        self.raw_name: str | None = None    # mode=edit: the raw LoRA (feedback adds vs_raw)
        # mode=edit, generation >= 1: the parent model is M_0 + the current LoRA (served as this adapter on the
        # M_0 server); candidates are full replacement LoRAs on M_0. None = the served model itself.
        self.parent_lora: str | None = None
        self.pinned: set = set()            # adapters never unloaded (the parent / editor LoRA)
        # merge.mode adapter (scratch, 2026-11-22): the current model is M_0 + the cumulative LoRA C_g (parent_lora
        # is its adapter); a candidate is a DELTA on M_g and is served as compose_cumulative(C_g, delta) (the one
        # place where deltas are composed; full adapters are registered with full=True). None = no composition.
        self.base_lora = None
        self.base_budget: int | None = None
        self.shown: dict[str, set] = {}
        self._loaded: OrderedDict = OrderedDict()
        self._paths: dict[str, Path] = {}
        self._max_loaded = max_loaded
        self._lock = threading.RLock()
        ec = cfg.get("feedback", {})
        self.n_examples = int(ec.get("n_examples", 3))
        self.mark_shown_examples = bool(ec.get("mark_shown", True))

    # ------------------------------------------------------------------ adapters
    def set_base(self, lora, budget: int | None = None) -> None:
        """merge.mode adapter: candidates registered from now on are deltas on M_0 + `lora` (composed at register)."""
        self.base_lora = lora if (lora is not None and lora.modules()) else None
        self.base_budget = budget

    def compose(self, lora):
        """What a delta is served as: compose_cumulative(C_g, lora, budget) in adapter mode (also with C_g empty when
        the delta alone is above the budget), else lora itself. Deterministic: run.py rebuilds the accepted C_{g+1}
        with the same call instead of keeping every composite in memory."""
        if getattr(lora, "kind", "lora") == "weights" or not lora.modules():
            return lora
        if self.base_lora is None and not (self.base_budget and lora.max_rank() > int(self.base_budget)):
            return lora
        from .primitives.lora import compose_cumulative
        return compose_cumulative(self.base_lora, lora, self.base_budget)

    def register(self, lora, name: str, *, full: bool = False) -> Path:
        """Save `lora` (PEFT dir, or mock JSON spec) and make it servable under `name`. Adapter mode: a delta is
        saved as C_g (+) delta (full=False); full=True registers a complete adapter on M_0 as is (C_g itself, the
        next C). Mock quality is always the delta's, relative to parent_quality (= the current model);
        a full adapter whose quality is already in the registry keeps it."""
        weights = getattr(lora, "kind", "lora") == "weights"
        serve = lora if full else self.compose(lora)
        with self._lock:
            if weights:
                self._kinds[name] = "weights"
            if self.mock:
                path = self.out_dir / "adapters" / f"{name}.json"
                write_json(path, serve.to_json())
                if full and name in (read_json(self.registry, {}) or {}):
                    self._paths[name] = path
                    return path
                # per-domain mock quality: the run's domain gains (the train and gate datasets both belong to it, so
                # both of them see it), the other domains take the damage
                tkey = getattr(self.rc, "domain", None) or self.target
                q = lora.quality(self.parent_quality if self.parent_quality is not None else 0.7, salt=name,
                                 target=tkey)
                reg = read_json(self.registry, {}) or {}
                reg[name] = q
                write_json(self.registry, reg)
            elif weights:
                path = self.out_dir / "adapters" / f"{name}.wedit.safetensors"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    lora.save_raw(path)
            else:
                from .primitives.lora import save_peft
                path = self.out_dir / "adapters" / name
                if not (path / "adapter_model.safetensors").exists():
                    save_peft(serve, path, self.store, base_model=str(getattr(self.store, "model_dir", "")),
                              dtype=self.cfg.get("lora", {}).get("save_dtype", "bfloat16"))
            self._paths[name] = path
            return path

    def known(self, name: str) -> bool:
        return name in self._paths

    def set_parent(self, lora, name: str) -> None:
        """The parent / editor model is M_0 + `lora` (registered as a full adapter and pinned as `name`): mode=edit's
        current LoRA, or merge.mode adapter's cumulative LoRA C_g."""
        if not self.known(name):
            self.register(lora, name, full=True)
        self.parent_lora = name
        self.pinned.add(name)
        self.ensure_loaded(name)

    def ensure_loaded(self, name: str) -> None:
        if self._kinds.get(name) == "weights":      # a weight edit is served as its own model, nothing to load
            return
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return
            while len(self._loaded) >= self._max_loaded:
                old = next((k for k in self._loaded if k not in self.pinned and not self._busy.get(k)), None)
                if old is None:
                    break
                self._loaded.pop(old)
                for s in self.replicas:
                    try:
                        s.unload_lora(old)
                    except Exception:  # noqa: BLE001
                        pass
            for s in self.replicas:
                s.load_lora(name, self._paths[name])
            self._loaded[name] = True

    # ------------------------------------------------------------------ scoring
    def level_decoding(self, level: str) -> dict | None:
        """The decoding dict of `level` (None = tasks.yaml greedy); full falls back to screen."""
        dec = self.decoding.get(level)
        if dec is None and level == "full":
            dec = self.decoding.get("screen")
        return dec

    def _key_lock(self, key: str, lv: str) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault((key, lv), threading.Lock())

    def run(self, name: str | None, level: str) -> dict:
        """{dataset: {"score","ci","n","n_scored","records":[{id,score,out,len,finish,parsed[,ns,trunc]}]}}
        (cached on disk; each (model, level) is scored by one thread at a time, the others then read the cache)."""
        key = name or "__parent__"
        lvs = {ds: self._cache_level(level, ds, ids) for ds, ids in self.ids[level].items()}
        locks = [self._key_lock(key, lv) for lv in sorted(set(lvs.values()))]
        for lk in locks:
            lk.acquire()
        try:
            return self._run(name, level, key, lvs)
        finally:
            for lk in reversed(locks):
                lk.release()

    def _run(self, name, level, key, lvs) -> dict:
        from rsicore.evaluate import score_split

        lora = name if name is not None else self.parent_lora
        dec = self.level_decoding(level)
        res = {}
        weights_real = name is not None and self._kinds.get(name) == "weights" and not self.mock
        pre = {}
        if weights_real:                            # every uncached dataset in one export + serve
            todo, dirs = {}, {}
            for ds, ids in sorted(self.ids[level].items()):
                lv = lvs[ds]
                if read_json(self.out_dir / "evals" / key / lv / f"{ds}.json") is None:
                    todo[ds] = list(ids)
                    dirs[ds] = self.out_dir / "evals" / key / lv / f"{ds}_raw"
            if todo:
                if self.weights_scorer is None:
                    raise RuntimeError("weight-edit candidate but no weights_scorer (method.mode weights)")
                pre = (self.weights_scorer(name, self._paths[name], todo, dirs, decoding=dec) if dec is not None
                       else self.weights_scorer(name, self._paths[name], todo, dirs))
        for ds, ids in sorted(self.ids[level].items()):
            lv = lvs[ds]
            cache = self.out_dir / "evals" / key / lv / f"{ds}.json"
            got = read_json(cache)
            if got is None:
                if ds in pre:
                    er = pre[ds]
                else:
                    if lora is not None:
                        with self._lock:
                            self._busy[lora] = self._busy.get(lora, 0) + 1
                    try:
                        if lora is not None:
                            self.ensure_loaded(lora)
                        er = score_split(self.endpoints, self.served, ds, pool="pool", ids=list(ids),
                                         out_dir=self.out_dir / "evals" / key / lv / f"{ds}_raw",
                                         lora=lora, mock=self.mock, decoding=dec)
                    finally:
                        if lora is not None:
                            with self._lock:
                                self._busy[lora] -= 1
                recs = [_compact(r) for r in er.records]
                scored = [r["score"] for r in recs if r["score"] is not None]
                m, ci = mean_ci(scored, seed=self.rc.seed)
                got = {"score": m, "ci": ci, "n": len(recs), "n_scored": len(scored), "records": recs,
                       "status": (er.summary or {}).get("status") if isinstance(er.summary, dict) else None}
                if dec is not None:
                    got["decoding"] = dict(dec)
                write_json(cache, got)
            res[ds] = got
        return res

    def _dec_tag(self, level: str) -> str:
        dec = self.level_decoding(level)
        return "" if dec is None else "@" + stable_hash(json.dumps(dec, sort_keys=True), n=8)

    def _cache_level(self, level: str, ds: str, ids) -> str:
        """Cache directory level of (level, ds): a level with the same dataset, the same items AND the same decoding
        as the screen level (full = screen in the simple flow) shares the screen result instead of scoring them again; a
        different dataset (the gate set) never does. A sampled decoding adds its tag (<level>@<hash>, 2026-11-21)."""
        scr = self.ids.get("screen") or {}
        if (level != "screen" and ds in scr and sorted(ids) == sorted(scr[ds])
                and self.level_decoding(level) == self.level_decoding("screen")):
            return "screen" + self._dec_tag("screen")
        return level + self._dec_tag(level)

    def full_records(self, name: str | None, level: str) -> dict:
        """{dataset: [run_eval records]} of an already scored (model, level): every sample's full text, thinking
        ("reasoning"), extracted answer, finish reason and score (2026-11-21: the current model's screen records for
        the editor's screen tools). Datasets without records on disk are left out; never evaluates."""
        key, out = name or "__parent__", {}
        for ds, ids in sorted(self.ids[level].items()):
            p = self.out_dir / "evals" / key / self._cache_level(level, ds, ids) / f"{ds}_raw" / f"{ds}.jsonl"
            if p.exists():
                out[ds] = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        return out

    def cached(self, name: str | None, level: str) -> dict | None:
        """run()'s result if every dataset of `level` is already on disk; None otherwise (never evaluates)."""
        key = name or "__parent__"
        res = {}
        for ds, ids in sorted(self.ids.get(level, {}).items()):
            got = read_json(self.out_dir / "evals" / key / self._cache_level(level, ds, ids) / f"{ds}.json")
            if got is None:
                return None
            res[ds] = got
        return res

    def adopt_parent(self, level: str, src: Path, *, layout: str = "cache") -> bool:
        """Seed the current model's `level` results from an earlier scoring of the SAME model (2026-11-24), so the
        parent is not sampled again: layout "cache" = another Evaluator's evals/<key> dir (a previous generation's
        __parent__ or the accepted candidate's), "baseline" = Runner.baseline's gate dir (<src>/<ds>/<ds>.jsonl, M_0
        with the gate decoding). Taken only when every dataset of `level` is there with exactly this level's items
        (and, for "cache", the same decoding, which the level directory's tag also encodes); otherwise nothing is
        written and run() scores the parent as before. -> True when the cache now holds every dataset."""
        import shutil
        key, src, dec = "__parent__", Path(src), self.level_decoding(level)
        plan = []
        for ds, ids in sorted(self.ids.get(level, {}).items()):
            lv = self._cache_level(level, ds, ids)
            dst = self.out_dir / "evals" / key / lv
            if read_json(dst / f"{ds}.json") is not None:
                continue
            if layout == "cache":
                got = read_json(src / lv / f"{ds}.json")
                raw = src / lv / f"{ds}_raw"
                if got is None or got.get("decoding") != (dict(dec) if dec is not None else None):
                    return False
            else:
                raw = src / ds / f"{ds}.jsonl"
                if not raw.exists():
                    return False
                recs = [_compact(json.loads(x)) for x in raw.read_text().splitlines() if x.strip()]
                summ = read_json(src / ds / "summary.json") or {}
                scored = [r["score"] for r in recs if r["score"] is not None]
                m, ci = mean_ci(scored, seed=self.rc.seed)
                got = {"score": m, "ci": ci, "n": len(recs), "n_scored": len(scored), "records": recs,
                       "status": summ.get("status") if isinstance(summ, dict) else None}
                if dec is not None:
                    got["decoding"] = dict(dec)
            if sorted(str(r.get("id")) for r in got.get("records") or []) != sorted(str(i) for i in ids):
                return False
            plan.append((ds, lv, dst, got, raw))
        for ds, lv, dst, got, raw in plan:
            with self._key_lock(key, lv):
                if read_json(dst / f"{ds}.json") is not None:
                    continue
                if raw.is_dir():
                    shutil.copytree(raw, dst / f"{ds}_raw", dirs_exist_ok=True)
                elif raw.exists():
                    (dst / f"{ds}_raw").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(raw, dst / f"{ds}_raw" / f"{ds}.jsonl")
                write_json(dst / f"{ds}.json", dict(got, adopted_from=str(raw)))
        return self.cached(None, level) is not None

    # ------------------------------------------------------------------ comparisons
    def _paired(self, cand: list[dict], ref: list[dict]) -> dict:
        """delta = candidate − reference, paired by item id (rsicore.paired_delta(a=ref, b=cand))."""
        from rsicore.evaluate import paired_delta
        cc = [r for r in cand if r.get("score") is not None]
        rr = [r for r in ref if r.get("score") is not None]
        if not cc or not rr:
            return {"delta": None, "ci": [None, None], "n": 0}
        return paired_delta(rr, cc, seed=self.rc.seed)

    def compare(self, cand: dict, ref: dict) -> dict:
        out = {}
        for ds, r in cand.items():
            if ds in ref:
                out[ds] = self._paired(r["records"], ref[ds]["records"])
        return out

    def feedback(self, name: str | None, level: str, lora_report: dict | None = None, cost: dict | None = None,
                 parent_gen0: dict | None = None) -> tuple[float | None, dict]:
        cand = self.run(name, level)
        par = self.run(None, level)
        vs_par = self.compare(cand, par)
        fb = {"level": level, "lora": name}
        T = self.gate_target if level == "gate" else self.target
        t = cand.get(T, {})
        tgt = {"dataset": T, "score": t.get("score"), "ci": t.get("ci"), "n": t.get("n_scored"),
               "parent": par.get(T, {}).get("score"), "vs_parent": vs_par.get(T)}
        raw = self.run(self.raw_name, level) if (self.raw_name and level != "gate") else None
        if raw is not None and T in raw:
            tgt["raw"] = raw[T]["score"]
            tgt["vs_raw"] = self._paired(t.get("records", []), raw[T]["records"])
        tgt["significant"] = _sig(tgt["vs_parent"])
        tgt["errors"] = self._examples(t.get("records", []), par.get(T, {}).get("records", []),
                                       T, mode="errors") if name else []
        fb["target"] = tgt
        fb["behavior"] = _behavior(cand, par)
        if lora_report is not None:
            fb["lora_stats"] = lora_report
        if cost is not None:
            fb["cost"] = cost
        gain = (tgt["vs_parent"] or {}).get("delta")
        if name is None:
            score = 0.0
        elif gain is None:
            score = None
        else:
            score = gain
        fb["score"] = score
        fb["score_formula"] = "target_gain (paired vs the current model)"
        return score, fb

    def _examples(self, cand_recs, par_recs, ds, mode="errors"):
        """errors: items the candidate gets wrong (avg@n records: mostly wrong, score <= 0.5), the ones the parent got
        right first."""
        pmap = {r["id"]: r for r in par_recs}

        def wrong(r):
            return r["score"] <= 0.5 if _multi(r) else r["score"] < 1
        pick = [r for r in cand_recs if r.get("score") is not None and wrong(r)]
        pick.sort(key=lambda r: (-(pmap.get(r["id"], {}).get("score") or 0), stable_hash(r["id"])))
        pick = pick[: self.n_examples]
        if not pick:
            return []
        items = load_records(ds, [r["id"] for r in pick])
        out = []
        for r in pick:
            it = items.get(r["id"], {})
            e = {"dataset": ds, "id": r["id"], "prompt": _short(it.get("prompt") or str(it.get("messages")), 500),
                 "output": r.get("out")}
            if _multi(r):                           # avg@n: the score is the share of correct samples
                e["score"] = r["score"]
            e["expected"] = _short(it.get("reference"), 200)
            out.append(e)
            if self.mark_shown_examples:
                self.shown.setdefault(ds, set()).add(r["id"])
        return out


def _sig(d) -> bool | None:
    if not d or d.get("ci") is None or d["ci"][0] is None:
        return None
    lo, hi = d["ci"]
    return bool(lo > 0 or hi < 0)


def _behavior(cand: dict, par: dict) -> dict:
    def stats(res):
        L, trunc, fmt, n = 0, 0, 0, 0
        for r in res.values():
            for x in r.get("records", []):
                n += 1
                L += x.get("len", len(x.get("out") or ""))
                trunc += _trunc(x)
                fmt += (x.get("score") is not None) and not x.get("parsed")
        return L, trunc, fmt, max(n, 1)
    lc, tc, fc, nc = stats(cand)
    lp, _, _, np_ = stats(par)
    return {"len_ratio": round((lc / nc) / max(lp / np_, 1e-9), 3), "truncated": round(tc / nc, 4),
            "format_fail": round(fc / nc, 4)}
