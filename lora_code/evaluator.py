"""Evaluation of LoRA candidates on the served model M_t (vLLM multi-LoRA, hot-loaded adapters).

Levels: `screen` (target screen + retention screen, ~200 items each), `full` (whole signal draw of the
generation), `gate` (fixed gate of every domain). Every result is cached on disk per (name, level), so a
resumed run never re-evaluates. Feedback follows RSI_LOOP_DESIGN §3.4.
"""
from __future__ import annotations

import random
import threading
from collections import OrderedDict
from pathlib import Path

from .common import DATASET_DOMAIN, load_records, read_json, stable_hash, write_json


def mean_ci(scores: list[float], seed: int = 42, n_boot: int = 500) -> tuple[float | None, list]:
    if not scores:
        return None, [None, None]
    m = sum(scores) / len(scores)
    rng = random.Random(seed)
    n = len(scores)
    bs = sorted(sum(scores[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return m, [bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]]


def _short(s, n=400):
    if s is None:
        return None
    if isinstance(s, list):
        s = s[0] if s else ""
    s = str(s)
    return s if len(s) <= n else s[: n // 2] + " ... " + s[-n // 2:]


class Evaluator:
    def __init__(self, rc, cfg: dict, *, out_dir: Path, endpoint: str, served: str, server, ids: dict,
                 target_dataset: str, kl_texts: list[str], store, mock: bool, registry: Path | None,
                 parent_quality: float | dict | None = None, max_loaded: int = 6, tag: str = ""):
        self.rc, self.cfg = rc, cfg
        self.out_dir = Path(out_dir)
        self.endpoint, self.served, self.server = endpoint, served, server
        self.ids = ids                      # level -> {dataset: [ids]}
        self.target = target_dataset
        self.kl_texts = kl_texts
        self.store = store
        self.mock = mock
        self.registry = registry
        self.parent_quality = parent_quality
        self.raw_name: str | None = None    # mode=edit: the raw LoRA (feedback adds vs_raw)
        # mode=edit, generation >= 1: the parent model is M_0 + the current LoRA (served as this adapter on the
        # M_0 server); candidates are full replacement LoRAs on M_0. None = the served model itself.
        self.parent_lora: str | None = None
        self.pinned: set = set()            # adapters never unloaded (the parent / editor LoRA)
        self.shown: dict[str, set] = {}
        self._loaded: OrderedDict = OrderedDict()
        self._paths: dict[str, Path] = {}
        self._max_loaded = max_loaded
        self._lock = threading.RLock()
        self._ref_cache = None
        self.tag = tag
        ec = cfg.get("feedback", {})
        self.lam = float(ec.get("lambda_retention", 1.0))
        self.lam_kl = float(ec.get("lambda_kl", 0.5))
        self.n_examples = int(ec.get("n_examples", 3))
        self.mark_shown_examples = bool(ec.get("mark_shown", True))

    # ------------------------------------------------------------------ adapters
    def register(self, lora, name: str) -> Path:
        """Save `lora` (PEFT dir, or mock JSON spec) and make it servable under `name`."""
        with self._lock:
            if self.mock:
                path = self.out_dir / "adapters" / f"{name}.json"
                write_json(path, lora.to_json())
                # per-domain mock quality: the target domain gains, retention domains take the damage
                q = lora.quality(self.parent_quality if self.parent_quality is not None else 0.7, salt=name,
                                 target=DATASET_DOMAIN.get(self.target, self.target))
                reg = read_json(self.registry, {}) or {}
                reg[name] = q
                write_json(self.registry, reg)
            else:
                from .primitives.lora import save_peft
                path = self.out_dir / "adapters" / name
                if not (path / "adapter_model.safetensors").exists():
                    save_peft(lora, path, self.store, base_model=str(getattr(self.store, "model_dir", "")),
                              dtype=self.cfg.get("lora", {}).get("save_dtype", "bfloat16"))
            self._paths[name] = path
            return path

    def known(self, name: str) -> bool:
        return name in self._paths

    def set_parent(self, lora, name: str) -> None:
        """mode=edit: the parent / editor model is M_0 + `lora` (registered and pinned as `name`)."""
        if not self.known(name):
            self.register(lora, name)
        self.parent_lora = name
        self.pinned.add(name)
        self.ensure_loaded(name)

    def ensure_loaded(self, name: str) -> None:
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return
            while len(self._loaded) >= self._max_loaded:
                old = next((k for k in self._loaded if k not in self.pinned), None)
                if old is None:
                    break
                self._loaded.pop(old)
                try:
                    self.server.unload_lora(old)
                except Exception:  # noqa: BLE001
                    pass
            self.server.load_lora(name, self._paths[name])
            self._loaded[name] = True

    # ------------------------------------------------------------------ scoring
    def run(self, name: str | None, level: str) -> dict:
        """{dataset: {"score","ci","n","n_scored","records":[{id,score,out,finish,parsed}]}} (cached)."""
        from rsicore.evaluate import score as rs_score

        key = name or "__parent__"
        lora = name if name is not None else self.parent_lora
        res = {}
        for ds, ids in sorted(self.ids[level].items()):
            cache = self.out_dir / "evals" / key / level / f"{ds}.json"
            got = read_json(cache)
            if got is None:
                if lora is not None:
                    self.ensure_loaded(lora)
                er = rs_score(self.endpoint, self.served, ds, pool="pool", ids=list(ids),
                              out_dir=self.out_dir / "evals" / key / level / f"{ds}_raw",
                              lora=lora, mock=self.mock)
                recs = []
                for r in er.records:
                    recs.append({"id": r.get("id"), "score": r.get("score"),
                                 "out": _short(r.get("raw_output"), 600),
                                 "len": len(str(r.get("raw_output") or "")),
                                 "finish": (r.get("finish_reason") or [None])[0] if isinstance(r.get("finish_reason"), list) else r.get("finish_reason"),
                                 "parsed": r.get("parsed_answer") is not None,
                                 "status": r.get("status")})
                scored = [r["score"] for r in recs if r["score"] is not None]
                m, ci = mean_ci(scored, seed=self.rc.seed)
                got = {"score": m, "ci": ci, "n": len(recs), "n_scored": len(scored), "records": recs,
                       "status": (er.summary or {}).get("status") if isinstance(er.summary, dict) else None}
                write_json(cache, got)
            res[ds] = got
        return res

    def cached(self, name: str | None, level: str) -> dict | None:
        """run()'s result if every dataset of `level` is already on disk; None otherwise (never evaluates)."""
        key = name or "__parent__"
        res = {}
        for ds in sorted(self.ids.get(level, {})):
            got = read_json(self.out_dir / "evals" / key / level / f"{ds}.json")
            if got is None:
                return None
            res[ds] = got
        return res

    def kl(self, name: str | None) -> float | None:
        if name is None or not self.kl_texts:
            return 0.0 if name is None else None
        from rsicore.evaluate import reference_logprobs, token_kl

        cache = self.out_dir / "evals" / name / "kl.json"
        got = read_json(cache)
        if got is not None:
            return got["mean"]
        with self._lock:
            if self._ref_cache is None:
                if self.parent_lora:   # reference = M_0 + current LoRA (adapter name as the model)
                    self.ensure_loaded(self.parent_lora)
                self._ref_cache = reference_logprobs(self.endpoint, self.parent_lora or self.served, self.kl_texts,
                                                     self.out_dir / "evals" / "__parent__" / "ref_logprobs.jsonl")
            self.ensure_loaded(name)
        out = token_kl(self._ref_cache, self.endpoint, self.served, self.kl_texts, lora=name)
        write_json(cache, {"mean": out.get("mean"), "per_text": out.get("per_text", [])[:256]})
        return out.get("mean")

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
        kl = self.kl(name)
        vs_par = self.compare(cand, par)
        fb = {"level": level, "lora": name}
        t = cand.get(self.target, {})
        tgt = {"dataset": self.target, "score": t.get("score"), "ci": t.get("ci"), "n": t.get("n_scored"),
               "parent": par.get(self.target, {}).get("score"), "vs_parent": vs_par.get(self.target)}
        raw = self.run(self.raw_name, level) if (self.raw_name and level != "gate") else None
        if raw is not None and self.target in raw:
            tgt["raw"] = raw[self.target]["score"]
            tgt["vs_raw"] = self._paired(t.get("records", []), raw[self.target]["records"])
        tgt["significant"] = _sig(tgt["vs_parent"])
        tgt["errors"] = self._examples(t.get("records", []), par.get(self.target, {}).get("records", []),
                                       self.target, mode="errors") if name else []
        fb["target"] = tgt
        ret, drops = {}, []
        for ds, r in cand.items():
            if ds == self.target:
                continue
            d = vs_par.get(ds) or {}
            ret[DATASET_DOMAIN.get(ds, ds)] = {"dataset": ds, "score": r.get("score"),
                                                "parent": par.get(ds, {}).get("score"), "vs_parent": d,
                                                "significant": _sig(d)}
            if d.get("delta") is not None:
                drops.append(max(0.0, -d["delta"]))
        fb["retention"] = ret
        fb["kl_vs_parent"] = kl
        regs = []
        if name:
            for ds, r in cand.items():
                if ds != self.target:
                    regs += self._examples(r.get("records", []), par.get(ds, {}).get("records", []), ds,
                                           mode="regressions")
        fb["regressions"] = regs[: self.n_examples]
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
            score = gain - self.lam * (sum(drops) / len(drops) if drops else 0.0) - self.lam_kl * float(kl or 0.0)
        fb["score"] = score
        fb["score_formula"] = f"target_gain - {self.lam} * mean_retention_drop - {self.lam_kl} * kl"
        return score, fb

    def _examples(self, cand_recs, par_recs, ds, mode):
        pmap = {r["id"]: r for r in par_recs}
        if mode == "errors":
            pick = [r for r in cand_recs if r.get("score") is not None and r["score"] < 1]
            pick.sort(key=lambda r: (-(pmap.get(r["id"], {}).get("score") or 0), stable_hash(r["id"])))
        else:
            pick = [r for r in cand_recs if r.get("score") is not None and r["score"] < 1
                    and (pmap.get(r["id"], {}).get("score") or 0) >= 1]
            pick.sort(key=lambda r: stable_hash(r["id"]))
        pick = pick[: self.n_examples]
        if not pick:
            return []
        items = load_records(ds, [r["id"] for r in pick])
        out = []
        for r in pick:
            it = items.get(r["id"], {})
            e = {"dataset": ds, "id": r["id"], "prompt": _short(it.get("prompt") or str(it.get("messages")), 500),
                 "output": r.get("out")}
            if mode == "errors":
                e["expected"] = _short(it.get("reference"), 200)
            else:
                e["before"] = pmap.get(r["id"], {}).get("out")
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
                trunc += x.get("finish") == "length"
                fmt += (x.get("score") is not None) and not x.get("parsed")
        return L, trunc, fmt, max(n, 1)
    lc, tc, fc, nc = stats(cand)
    lp, _, _, np_ = stats(par)
    return {"len_ratio": round((lc / nc) / max(lp / np_, 1e-9), 3), "truncated": round(tc / nc, 4),
            "format_fail": round(fc / nc, 4)}
