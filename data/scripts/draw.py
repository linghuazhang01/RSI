"""Per-generation sampling from the training-set draw pools (data/splits/<dataset>/pool.json).

Each generation t of a run draws a sample per domain (1,000-2,000 items) from the pools.
Reuse policy (config key `reuse`, defaults shown; revised 2026-11-19):
  across_generations: true     items may be drawn again in later generations; never-used items are
                               taken first, then the least recently used ones
  never_signal_after: [train, shown]
                               an item that was ever used as `train` (Paper B gradient data) or ever
                               shown to the editor as a feedback example (recorded with --mark-shown)
                               is never again used as signal / screen / kl. Such items may still be
                               drawn as `train` again (training data is reusable).
Within one generation all roles are disjoint (screen is a subset of signal).

Roles written per domain:
  Paper A: signal (L1 feedback), screen (stratified subset of signal for the first evaluation
           stage), kl (KL / calibration contexts, disjoint from signal when the pool allows)
  Paper B: target domain -> train (gradient-tool data) + signal + screen
           other domains -> signal (retention probes) + screen + kl (token-KL prompts)
Fixed gate (revised 2026-11-19: split dynamically from the config): before the first draw, each
domain's `gate_n` items are taken from its training pool (data/splits/<dataset>/pool.json),
deterministically from `gate.seed` (default 20261118) and the gate sizes only -- so every run with
the same gate sizes gets the same gate, whatever its experiment seed. The gate is written to
draws/gate/ and is never drawn for anything else; held-out test pools are never touched.

Usage (run with -I; nothing here executes downloaded data):
  PY="eval/.venv/bin/python -I"
  $PY data/scripts/draw.py --run experiment_records/<run_id> --gen 0 --config data/draw/paper_a.yaml
  $PY data/scripts/draw.py --run experiment_records/<run_id> --gen 1                 # config is locked in the run
  $PY data/scripts/draw.py --run experiment_records/<run_id> --gen 0 --target math    # Paper B
  $PY data/scripts/draw.py --run experiment_records/<run_id> --gen 1 --export-jsonl   # also write item JSONL
  $PY data/scripts/draw.py --run experiment_records/<run_id> --gate-only --config data/draw/paper_a.yaml
      # split and write the fixed gate only (e.g. to evaluate generation 0 on the gate first)
  $PY data/scripts/draw.py --run experiment_records/<run_id> --status
  $PY data/scripts/draw.py --run experiment_records/<run_id> --gen 1 --mark-shown shown.json
      # shown.json = {"<dataset>": ["<id>", ...]}: items whose content was shown to the editor in
      # generation 1 (error examples); they are excluded from all later signal draws

Outputs (inside the run directory):
  draws/draw_config.yaml           locked copy of the config
  draws/ledger.json                every drawn id -> [[gen, role], ...]
  draws/gen_XX/manifest.json       ids per domain / role / dataset, counts, reuse, hashes
  draws/gen_XX/ids/<dataset>.<role>.json   id lists for eval/run_eval.py --pool pool --ids-file ...
  draws/gen_XX/data/<domain>.<role>.jsonl  (with --export-jsonl) full processed records + draw_role

Determinism: given the same splits, config, run name and sequence of draws, every id list is
identical (RNG seeded by seed:run:gen:domain:dataset:role; candidates sorted before shuffling).
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parents[1]
PROC, SPL = DATA / "processed", DATA / "splits"
LEDGER_VERSION = 1


# ----------------------------------------------------------------------------- small helpers
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json_atomic(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, p)


def alloc(weights: dict, total: int, caps: dict | None = None) -> dict:
    """Largest-remainder allocation of `total` over keys by weight, capped per key; leftover from
    capped keys is redistributed over keys with spare capacity. Deterministic (ties by key)."""
    caps = caps or {k: total for k in weights}
    out = {k: 0 for k in weights}
    left = total
    active = {k for k, w in weights.items() if w > 0 and caps.get(k, 0) > 0}
    while left > 0 and active:
        W = sum(weights[k] for k in active)
        raw = {k: left * weights[k] / W for k in active}
        add = {k: min(int(raw[k]), caps[k] - out[k]) for k in active}
        rem = left - sum(add.values())
        for k in sorted(active, key=lambda k: (-(raw[k] - int(raw[k])), k)):
            if rem <= 0:
                break
            if out[k] + add[k] < caps[k]:
                add[k] += 1
                rem -= 1
        if sum(add.values()) == 0:
            break
        for k, a in add.items():
            out[k] += a
        left = total - sum(out.values())
        active = {k for k in active if out[k] < caps[k]}
    return out


class Store:
    """Processed records and split pools, loaded lazily."""

    def __init__(self):
        self._rows, self._pools = {}, {}
        self.gate = {}  # ds -> set(ids) excluded from every draw

    def rows(self, ds):
        if ds not in self._rows:
            with open(PROC / f"{ds}.jsonl") as f:
                self._rows[ds] = {r["id"]: r for r in map(json.loads, f)}
        return self._rows[ds]

    def pool(self, ds):
        """Draw pool = split pool minus this run's fixed gate."""
        full = self.full_pool(ds)
        g = self.gate.get(ds)
        return [i for i in full if i not in g] if g else full

    def full_pool(self, ds):
        if ds not in self._pools:
            p = SPL / ds / "pool.json"
            if not p.exists():
                sys.exit(f"dataset {ds} has no draw pool ({p}); see data/splits/split_config.yaml")
            ids = json.load(open(p))
            rows = self.rows(ds)
            bad = [i for i in ids if rows[i]["split_pool"] != "pool"]
            assert not bad, f"{ds}: pool.json disagrees with processed split_pool ({len(bad)} ids)"
            self._pools[ds] = sorted(ids)
        return self._pools[ds]


# ----------------------------------------------------------------------------- drawing
class Drawer:
    def __init__(self, cfg, run_name, gen, ledger, store):
        self.cfg, self.run, self.gen, self.store = cfg, run_name, gen, store
        self.items = ledger["items"]          # ds -> id -> [[gen, role], ...]
        self.taken = defaultdict(set)         # ids already chosen in this generation
        self.reused = defaultdict(int)
        pol = cfg.get("reuse") or {}
        self.across = bool(pol.get("across_generations", True))
        self.never_signal_after = set(pol.get("never_signal_after", ["train", "shown"]))

    def rng(self, *parts):
        return random.Random(":".join(str(p) for p in (self.cfg["seed"], self.run, self.gen) + parts))

    def blocked(self, ds, i, role):
        """True if item i may not take `role` because of its history in this run."""
        if role == "train":
            return False  # training data is reusable
        return any(r in self.never_signal_after for _, r in self.items.get(ds, {}).get(i, []))

    def candidates(self, ds, role):
        """Never-used pool items, and (if reuse across generations) eligible previously used ones."""
        pool, used = self.store.pool(ds), self.items.get(ds, {})
        fresh = [i for i in pool if i not in used and i not in self.taken[ds]]
        old = []
        if self.across:
            old = [i for i in pool if i in used and i not in self.taken[ds] and not self.blocked(ds, i, role)]
        return fresh, old

    def take_from(self, ds, k, key, role, strat_key=None):
        """Take k ids from dataset ds (stratified by metadata[strat_key] if given): never-used items
        first, then previously used eligible items, least recently used first."""
        if k <= 0:
            return []
        fresh, old = self.candidates(ds, role)
        rows = self.store.rows(ds)

        def stratum(i):
            return str(rows[i]["metadata"].get(strat_key)) if strat_key else "_"

        def pick(cands, n, tag):
            by = defaultdict(list)
            for i in cands:
                by[stratum(i)].append(i)
            sizes = {s: len(v) for s, v in by.items()}
            a = alloc(sizes, min(n, len(cands)), caps=sizes)
            out = []
            for s in sorted(by):
                lst = sorted(by[s])
                self.rng(key, ds, role, tag, s).shuffle(lst)
                out += lst[:a[s]]
            return out

        got = pick(fresh, k, "fresh")
        if len(got) < k and old:
            last = {i: max(g for g, _ in self.items[ds][i]) for i in old}
            # oldest-used first, generation by generation
            for g in sorted(set(last.values())):
                if len(got) >= k:
                    break
                batch = [i for i in old if last[i] == g]
                more = pick(batch, k - len(got), f"reuse{g}")
                self.reused[(key, role)] += len(more)
                got += more
        self.taken[ds].update(got)
        return got

    def capacity(self, ds, role):
        fresh, old = self.candidates(ds, role)
        return len(fresh) + len(old)

    def draw_role(self, domain, dcfg, role, n):
        weights = {ds: float(w) for ds, w in dcfg["datasets"].items()}
        caps = {ds: self.capacity(ds, role) for ds in weights}
        a = alloc(weights, n, caps)
        out = {ds: self.take_from(ds, a[ds], domain, role, dcfg.get("stratify_by")) for ds in weights}
        got = sum(len(v) for v in out.values())
        return {ds: sorted(v) for ds, v in out.items() if v}, n - got

    def subset(self, domain, role_ids, n, tag):
        """Stratified (by dataset) deterministic subset of already drawn ids."""
        sizes = {ds: len(v) for ds, v in role_ids.items()}
        a = alloc(sizes, min(n, sum(sizes.values())), caps=sizes)
        out = {}
        for ds, ids in role_ids.items():
            lst = sorted(ids)
            self.rng(domain, ds, tag).shuffle(lst)
            if a[ds]:
                out[ds] = sorted(lst[:a[ds]])
        return out


def draw_generation(cfg, run_name, gen, ledger, store, target=None):
    paper = cfg["paper"]
    dr = Drawer(cfg, run_name, gen, ledger, store)
    domains, problems = {}, []
    names = list(cfg["domains"])
    if paper == "B":
        if target not in cfg["domains"]:
            sys.exit(f"Paper B needs --target, one of {names}")
    elif target:
        sys.exit("--target is only used for Paper B")
    kl_alloc = alloc({d: 1 for d in names}, int(cfg.get("kl_contexts", 0))) if paper == "A" else {}

    # target domain first so its train/signal get first pick of fresh items
    order = ([target] + [d for d in names if d != target]) if target else names
    for d in order:
        dc = cfg["domains"][d]
        roles, short = {}, {}
        if paper == "A":
            roles["signal"], short["signal"] = dr.draw_role(d, dc, "signal", int(dc["n"]))
            k = kl_alloc.get(d, 0)
            roles["kl"], short["kl"] = dr.draw_role(d, dc, "kl", k)
            if short["kl"]:  # pool too small: take KL contexts from this generation's signal
                roles["kl"] = merge(roles["kl"], dr.subset(d, roles["signal"], short["kl"], "kl_from_signal"))
                short["kl_from_signal"] = short["kl"]
                short["kl"] = 0
        elif d == target:
            n_train = round(int(dc["n"]) * float(cfg["target_train_frac"]))
            roles["train"], short["train"] = dr.draw_role(d, dc, "train", n_train)
            roles["signal"], short["signal"] = dr.draw_role(d, dc, "signal", int(dc["n"]) - n_train)
        else:
            roles["signal"], short["signal"] = dr.draw_role(d, dc, "signal", min(int(cfg["retention_n"]), int(dc["n"])))
            roles["kl"], short["kl"] = dr.draw_role(d, dc, "kl", int(cfg["retention_kl"]))
            if short["kl"]:
                roles["kl"] = merge(roles["kl"], dr.subset(d, roles["signal"], short["kl"], "kl_from_signal"))
                short["kl_from_signal"] = short["kl"]
                short["kl"] = 0
        roles["screen"] = dr.subset(d, roles["signal"], int(cfg["screen_n"]), "screen")
        for r, s in short.items():
            if s and not r.startswith("kl_from"):
                problems.append(f"{d}/{r}: short by {s} items (pool exhausted: too few eligible items)")
        domains[d] = {
            "roles": roles,
            "counts": {r: sum(len(v) for v in ids.values()) for r, ids in roles.items()},
            "reused": {r: dr.reused.get((d, r), 0) for r in roles if dr.reused.get((d, r), 0)},
            "shortfall": {r: s for r, s in short.items() if s},
        }
    return domains, problems


def compute_gate(cfg, store):
    """Fixed gate per domain: `gate_n` items split over the domain's datasets by weight, stratified by
    `stratify_by`, chosen with an RNG seeded only by gate.seed, dataset and stratum (not by the run)."""
    gseed = (cfg.get("gate") or {}).get("seed", 20261118)
    out = {}
    for d, dc in cfg["domains"].items():
        n = int(dc.get("gate_n", 0))
        if n <= 0:
            continue
        weights = {ds: float(w) for ds, w in dc["datasets"].items()}
        caps = {ds: len(store.full_pool(ds)) for ds in weights}
        a = alloc(weights, n, caps)
        if sum(a.values()) < n:
            sys.exit(f"gate for {d}: pools too small for gate_n={n}")
        out[d] = {}
        key = dc.get("stratify_by")
        for ds, k in a.items():
            if not k:
                continue
            rows = store.rows(ds)
            by = defaultdict(list)
            for i in store.full_pool(ds):
                by[str(rows[i]["metadata"].get(key)) if key else "_"].append(i)
            sizes = {st: len(v) for st, v in by.items()}
            per = alloc(sizes, k, caps=sizes)
            ids = []
            for st in sorted(by):
                lst = sorted(by[st])
                random.Random(f"{gseed}:gate:{ds}:{st}").shuffle(lst)
                ids += lst[:per[st]]
            out[d][ds] = sorted(ids)
    return out


def load_or_make_gate(ddir, cfg, store):
    """Write draws/gate/ on first use; afterwards verify it is unchanged. Installs it in `store`."""
    gate = compute_gate(cfg, store)
    gdir = ddir / "gate"
    man = gdir / "manifest.json"
    if man.exists():
        old = json.load(open(man))["domains"]
        if old != gate:
            sys.exit(f"{man} differs from the gate implied by the locked config and current pools "
                     "(re-split?). Start a new run.")
    else:
        gdir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(man, {"seed": (cfg.get("gate") or {}).get("seed", 20261118),
                                "sizes": {d: sum(len(v) for v in g.values()) for d, g in gate.items()},
                                "domains": gate})
        per = defaultdict(list)
        for g in gate.values():
            for ds, ids in g.items():
                per[ds] += ids
        for ds, ids in per.items():
            write_json_atomic(gdir / f"{ds}.gate.json", sorted(ids))
    for g in gate.values():
        for ds, ids in g.items():
            store.gate.setdefault(ds, set()).update(ids)
    return gate


def merge(a, b):
    out = {ds: list(v) for ds, v in a.items()}
    for ds, v in b.items():
        out[ds] = sorted(set(out.get(ds, [])) | set(v))
    return out


# ----------------------------------------------------------------------------- run directory
def load_ledger(path: Path):
    if path.exists():
        led = json.load(open(path))
        assert led.get("version") == LEDGER_VERSION, "unknown ledger version"
        return led
    return {"version": LEDGER_VERSION, "items": {}, "gens": {}}


def remove_gen(ledger, gen):
    for ds in list(ledger["items"]):
        for i in list(ledger["items"][ds]):
            uses = [u for u in ledger["items"][ds][i] if u[0] != gen]
            if uses:
                ledger["items"][ds][i] = uses
            else:
                del ledger["items"][ds][i]
    ledger["gens"].pop(str(gen), None)


def pool_hashes(cfg):
    dss = sorted({ds for d in cfg["domains"].values() for ds in d["datasets"]})
    return {ds: sha256_file(SPL / ds / "pool.json") for ds in dss}


def export_jsonl(store, gdir, domains):
    out = gdir / "data"
    out.mkdir(exist_ok=True)
    for d, info in domains.items():
        for role, ids in info["roles"].items():
            with open(out / f"{d}.{role}.jsonl", "w") as f:
                for ds in sorted(ids):
                    rows = store.rows(ds)
                    for i in ids[ds]:
                        r = dict(rows[i])
                        r["draw_role"] = role
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def status(run: Path):
    ddir = run / "draws"
    if not (ddir / "ledger.json").exists():
        sys.exit(f"no draws in {run}")
    cfg = yaml.safe_load(open(ddir / "draw_config.yaml"))
    led = load_ledger(ddir / "ledger.json")
    store = Store()
    gate = load_or_make_gate(ddir, cfg, store)
    print(f"run {run.name}  paper {cfg['paper']}  generations {sorted(int(g) for g in led['gens'])}")
    print("  fixed gate: " + "  ".join(f"{d}:{sum(len(v) for v in g.values())}" for d, g in gate.items()))
    for g in sorted(led["gens"], key=int):
        m = json.load(open(ddir / f"gen_{int(g):02d}" / "manifest.json"))
        print(f"  gen {int(g):2d} target={m.get('target')} " +
              "  ".join(f"{d}:{info['counts']}" for d, info in m["domains"].items()))
    for ds in sorted({ds for d in cfg["domains"].values() for ds in d["datasets"]}):
        pool = store.pool(ds)
        used = led["items"].get(ds, {})
        print(f"  {ds:16s} pool={len(pool):6d} used={len(used):6d} fresh_left={len(pool) - len(used):6d}")


def mark_shown(run: Path, gen: int, path: Path):
    """Record items of generation `gen` whose content was shown to the editor (e.g. error examples).
    They must have been drawn in that generation; they are then excluded from later signal draws."""
    ddir = run / "draws"
    led_path = ddir / "ledger.json"
    ledger = load_ledger(led_path)
    if str(gen) not in ledger["gens"]:
        sys.exit(f"gen {gen} has not been drawn")
    shown = json.load(open(path))
    if not isinstance(shown, dict):
        sys.exit('--mark-shown expects {"<dataset>": ["<id>", ...]}')
    n_new = 0
    for ds, ids in shown.items():
        for i in ids:
            uses = ledger["items"].get(ds, {}).get(i)
            if not uses or not any(g == gen for g, _ in uses):
                sys.exit(f"{ds}/{i} was not drawn in gen {gen}")
            if [gen, "shown"] not in uses:
                uses.append([gen, "shown"])
                n_new += 1
    write_json_atomic(led_path, ledger)
    rec = ddir / f"gen_{gen:02d}" / "shown.json"
    old = json.load(open(rec)) if rec.exists() else {}
    for ds, ids in shown.items():
        old[ds] = sorted(set(old.get(ds, [])) | set(ids))
    write_json_atomic(rec, old)
    print(f"gen {gen}: marked {n_new} new shown items ({sum(len(v) for v in old.values())} total) -> {rec}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory, e.g. experiment_records/<run_id>")
    ap.add_argument("--gen", type=int)
    ap.add_argument("--config", help="draw config (first draw only; later draws use the locked copy)")
    ap.add_argument("--target", help="Paper B: target domain of this generation")
    ap.add_argument("--export-jsonl", action="store_true")
    ap.add_argument("--force", action="store_true", help="redraw the LATEST generation")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--gate-only", action="store_true", help="only split / verify the fixed gate")
    ap.add_argument("--mark-shown", metavar="JSON",
                    help='record items shown to the editor in --gen: {"<dataset>": ["<id>", ...]}')
    a = ap.parse_args()
    run = Path(a.run).resolve()
    if a.status:
        return status(run)
    if a.gen is None and not a.gate_only or (a.gen is not None and a.gen < 0):
        sys.exit("--gen N (>= 0) is required")
    if a.mark_shown:
        return mark_shown(run, a.gen, Path(a.mark_shown))
    ddir = run / "draws"
    ddir.mkdir(parents=True, exist_ok=True)
    locked = ddir / "draw_config.yaml"
    if locked.exists():
        if a.config and sha256_file(a.config) != sha256_file(locked):
            sys.exit(f"{locked} is locked for this run and differs from --config")
    else:
        if not a.config:
            sys.exit("first draw of a run needs --config (e.g. data/draw/paper_a.yaml)")
        shutil.copyfile(a.config, locked)
    cfg = yaml.safe_load(open(locked))
    store = Store()
    gate = load_or_make_gate(ddir, cfg, store)
    if a.gate_only:
        print("fixed gate -> " + str(ddir / "gate") + "  " +
              "  ".join(f"{d}:{sum(len(v) for v in g.values())}" for d, g in gate.items()))
        return 0
    led_path = ddir / "ledger.json"
    ledger = load_ledger(led_path)
    gens = sorted(int(g) for g in ledger["gens"])
    gdir = ddir / f"gen_{a.gen:02d}"

    if a.gen in gens and not a.force:
        m = json.load(open(gdir / "manifest.json"))
        if (m.get("target") or None) != (a.target or None):
            sys.exit(f"gen {a.gen} was drawn with target={m.get('target')}; use --force to redraw it")
        print(f"gen {a.gen} already drawn; nothing changed ({gdir / 'manifest.json'})")
        if a.export_jsonl:
            export_jsonl(store, gdir, m["domains"])
        return 0
    if a.gen in gens and a.force:
        if a.gen != gens[-1]:
            sys.exit(f"--force only redraws the latest generation ({gens[-1]})")
        remove_gen(ledger, a.gen)
        shutil.rmtree(gdir, ignore_errors=True)
        gens = gens[:-1]
    expected = (gens[-1] + 1) if gens else 0
    if a.gen != expected:
        sys.exit(f"generations are drawn in order; next is {expected}")

    hashes = pool_hashes(cfg)
    if gens:
        first = json.load(open(ddir / f"gen_{gens[0]:02d}" / "manifest.json"))
        changed = [ds for ds, h in hashes.items() if first["pool_sha256"].get(ds) != h]
        if changed:
            sys.exit(f"draw pools changed since gen {gens[0]} (re-split?): {changed}. Start a new run.")

    domains, problems = draw_generation(cfg, run.name, a.gen, ledger, store, a.target)
    if problems:
        sys.exit("cannot draw this generation:\n  " + "\n  ".join(problems) +
                 "\nLower n in the config (new run) or allow reuse (reuse.across_generations: true).")

    # sanity: drawn ids are pool ids, disjoint across roles within this generation (except screen
    # subset of signal and kl_from_signal)
    for d, info in domains.items():
        for role, ids in info["roles"].items():
            for ds, lst in ids.items():
                pool = set(store.pool(ds))
                assert all(i in pool for i in lst), f"{d}/{role}/{ds}: id outside draw pool"
        for role, ids in info["roles"].items():
            for ds, lst in ids.items():
                assert not (set(lst) & store.gate.get(ds, set())), f"{d}/{role}/{ds}: drew a gate item"
        sig = {(ds, i) for ds, v in info["roles"].get("signal", {}).items() for i in v}
        scr = {(ds, i) for ds, v in info["roles"]["screen"].items() for i in v}
        assert scr <= sig, f"{d}: screen is not a subset of signal"
        tr = {(ds, i) for ds, v in info["roles"].get("train", {}).items() for i in v}
        assert not (tr & sig), f"{d}: train and signal overlap"

    for d, info in domains.items():
        for role, ids in info["roles"].items():
            if role == "screen":
                continue  # subset of signal; recorded via signal
            for ds, lst in ids.items():
                for i in lst:
                    ledger["items"].setdefault(ds, {}).setdefault(i, []).append([a.gen, role])
    manifest = {"run": run.name, "gen": a.gen, "paper": cfg["paper"], "target": a.target,
                "seed": cfg["seed"], "config_sha256": sha256_file(locked), "pool_sha256": hashes,
                "domains": domains, "warnings": problems}
    gdir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(gdir / "manifest.json", manifest)
    idir = gdir / "ids"
    idir.mkdir(exist_ok=True)
    per = defaultdict(list)
    for d, info in domains.items():
        for role, ids in info["roles"].items():
            for ds, lst in ids.items():
                per[(ds, role)] += lst
    for (ds, role), lst in per.items():
        write_json_atomic(idir / f"{ds}.{role}.json", sorted(set(lst)))
    ledger["gens"][str(a.gen)] = {"target": a.target, "manifest_sha256": sha256_file(gdir / "manifest.json")}
    write_json_atomic(led_path, ledger)
    if a.export_jsonl:
        export_jsonl(store, gdir, domains)
    print(f"gen {a.gen} drawn -> {gdir}")
    for d, info in domains.items():
        extra = f" reused={info['reused']}" if info["reused"] else ""
        print(f"  {d:22s} {info['counts']}{extra}")
    for p in problems:
        print("  note:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
