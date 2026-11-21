"""Wrappers around data/scripts/draw.py (run as a subprocess with `python -I`).

Layout written by draw.py inside the run dir: draws/draw_config.yaml (locked), draws/gate/,
draws/gen_XX/{manifest.json, ids/<dataset>.<role>.json}, draws/ledger.json.

objective eval_overfit (2026-11-20, paper config `eval_target`): no draws at all. The run's dataset is the
eval-set target (data/scripts/build_eval_target.py, e.g. math_eval = MATH-500 + GSM8K test, built on demand);
every role (signal, screen, train) and the gate are ALL of its items, the kl role a fixed seeded subset of
eval_target.kl_n items; nothing is marked shown (the editor may see everything).

data roles (2026-11-21, paper config `data_roles`, rsicore.evalsets): train_dataset(rc) / gate_dataset(rc) say which
dataset the editor works from (signal / screen / train / kl) and which one accepts candidates. Each is a training set
(per-generation draws / the fixed gate split, as before) or an eval set K_eval (every item, nothing drawn or marked
shown). Defaults: the domain's training set for both; objective eval_overfit -> the eval set for both.

objective simple (2026-11-21; train role simple_<d>_pool, gate simple_<d>_eval, held-out simple_<d>_test): no
draw.py. Generation g: candidates = pool ids - gate ids - test ids; screen = signal = a sample of simple.screen_n
(else data.screen_n, else 500) candidates with random.Random(f"{seed}:simple:{domain}:{r}"), r = search_round(rc, g)
(no run id or paper; Paper A's generation g and Paper B's g-1 are the same round r, so both papers' k-th search
screens the same items); train = the other candidates; kl = eval_target.kl_n (default 400) of train with
random.Random(f"{seed}:simple_kl:{domain}:{r}"). Written to draws/gen_XX/ids/simple_<d>_pool.<role>.json (+
manifest.json; reused on resume). Screens are drawn independently every generation (they may overlap earlier
generations' screens / train items); mark_shown does nothing. The gate is all 500 eval items (as before).
"""
import hashlib
import json
import random
import subprocess
from pathlib import Path

from .config import ROOT, eval_python

DRAW_PY = ROOT / "data" / "scripts" / "draw.py"
DOMAIN_TRAIN = {"math": ["dapo_math"], "code": ["taco_verified"], "knowledge": ["triviaqa_train"],
                "instruction_following": ["ifeval_train"], "tool_calling": ["sealtool_train"]}
ROLES = ("signal", "screen", "train", "kl")


def _ddir(rc) -> Path:
    return Path(rc.run_dir) / "draws"


def eval_target(rc) -> dict | None:
    """The paper config's eval_target block when the objective optimises on the eval set itself, else None."""
    et = (getattr(rc, "paper_cfg", None) or {}).get("eval_target") or {}
    return et if et.get("enabled") else None


def eval_dataset(rc) -> str | None:
    et = eval_target(rc)
    return et.get("dataset", "math_eval") if et else None


def roles(rc) -> dict:
    """{"train": dataset, "gate": dataset} of the run (data_roles over the objective's defaults)."""
    from .evalsets import default_roles
    cfg = getattr(rc, "paper_cfg", None) or {}
    dr = cfg.get("data_roles") or {}
    base = default_roles(cfg, rc.domain) if rc.domain != "all" else {"train": None, "gate": None}
    return {k: dr.get(k) or base[k] for k in ("train", "gate")}


def train_dataset(rc) -> str | None:
    return roles(rc)["train"]


def gate_dataset(rc) -> str | None:
    return roles(rc)["gate"]


def is_eval_set(dataset) -> bool:
    from .evalsets import is_eval_set as _is
    return _is(dataset)


def _needs_draws(rc) -> bool:
    """A training set in the train or gate role (or no single domain): draw.py runs."""
    r = roles(rc)
    return rc.domain == "all" or not (is_eval_set(r["train"]) and is_eval_set(r["gate"]))


def _kl_n(rc) -> int:
    cfg = getattr(rc, "paper_cfg", None) or {}
    return int((cfg.get("eval_target") or {}).get("kl_n", 400))


_BUILT: dict[str, list[str]] = {}                   # eval set -> its ids, once per process (2026-11-21)


def build_eval_set(name: str) -> list[str]:
    """Build (data/scripts/build_eval_target.py; a no-op when up to date) the eval-set dataset `name` into
    data/processed/<name>.jsonl + data/splits/<name>/pool.json, once per process; returns its sorted ids."""
    if name not in _BUILT:
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_eval_target",
                                                      ROOT / "data" / "scripts" / "build_eval_target.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.build(name)
        _BUILT[name] = json.loads((ROOT / "data" / "splits" / name / "pool.json").read_text())
    return list(_BUILT[name])


def _eval_ids(rc, name: str | None = None) -> list[str]:
    return build_eval_set(name or eval_dataset(rc))


# ---------------------------------------------------------------------------- objective simple (2026-11-21)
def is_simple(rc) -> bool:
    return (getattr(rc, "paper_cfg", None) or {}).get("objective") == "simple" and rc.domain != "all"


def simple_pool(rc) -> str | None:
    """The dataset drawn per generation (simple_<d>_pool) when objective simple uses it as the train role, else None."""
    if not is_simple(rc):
        return None
    tds = train_dataset(rc)
    return tds if tds == f"simple_{rc.domain}_pool" else None


def test_dataset(rc) -> str | None:
    """objective simple's held-out test set (simple_<d>_test), else None."""
    return f"simple_{rc.domain}_test" if is_simple(rc) else None


def simple_screen_n(rc) -> int:
    """Screen size of objective simple: simple.screen_n, else data.screen_n, else 500."""
    cfg = getattr(rc, "paper_cfg", None) or {}
    v = (cfg.get("simple") or {}).get("screen_n") or (cfg.get("data") or {}).get("screen_n") or 500
    return int(v)


_SIMPLE: dict[tuple, dict] = {}                     # (run dir, gen) -> {role: ids}


def search_round(rc, gen: int) -> int:
    """The paper-independent index of generation `gen` for objective simple's screen seed: Paper A searches in
    generations 1..N (0 = baseline), Paper B in 0..N-1, so A's g and B's g-1 are the same round (A's g=1 and B's g=0
    screen identical items)."""
    return int(gen) if getattr(rc, "paper", "A") == "A" else int(gen) + 1


def simple_draw(rc, gen: int) -> dict:
    """objective simple's draw of generation `gen` (see the module doc): {role: sorted ids} for ROLES. Written once to
    draws/gen_XX/ (ids/<pool>.<role>.json, manifest.json) and reused (resume, audit)."""
    pool = simple_pool(rc)
    if pool is None:
        raise ValueError("simple_draw needs objective simple with train role simple_<domain>_pool")
    key = (str(Path(rc.run_dir).resolve()), int(gen))
    if key in _SIMPLE:
        return _SIMPLE[key]
    files = {r: ids_file(rc, gen, pool, r) for r in ROLES}
    man = _ddir(rc) / f"gen_{int(gen):02d}" / "manifest.json"
    if man.exists() and all(p.exists() for p in files.values()):
        out = {r: json.loads(p.read_text()) for r, p in files.items()}
    else:
        seed, dom, gds, tst = getattr(rc, "seed", 42), rc.domain, gate_dataset(rc), test_dataset(rc)
        rnd = search_round(rc, gen)
        allids = build_eval_set(pool)
        excl = set(gate_ids(rc, gds)) | set(build_eval_set(tst))
        cand = sorted(set(allids) - excl)
        n = min(simple_screen_n(rc), len(cand))
        screen = sorted(random.Random(f"{seed}:simple:{dom}:{rnd}").sample(cand, n))
        sset = set(screen)
        train = [i for i in cand if i not in sset]
        kl = sorted(random.Random(f"{seed}:simple_kl:{dom}:{rnd}").sample(train, min(_kl_n(rc), len(train))))
        out = {"signal": screen, "screen": screen, "train": train, "kl": kl}
        files["signal"].parent.mkdir(parents=True, exist_ok=True)
        for r, p in files.items():
            p.write_text(json.dumps(out[r]))
        man.write_text(json.dumps({
            "gen": int(gen), "target": None, "simple": True, "dataset": pool, "screen_n": n,
            "round": rnd, "seed": f"{seed}:simple:{dom}:{rnd}", "kl_seed": f"{seed}:simple_kl:{dom}:{rnd}",
            "excluded": {"gate": gds, "test": tst}, "pool": len(allids),
            "candidates": len(cand), "n": {r: len(v) for r, v in out.items()}}, indent=1))
    _SIMPLE[key] = out
    return out


def _run(args: list[str]) -> str:
    cmd = [eval_python(), "-I", str(DRAW_PY)] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"draw.py failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return r.stdout


def ensure_gate(rc) -> dict:
    """Split (first call) or reuse the fixed gate. Returns {domain: {dataset: [ids]}}."""
    gds = gate_dataset(rc)
    if test_dataset(rc):                             # objective simple: build the held-out test set early
        build_eval_set(test_dataset(rc))
    if not _needs_draws(rc):
        return {rc.domain: {gds: _eval_ids(rc, gds)}}
    man = _ddir(rc) / "gate" / "manifest.json"
    if not (man.exists() and (_ddir(rc) / "draw_config.yaml").exists()):
        _run(["--run", rc.run_dir, "--gate-only", "--config", rc.draw_config])
    doms = json.loads(man.read_text())["domains"]
    if is_eval_set(gds):                             # an eval-set gate next to a drawn training set
        doms = dict(doms, **{rc.domain: dict(doms.get(rc.domain) or {}, **{gds: _eval_ids(rc, gds)})})
    return doms


def draw_gen(rc, gen: int, target: str | None = None) -> dict:
    """Draw generation `gen` (idempotent: an existing draw with the same target is reused). Returns the manifest."""
    if simple_pool(rc):                              # objective simple: rsicore draws it (no draw.py)
        simple_draw(rc, gen)
        m = json.loads((_ddir(rc) / f"gen_{int(gen):02d}" / "manifest.json").read_text())
        return dict(m, target=target)
    tds = train_dataset(rc) if rc.domain != "all" else None
    if is_eval_set(tds) and not _needs_draws(rc):
        return {"gen": int(gen), "target": target, "eval_target": tds, "n": len(_eval_ids(rc, tds))}
    man = _ddir(rc) / f"gen_{int(gen):02d}" / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        if (m.get("target") or None) == (target or None):
            return m
    args = ["--run", rc.run_dir, "--gen", int(gen), "--config", rc.draw_config]
    if target:
        args += ["--target", target]
    _run(args)
    return json.loads(man.read_text())


def ids_file(rc, gen: int, dataset: str, role: str) -> Path:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    return _ddir(rc) / f"gen_{int(gen):02d}" / "ids" / f"{dataset}.{role}.json"


def ids(rc, gen: int, dataset: str, role: str) -> list[str]:
    if dataset and dataset == simple_pool(rc):       # objective simple: the generation's own draw (before eval sets)
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        return list(simple_draw(rc, gen)[role])
    if is_eval_set(dataset):
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        allids = _eval_ids(rc, dataset)
        if role == "kl":                             # fixed seeded subset (the same in every generation)
            import random
            n = _kl_n(rc)
            return sorted(random.Random(f"{getattr(rc, 'seed', 42)}:eval_kl").sample(allids, min(n, len(allids))))
        return allids
    p = ids_file(rc, gen, dataset, role)
    return json.loads(p.read_text()) if p.exists() else []


def gate_ids_file(rc, dataset: str) -> Path:
    return _ddir(rc) / "gate" / f"{dataset}.gate.json"


def gate_ids(rc, dataset: str) -> list[str]:
    if is_eval_set(dataset):
        return _eval_ids(rc, dataset)
    p = gate_ids_file(rc, dataset)
    return json.loads(p.read_text()) if p.exists() else []


def mark_shown(rc, gen: int, shown: dict[str, list[str]]) -> None:
    """Record items whose content was shown to the editor in generation `gen` (never signal again). Eval sets
    (incl. objective simple's pool, gate and test sets: screens are redrawn independently every generation) are
    ignored."""
    shown = {ds: sorted(set(v)) for ds, v in (shown or {}).items() if v and not is_eval_set(ds)}
    if not shown:                                    # eval sets: the editor may see everything
        return
    blob = json.dumps(shown, sort_keys=True)
    p = _ddir(rc) / f"gen_{int(gen):02d}" / f"shown_req_{hashlib.sha256(blob.encode()).hexdigest()[:12]}.json"
    p.write_text(blob)
    _run(["--run", rc.run_dir, "--gen", int(gen), "--mark-shown", p])
