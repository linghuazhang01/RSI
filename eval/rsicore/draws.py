"""Per-generation draws of the simple flow (2026-11-21; the only flow since 2026-11-22).

Datasets of a run of domain d (rsicore.evalsets.simple_sets): train role simple_<d>_pool (the whole clean pool),
gate simple_<d>_eval (all 500 items), held-out simple_<d>_test (500). Generation g: candidates = pool ids - gate ids
- test ids; screen = signal = a sample of simple.screen_n (default 500) candidates with
random.Random(f"{seed}:simple:{domain}:{r}"), r = search_round(rc, g) (no run id or paper; Paper A's generation g and
Paper B's g-1 are the same round r, so both papers' k-th search screens the same items); train = the other
candidates; kl = KL_N (400) of train with random.Random(f"{seed}:simple_kl:{domain}:{r}"). Written to
draws/gen_XX/ids/simple_<d>_pool.<role>.json (+ manifest.json; reused on resume). Screens are drawn independently
every generation (they may overlap earlier generations' screens / train items); nothing is marked shown.
"""
import json
import random
from pathlib import Path

from .config import ROOT

# training-set names per domain (the raw datasets the simple_* sets come from); read only by gate.fixed_noise_band
# (noise.fixed {dataset | domain: band}) and kept for older callers
DOMAIN_TRAIN = {"math": ["dapo_math"], "code": ["taco_verified"], "knowledge": ["triviaqa_train"],
                "instruction_following": ["ifeval_train"], "tool_calling": ["sealtool_train"]}
ROLES = ("signal", "screen", "train", "kl")
KL_N = 400                                          # kl role size (Paper A's generation-0 calibration texts)


def _ddir(rc) -> Path:
    return Path(rc.run_dir) / "draws"


def train_dataset(rc) -> str:
    """The train role: simple_<domain>_pool (drawn per generation)."""
    return f"simple_{rc.domain}_pool"


def gate_dataset(rc) -> str:
    """The gate: simple_<domain>_eval (every item)."""
    return f"simple_{rc.domain}_eval"


def test_dataset(rc) -> str:
    """The held-out test set: simple_<domain>_test (every item; report only)."""
    return f"simple_{rc.domain}_test"


def is_eval_set(dataset) -> bool:
    from .evalsets import is_eval_set as _is
    return _is(dataset)


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


def simple_screen_n(rc) -> int:
    """Screen size: simple.screen_n, else 500."""
    cfg = getattr(rc, "paper_cfg", None) or {}
    return int((cfg.get("simple") or {}).get("screen_n") or 500)


_SIMPLE: dict[tuple, dict] = {}                     # (run dir, gen) -> {role: ids}


def search_round(rc, gen: int) -> int:
    """The paper-independent index of generation `gen` for the screen seed: Paper A searches in generations 1..N
    (0 = baseline), Paper B in 0..N-1, so A's g and B's g-1 are the same round (A's g=1 and B's g=0 screen identical
    items)."""
    return int(gen) if getattr(rc, "paper", "A") == "A" else int(gen) + 1


def simple_draw(rc, gen: int) -> dict:
    """The draw of generation `gen` (see the module doc): {role: sorted ids} for ROLES. Written once to
    draws/gen_XX/ (ids/<pool>.<role>.json, manifest.json) and reused (resume, audit)."""
    pool = train_dataset(rc)
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
        kl = sorted(random.Random(f"{seed}:simple_kl:{dom}:{rnd}").sample(train, min(KL_N, len(train))))
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


def ensure_gate(rc) -> dict:
    """Build the gate and the held-out test set (once per process). Returns {domain: {gate dataset: [ids]}}."""
    gds = gate_dataset(rc)
    build_eval_set(test_dataset(rc))
    build_eval_set(train_dataset(rc))        # also on a resume that finds the draw files already written
    return {rc.domain: {gds: build_eval_set(gds)}}


def draw_gen(rc, gen: int, target: str | None = None) -> dict:
    """Draw generation `gen` (idempotent: an existing draw is reused). Returns the manifest (target = `target`)."""
    simple_draw(rc, gen)
    m = json.loads((_ddir(rc) / f"gen_{int(gen):02d}" / "manifest.json").read_text())
    return dict(m, target=target)


def ids_file(rc, gen: int, dataset: str, role: str) -> Path:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    return _ddir(rc) / f"gen_{int(gen):02d}" / "ids" / f"{dataset}.{role}.json"


def ids(rc, gen: int, dataset: str, role: str) -> list[str]:
    """The ids of `role` in generation `gen`'s draw of the train role (dataset must be train_dataset(rc))."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if dataset != train_dataset(rc):
        raise ValueError(f"draws.ids: only the train role {train_dataset(rc)} is drawn, got {dataset!r}")
    return list(simple_draw(rc, gen)[role])


def gate_ids(rc, dataset: str) -> list[str]:
    """Every item of the eval-set `dataset` (the gate: simple_<domain>_eval)."""
    if not is_eval_set(dataset):
        raise ValueError(f"draws.gate_ids: {dataset!r} is not an eval set (eval/configs/tasks.yaml eval_targets)")
    return build_eval_set(dataset)
