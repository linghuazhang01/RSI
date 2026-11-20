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
"""
import hashlib
import json
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


def _eval_ids(rc, name: str | None = None) -> list[str]:
    name = name or eval_dataset(rc)
    split = ROOT / "data" / "splits" / name / "pool.json"
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_eval_target", ROOT / "data" / "scripts" / "build_eval_target.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build(name)                                  # no-op when up to date
    return json.loads(split.read_text())


def _run(args: list[str]) -> str:
    cmd = [eval_python(), "-I", str(DRAW_PY)] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"draw.py failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return r.stdout


def ensure_gate(rc) -> dict:
    """Split (first call) or reuse the fixed gate. Returns {domain: {dataset: [ids]}}."""
    gds = gate_dataset(rc)
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
    """Record items whose content was shown to the editor in generation `gen` (never signal again)."""
    shown = {ds: sorted(set(v)) for ds, v in (shown or {}).items() if v and not is_eval_set(ds)}
    if not shown:                                    # eval sets: the editor may see everything
        return
    blob = json.dumps(shown, sort_keys=True)
    p = _ddir(rc) / f"gen_{int(gen):02d}" / f"shown_req_{hashlib.sha256(blob.encode()).hexdigest()[:12]}.json"
    p.write_text(blob)
    _run(["--run", rc.run_dir, "--gen", int(gen), "--mark-shown", p])
