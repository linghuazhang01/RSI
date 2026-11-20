"""Run configuration: common CLI, merge of <paper_dir>/configs/{base,domains/<domain>}.yaml, lock file.

    ap = build_arg_parser("A"); ap.add_argument("--budgets", ...)    # papers may add arguments
    rc = load_run_config("A", ap.parse_args())

run_config.yaml is written on the first run of a run-id and locked: later runs with a different
paper / domain / model / seed / mock / draw_config / paper_cfg are refused. `gens` and `gpus` may change
(extending a run, moving machines); the stored values are updated.
"""
import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval"
DOMAINS = ["math", "code", "knowledge", "instruction_following", "tool_calling"]
PAPER_DIRS = {"A": ROOT / "edit_code", "B": ROOT / "lora_code"}
DRAW_CONFIGS = {"A": ROOT / "data" / "draw" / "paper_a.yaml", "B": ROOT / "data" / "draw" / "paper_b.yaml"}
LOCKED = ("paper", "domain", "model", "model_path", "seed", "mock", "draw_config", "paper_cfg")


@dataclass
class RunConfig:
    paper: str            # "A" | "B"
    domain: str
    gens: int
    run_id: str
    run_dir: Path
    model: str            # name or path, as given
    model_path: Path      # resolved path (may not exist in mock mode)
    seed: int
    gpus: list
    mock: bool
    draw_config: Path     # data/draw/paper_a.yaml or paper_b.yaml
    paper_cfg: dict       # base.yaml + domains/<domain>.yaml (+ --set overrides)
    cli_extra: dict = field(default_factory=dict)   # paper-specific CLI args (informational, not locked)


def eval_python() -> str:
    """Interpreter for eval/run_eval.py and data/scripts/draw.py (the CPU eval env)."""
    if os.environ.get("RSI_EVAL_PYTHON"):
        return os.environ["RSI_EVAL_PYTHON"]
    p = EVAL_DIR / ".venv" / "bin" / "python"
    return str(p) if p.exists() else sys.executable


def build_arg_parser(paper: str) -> argparse.ArgumentParser:
    paper = paper.upper()
    doms = DOMAINS + (["all"] if paper == "A" else [])
    ap = argparse.ArgumentParser(description=f"RSI Paper {paper} run (see eval/rsicore/INTERFACES.md)")
    ap.add_argument("--domain", required=True, choices=doms)
    ap.add_argument("--gens", type=int, default=5, help="number of generations")
    ap.add_argument("--model", default="qwen3.8-27b", help="download_models.sh name or a path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpus", default="0,1", help="comma-separated GPU ids")
    ap.add_argument("--run-id", default=None, help="default <paper>_<domain>_s<seed>[_single][_mock]")
    ap.add_argument("--mock", action="store_true", help="no real model: mock OpenAI server + mock builders")
    ap.add_argument("--runs-dir", default=str(ROOT / "experiment_records"), help="parent of run directories")
    ap.add_argument("--paper-dir", default=None, help="override the paper code dir holding configs/")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a paper config value (dotted key, YAML value); repeatable")
    return ap


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _apply_set(cfg: dict, items) -> dict:
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            if not isinstance(d.get(p), dict):   # e.g. gpu_roles: null in base.yaml + --set gpu_roles.train=[2,3]
                d[p] = {}
            d = d[p]
        d[parts[-1]] = yaml.safe_load(v)
    return cfg


def _apply_profile(cfg: dict, sets, key: str, default: str) -> dict:
    """Merge cfg[f"{key}_profiles"][name] (name = --set key=... or cfg[key] or default) BEFORE the --set
    overrides, so explicit --set values win over the profile. The result is locked in run_config.yaml."""
    profiles = cfg.get(f"{key}_profiles") or {}
    if not profiles:
        return cfg
    name = cfg.get(key) or default
    for it in sets:
        k, _, v = it.partition("=")
        if k.strip() == key:
            name = str(yaml.safe_load(v))
    if name not in profiles:
        raise SystemExit(f"unknown {key} profile {name!r}; known: {sorted(profiles)}")
    cfg = _deep_merge(cfg, profiles[name] or {})
    cfg[key] = name
    return cfg


def _apply_hardware(cfg: dict, sets) -> dict:
    """hardware: b300 | h20 ... (hardware_profiles)."""
    return _apply_profile(cfg, sets, "hardware", "b300")


def _apply_scope(cfg: dict, sets, domain: str) -> dict:
    """scope: single | all (scope_profiles). single = one domain only: training, signal, gate, feedback and
    held-out evaluation all on the run's --domain (the paper's scope_profiles.single says which keys that
    sets). --domain all (Paper A) runs with scope all unless --set scope=single is given (refused)."""
    explicit = any(it.partition("=")[0].strip() == "scope" for it in sets)
    if domain == "all" and cfg.get("scope_profiles"):
        if explicit and any(it.partition("=")[0].strip() == "scope" and yaml.safe_load(it.partition("=")[2]) == "single"
                            for it in sets):
            raise SystemExit("scope=single needs one --domain, not 'all'")
        cfg = dict(cfg, scope="all")
    return _apply_profile(cfg, sets, "scope", "single")


def _resolve_run_domain(cfg: dict, domain: str) -> dict:
    """heldout.domains: run -> the run's domain ('all' for Paper A's --domain all)."""
    h = cfg.get("heldout")
    if isinstance(h, dict) and h.get("domains") == "run":
        h["domains"] = domain
    return cfg


DATA_TOP = {"A": ("screen_n", "kl_contexts"), "B": ("screen_n", "target_train_frac", "retention_n", "retention_kl")}
DATA_DOMAIN = ("n", "gate_n")


def kl_domains_for(cfg: dict, paper: str, domain: str) -> list[str] | None:
    """Paper A: the domains whose KL / calibration texts the run uses. kl.domains other than `all` (e.g. `run`,
    set by scope=single) with one --domain -> [domain], so the draw puts every kl_contexts text in that domain.
    None = the draw's default (kl_contexts split over all five domains)."""
    if paper != "A" or domain == "all":
        return None
    return None if (cfg.get("kl") or {}).get("domains", "all") == "all" else [domain]


def apply_data_sizes(draw: dict, data: dict | None, paper: str, kl_domains: list[str] | None = None) -> dict:
    """Draw config with the run config's `data:` sizes applied (null = keep the draw config's value).
    Top-level sizes (DATA_TOP) are numbers; per-domain sizes (n, gate_n) are a number (every domain) or
    {domain: number} (the domains not listed keep theirs). kl_domains (Paper A, kl_domains_for) restricts the
    domains kl_contexts is split over."""
    out = copy.deepcopy(draw)
    if kl_domains:
        out["kl_domains"] = list(kl_domains)
    data = {k: v for k, v in (data or {}).items() if v is not None}
    known = DATA_TOP[paper] + DATA_DOMAIN
    bad = sorted(set(data) - set(known))
    if bad:
        raise SystemExit(f"data: unknown size key(s) {bad} for Paper {paper}; known: {list(known)}")

    def num(key, v, frac=False):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SystemExit(f"data.{key} must be a number, got {v!r}")
        if frac:
            if not 0 < float(v) < 1:
                raise SystemExit(f"data.{key} must be in (0, 1), got {v!r}")
            return float(v)
        if int(v) != v or int(v) < 1:
            raise SystemExit(f"data.{key} must be a positive integer, got {v!r}")
        return int(v)

    for k in DATA_TOP[paper]:
        if k in data:
            out[k] = num(k, data[k], frac=(k == "target_train_frac"))
    doms = out.get("domains") or {}
    for k in DATA_DOMAIN:
        if k not in data:
            continue
        spec = data[k]
        if isinstance(spec, dict):
            unknown = sorted(set(spec) - set(doms))
            if unknown:
                raise SystemExit(f"data.{k}: unknown domain(s) {unknown}; known: {sorted(doms)}")
            vals = {d: num(f"{k}.{d}", v) for d, v in spec.items() if v is not None}
        else:
            vals = {d: num(k, spec) for d in doms}
        for d, v in vals.items():
            doms[d][k] = v
    return out


def _effective_draw_config(base: Path, data: dict | None, paper: str, run_dir: Path,
                           kl_domains: list[str] | None = None, seed: int | None = None) -> Path:
    """Write run_dir/draw_config.effective.yaml: `base` with the `data:` sizes, kl_domains and the run's seed
    (the per-generation draws; the gate split keeps gate.seed) applied
    (deterministic content, so a resume with the same settings matches the copy data/scripts/draw.py locked in
    draws/draw_config.yaml; different settings are refused there)."""
    eff = apply_data_sizes(_load_yaml(base), data, paper, kl_domains)
    if seed is not None:
        eff["seed"] = int(seed)
    eff["data_sizes_from"] = {"draw_config": str(base), "run_config_data": copy.deepcopy(data or {}),
                              "kl_domains": kl_domains}
    p = run_dir / "draw_config.effective.yaml"
    txt = yaml.safe_dump(eff, sort_keys=True, allow_unicode=True)
    if not p.exists() or p.read_text() != txt:
        p.write_text(txt)
    return p


def resolve_model(model: str) -> Path:
    p = Path(model).expanduser()
    if p.exists() or "/" in model or model.startswith("."):
        return p.resolve()
    return Path(os.environ.get("MODELS_DIR", str(Path.home() / "models"))).expanduser() / model


def _plain(rc: RunConfig) -> dict:
    d = {k: getattr(rc, k) for k in rc.__dataclass_fields__}
    for k in ("run_dir", "model_path", "draw_config"):
        d[k] = str(d[k])
    return d


def load_run_config(paper: str, args) -> RunConfig:
    paper = paper.upper()
    if paper not in PAPER_DIRS:
        raise SystemExit(f"paper must be A or B, got {paper}")
    pdir = Path(args.paper_dir).resolve() if getattr(args, "paper_dir", None) else PAPER_DIRS[paper]
    cdir = pdir / "configs"
    if not (cdir / "base.yaml").exists():
        print(f"[rsicore] warning: {cdir / 'base.yaml'} not found; using empty paper config", file=sys.stderr)
    cfg = _deep_merge(_load_yaml(cdir / "base.yaml"), _load_yaml(cdir / "domains" / f"{args.domain}.yaml"))
    sets = list(getattr(args, "set", []) or [])
    cfg = _apply_hardware(cfg, sets)
    cfg = _apply_scope(cfg, sets, args.domain)
    cfg = _apply_set(cfg, sets)
    cfg = _resolve_run_domain(cfg, args.domain)
    # one experiment seed (2026-11-20): the held-out decoding seed null = the run's --seed
    hd = (cfg.get("heldout") or {}).get("decoding")
    if isinstance(hd, dict) and hd.get("seed") is None:
        hd["seed"] = int(args.seed)
    draw_cfg = cfg.get("draw_config")
    draw_cfg = base_draw = (ROOT / draw_cfg).resolve() if draw_cfg else DRAW_CONFIGS[paper]
    apply_data_sizes(_load_yaml(base_draw), cfg.get("data"), paper)   # validate the `data:` sizes early
    kl_doms = kl_domains_for(cfg, paper, args.domain)
    # the per-generation draws use the run's --seed too: a seed other than the draw config's -> effective copy
    seed_differs = int(_load_yaml(base_draw).get("seed", args.seed)) != int(args.seed)
    sized = kl_doms is not None or seed_differs or any(v is not None for v in (cfg.get("data") or {}).values())
    single = cfg.get("scope") == "single" and args.domain != "all"
    run_id = args.run_id or (f"{paper}_{args.domain}_s{args.seed}" + ("_single" if single else "")
                             + ("_mock" if args.mock else ""))
    run_dir = Path(getattr(args, "runs_dir", None) or ROOT / "experiment_records").resolve() / run_id
    if sized:
        draw_cfg = run_dir / "draw_config.effective.yaml"   # written below, once the lock check passed
    gpus = [int(g) for g in str(args.gpus).split(",") if str(g).strip() != ""]
    common = {"domain", "gens", "model", "seed", "gpus", "run_id", "mock", "runs_dir", "paper_dir", "set"}
    extra = {k: v for k, v in vars(args).items() if k not in common}
    rc = RunConfig(paper=paper, domain=args.domain, gens=int(args.gens), run_id=run_id, run_dir=run_dir,
                   model=args.model, model_path=resolve_model(args.model), seed=int(args.seed), gpus=gpus,
                   mock=bool(args.mock), draw_config=draw_cfg, paper_cfg=cfg,
                   cli_extra=yaml.safe_load(yaml.safe_dump(extra, default_flow_style=False)) if extra else {})
    if not rc.mock and not rc.model_path.exists():
        raise SystemExit(f"model not found: {rc.model_path} (download with ./download_models.sh, or use --mock)")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "run_config.yaml"
    new = _plain(rc)
    if lock.exists():
        old = yaml.safe_load(lock.read_text()) or {}
        diff = [k for k in LOCKED if old.get(k) != new[k]]
        if diff:
            lines = "\n".join(f"  {k}: locked={old.get(k)!r} now={new[k]!r}" for k in diff)
            raise SystemExit(f"{lock} is locked and differs from this invocation:\n{lines}\n"
                             "Use a new --run-id, or rerun with the original arguments.")
        new["created"] = old.get("created")
    else:
        new["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if sized:
        _effective_draw_config(base_draw, cfg.get("data"), paper, run_dir, kl_doms, seed=int(args.seed))
    tmp = lock.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True))
    os.replace(tmp, lock)
    return rc
