"""Run configuration: common CLI, merge of <paper_dir>/configs/{base,domains/<domain>}.yaml, lock file.

    ap = build_arg_parser("A"); ap.add_argument("--budgets", ...)    # papers may add arguments
    rc = load_run_config("A", ap.parse_args())

run_config.yaml is written on the first run of a run-id and locked: later runs with a different
paper / domain / model / seed / mock / paper_cfg are refused. `gens` and `gpus` may change
(extending a run, moving machines); the stored values are updated. The paper_cfg comparison ignores the keys of
removed features (LOCK_IGNORED_*, 2026-11-22: objective / scope profiles, data sizes, data roles, retention, E_t,
eval_target, draw_config; 2026-11-23: the search / gate / noise-band / KL-score knobs of REMOVED_KNOBS, and the
paper-only PAPER_REMOVED_KNOBS such as Paper A's tools.compare) and execution knobs that only change batching
(LOCK_IGNORED_SUFFIXES: scoring.concurrency, at any depth), so runs made before their removal resume with the same
command.

The search flow is hard-wired (2026-11-23): candidates are never screened, every candidate gets ONE gate evaluation
during the search (paired vs the current model), the passing candidate with the largest paired gain is accepted; no
api.evaluate, no full / rerun stage, no submissions after the search, no noise band, no KL score term. The knobs of
the removed variants (REMOVED_KNOBS) are refused on --set / --config unless given their hard-wired value (a no-op),
and a lock that used another value cannot be resumed.

One flow (2026-11-22, formerly `objective: simple` + `scope: single`): train role simple_<d>_pool, gate
simple_<d>_eval, held-out simple_<d>_test (rsicore.draws / rsicore.heldout). The default run id keeps the old
suffix: <paper>_<domain>_s<seed>_single_simple[_mock].
"""
import argparse
import copy
import json
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
LOCKED = ("paper", "domain", "model", "model_path", "seed", "mock", "paper_cfg")
RUN_SUFFIX = "_single_simple"     # default run-id suffix (kept from the objective simple / scope single era)

# paper_cfg keys of removed features (2026-11-22), ignored by the run lock: names at any depth, names containing a
# token, dotted paths, and the defaults of keys a newer base.yaml may add (absent = the default)
LOCK_IGNORED_KEYS = frozenset({
    "objective", "objective_profiles", "scope", "scope_profiles", "mock_objective_overrides", "data", "data_roles",
    "eval_target", "draw_config", "e_t", "e_task", "e_ref", "e_branch", "kl_contexts", "lambda_kl", "kl_limit"})
LOCK_IGNORED_TOKENS = ("retention", "theta0", "regmean")
# 2026-11-23: knobs of the removed search / gate variants, the noise band and Paper A's KL score term, by dotted path
# (also matched under mock_overrides.) -> the values equal to the hard-wired flow. --set / --config of such a key is
# dropped as a no-op when its value is one of these and refused otherwise (an empty tuple: always refused). Whole
# removed blocks: REMOVED_BLOCKS (any key under them is a removed knob; the block given as null is a no-op).
REMOVED_KNOBS = {
    "search.screen_candidates": (False,), "search.gate_every_candidate": (True,), "search.charge_screens": (False,),
    "search.program_evaluate_calls": (0,), "search.program_evaluate_total": (0, None), "search.max_evaluate": (),
    "search.top_full": (0,), "search.rerun_seeds": (0,), "gate.max_submissions": (),
    "search.kl_weight": (0,), "kl.eval_texts": (), "kl.top_k": (20,),                # Paper A KL score term
    "noise.repeats": (), "noise.temperature": (0,), "noise.top_p": (), "noise.n": (), "noise.fixed": (None,),
    "baseline.noise_repeats": (), "baseline.noise_temperature": (0,), "baseline.noise_top_p": (),
    "baseline.noise_n": (), "baseline.noise_fixed": (None,),                         # Paper B noise band
    "gate.use_noise_band": (), "gate.reference": ("auto", "parent"), "gate.epsilon": ()}   # Paper B raw reference
REMOVED_BLOCKS = ("noise",)                                                          # Paper A's noise: block
# removed knobs of one paper only (2026-11-23): the path is live in the other paper. Paper A's tools.compare (no
# candidate screens to compare; Paper B's api.compare is live): either boolean is a no-op.
PAPER_REMOVED_KNOBS = {"A": {"tools.compare": (True, False)}}
# removed knobs whose locked value is checked (_removed_features_used): another explicit value changed the search
LOCK_REFUSED_KNOBS = ("search.screen_candidates", "search.gate_every_candidate", "search.program_evaluate_calls",
                      "search.top_full", "search.rerun_seeds", "search.kl_weight", "noise.temperature", "noise.fixed",
                      "baseline.noise_temperature", "baseline.noise_fixed")
_REMOVED_LOCK_PATHS = tuple(REMOVED_KNOBS) + REMOVED_BLOCKS
# baselines (Paper A, 2026-11-23): the classic baselines are report-only (run once after the generations)
# reuse (2026-11-24): reuse.baseline_from copies M_0's generation-0 scores from a matching earlier run (checked by
# rsicore.reuse); it changes no number, so a run may be started or resumed with or without it
LOCK_IGNORED_PATHS = ("kl.domains", "heldout.domains", "max_tokens.math_eval", "baselines",
                      "mock_overrides.baselines", "reuse", "mock_overrides.reuse") + _REMOVED_LOCK_PATHS + tuple(
    f"mock_overrides.{p}" for p in _REMOVED_LOCK_PATHS)
# execution knobs that change only batching, like --gpus, ignored at this dotted path anywhere in the config (top
# level, under mock_overrides., under hardware_profiles.<name>.): scoring.concurrency (Paper A, 2026-11-23),
# gpu_layout.gate_split (Paper A, 2026-11-24: GPUs per candidate gate)
LOCK_IGNORED_SUFFIXES = ("scoring.concurrency", "gpu_layout.gate_split", "gate.kl.max_num_batched_tokens")
HARDWIRED_FLOW = ("candidates are never screened; every candidate gets one gate evaluation during the search, paired "
                  "vs the current model; the passing candidate with the largest paired gain is accepted; no "
                  "api.evaluate, full / rerun stage, gate submissions after the search, noise band or KL score term")
# Paper A KL gate (2026-11-24): runs locked before these keys existed used the accuracy gate
LOCK_DEFAULTS = {"simple.screen_n": 500, "gate.metric": "accuracy", "gate.kl.traces": 256, "gate.kl.max_tokens": 8192,
                 "gate.kl.top_k": 20, "gate.kl.temperature": 0.6, "gate.kl.top_p": 0.95, "gate.kl.sample_top_k": 20}
# --set / --config values of the removed selectors that are accepted as no-ops (the one remaining flow)
LEGACY_SELECTORS = {"objective": "simple", "scope": "single"}


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
    paper_cfg: dict       # base.yaml + domains/<domain>.yaml (+ --set overrides)
    cli_extra: dict = field(default_factory=dict)   # paper-specific CLI args (informational, not locked)


def eval_python() -> str:
    """Interpreter for eval/run_eval.py (the CPU eval env)."""
    if os.environ.get("RSI_EVAL_PYTHON"):
        return os.environ["RSI_EVAL_PYTHON"]
    p = EVAL_DIR / ".venv" / "bin" / "python"
    return str(p) if p.exists() else sys.executable


def build_arg_parser(paper: str) -> argparse.ArgumentParser:
    paper = paper.upper()
    ap = argparse.ArgumentParser(description=f"RSI Paper {paper} run (see eval/rsicore/INTERFACES.md)")
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--gens", type=int, default=5, help="number of generations")
    ap.add_argument("--model", default="qwen3.8-27b", help="download_models.sh name or a path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpus", default="0,1", help="comma-separated GPU ids")
    ap.add_argument("--run-id", default=None, help=f"default <paper>_<domain>_s<seed>{RUN_SUFFIX}[_mock]")
    ap.add_argument("--mock", action="store_true", help="no real model: mock OpenAI server + mock builders")
    ap.add_argument("--runs-dir", default=str(ROOT / "experiment_records"), help="parent of run directories")
    ap.add_argument("--paper-dir", default=None, help="override the paper code dir holding configs/")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a paper config value (dotted key, YAML value); repeatable")
    ap.add_argument("--config", action="append", default=[], metavar="YAML",
                    help="experiment config file(s) (2026-11-21): merged after the hardware profile and before --set "
                         "(later files win); its hardware key selects that profile")
    return ap


PROFILE_KEYS = ("hardware",)


def _legacy_selector(key: str, value, where: str) -> bool:
    """True for objective / scope (removed 2026-11-22) given with the one remaining value (simple / single): a no-op.
    Any other value is refused."""
    if key not in LEGACY_SELECTORS:
        return False
    if value != LEGACY_SELECTORS[key]:
        raise SystemExit(f"{where}: {key}={value!r} was removed (2026-11-22); the only flow is the former "
                         f"{key} {LEGACY_SELECTORS[key]!r} (drop the setting)")
    return True


def _load_config_files(paths) -> list[dict]:
    """--config files (2026-11-21): YAML mappings of paper config overrides. *_profiles keys are refused (a config file
    selects a profile by name, e.g. hardware: h20; it does not redefine one). objective: simple / scope: single (removed
    selectors) are dropped as no-ops."""
    out = []
    for p in paths or []:
        path = Path(p)
        if not path.is_absolute() and not path.exists() and (ROOT / path).exists():
            path = ROOT / path
        if not path.exists():
            raise SystemExit(f"--config {p}: file not found")
        d = yaml.safe_load(path.read_text()) or {}
        if not isinstance(d, dict):
            raise SystemExit(f"--config {p}: expected a YAML mapping, got {type(d).__name__}")
        bad = sorted(k for k in d if str(k).endswith("_profiles"))
        if bad:
            raise SystemExit(f"--config {p}: profile definitions are not allowed in a config file: {bad}")
        out.append({k: v for k, v in d.items() if not _legacy_selector(k, v, f"--config {p}")})
    return out


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


# ---------------------------------------------------------------------------- decoding (2026-11-21)
# A decoding dict (rsicore.evaluate.score / score_split `decoding=`): any of these keys; absent / null = the task's
# eval/configs/tasks.yaml value (greedy, thinking off). max_tokens wins over the run-wide max_tokens table. n > 1:
# each record's score is the mean over its n samples (avg@n). thinking: chat_template_kwargs enable_thinking; only
# the final answer is graded. min_p: only 0 / null (run_eval sends none; vLLM's default is 0).
# Paper config: `decoding: {screen: ..., full: ..., gate: ...}` (null = tasks.yaml; full falls back to screen) and
# heldout.decoding (also n_by_task). A null seed = the run's --seed.
DECODING_KEYS = ("temperature", "top_p", "top_k", "min_p", "n", "thinking", "max_tokens", "seed")
DECODING_STAGES = ("screen", "full", "gate")


def check_decoding(dec, where: str = "decoding", extra_keys=()) -> dict | None:
    """Validated copy of a decoding dict without its null values; None / {} -> None (tasks.yaml decoding)."""
    if dec is None:
        return None
    if not isinstance(dec, dict):
        raise SystemExit(f"{where} must be a mapping or null, got {dec!r}")
    bad = sorted(set(dec) - set(DECODING_KEYS) - set(extra_keys))
    if bad:
        raise SystemExit(f"{where}: unknown key(s) {bad}; known: {list(DECODING_KEYS) + list(extra_keys)}")
    out = {k: v for k, v in dec.items() if v is not None}
    for k in ("n", "max_tokens", "top_k", "seed"):
        lo = 1 if k in ("n", "max_tokens") else 0
        if k in out and (isinstance(out[k], bool) or not isinstance(out[k], int) or out[k] < lo):
            raise SystemExit(f"{where}.{k} must be an integer >= {lo}, got {out[k]!r}")
    for k in ("temperature", "top_p", "min_p"):
        if k in out and (isinstance(out[k], bool) or not isinstance(out[k], (int, float)) or out[k] < 0):
            raise SystemExit(f"{where}.{k} must be a non-negative number, got {out[k]!r}")
    if "thinking" in out and not isinstance(out["thinking"], bool):
        raise SystemExit(f"{where}.thinking must be true / false, got {out['thinking']!r}")
    if out.get("min_p"):
        raise SystemExit(f"{where}.min_p: only 0 / null is supported (run_eval sends no min_p; vLLM's default is 0)")
    return out or None


def stage_decoding(cfg: dict, stage: str, run_seed: int | None = None) -> dict | None:
    """Decoding dict of a scoring stage (screen / full / gate) from the paper config's `decoding` block, for
    rsicore.evaluate.score(..., decoding=...). None = tasks.yaml decoding (greedy). full falls back to screen.
    A null seed = run_seed (one experiment seed). The same dict must be used for the reference (incumbent / parent)
    and the candidates (common random numbers)."""
    if stage not in DECODING_STAGES:
        raise ValueError(f"stage must be one of {DECODING_STAGES}")
    block = cfg.get("decoding") or {}
    raw = block.get(stage)
    if raw is None and stage == "full":
        raw = block.get("screen")
    dec = check_decoding(raw, f"decoding.{stage}")
    if dec is not None and dec.get("seed") is None and run_seed is not None:
        dec["seed"] = int(run_seed)
    return dec


def _resolve_decoding(cfg: dict, seed: int) -> dict:
    """Validate the decoding blocks; a null seed = the run's --seed (written into the locked paper config)."""
    block = cfg.get("decoding")
    if block is not None:
        if not isinstance(block, dict):
            raise SystemExit(f"decoding must be a mapping {{screen, full, gate}}, got {block!r}")
        bad = sorted(set(block) - set(DECODING_STAGES))
        if bad:
            raise SystemExit(f"decoding: unknown stage(s) {bad}; known: {list(DECODING_STAGES)}")
        for st, d in block.items():
            if check_decoding(d, f"decoding.{st}") is not None and d.get("seed") is None:
                d["seed"] = int(seed)
    hd = (cfg.get("heldout") or {}).get("decoding")
    if isinstance(hd, dict):
        check_decoding(hd, "heldout.decoding", extra_keys=("n_by_task",))
        if hd.get("seed") is None:                   # one experiment seed (2026-11-20)
            hd["seed"] = int(seed)
    return cfg


def _resolve_run_domain(cfg: dict, domain: str) -> dict:
    """heldout.domains: run -> the run's domain."""
    h = cfg.get("heldout")
    if isinstance(h, dict) and h.get("domains") == "run":
        h["domains"] = domain
    return cfg


def resolve_model(model: str) -> Path:
    p = Path(model).expanduser()
    if p.exists() or "/" in model or model.startswith("."):
        return p.resolve()
    return Path(os.environ.get("MODELS_DIR", str(Path.home() / "models"))).expanduser() / model


def _plain(rc: RunConfig) -> dict:
    d = {k: getattr(rc, k) for k in rc.__dataclass_fields__}
    for k in ("run_dir", "model_path"):
        d[k] = str(d[k])
    return d


# top-level paper config keys that rsicore no longer reads (2026-11-22); setting them is refused
REMOVED_SET_KEYS = {"data": "simple.screen_n", "data_roles": None, "eval_target": None, "draw_config": None}


def _check_removed(key: str, where: str) -> None:
    top = str(key).split(".")[0]
    if top in REMOVED_SET_KEYS:
        alt = REMOVED_SET_KEYS[top]
        raise SystemExit(f"{where}: {top} was removed (2026-11-22; the train / gate / held-out sets are fixed to "
                         f"simple_<domain>_pool / _eval / _test)" + (f"; use {alt}" if alt else ""))


def _removed_table(paper: str | None) -> dict:
    """REMOVED_KNOBS plus the paper's PAPER_REMOVED_KNOBS (paper None: the shared table only)."""
    return {**REMOVED_KNOBS, **PAPER_REMOVED_KNOBS.get(str(paper or "").upper(), {})}


def _removed_knob(path: str, paper: str | None = None) -> str | None:
    """The removed-knob path (REMOVED_KNOBS, the paper's PAPER_REMOVED_KNOBS, or a path under a REMOVED_BLOCKS
    block) that dotted `path` names, also under mock_overrides.; None for a live key."""
    p = path[len("mock_overrides."):] if path.startswith("mock_overrides.") else path
    return p if (p in _removed_table(paper) or p.split(".")[0] in REMOVED_BLOCKS) else None


def _is_noop(rp: str, value, paper: str | None = None) -> bool:
    """value is a hard-wired value of removed knob rp (booleans only match booleans, None only None, numbers by
    value: 0 == 0.0)."""
    for v in _removed_table(paper).get(rp, (None,) if rp in REMOVED_BLOCKS else ()):
        if isinstance(v, bool) or isinstance(value, bool) or v is None or value is None:
            if type(v) is type(value) and v == value:
                return True
        elif value == v:
            return True
    return False


def _clean_removed(value, path: str, where: str, paper: str | None = None):
    """(keep, value) for a --set / --config value at dotted `path` (2026-11-23): a removed knob with its hard-wired
    value -> (False, None) (a no-op, dropped); with another value -> SystemExit; mappings are cleaned key by key."""
    rp = _removed_knob(path, paper) if path else None
    if rp is not None:
        if rp in REMOVED_BLOCKS and isinstance(value, dict):
            for k, v in value.items():
                _clean_removed(v, f"{path}.{k}", where, paper)
            return False, None
        if _is_noop(rp, value, paper):
            print(f"[rsicore] note: {where} {path}={value!r} is the hard-wired value of a removed knob (2026-11-23); "
                  "ignored", file=sys.stderr)
            return False, None
        noop = _removed_table(paper).get(rp, ())
        raise SystemExit(f"{where}: {path}={value!r}: this knob was removed (2026-11-23; {HARDWIRED_FLOW})"
                         + (f"; only {noop[0]!r} is accepted (a no-op)" if noop else "") + ". Drop the setting.")
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            keep, nv = _clean_removed(v, f"{path}.{k}" if path else str(k), where, paper)
            if keep:
                out[k] = nv
        return True, out
    return True, value


def _lock_ignored(key: str, path: str, paper: str | None = None) -> bool:
    if key in LOCK_IGNORED_KEYS or any(t in key for t in LOCK_IGNORED_TOKENS) or path in LOCK_IGNORED_PATHS:
        return True
    if any(path == s or path.endswith("." + s) for s in LOCK_IGNORED_SUFFIXES):
        return True
    extra = PAPER_REMOVED_KNOBS.get(str(paper or "").upper(), {})
    return bool(extra) and (path in extra or (path.startswith("mock_overrides.")
                                              and path[len("mock_overrides."):] in extra))


def _strip_ignored(x, path: str = "", paper: str | None = None):
    if not isinstance(x, dict):
        return x
    out = {}
    for k, v in x.items():
        p = f"{path}.{k}" if path else str(k)
        if not _lock_ignored(str(k), p, paper):
            sv = _strip_ignored(v, p, paper)
            if not (isinstance(v, dict) and v and not sv):      # a mapping that held only ignored keys: absent
                out[k] = sv
    return out


def lock_view(cfg: dict | None, paper: str | None = None) -> dict:
    """paper_cfg as the run lock compares it (2026-11-22): the former mock_objective_overrides.simple folded into
    mock_overrides (both were applied after --set, in this order), the former data.screen_n read as simple.screen_n,
    LOCK_DEFAULTS filled in, then every LOCK_IGNORED_* key (and the paper's PAPER_REMOVED_KNOBS) dropped. A run locked before the removals and the same
    command today give the same view."""
    c = copy.deepcopy(cfg or {})
    moo = (c.get("mock_objective_overrides") or {}).get("simple") if isinstance(c.get("mock_objective_overrides"), dict) else None
    if moo:
        c["mock_overrides"] = _deep_merge(c.get("mock_overrides") or {}, moo)
    sn = (c.get("data") or {}).get("screen_n") if isinstance(c.get("data"), dict) else None
    if sn is not None and not (isinstance(c.get("simple"), dict) and c["simple"].get("screen_n") is not None):
        c["simple"] = dict(c.get("simple") or {}, screen_n=sn)
    for path, v in LOCK_DEFAULTS.items():
        *head, leaf = path.split(".")
        d = c
        for h in head:
            if d.get(h) is None:
                d[h] = {}
            d = d[h]
            if not isinstance(d, dict):
                break
        else:
            if d.get(leaf) is None:
                d[leaf] = v
    return _strip_ignored(c, "", paper)


_ABSENT = object()


def _get_path(cfg: dict, path: str):
    d = cfg
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return _ABSENT
        d = d[part]
    return d


def _removed_features_used(ocfg: dict, domain: str, mock: bool = False) -> list[str]:
    """Removed features a locked paper_cfg actually used: E_t on, retention domains, data_roles other than the fixed
    simple sets (2026-11-22); an explicit LOCK_REFUSED_KNOBS value other than the hard-wired flow's (2026-11-23:
    screened candidates, no gate per candidate, api.evaluate, a full / rerun stage, a KL score term, a sampled or
    fixed noise band), read with mock_overrides applied when the lock is a mock run. Absent knobs are not refused
    (every lock made since they were added to base.yaml has them). The lock view ignores these keys, so they are
    checked by value here."""
    out = []
    eff = ocfg
    if mock:        # the effective mock values: mock_overrides, then the former mock_objective_overrides.simple
        mo = ocfg.get("mock_overrides") if isinstance(ocfg.get("mock_overrides"), dict) else {}
        moo = ocfg.get("mock_objective_overrides")
        moo = moo.get("simple") if isinstance(moo, dict) else None
        if isinstance(moo, dict):
            mo = _deep_merge(mo, moo)
        if mo:
            eff = _deep_merge(ocfg, mo)
    for path in LOCK_REFUSED_KNOBS:
        v = _get_path(eff, path)
        if v is not _ABSENT and not _is_noop(path, v):
            out.append(f"{path} {v!r}")
    for k in ("e_t", "e_task"):
        if isinstance(ocfg.get(k), dict) and ocfg[k].get("enabled"):
            out.append(f"{k}.enabled true")
    if isinstance(ocfg.get("retention"), dict) and ocfg["retention"].get("domains"):
        out.append("retention.domains " + repr(ocfg["retention"]["domains"]))
    fixed = {"train": f"simple_{domain}_pool", "gate": f"simple_{domain}_eval"}
    for role, want in fixed.items():
        v = (ocfg.get("data_roles") or {}).get(role) if isinstance(ocfg.get("data_roles"), dict) else None
        if v not in (None, want, f"eval:{want}"):
            out.append(f"data_roles.{role} {v!r}")
    return out


def _diff_paths(a, b, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in list(a) + [k for k in b if k not in a]:
            out += _diff_paths(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}.{k}" if path else str(k))
        return out
    return [] if a == b else [f"{path}: locked={a!r} now={b!r}"]


def load_run_config(paper: str, args) -> RunConfig:
    paper = paper.upper()
    if paper not in PAPER_DIRS:
        raise SystemExit(f"paper must be A or B, got {paper}")
    pdir = Path(args.paper_dir).resolve() if getattr(args, "paper_dir", None) else PAPER_DIRS[paper]
    cdir = pdir / "configs"
    if not (cdir / "base.yaml").exists():
        print(f"[rsicore] warning: {cdir / 'base.yaml'} not found; using empty paper config", file=sys.stderr)
    cfg = _deep_merge(_load_yaml(cdir / "base.yaml"), _load_yaml(cdir / "domains" / f"{args.domain}.yaml"))
    sets = []
    for it in list(getattr(args, "set", []) or []):   # objective=simple / scope=single: accepted no-ops
        k, eq, v = it.partition("=")
        if not (eq and _legacy_selector(k.strip(), yaml.safe_load(v), "--set")):
            if eq and k.strip() == "data.screen_n":     # removed 2026-11-22: same value, new key
                print("[rsicore] note: --set data.screen_n is now simple.screen_n", file=sys.stderr)
                sets.append(f"simple.screen_n={v}")
                continue
            _check_removed(k.strip(), "--set")
            if eq:                                      # removed search / gate / noise / KL knobs (2026-11-23)
                val = yaml.safe_load(v)
                keep, nv = _clean_removed(val, k.strip(), "--set", paper)
                if not keep:
                    continue
                if nv != val:
                    it = f"{k.strip()}={json.dumps(nv)}"
            sets.append(it)
    files = _load_config_files(getattr(args, "config", None))
    for f in files:
        for k in f:
            _check_removed(k, "--config")
    files = [_clean_removed(f, "", "--config", paper)[1] for f in files]
    for f in files:                              # a config file's hardware selects the profile
        cfg.update({k: f[k] for k in PROFILE_KEYS if f.get(k) is not None})   # (--set key=... still wins)
    cfg = _apply_hardware(cfg, sets)
    for f in files:                              # then the file's values, over the profile, under --set
        cfg = _deep_merge(cfg, {k: v for k, v in f.items() if k not in PROFILE_KEYS})
    cfg = _apply_set(cfg, sets)
    cfg = _resolve_run_domain(cfg, args.domain)
    # one experiment seed (2026-11-20): a null decoding seed (decoding.<stage>, heldout.decoding) = the run's --seed
    cfg = _resolve_decoding(cfg, int(args.seed))
    run_id = args.run_id or f"{paper}_{args.domain}_s{args.seed}{RUN_SUFFIX}" + ("_mock" if args.mock else "")
    run_dir = Path(getattr(args, "runs_dir", None) or ROOT / "experiment_records").resolve() / run_id
    gpus = [int(g) for g in str(args.gpus).split(",") if str(g).strip() != ""]
    common = {"domain", "gens", "model", "seed", "gpus", "run_id", "mock", "runs_dir", "paper_dir", "set"}
    extra = {k: v for k, v in vars(args).items() if k not in common}
    rc = RunConfig(paper=paper, domain=args.domain, gens=int(args.gens), run_id=run_id, run_dir=run_dir,
                   model=args.model, model_path=resolve_model(args.model), seed=int(args.seed), gpus=gpus,
                   mock=bool(args.mock), paper_cfg=cfg,
                   cli_extra=yaml.safe_load(yaml.safe_dump(extra, default_flow_style=False)) if extra else {})
    if not rc.mock and not rc.model_path.exists():
        raise SystemExit(f"model not found: {rc.model_path} (download with ./download_models.sh, or use --mock)")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "run_config.yaml"
    new = _plain(rc)
    if lock.exists():
        old = yaml.safe_load(lock.read_text()) or {}
        ocfg = old.get("paper_cfg") or {}
        for k, want in LEGACY_SELECTORS.items():    # runs of a removed objective / scope cannot be resumed
            if ocfg.get(k) not in (None, want):
                raise SystemExit(f"{lock} was made with {k} {ocfg[k]!r}, which was removed (2026-11-22); "
                                 "it cannot be resumed. Use a new --run-id.")
        used = _removed_features_used(ocfg, rc.domain, mock=bool(old.get("mock")))
        if used:
            raise SystemExit(f"{lock} was made with {', '.join(used)}, removed (2026-11-22 / 2026-11-23); it cannot "
                             "be resumed. Use a new --run-id.")
        lines = [f"  {k}: locked={old.get(k)!r} now={new[k]!r}" for k in LOCKED
                 if k != "paper_cfg" and old.get(k) != new[k]]
        lines += [f"  paper_cfg.{d}" for d in _diff_paths(lock_view(ocfg, paper), lock_view(cfg, paper))]
        if lines:
            raise SystemExit(f"{lock} is locked and differs from this invocation:\n" + "\n".join(lines) + "\n"
                             "Use a new --run-id, or rerun with the original arguments.")
        new["created"] = old.get("created")
    else:
        new["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = lock.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True))
    os.replace(tmp, lock)
    return rc
