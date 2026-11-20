"""Eval-set targets of objective eval_overfit (2026-11-21): which eval set a run optimises on directly.

eval/configs/tasks.yaml `eval_targets` lists them: key K (e.g. math, humanevalplus, ifeval) -> source (task, pool)
pairs. The run's dataset and run_eval task is K_eval (built by data/scripts/build_eval_target.py; the task entry is
derived by evallib.tasks.eval_target_tasks). One eval set per run; its domain is the source task's category.
"""
from functools import lru_cache

import yaml

from .config import EVAL_DIR

TASKS_YAML = EVAL_DIR / "configs" / "tasks.yaml"


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return yaml.safe_load(TASKS_YAML.read_text()) or {}


def targets() -> dict:
    """{key: {"dataset": K_eval, "domain", "sources": [(task, pool)], "source_datasets": [(dataset, pool)]}}."""
    tasks, out = _cfg().get("tasks") or {}, {}
    for key, sources in (_cfg().get("eval_targets") or {}).items():
        if isinstance(sources, dict):              # a fixed item file (simple_data): {task, file, dataset}
            t = sources["task"]
            out[key] = {"dataset": sources["dataset"], "domain": tasks[t]["category"], "sources": [(t, "file")],
                        "source_datasets": [], "file": sources["file"]}
            continue
        src = [(str(t), str(p)) for t, p in sources]
        out[key] = {"dataset": f"{key}_eval", "domain": tasks[src[0][0]]["category"], "sources": src,
                    "source_datasets": [(tasks[t]["dataset"], p) for t, p in src]}
    return out


def key_of(name: str) -> str | None:
    """Target key of a key or a dataset name (humanevalplus / humanevalplus_eval -> humanevalplus); None if unknown."""
    t = targets()
    if name in t:
        return name
    for k, v in t.items():
        if v["dataset"] == name:
            return k
    return None


def resolve(name: str | None, domain: str) -> tuple[str, dict]:
    """(key, target) for eval_target.dataset `name` in a run of `domain`; name None -> the domain's default."""
    t = targets()
    if not name:
        name = (_cfg().get("eval_target_defaults") or {}).get(domain)
        if not name:
            raise SystemExit(f"objective eval_overfit: no default eval set for domain {domain!r}; "
                             f"set eval_target.dataset (one of {sorted(t)})")
    key = key_of(str(name))
    if key is None:
        raise SystemExit(f"unknown eval set {name!r}; known (eval/configs/tasks.yaml eval_targets): {sorted(t)}")
    if t[key]["domain"] != domain:
        raise SystemExit(f"eval set {key} belongs to domain {t[key]['domain']!r}, not the run's --domain {domain!r}")
    return key, t[key]


def run_suffix(key: str) -> str:
    """Run-id suffix: _evalfit for math (MATH-500 + GSM8K, the original eval_overfit run), else _evalfit_<key>."""
    return "_evalfit" if key == "math" else f"_evalfit_{key}"


def task_names() -> set:
    """Every run_eval task name: tasks.yaml `tasks` plus the derived K_eval tasks."""
    return set(_cfg().get("tasks") or {}) | {v["dataset"] for v in targets().values()}


# ---------------------------------------------------------------------------- data roles (2026-11-21)
# Which dataset each role of a run uses, paper config `data_roles` (absent / null = the objective's default):
#   train : what the editor works from: signal / screen items, Paper B's train_example / acts / train_loss,
#           Paper A's KL / calibration prompts. A training set (its per-generation draws) or an eval set (all items).
#   gate  : what accepts a candidate. A training set (its fixed gate split) or an eval set (all items).
#   eval  : the held-out report, comma-separated: domains (math = math500,gsm8k,aime24,aime25), held-out tasks
#           (math500, humanevalplus, ...; pool test), eval sets (math, humanevalplus, ...: the K_eval copy),
#           or train / gate (the role's dataset, when it is an eval set).
# train and gate must belong to the run's --domain; an eval-set train or gate needs scope single.
TRAIN_SETS = {"math": "dapo_math", "code": "taco_verified", "knowledge": "triviaqa_train",
              "instruction_following": "ifeval_train", "tool_calling": "sealtool_train"}


def train_set_domain(name: str) -> str | None:
    for d, ds in TRAIN_SETS.items():
        if ds == name:
            return d
    return None


def is_eval_set(dataset: str | None) -> bool:
    """True for an eval-set dataset (K_eval): every item is used, nothing is drawn or marked shown."""
    return bool(dataset) and any(v["dataset"] == dataset for v in targets().values())


def resolve_set(name: str, domain: str, role: str) -> str:
    """Dataset name of a train / gate role value: a training set (dapo_math, ...) or an eval set (key or K_eval)."""
    name = str(name)
    if name in ("simple_train", "simple_eval"):    # simple_data's 1,500 / 500 items of the run's domain
        name = f"simple_{domain}_{name.split('_')[1]}"
    d = train_set_domain(name)
    if d is None:
        key = key_of(name)
        if key is None:
            raise SystemExit(f"data_roles.{role}: unknown dataset {name!r}; training sets {sorted(TRAIN_SETS.values())}, "
                             f"eval sets {sorted(targets())}")
        name, d = targets()[key]["dataset"], targets()[key]["domain"]
    if d != domain:
        raise SystemExit(f"data_roles.{role}: {name} belongs to domain {d!r}, not the run's --domain {domain!r}")
    return name


def short(dataset: str) -> str:
    """Name used in run ids: a training set as is, an eval set by its key (humanevalplus_eval -> humanevalplus)."""
    return key_of(dataset) or dataset


def default_roles(cfg: dict, domain: str) -> dict:
    """{train, gate} datasets of the objective: eval_overfit -> the eval set for both; else the domain's training set."""
    et = cfg.get("eval_target") or {}
    if cfg.get("objective") == "simple":           # simple_data: train on the 1,500, accept on the 500
        return {"train": f"simple_{domain}_train", "gate": f"simple_{domain}_eval"}
    if cfg.get("objective") == "eval_overfit" and et.get("enabled"):
        ds = resolve(et.get("dataset"), domain)[1]["dataset"]
        return {"train": ds, "gate": ds}
    ds = TRAIN_SETS.get(domain)
    return {"train": ds, "gate": ds}


def eval_tokens(spec, roles: dict) -> str:
    """Normalise a data_roles.eval value into heldout_tasks tokens (comma-separated)."""
    from .heldout import HELDOUT
    toks = [t.strip() for t in (spec.split(",") if isinstance(spec, str) else spec) if str(t).strip()]
    out = []
    for t in toks:
        if t in ("train", "gate"):
            if not is_eval_set(roles[t]):
                raise SystemExit(f"data_roles.eval: {t} = {roles[t]} is a training set; name eval sets / tasks instead")
            out.append(f"eval:{roles[t]}")
        elif t in HELDOUT or t.startswith("eval:"):
            out.append(t)
        elif any(t == task for v in HELDOUT.values() for task, _ in v):
            out.append(t)
        elif key_of(t):
            out.append(f"eval:{targets()[key_of(t)]['dataset']}")
        else:
            raise SystemExit(f"data_roles.eval: unknown {t!r} (a domain, a held-out task, an eval set, train or gate)")
    return ",".join(out)
