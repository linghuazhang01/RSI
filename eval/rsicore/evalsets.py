"""Eval-set datasets (2026-11-21): the fixed simple_data sets registered in eval/configs/tasks.yaml `eval_targets`.

Each entry is {task, file, dataset} (the items of a jsonl file, simple_data/<d>/{train,eval,test}.jsonl) or
{task, ids_file, dataset} (the task dataset's records listed in ids_file; simple_<d>_pool = the whole clean pool).
data/scripts/build_eval_target.py builds them into data/processed/<dataset>.jsonl (pool `pool`); the run_eval task of
the same name is derived by evallib.tasks.eval_target_tasks. A run of domain d uses simple_<d>_pool (train role,
per-generation draws, rsicore.draws), simple_<d>_eval (gate) and simple_<d>_test (held-out report).
"""
from functools import lru_cache

import yaml

from .config import EVAL_DIR

TASKS_YAML = EVAL_DIR / "configs" / "tasks.yaml"


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return yaml.safe_load(TASKS_YAML.read_text()) or {}


def targets() -> dict:
    """{key: {"dataset", "domain", "sources": [(task, "file" | "ids_file")], "source_datasets", "file"}}."""
    tasks, out = _cfg().get("tasks") or {}, {}
    for key, src in (_cfg().get("eval_targets") or {}).items():
        t, form = src["task"], ("ids_file" if src.get("ids_file") else "file")
        out[key] = {"dataset": src["dataset"], "domain": tasks[t]["category"], "sources": [(t, form)],
                    "source_datasets": [(tasks[t]["dataset"], "pool")] if form == "ids_file" else [],
                    "file": src[form]}
    return out


def task_names() -> set:
    """Every run_eval task name: tasks.yaml `tasks` plus the derived eval-set tasks (simple_*)."""
    return set(_cfg().get("tasks") or {}) | {v["dataset"] for v in targets().values()}


def simple_sets(domain: str) -> dict:
    """The datasets of a run of `domain`: {train: pool, gate: eval, test: test}."""
    return {"train": f"simple_{domain}_pool", "gate": f"simple_{domain}_eval", "test": f"simple_{domain}_test"}


def is_eval_set(dataset: str | None) -> bool:
    """True for a registered eval-set dataset (simple_*): every item is used (the train role simple_<d>_pool is
    drawn per generation by rsicore.draws) and nothing is marked shown."""
    return bool(dataset) and any(v["dataset"] == dataset for v in targets().values())
