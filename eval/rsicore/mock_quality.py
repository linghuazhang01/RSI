"""Mock registry quality values (INTERFACES.md §5): resolution of per-task / per-domain qualities.

A registry value (eval/rsicore/mock_server.py --registry JSON, {name: value}) is one of
  0.7                                              same quality for every request (original behaviour)
  {"default": 0.6, "math": 0.7, "dapo_math": 0.72}  keys: a run_eval task name, a domain
                                                   (math | code | knowledge | instruction_following |
                                                   tool_calling) or "default"
  {"quality": <either of the above>, ...}          extra keys are metadata (ignored)
Resolution for a request of task T in domain D: T > D > "default" > mean of the numeric values > default.
`scalar()` (used for fake prompt logprobs / KL and as a paper's single "skill" number) is the mean over
the five domains, each resolved as above; for {"default": x} or a float it is that number.
No imports beyond the standard library, so papers can use it without the eval environment.
"""
DOMAINS = ("math", "code", "knowledge", "instruction_following", "tool_calling")
# eval/configs/tasks.yaml `category` -> domain (other categories, e.g. knowledge_injection, use "default")
CATEGORY_DOMAIN = {d: d for d in DOMAINS}


def unwrap(v):
    """Strip the {"quality": ...} wrapper."""
    if isinstance(v, dict) and "quality" in v:
        return v["quality"]
    return v


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def resolve(v, task: str | None = None, domain: str | None = None, default: float = 0.7) -> float:
    v = unwrap(v)
    if v is None:
        return float(default)
    if _num(v):
        return float(v)
    if isinstance(v, dict):
        for k in (task, domain, "default"):
            if k is not None and _num(v.get(k)):
                return float(v[k])
        vals = [float(x) for x in v.values() if _num(x)]
        return sum(vals) / len(vals) if vals else float(default)
    return float(v)


def scalar(v, default: float = 0.7) -> float:
    v = unwrap(v)
    if v is None:
        return float(default)
    if _num(v):
        return float(v)
    return sum(resolve(v, domain=d, default=default) for d in DOMAINS) / len(DOMAINS)


def as_map(v, default: float = 0.7) -> dict:
    """Value -> dict form ({"default": x} for a float)."""
    v = unwrap(v)
    if v is None:
        return {"default": float(default)}
    if _num(v):
        return {"default": float(v)}
    out = {k: float(x) for k, x in v.items() if _num(x)}
    if "default" not in out:
        out["default"] = scalar(v, default)
    return out
