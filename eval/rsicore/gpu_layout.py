"""Automatic GPU layout: spread every GPU of --gpus over a paper's roles.

    roles, unused = layout(4, own=["editor", "candidate"],
                           growth=[("candidate", 2), ("editor", 2), ("candidate", 4), ("editor", 4)])
    # -> {"editor": [0, 1], "candidate": [2, 3]}, []

- Every role in `own` gets one GPU of its own; the remaining GPUs are handed out in the order of `growth`
  (role, target count): a step is taken when enough GPUs are left to bring the role up to its target, so
  vLLM roles grow in powers of two (tensor parallelism needs a divisor of the head counts).
- `spill`: a role that can use any number of GPUs (an HF model split layer-wise across cards) takes every
  GPU still left afterwards, so nothing stays idle.
- Roles in `shared` ({role: host}) get no GPU of their own and sit on the first GPU of `host`.
- `explicit` ({role: [indices]}, cfg["gpu_roles"]; negative = from the end) pins those roles; the other
  roles are laid out automatically on the GPUs no pinned role uses.
Values are indices into --gpus. Blocks are contiguous in `own` order. `unused` lists the indices no role
got (the caller warns).
"""
from __future__ import annotations


def layout(n: int, own: list[str], growth: list[tuple[str, int]], shared: dict | None = None,
           spill: str | None = None, share_single: bool = False,
           explicit: dict | None = None) -> tuple[dict, list[int]]:
    """Returns ({role: [indices]}, [unused indices]). Raises ValueError when there are not enough GPUs."""
    if n < 1:
        raise ValueError("no GPUs given")
    shared = dict(shared or {})
    out = {}
    for role, ii in (explicit or {}).items():
        ii = [int(i) for i in ii]
        bad = [i for i in ii if not -n <= i < n]
        if bad:
            raise ValueError(f"gpu_roles.{role} = {ii} needs more GPUs than the {n} given")
        out[role] = [i % n for i in ii]
    pinned = {i for v in out.values() for i in v}
    free = [i for i in range(n) if i not in pinned]
    todo = [r for r in own if r not in out]
    if todo:
        if len(free) < len(todo):
            if share_single and n == 1 and not out:
                out = {r: [0] for r in own}
                out.update({r: [0] for r in shared})
                return out, []
            what = f"{len(free)} free after gpu_roles" if out else str(n)
            raise ValueError(f"needs at least {len(todo)} GPUs (one each for {', '.join(todo)}), got {what}")
        count = {r: 1 for r in todo}
        left = len(free) - len(todo)
        for role, target in growth:
            if role in count:
                inc = int(target) - count[role]
                if 0 < inc <= left:
                    count[role] += inc
                    left -= inc
        if spill in count and left > 0:
            count[spill] += left
            left = 0
        i = 0
        for r in todo:
            out[r] = free[i:i + count[r]]
            i += count[r]
    for r, host in shared.items():
        if r not in out:
            out[r] = [out[host][0]]
    used = {i for v in out.values() for i in v}
    return out, [i for i in range(n) if i not in used]
