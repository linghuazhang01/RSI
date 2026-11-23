"""Inner search of one generation (threads x steps).

inner_search(): steps are run round-robin over threads (step 0 of every thread, then step 1, ...), so
every thread sees all candidates recorded so far. For each (thread, step):
    msg = render_context(history, thread)
    program = editor.propose(msg, ctx={"cid", "gen_dir", "visible_cids"})   # ctx only if propose accepts it
    (rsicore.editor.EditorReplyError -> empty program, error "editor failed: ..."; other exceptions propagate)
    artifact, error = build_fn(program, cid)          # exceptions are caught and recorded as errors
    score, feedback = screen_fn(artifact, cid)        # only if built; exceptions propagate (infra)
screen_fn is the candidate's ONE evaluation: in both papers (2026-11-23, the only flow) it is the candidate's gate
evaluation, paired vs the current model, which the paper records in gen_XX/gate.json at once (so later steps and
generations see it); score = the paired gate gain. Candidates are never screened otherwise and there is no
second (full / rerun) stage: Candidate.score = screen.score and Candidate.full is always None (the field and the
"full" record key are kept so older readers and records stay valid). The paper accepts the passing candidate with
the largest paired gain after the search.
Records: gen_XX/candidates.jsonl (rewritten atomically; one line per candidate), programs/<cid>.py,
feedback/<cid>.json. Resumable at candidate granularity: recorded candidates are skipped.
cid = g{gen:02d}_t{thread}_s{step:02d}.
Output directory: gen_dir(rc, gen) by default; pass out_dir=... (same value to inner_search and
load_candidates) to keep a search's records in another directory.
"""
import inspect
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .editor import EditorReplyError, describe_stop
from .records import gen_dir, read_jsonl, write_json

EMPTY_PROGRAM = "no ```python block in the reply"
STAT_COST = {"rounds": "editor_rounds", "tool_calls": "editor_tool_calls", "tool_s": "editor_tool_s",
             "prompt_tokens": "editor_prompt_tokens", "completion_tokens": "editor_completion_tokens",
             "stop": "editor_stop"}


def _takes_ctx(fn) -> bool:
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "ctx" in ps or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps.values())


@dataclass
class Candidate:
    cid: str
    thread: int
    step: int
    program: str                    # program text (file: programs/<cid>.py)
    artifact: dict | None
    error: str | None
    screen: dict | None             # {"score": float, "feedback": ...}: screen_fn's result (the gate evaluation)
    full: dict | None               # always None (2026-11-23: no full / rerun stage); kept for older readers
    score: float | None             # = screen score
    cost: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        d = asdict(self)
        d["program"] = f"programs/{self.cid}.py"
        return d


def _cid(gen, thread, step):
    return f"g{int(gen):02d}_t{int(thread)}_s{int(step):02d}"


def _sc(x):
    return float("-inf") if x is None else float(x)


def _out(rc, gen, out_dir) -> Path:
    if out_dir is None:
        return gen_dir(rc, gen)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_candidates(rc, gen: int, out_dir: Path | None = None) -> dict[str, Candidate]:
    gd = _out(rc, gen, out_dir)
    out = {}
    for r in read_jsonl(gd / "candidates.jsonl"):
        p = gd / r["program"] if isinstance(r.get("program"), str) and r["program"].startswith("programs/") else None
        r["program"] = p.read_text() if p and p.exists() else (r.get("program") or "")
        out[r["cid"]] = Candidate(**{k: r.get(k) for k in Candidate.__dataclass_fields__ if k in r})
    return out


def _save(gd: Path, cands: dict):
    tmp = gd / f"candidates.jsonl.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        for c in cands.values():
            f.write(json.dumps(c.to_record(), ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, gd / "candidates.jsonl")


def _write_feedback(gd: Path, c: Candidate):
    write_json(gd / "feedback" / f"{c.cid}.json", {"cid": c.cid, "error": c.error, "artifact": c.artifact,
                                                   "screen": c.screen, "full": c.full, "score": c.score})


def ranked(cands) -> list[Candidate]:
    return sorted(cands, key=lambda c: (_sc(c.score), _sc((c.screen or {}).get("score"))), reverse=True)


def inner_search(rc, gen: int, *, editor_factory, render_context, build_fn, screen_fn,
                 threads: int, steps: int, max_candidates: int | None = None,
                 out_dir: Path | None = None, parallel: int = 1, phases=None) -> list[Candidate]:
    """parallel=1: one thread after another (a thread sees the candidates of the threads before it in the
    same step). parallel=N>1: within a step, up to N threads propose, build and screen at the same time
    (they all see the history up to the end of the previous step; the paper's build/screen functions must
    be thread-safe, e.g. a pool of N candidate servers). Results are recorded in thread order, so the record
    does not depend on which job finished first.
    phases (2026-11-20, parallel > 1): an object with editor_phase() and eval_phase() -> int. Each step then
    runs in two phases: every thread proposes at once (editor_phase() first: e.g. wake the editor), then
    eval_phase() (e.g. put the editor to sleep and lend its GPU as one more candidate slot; returns how many
    builds / screens run at once) and all builds + screens. The search ends in the eval phase (eval_phase() is
    called once more at the end; the caller wakes or stops the editor).
    max_candidates: at most this many candidates (None = threads x steps).
    Returns ranked(all candidates)."""
    gd = _out(rc, gen, out_dir)
    (gd / "programs").mkdir(exist_ok=True)
    (gd / "feedback").mkdir(exist_ok=True)
    cands = load_candidates(rc, gen, gd)
    editors = {}
    if max_candidates is None:
        max_candidates = int(threads) * int(steps)

    def propose(th: int, cid: str, history: list) -> tuple[str, dict, str | None]:
        msg = render_context(history, th)
        t0 = time.time()
        ed, perr = editors[th], None
        try:
            if _takes_ctx(ed.propose):
                program = ed.propose(msg, ctx={"cid": cid, "gen_dir": str(gd), "visible_cids": [c.cid for c in history]})
            else:
                program = ed.propose(msg)
        except EditorReplyError as e:      # this candidate only; dead servers (RuntimeError) stay fatal
            program, perr = "", f"editor failed: {e}"
        program = program or ""
        (gd / "programs" / f"{cid}.py").write_text(program)
        cost = {"propose_s": round(time.time() - t0, 2)}
        stats = getattr(ed, "last_stats", None) if perr is None else None
        if isinstance(stats, dict):
            cost.update({ck: stats.get(k) for k, ck in STAT_COST.items() if k in stats})
        if perr is None and not program.strip():
            why = describe_stop(stats) if isinstance(stats, dict) and stats.get("stop") not in (None, "no_code") else ""
            perr = EMPTY_PROGRAM + (f" ({why})" if why else "")
        return program, cost, perr

    def one(th: int, step: int, cid: str, history: list) -> Candidate:
        return evaluate(th, step, cid, *propose(th, cid, history))

    def evaluate(th: int, step: int, cid: str, program: str, cost: dict, perr: str | None = None) -> Candidate:
        cost = dict(cost)
        t0 = time.time()
        try:
            artifact, error = build_fn(program, cid) if program.strip() else (None, perr or EMPTY_PROGRAM)
        except Exception as e:  # noqa: BLE001 - a failing program must not stop the search
            artifact, error = None, f"build_fn raised {type(e).__name__}: {e}\n{traceback.format_exc(limit=5)[-3000:]}"
        cost["build_s"] = round(time.time() - t0, 2)
        if artifact is not None and isinstance(artifact.get("cost"), dict):
            cost.update(artifact["cost"])
        screen = None
        if artifact is not None and error is None:
            t0 = time.time()
            sc, fb = screen_fn(artifact, cid)
            cost["screen_s"] = round(time.time() - t0, 2)
            screen = {"score": None if sc is None else float(sc), "feedback": fb}
        return Candidate(cid=cid, thread=th, step=step, program=program, artifact=artifact, error=error,
                         screen=screen, full=None, score=(screen or {}).get("score"), cost=cost)

    def record(c: Candidate):
        cands[c.cid] = c
        _write_feedback(gd, c)
        _save(gd, cands)

    for step in range(steps):
        batch = []
        for th in range(threads):
            cid = _cid(gen, th, step)
            if cid in cands:
                continue
            if len(cands) + len(batch) >= max_candidates:
                break
            batch.append((th, cid))
        for th, _ in batch:
            if th not in editors:
                editors[th] = editor_factory(th)
        if phases is not None and parallel > 1 and batch:
            history = list(cands.values())
            phases.editor_phase()
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:      # all proposals at once (one editor server)
                props = list(ex.map(lambda tc: propose(tc[0], tc[1], history), batch))
            n_eval = max(1, int(phases.eval_phase() or parallel))
            with ThreadPoolExecutor(max_workers=min(n_eval, len(batch))) as ex:
                futs = [ex.submit(evaluate, th, step, cid, *pr) for (th, cid), pr in zip(batch, props)]
                for f in futs:          # thread order
                    record(f.result())
        elif parallel <= 1 or len(batch) <= 1:
            for th, cid in batch:
                record(one(th, step, cid, list(cands.values())))
        else:
            history = list(cands.values())
            with ThreadPoolExecutor(max_workers=min(int(parallel), len(batch))) as ex:
                futs = [ex.submit(one, th, step, cid, history) for th, cid in batch]
                for f in futs:          # thread order
                    record(f.result())

    if phases is not None and parallel > 1:
        phases.eval_phase()             # the search ends in the eval phase (also when every candidate was resumed)
    return ranked(cands.values())
