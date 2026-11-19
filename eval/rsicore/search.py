"""Inner search of one generation (threads x steps) and gate submission.

inner_search(): steps are run round-robin over threads (step 0 of every thread, then step 1, ...), so
every thread sees all candidates recorded so far. For each (thread, step):
    msg = render_context(history, thread); program = editor.propose(msg)
    artifact, error = build_fn(program, cid)          # exceptions are caught and recorded as errors
    score, feedback = screen_fn(artifact, cid)        # only if built; exceptions propagate (infra)
After all steps: full_fn(artifact, cid) on the `top_full` best by screen score; with rerun_seeds=k,
full_fn is called k more times as full_fn(artifact, f"{cid}_r{i}") and full.score is the mean.
Records: gen_XX/candidates.jsonl (rewritten atomically; one line per candidate), programs/<cid>.py,
feedback/<cid>.json. Resumable at candidate granularity: recorded candidates are skipped.
cid = g{gen:02d}_t{thread}_s{step:02d}.
Output directory: gen_dir(rc, gen) by default; pass out_dir=... (same value to inner_search,
load_candidates and submit_to_gate) for additional searches of one generation (e.g. E_t, one per branch).
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .records import gen_dir, read_json, read_jsonl, write_json


@dataclass
class Candidate:
    cid: str
    thread: int
    step: int
    program: str                    # program text (file: programs/<cid>.py)
    artifact: dict | None
    error: str | None
    screen: dict | None             # {"score": float, "feedback": ...}
    full: dict | None               # {"score": float, "feedback": ..., "reruns": [...]}
    score: float | None             # full score if available, else screen score
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


def inner_search(rc, gen: int, *, editor_factory, render_context, build_fn, screen_fn, full_fn,
                 threads: int, steps: int, max_candidates: int, top_full: int = 3, rerun_seeds: int = 0,
                 out_dir: Path | None = None, parallel: int = 1) -> list[Candidate]:
    """parallel=1: one thread after another (a thread sees the candidates of the threads before it in the
    same step). parallel=N>1: within a step, up to N threads propose, build and screen at the same time
    (they all see the history up to the end of the previous step; the paper's build/screen functions must
    be thread-safe, e.g. a pool of N candidate servers), and the top_full stage runs N at a time. Results
    are recorded in thread order, so the record does not depend on which job finished first."""
    gd = _out(rc, gen, out_dir)
    (gd / "programs").mkdir(exist_ok=True)
    (gd / "feedback").mkdir(exist_ok=True)
    cands = load_candidates(rc, gen, gd)
    editors = {}

    def one(th: int, step: int, cid: str, history: list) -> Candidate:
        msg = render_context(history, th)
        t0 = time.time()
        program = editors[th].propose(msg)
        cost = {"propose_s": round(time.time() - t0, 2)}
        (gd / "programs" / f"{cid}.py").write_text(program)
        t0 = time.time()
        try:
            artifact, error = build_fn(program, cid) if program.strip() else (None, "no ```python block in the reply")
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
        if parallel <= 1 or len(batch) <= 1:
            for th, cid in batch:
                record(one(th, step, cid, list(cands.values())))
        else:
            history = list(cands.values())
            with ThreadPoolExecutor(max_workers=min(int(parallel), len(batch))) as ex:
                futs = [ex.submit(one, th, step, cid, history) for th, cid in batch]
                for f in futs:          # thread order
                    record(f.result())

    # second stage: full evaluation of the best screened candidates
    screened = [c for c in cands.values() if c.screen and c.screen.get("score") is not None]
    top = sorted(screened, key=lambda c: c.screen["score"], reverse=True)[:top_full]
    todo = [c for c in top if not (c.full is not None and len(c.full.get("reruns", [])) >= rerun_seeds)]

    def full_of(c: Candidate) -> tuple[dict, float]:
        t0 = time.time()
        full = dict(c.full or {})
        if "first" not in full:
            sc, fb = full_fn(c.artifact, c.cid)
            full = {"first": None if sc is None else float(sc), "feedback": fb, "reruns": []}
        full["reruns"] = list(full.get("reruns", []))
        for i in range(len(full["reruns"]) + 1, rerun_seeds + 1):
            sc, _ = full_fn(c.artifact, f"{c.cid}_r{i}")
            full["reruns"].append(None if sc is None else float(sc))
        vals = [v for v in [full["first"]] + full["reruns"] if v is not None]
        full["score"] = sum(vals) / len(vals) if vals else None
        return full, time.time() - t0

    def finish(c: Candidate, full: dict, secs: float):
        c.full = full
        c.score = full["score"] if full["score"] is not None else c.screen["score"]
        c.cost["full_s"] = round(c.cost.get("full_s", 0) + secs, 2)
        _write_feedback(gd, c)
        _save(gd, cands)

    if parallel <= 1 or len(todo) <= 1:
        for c in todo:
            finish(c, *full_of(c))
    else:
        with ThreadPoolExecutor(max_workers=min(int(parallel), len(todo))) as ex:
            futs = [(c, ex.submit(full_of, c)) for c in todo]
            for c, f in futs:
                finish(c, *f.result())
    return ranked(cands.values())


def submit_to_gate(rc, gen: int, ranked: list[Candidate], gate_fn, max_submissions: int = 3,
                   out_dir: Path | None = None) -> Candidate | None:
    """Submit candidates in order (skipping failed ones) until one passes or max_submissions is reached.
    gate_fn(candidate) -> (passed, info). Appends to <out_dir or gen_XX>/gate.json; resumable (already
    submitted candidates are not resubmitted)."""
    p = _out(rc, gen, out_dir) / "gate.json"
    log = read_json(p, None) or {"submissions": []}
    by_cid = {c.cid: c for c in ranked}
    for s in log["submissions"]:
        if s["passed"]:
            return by_cid.get(s["cid"])
    done = {s["cid"] for s in log["submissions"]}
    for c in ranked:
        if len(log["submissions"]) >= max_submissions:
            break
        if c.cid in done or c.artifact is None or c.error is not None or c.score is None:
            continue
        passed, info = gate_fn(c)
        log["submissions"].append({"cid": c.cid, "passed": bool(passed), "info": info,
                                   "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
        write_json(p, log)
        if passed:
            return c
    if not log["submissions"]:
        write_json(p, log)
    return None
