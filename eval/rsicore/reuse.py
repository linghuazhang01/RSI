"""Reuse of generation 0's M_0 scores from an earlier run (2026-11-24, paper config `reuse.baseline_from`).

Generation 0 scores the unmodified model M_0 on the gate set and on the held-out set. Two runs with the same model,
seed, domain and decoding sample exactly the same thing, so a later run (another --tag / --config, or the other
paper) can take these scores from an earlier run instead of sampling M_0 again:

  --set reuse.baseline_from=B_math_s42_single_simple            (a run id in the same runs dir, or a path)

Only M_0's gate and held-out results are taken (Paper A's Q_0, calibration texts and layer cache are not). The source
must have finished the step, with the same model path, seed, domain and mock flag; the same configured decoding
(both directions, nulls included) and max_tokens table; records actually sampled with this run's settings (run_eval's
stored summary decoding); exactly this run's gate items, or every held-out task complete with the same limit; the same
data (stored processed / split-manifest sha256 vs this checkout), grader package versions and grading-code fingerprint
(baseline/eval_code.json, written whenever M_0's gate is scored or reused; a source without one, scored before
2026-11-24, needs reuse.trust_unrecorded_code=true). Anything else stops the run with the list of differences: an
explicit reuse never falls back to sampling silently; every pending step is checked before anything is copied.
Both papers' layouts are read, so Paper A can reuse Paper B's M_0 scores and the reverse when everything matches
(the base configs differ in held-out n: Paper A 8, Paper B 4), and a run that reused can itself be a source.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from .config import ROOT, eval_python, stage_decoding

# decoding fields compared with run_eval's stored summary decoding (the sampling settings that change the records)
_DEC_KEYS = ("temperature", "top_p", "top_k", "n", "seed", "thinking", "max_tokens")
_VERSIONS: dict | None = None


def _sha(p: Path):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _current_versions() -> dict | None:
    """The grader package versions run_eval records (run_eval.py 'versions'), read in the eval interpreter
    (eval_python(), the one run_eval runs in). None = the interpreter could not be run (reported as such)."""
    global _VERSIONS
    if _VERSIONS is None:
        import subprocess
        code = ("import json, importlib.metadata as md\n"
                "def v(p):\n    try:\n        return md.version(p)\n    except Exception:\n        return None\n"
                "print(json.dumps({p: v(p) for p in ('math-verify', 'bfcl-eval', 'ifbench', 'httpx')}))")
        try:
            r = subprocess.run([eval_python(), "-I", "-c", code], capture_output=True, text=True, timeout=120)
            _VERSIONS = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() \
                else {"__error__": (r.stderr or "").strip()[-300:] or f"exit code {r.returncode}"}
        except (OSError, subprocess.TimeoutExpired, ValueError) as e:
            _VERSIONS = {"__error__": f"{type(e).__name__}: {e}"}
    return None if "__error__" in _VERSIONS else _VERSIONS


def _provenance(entry: dict, label: str) -> list[str]:
    """Differences between the data / grader a stored run_eval summary entry was scored with and what this checkout
    would score with now (processed data sha256, split manifest sha256, grader package versions)."""
    prov, ds = entry.get("data_provenance") or {}, entry.get("dataset")
    if not prov or not ds:
        return [f"{label}: no data_provenance stored, the data it was scored on cannot be checked"]
    bad = []
    now = {"processed_sha256": _sha(ROOT / "data" / "processed" / f"{ds}.jsonl"),
           "split_manifest_sha256": _sha(ROOT / "data" / "splits" / "MANIFEST.json")}
    for k, v in now.items():
        if prov.get(k) != v:
            bad.append(f"{label} {k}: {str(prov.get(k))[:12]} there, {str(v)[:12]} here (data rebuilt since)")
    vers = entry.get("versions") or {}
    cur = _current_versions() if vers else {}
    if cur is None:
        bad.append(f"{label}: the eval interpreter {eval_python()} could not be run to read the grader package "
                   f"versions ({_VERSIONS.get('__error__')})")
        return bad
    for k, v in vers.items():
        if cur.get(k) != v:
            bad.append(f"{label} grader package {k}: {v} there, {cur.get(k)} here")
    return bad


def _norm_dec(d, seed):
    """A decoding block with nulls dropped and seed null = the run's seed (for comparing two runs' configs)."""
    if not d:
        return None
    d = {k: v for k, v in dict(d).items() if v is not None}
    d.setdefault("seed", seed)
    return d


def baseline_from(rc, cfg: dict) -> Path | None:
    """The source run dir of `reuse.baseline_from` (a run id under this run's runs dir, a run directory, or its
    baseline/ folder); None = off."""
    v = (cfg.get("reuse") or {}).get("baseline_from")
    if not v:
        return None
    p = Path(str(v)).expanduser()
    if not p.is_absolute():                  # a run id under the runs dir, else a path from the working directory
        p = Path(rc.run_dir).parent / p if (Path(rc.run_dir).parent / p).exists() else Path.cwd() / p
    p = p.resolve()
    if p.name == "baseline" and (p.parent / "run_config.yaml").exists():   # a run's baseline/ folder given
        p = p.parent
    if p == Path(rc.run_dir).resolve():
        raise SystemExit(f"reuse.baseline_from={v!r} is this run itself")
    if not (p / "run_config.yaml").exists():
        raise SystemExit(f"reuse.baseline_from={v!r}: {p} is not a run directory (no run_config.yaml)")
    return p


def _source(src: Path) -> tuple[dict, dict]:
    sc = yaml.safe_load((src / "run_config.yaml").read_text()) or {}
    st = json.loads((src / "state.json").read_text()) if (src / "state.json").exists() else {}
    return sc, st


# the grading / prompting code a stored score depends on (2026-11-24): tasks.yaml (templates, default decoding), the
# graders and run_eval, rsicore's scoring (truncated-thinking rule) and eval-set building. A change to any of them
# makes an earlier score incomparable, so a reuse source must carry the same fingerprint (baseline/eval_code.json)
# + every .py under evallib/. sandbox/ = the code-domain Docker grader and its Dockerfile (the image is built from
# them; the built image and the third_party clones pins.yaml names are not hashed)
_CODE = ("configs/tasks.yaml", "run_eval.py", "rsicore/evaluate.py", "rsicore/heldout.py", "rsicore/draws.py",
         "rsicore/evalsets.py", "rsicore/kl.py", "third_party/pins.yaml", "sandbox/Dockerfile",
         "sandbox/inside_grade.py")
_MOCK_CODE = ("rsicore/mock_server.py", "rsicore/mock_quality.py", "rsicore/_mock_answers.py")   # --mock runs only


def code_fingerprint(mock: bool = False) -> str:
    import hashlib
    eval_dir = ROOT / "eval"
    files = [eval_dir / f for f in _CODE + (_MOCK_CODE if mock else ())]
    files += sorted(f for f in (eval_dir / "evallib").rglob("*.py") if "__pycache__" not in f.parts)
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(eval_dir)).encode() + b"\0" + (f.read_bytes() if f.exists() else b"") + b"\0")
    return h.hexdigest()


def _code_file(run_dir: Path) -> Path:
    return Path(run_dir) / "baseline" / "eval_code.json"


def record_code(run_dir: Path, src: Path | None = None, *, mock: bool = False) -> None:
    """Write baseline/eval_code.json, the grading-code fingerprint of M_0's generation-0 scores: this checkout's
    when M_0 is scored here; the source's file when they were reused (sha256 null when the source had none and
    reuse.trust_unrecorded_code accepted it, so a later reuse of this run needs the same flag)."""
    p = _code_file(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if src is None:
        rec = {"sha256": code_fingerprint(mock)}
    else:
        f = _code_file(src)
        rec = json.loads(f.read_text()) if f.exists() else {
            "sha256": None, "note": f"reused from {src}, which had no fingerprint (reuse.trust_unrecorded_code)"}
    p.write_text(json.dumps(rec, indent=1))


def note_heldout_code(run_dir: Path, src: Path | None = None, *, mock: bool = False) -> None:
    """After M_0's held-out (or Paper A's KL gate reference) was made here (src None) or reused from `src`: if it
    came from other grading code than
    the gate's fingerprint says (a resume on another checkout, or a source with another / no fingerprint), the two
    scores are not from one code version, so the fingerprint is dropped (null; a later reuse then needs
    reuse.trust_unrecorded_code)."""
    p = _code_file(run_dir)
    if not p.exists():
        return
    rec = json.loads(p.read_text())
    if src is None:
        other = code_fingerprint(mock)
    else:
        f = _code_file(src)
        other = (json.loads(f.read_text()) if f.exists() else {}).get("sha256")
    if rec.get("sha256") and rec["sha256"] != other:
        p.write_text(json.dumps({"sha256": None, "note": "gate and held-out scored with different eval code"},
                                indent=1))


def _common(rc, cfg: dict, src: Path, step: str) -> tuple[dict, list[str]]:
    sc, st = _source(src)
    bad = []
    for k, mine in (("domain", rc.domain), ("model_path", str(rc.model_path)), ("seed", rc.seed),
                    ("mock", bool(rc.mock))):
        theirs = sc.get(k)
        if (bool(theirs) if k == "mock" else str(theirs)) != (mine if k == "mock" else str(mine)):
            bad.append(f"{k}: {theirs!r} there, {mine!r} here")
    if step not in st:
        bad.append(f"step {step} is not finished there")
    # the run-wide max_tokens table fills any max_tokens a decoding block leaves null: compared as a whole
    from .config import LOCK_IGNORED_PATHS
    dead = {p.split(".", 1)[1] for p in LOCK_IGNORED_PATHS if p.startswith("max_tokens.")}   # removed no-op keys
    t_theirs, t_mine = ({k: v for k, v in (t or {}).items() if k not in dead}
                        for t in ((sc.get("paper_cfg") or {}).get("max_tokens"), cfg.get("max_tokens")))
    if t_theirs != t_mine:
        bad.append(f"max_tokens table: {t_theirs} there, {t_mine} here")
    fp = _code_file(src)
    rec = json.loads(fp.read_text()) if fp.exists() else None
    if rec and rec.get("sha256"):
        if rec["sha256"] != code_fingerprint(bool(rc.mock)):
            files = ", ".join(_CODE + (_MOCK_CODE if rc.mock else ()) + ("evallib/**/*.py",))
            bad.append(f"the grading / prompting code changed since the source scored M_0 (eval/: {files})")
    elif not (cfg.get("reuse") or {}).get("trust_unrecorded_code"):
        bad.append("the source has no grading-code fingerprint (baseline/eval_code.json missing or null: scored "
                   "before 2026-11-24, or itself reused under reuse.trust_unrecorded_code), so the grading code its "
                   "scores came from cannot be checked; set reuse.trust_unrecorded_code=true if eval/ has not "
                   "changed since")
    return sc, bad


def _refuse(src: Path, what: str, bad: list[str]):
    raise SystemExit(f"reuse.baseline_from={src}: cannot reuse M_0's {what}:\n  - " + "\n  - ".join(bad)
                     + "\n(remove reuse.baseline_from to score M_0 in this run)")


def _gate_file(src: Path, paper: str, ds: str) -> Path:
    """M_0's gate records of a run: Paper A baseline/gate/<ds>.jsonl, Paper B baseline/gate/<ds>/<ds>.jsonl."""
    return src / "baseline" / "gate" / (f"{ds}.jsonl" if str(paper).upper() == "A" else f"{ds}/{ds}.jsonl")


def gate_records(rc, cfg: dict, src: Path, ids_by_ds: dict, dst_dir_of, *, copy: bool = True) -> dict:
    """M_0's gate records {ds: [run_eval records]} from run `src`, copied into this run (dst_dir_of(ds) = the
    directory that holds <ds>.jsonl in this paper's layout). Refuses (SystemExit) unless the source scored M_0 with
    the same gate decoding on exactly these items."""
    sc, bad = _common(rc, cfg, src, "baseline/gate")
    mine = stage_decoding(cfg, "gate", rc.seed)
    theirs = stage_decoding(sc.get("paper_cfg") or {}, "gate", sc.get("seed"))
    if theirs != mine:                # both runs' configured gate decoding, both directions (nulls included)
        bad.append(f"decoding.gate: {theirs} there, {mine} here")
    entries = {}
    out = {}
    for ds, ids in sorted(ids_by_ds.items()):
        f = _gate_file(src, sc.get("paper"), ds)
        if not f.exists():
            bad.append(f"no gate records {f}")
            continue
        # the decoding / data / grader the records were scored with (run_eval's summary) against this run's
        sp = f.parent / "summary.json"
        entry = (json.loads(sp.read_text()) if sp.exists() else {}).get(ds) or {}
        if not entry:
            bad.append(f"{ds}: no run_eval summary next to {f}")
        stored = entry.get("decoding") or {}
        if mine is not None and entry:  # the records were really sampled with this run's settings
            diff = {k: (stored.get(k), v) for k, v in mine.items() if k in _DEC_KEYS and stored.get(k) != v}
            if diff:
                bad.append(f"{ds} gate decoding (stored there, here): {diff}")
        entries[ds] = entry
        if entry:
            bad += _provenance(entry, f"{ds} gate")
            if entry.get("n_pending"):     # e.g. code without Docker there: ungraded records, this run would grade
                bad.append(f"{ds}: {entry['n_pending']} gate items were never graded there (pending)")
        recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        got, want = sorted({str(r.get("id")) for r in recs}), sorted({str(i) for i in ids})
        if got != want:
            bad.append(f"{ds}: {len(got)} gate items there, {len(want)} here, not the same ids")
        out[ds] = (f, recs)
    if bad:
        _refuse(src, "gate scores", bad)
    if not copy:
        return {ds: recs for ds, (_f, recs) in out.items()}
    res = {}
    for ds, (f, recs) in out.items():
        dst = Path(dst_dir_of(ds))
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / f"{ds}.jsonl")
        sp = dst / "summary.json"           # the run_eval summary entry travels with the records (either layout)
        summ = json.loads(sp.read_text()) if sp.exists() else {}
        summ[ds] = dict(entries[ds], reused_from=str(f))
        sp.write_text(json.dumps(summ, indent=1, ensure_ascii=False))
        res[ds] = recs
    return res


def heldout(rc, cfg: dict, src: Path, domains, dst: Path | None, *, copy: bool = True) -> dict:
    """M_0's held-out result (run_heldout's {task: {accuracy, avg_at_n, pass_at_n, ...}}) from run `src`, its
    baseline/heldout dir copied to `dst`. Refuses (SystemExit) unless every held-out task is there, complete, with
    this run's heldout.decoding (n included) and heldout.limit."""
    from .heldout import ensure_built, heldout_tasks, stored_heldout
    ensure_built(domains)             # the test set's processed file (built on demand) for the provenance check
    sc, bad = _common(rc, cfg, src, "baseline/heldout")
    hcfg = cfg.get("heldout") or {}
    theirs = _norm_dec(((sc.get("paper_cfg") or {}).get("heldout") or {}).get("decoding"), sc.get("seed"))
    mine = _norm_dec(hcfg.get("decoding"), rc.seed)
    if theirs != mine:                # both directions, a null (tasks.yaml greedy) block included
        bad.append(f"heldout.decoding: {theirs} there, {mine} here")
    hdir = src / "baseline" / "heldout"
    h = stored_heldout(hdir, hcfg.get("decoding"), domains, hcfg.get("limit"))
    if h is None:
        bad.append(f"{hdir}: not every held-out task stored complete with this run's heldout.decoding / limit")
    summ = json.loads((hdir / "summary.json").read_text()) if (hdir / "summary.json").exists() else {}
    for task, _pool in heldout_tasks(domains):
        if isinstance(summ.get(task), dict):
            bad += _provenance(dict(summ[task], dataset=summ[task].get("dataset") or task), f"{task} held-out")
    if bad:
        _refuse(src, "held-out scores", bad)
    if copy:
        shutil.copytree(src / "baseline" / "heldout", dst, dirs_exist_ok=True)
    return h


def kl_gate(rc, cfg: dict, src: Path, dst: Path | None, *, copy: bool = True) -> None:
    """Paper A's KL gate reference (baseline/kl_gate: M_0's reference traces and its top-k logprobs on them) from
    run `src`, copied to `dst`. Refuses (SystemExit) unless the source wrote it with the same gate.kl settings (both
    directions), every trace has its cached reference logprobs, and the common checks pass."""
    sc, bad = _common(rc, cfg, src, "baseline/kl_gate")
    batch_only = ("max_num_batched_tokens",)     # server batching, not what the reference traces contain
    theirs, mine = ({k: v for k, v in (d or {}).items() if k not in batch_only} or None
                    for d in (((sc.get("paper_cfg") or {}).get("gate") or {}).get("kl"), (cfg.get("gate") or {}).get("kl")))
    if theirs != mine:
        bad.append(f"gate.kl: {theirs} there, {mine} here")
    kdir = src / "baseline" / "kl_gate"
    tr = json.loads((kdir / "traces.json").read_text()) if (kdir / "traces.json").exists() else {}
    items = tr.get("items") or []
    if not items:
        bad.append(f"{kdir}: no reference traces")
    else:
        import hashlib
        top_k = int((mine or {}).get("top_k", 20))
        have = set()
        if (kdir / "ref.jsonl").exists():
            for line in (kdir / "ref.jsonl").read_text().splitlines():
                try:
                    r = json.loads(line) if line.strip() else {}
                except json.JSONDecodeError:   # a torn line (crash mid-write): that trace counts as missing
                    continue
                if r.get("top_k") == top_k:
                    have.add(r.get("sha"))
        miss = [it["id"] for it in items if hashlib.sha256(it["text"].encode("utf-8")).hexdigest() not in have]
        if miss:
            bad.append(f"{kdir}: {len(miss)} traces without cached reference logprobs (top_k {top_k})")
    if bad:
        _refuse(src, "KL gate reference", bad)
    if copy:
        shutil.copytree(kdir, dst, dirs_exist_ok=True)


def preflight(rc, cfg: dict, src: Path, *, gate_ids: dict | None, heldout_domains, kl_ref: bool = False) -> None:
    """Every check of the steps still to reuse, before anything is copied (a held-out mismatch must not stop the
    run after the gate was already reused). gate_ids None / heldout_domains None / kl_ref False = that step is not
    pending."""
    if gate_ids is not None:
        gate_records(rc, cfg, src, gate_ids, None, copy=False)
    if heldout_domains is not None:
        heldout(rc, cfg, src, heldout_domains, None, copy=False)
    if kl_ref:
        kl_gate(rc, cfg, src, None, copy=False)
