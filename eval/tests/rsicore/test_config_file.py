"""--config experiment files (2026-11-21): merge order (base + hardware profile < file < --set), hardware selection
from the file, the removed objective / scope selectors (simple / single: no-ops), refusals, and equality with the
equivalent --set command for the shipped plan/configs/run files.
    python -I -m pytest -q eval/tests/rsicore/test_config_file.py
"""
import copy
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from rsicore.config import build_arg_parser, load_run_config  # noqa: E402


def _cfg(paper, *extra):
    ap = build_arg_parser(paper)
    if paper == "A":
        ap.add_argument("--budget-gib", type=float, default=None)
    a = ap.parse_args(["--domain", "math", "--runs-dir", tempfile.mkdtemp(), "--mock", *extra])
    return load_run_config(paper, a)


def test_merge_order_and_profile_selection(tmp_path):
    f = tmp_path / "exp.yaml"
    f.write_text("objective: simple\nhardware: h20\neditor: {tool_rounds: 7}\nsearch: {threads: 2}\n")
    rc = _cfg("B", "--config", str(f))
    c = rc.paper_cfg
    assert c["hardware"] == "h20"                                                                  # profile selected
    assert c["editor"]["tool_rounds"] == 7 and c["search"]["threads"] == 2                        # file over base
    assert c.get("objective") != "simple"                                                         # dropped (no-op)
    assert rc.cli_extra.get("config") == [str(f)]                                                 # recorded
    c = _cfg("B", "--config", str(f), "--set", "editor.tool_rounds=9", "--set", "hardware=b300").paper_cfg
    assert c["editor"]["tool_rounds"] == 9 and c["hardware"] == "b300"                            # --set wins
    g = tmp_path / "later.yaml"
    g.write_text("editor: {tool_rounds: 11}\n")
    assert _cfg("B", "--config", str(f), "--config", str(g)).paper_cfg["editor"]["tool_rounds"] == 11   # later wins


def test_refusals(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("objective_profiles: {simple: {}}\n")
    with pytest.raises(SystemExit):
        _cfg("A", "--config", str(bad))
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n")
    with pytest.raises(SystemExit):
        _cfg("A", "--config", str(lst))
    with pytest.raises(SystemExit):
        _cfg("A", "--config", str(tmp_path / "missing.yaml"))
    for txt in ("objective: eval_overfit\n", "scope: all\n", "data_roles: {eval: test}\n"):   # removed (2026-11-22)
        bad.write_text(txt)
        with pytest.raises(SystemExit):
            _cfg("A", "--config", str(bad))


@pytest.mark.parametrize("paper", ["A", "B"])
def test_shipped_run_configs_equal_the_set_command(paper):
    sets = ["--set", "search.threads=8", "--set", "search.steps=1", "--set", "editor.tool_rounds=30",
            "--set", "editor.turn_max_tokens=16384",           # 49a5fe4: the run configs cap every round at 16k
            "--set", "editor.max_tokens=16384"]
    if paper == "B":                                            # 2026-11-25: Paper A has no accuracy gate
        sets += ["--set", "method.mode=scratch", "--set", "gate.min_gain=0.0", "--set", "gate.require_significant=false"]
    for hw in ("b300", "h20"):
        a = _cfg(paper, "--config", f"plan/configs/run/{paper}_math_r30_nosig.yaml", "--set", f"hardware={hw}").paper_cfg
        b = _cfg(paper, *sets, "--set", f"hardware={hw}").paper_cfg
        assert a == b and a["hardware"] == hw and a["search"]["threads"] == 8 and a["editor"]["tool_rounds"] == 30


def test_removed_feature_locks_and_legacy_screen_n(tmp_path):
    """(2026-11-22 review) --set data.screen_n means simple.screen_n; a lock that used a removed feature (E_t on,
    retention probes, other data_roles) is refused instead of silently resumed; feedback.mark_shown stays locked."""
    import yaml
    c = _cfg("B", "--set", "data.screen_n=40").paper_cfg
    assert c["simple"]["screen_n"] == 40 and "data" not in c
    runs = tmp_path / "runs"

    def load(run_id, *extra):
        ap = build_arg_parser("B")
        return load_run_config("B", ap.parse_args(["--domain", "math", "--runs-dir", str(runs), "--mock",
                                                   "--run-id", run_id, *extra]))
    for i, bad in enumerate(({"e_t": {"enabled": True}}, {"retention": {"domains": ["code"]}},
                             {"data_roles": {"gate": "eval:math500"}})):
        rc = load(f"r{i}")
        lock = rc.run_dir / "run_config.yaml"
        d = yaml.safe_load(lock.read_text())
        d["paper_cfg"].update(bad)
        lock.write_text(yaml.safe_dump(d))
        with pytest.raises(SystemExit):
            load(f"r{i}")
    load("ok")
    with pytest.raises(SystemExit):                                   # a live key: changing it is refused
        load("ok", "--set", "feedback.mark_shown=false")


def _load_b(runs, run_id, *extra, paper="B"):
    ap = build_arg_parser(paper)
    if paper == "A":
        ap.add_argument("--budget-gib", type=float, default=None)
    return load_run_config(paper, ap.parse_args(["--domain", "math", "--runs-dir", str(runs), "--mock",
                                                 "--run-id", run_id, *extra]))


def test_removed_knobs_set_and_config(tmp_path, capsys):
    """(2026-11-23) the search / gate / noise / KL knobs of the removed flows: --set / --config with the hard-wired
    value is a no-op (dropped, noted), any other value is refused, also under mock_overrides. and inside a block."""
    import yaml
    base = _cfg("A").paper_cfg
    for kv in ("search.top_full=0", "search.screen_candidates=false", "search.gate_every_candidate=true",
               "search.program_evaluate_calls=0", "search.kl_weight=0.0", "noise.temperature=0", "noise=null",
               "mock_overrides.search.top_full=0", "search.program_evaluate_total=null"):
        assert _cfg("A", "--set", kv).paper_cfg == base, kv
    assert "hard-wired value" in capsys.readouterr().err
    for kv in ("search.top_full=3", "search.screen_candidates=true", "search.gate_every_candidate=false",
               "search.program_evaluate_calls=2", "search.program_evaluate_calls=null", "search.max_evaluate=12",
               "search.rerun_seeds=1", "gate.max_submissions=3", "search.kl_weight=0.5", "kl.eval_texts=100",
               "noise.repeats=3", "noise.fixed=0.03", "noise={temperature: 0.6}", "mock_overrides.search.top_full=2",
               "search={threads: 2, top_full: 1}", "search.top_full=false"):
        with pytest.raises(SystemExit, match="removed"):
            _cfg("A", "--set", kv)
    for kv in ("baseline.noise_temperature=0.6", "baseline.noise_fixed=0.05", "gate.use_noise_band=true",
               "gate.epsilon=0.01", "gate.reference=raw"):
        with pytest.raises(SystemExit, match="removed"):
            _cfg("B", "--set", kv)
    assert _cfg("B", "--set", "gate.reference=parent").paper_cfg == _cfg("B").paper_cfg
    c = _cfg("B", "--set", "search={threads: 2, top_full: 0}").paper_cfg       # a no-op leaf dropped from a block
    assert c["search"] == {"threads": 2}
    f = tmp_path / "exp.yaml"
    f.write_text(yaml.safe_dump({"search": {"threads": 2, "top_full": 0, "gate_every_candidate": True},
                                 "baseline": {"noise_temperature": 0.0}}))
    c = _cfg("B", "--config", str(f)).paper_cfg
    assert c["search"]["threads"] == 2 and c["search"] == dict(_cfg("B").paper_cfg["search"], threads=2)
    f.write_text(yaml.safe_dump({"search": {"rerun_seeds": 2}}))
    with pytest.raises(SystemExit, match="removed"):
        _cfg("B", "--config", str(f))
    f.write_text(yaml.safe_dump({"mock_overrides": {"baseline": {"noise_fixed": 0.02}}}))
    with pytest.raises(SystemExit, match="removed"):
        _cfg("B", "--config", str(f))
    assert _cfg("B", "--set", "search.threads=3").paper_cfg["search"]["threads"] == 3      # live keys untouched


@pytest.mark.parametrize("paper", ["A", "B"])
def test_removed_knob_lock_continuity(tmp_path, paper):
    """(2026-11-23) an old *_single_simple lock still holding the removed knobs at their hard-wired values resumes with
    the same command (the knobs are ignored by dotted path, also under mock_overrides., even when base.yaml no longer
    has them); one whose locked value differs from the hard-wired flow is refused; a live key of the same name
    elsewhere (decoding.*.top_k, editor.extra_body.top_k) stays locked."""
    import yaml
    from rsicore import config as C
    runs = tmp_path / "runs"
    rc = _load_b(runs, "old", paper=paper)
    lock = rc.run_dir / "run_config.yaml"
    d = yaml.safe_load(lock.read_text())
    pc = d["paper_cfg"]
    old_search = {"screen_candidates": False, "gate_every_candidate": True, "charge_screens": False,
                  "program_evaluate_calls": 0, "program_evaluate_total": 0, "max_evaluate": 24, "top_full": 0,
                  "rerun_seeds": 0}
    pc.setdefault("search", {}).update(old_search)
    pc.setdefault("gate", {})["max_submissions"] = 3
    pc.setdefault("mock_overrides", {}).setdefault("search", {}).update(max_evaluate=4, top_full=0)
    if paper == "A":
        pc["search"]["kl_weight"] = 0.0
        pc["noise"] = {"repeats": 3, "temperature": 0.0, "top_p": 1.0, "n": None, "fixed": None}
        pc.setdefault("kl", {}).update(eval_texts=400, top_k=20)
        pc["mock_overrides"]["noise"] = {"repeats": 3}
    else:
        pc["gate"].update(reference="auto", epsilon=0.01, use_noise_band=True)
        pc.setdefault("baseline", {}).update(noise_repeats=3, noise_temperature=0.0, noise_top_p=1.0, noise_n=None,
                                             noise_fixed=None)
        pc["mock_overrides"].setdefault("baseline", {})["noise_repeats"] = 3
    lock.write_text(yaml.safe_dump(d))
    _load_b(runs, "old", paper=paper)                                          # resumes with the same command
    # the same lock compared with a paper config that no longer has the knobs (what base.yaml will look like)
    stripped = copy.deepcopy(pc)
    for k in old_search:
        stripped["search"].pop(k)
    stripped["gate"].pop("max_submissions")
    for k in old_search:
        stripped["mock_overrides"]["search"].pop(k, None)
    stripped.pop("noise", None)
    stripped["mock_overrides"].pop("noise", None)
    stripped["search"].pop("kl_weight", None)
    if paper == "A":
        stripped["kl"].pop("eval_texts")
        stripped["kl"].pop("top_k")
    else:
        for k in ("reference", "epsilon", "use_noise_band"):
            stripped["gate"].pop(k)
        for k in list(stripped["baseline"]):
            if k.startswith("noise_"):
                stripped["baseline"].pop(k)
        stripped["mock_overrides"]["baseline"].pop("noise_repeats")
    for d_ in (stripped["mock_overrides"], stripped):                       # a block left empty is deleted
        for blk in [k for k, v in d_.items() if v == {}]:
            d_.pop(blk)
    assert C.lock_view(pc) == C.lock_view(stripped)
    changed = copy.deepcopy(pc)
    changed.setdefault("decoding", {}).setdefault("gate", {})["top_k"] = 7     # same leaf name, live path
    assert C.lock_view(changed) != C.lock_view(pc)
    bad = [("search", "screen_candidates", True), ("search", "gate_every_candidate", False),
           ("search", "program_evaluate_calls", 2), ("search", "program_evaluate_calls", None),
           ("search", "top_full", 3), ("search", "rerun_seeds", 1)]
    bad += ([("search", "kl_weight", 0.1), ("noise", "temperature", 0.6), ("noise", "fixed", 0.03)] if paper == "A"
            else [("baseline", "noise_temperature", 0.6), ("baseline", "noise_fixed", {"math": 0.03})])
    for i, (blk, k, v) in enumerate(bad):
        rc = _load_b(runs, f"bad{i}", paper=paper)
        lock = rc.run_dir / "run_config.yaml"
        d = yaml.safe_load(lock.read_text())
        d["paper_cfg"].setdefault(blk, {})[k] = v
        mo = (d["paper_cfg"].get("mock_overrides") or {}).get(blk)
        if isinstance(mo, dict) and k in mo:                                 # a mock lock: the override wins
            mo.pop(k)
        lock.write_text(yaml.safe_dump(d))
        with pytest.raises(SystemExit, match="cannot"):
            _load_b(runs, f"bad{i}", paper=paper)
    rc = _load_b(runs, "mockbad", paper=paper)                                # a mock lock: mock_overrides applied
    lock = rc.run_dir / "run_config.yaml"
    d = yaml.safe_load(lock.read_text())
    d["paper_cfg"].setdefault("mock_overrides", {}).setdefault("search", {})["top_full"] = 2
    lock.write_text(yaml.safe_dump(d))
    with pytest.raises(SystemExit, match="top_full"):
        _load_b(runs, "mockbad", paper=paper)


def test_mock_lock_check_folds_old_objective_overrides():
    """(2026-11-23 review) an old mock lock's mock_objective_overrides.simple won over mock_overrides: its effective
    top_full 0 is the hard-wired flow, so the lock is not refused; kl.top_k 20 is a no-op, other values refused."""
    from rsicore.config import _removed_features_used
    old = {"search": {"top_full": 0}, "mock_overrides": {"search": {"top_full": 1}},
           "mock_objective_overrides": {"simple": {"search": {"top_full": 0}}}}
    assert _removed_features_used(old, "math", mock=True) == []
    assert _removed_features_used(dict(old, mock_objective_overrides={}), "math", mock=True) == ["search.top_full 1"]
    _cfg("A", "--set", "kl.top_k=20")
    with pytest.raises(SystemExit):
        _cfg("A", "--set", "kl.top_k=5")


def test_paper_a_tools_compare_removed_paper_b_live(tmp_path, capsys):
    """(2026-11-23) tools.compare is removed for Paper A only (PAPER_REMOVED_KNOBS): --set of either boolean is a
    no-op and a lock still holding it resumes; another value is refused. Paper B's tools.compare is live: --set keeps
    it and the lock still compares it."""
    import yaml
    assert "compare" not in (_cfg("A").paper_cfg.get("tools") or {})
    for v in ("true", "false"):
        assert "compare" not in (_cfg("A", "--set", f"tools.compare={v}").paper_cfg.get("tools") or {})
    with pytest.raises(SystemExit, match="removed"):
        _cfg("A", "--set", "tools.compare=2")
    assert _cfg("B", "--set", "tools.compare=false").paper_cfg["tools"]["compare"] is False
    runs = tmp_path / "runs"
    for paper in ("A", "B"):
        rc = _load_b(runs, f"old{paper}", paper=paper)
        lock = rc.run_dir / "run_config.yaml"
        d = yaml.safe_load(lock.read_text())
        d["paper_cfg"].setdefault("tools", {})["compare"] = paper == "A"      # A: old lock value; B: changed
        lock.write_text(yaml.safe_dump(d))
        if paper == "A":
            _load_b(runs, "oldA", paper="A")                                   # resumes with the same command
        else:
            with pytest.raises(SystemExit, match="tools.compare"):
                _load_b(runs, "oldB", paper="B")
