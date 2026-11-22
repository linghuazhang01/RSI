"""--config experiment files (2026-11-21): merge order (base + hardware profile < file < --set), hardware selection
from the file, the removed objective / scope selectors (simple / single: no-ops), refusals, and equality with the
equivalent --set command for the shipped plan/configs/run files.
    python -I -m pytest -q eval/tests/rsicore/test_config_file.py
"""
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
    sets = ["--set", "editor.tool_rounds=30", "--set", "gate.require_significant=false",
            "--set", "editor.turn_max_tokens=16384"]          # 49a5fe4: the run configs cap every round at 16k
    if paper == "B":
        sets += ["--set", "nonllm.enabled=false"]
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


def test_fixed_noise_band_domain_keys():
    from rsicore.gate import fixed_noise_band
    assert fixed_noise_band({"math": 0.03}, "simple_math_eval") == 0.03
    assert fixed_noise_band({"instruction_following": 0.04}, "simple_instruction_following_pool") == 0.04
    assert fixed_noise_band({"simple_code_eval": 0.05, "code": 0.01}, "simple_code_eval") == 0.05   # dataset first
    assert fixed_noise_band({"math": 0.03}, "simple_code_eval") is None
