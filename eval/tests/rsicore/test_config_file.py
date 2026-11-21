"""--config experiment files (2026-11-21): merge order (profiles < file < --set), profile selection from the file,
refusals, and equality with the equivalent --set command for the shipped plan/configs/run files.
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
    f.write_text("objective: simple\neditor: {tool_rounds: 7}\nsearch: {threads: 2}\n")
    rc = _cfg("B", "--config", str(f))
    c = rc.paper_cfg
    assert c["objective"] == "simple" and c["data_roles"]["eval"] == "eval:simple_math_test"   # profile selected
    assert c["editor"]["tool_rounds"] == 7 and c["search"]["threads"] == 2                        # file over profile
    assert c["search"]["steps"] == 1                                                              # profile kept
    assert rc.cli_extra.get("config") == [str(f)]                                                 # recorded
    c = _cfg("B", "--config", str(f), "--set", "editor.tool_rounds=9").paper_cfg
    assert c["editor"]["tool_rounds"] == 9                                                        # --set wins
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


@pytest.mark.parametrize("paper", ["A", "B"])
def test_shipped_run_configs_equal_the_set_command(paper):
    sets = ["--set", "objective=simple", "--set", "editor.tool_rounds=30", "--set", "gate.require_significant=false"]
    if paper == "B":
        sets += ["--set", "nonllm.enabled=false"]
    for hw in ("b300", "h20"):
        a = _cfg(paper, "--config", f"plan/configs/run/{paper}_math_r30_nosig.yaml", "--set", f"hardware={hw}").paper_cfg
        b = _cfg(paper, *sets, "--set", f"hardware={hw}").paper_cfg
        assert a == b and a["hardware"] == hw and a["search"]["threads"] == 8 and a["editor"]["tool_rounds"] == 30
