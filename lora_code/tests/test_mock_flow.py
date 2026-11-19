"""Paper B orchestration in mock mode (no weights, no GPU), against the real eval/rsicore runtime.

  eval/.venv/bin/python -I -m pytest -q lora_code/tests/test_mock_flow.py

- test_mock_workspace_*: the sandbox tool surface in mock mode (train-id restriction, budgets, a program
  run through rsicore.sandbox.run_program, checks).
- test_mock_run_end_to_end: lora_code/run.py --mock for 1 generation (reduced branches) in a temp runs dir,
  then a second invocation of the same run-id must skip everything.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]

try:
    import rsicore.sandbox  # noqa: F401
    import rsicore.search  # noqa: F401
    HAVE_CORE = True
except Exception:  # noqa: BLE001
    HAVE_CORE = False
pytestmark = pytest.mark.skipif(not HAVE_CORE, reason="eval/rsicore not available")


def _ws(tmp_path, budget=None):
    from lora_code.mock import MockStore
    from lora_code.primitives.train import TrainData
    from lora_code.tools import Workspace
    train = TrainData({f"t/{i}": {"messages": [{"role": "user", "content": "q"}], "target": "a", "source": "gold"}
                       for i in range(10)})
    ws = Workspace(mock=True, store=MockStore(), train=train, cfg={"lora": {"rank_cap": 64, "norm_cap_tau": 0.1}},
                   out_dir=tmp_path, budget=budget or {"evaluate": 2, "gpu_seconds": 100})
    ws.build_api()
    return ws


def test_mock_workspace_restriction_and_budget(tmp_path):
    ws = _ws(tmp_path)
    call = ws.api.call                                   # budgets are charged by the rsicore ToolAPI
    with pytest.raises(PermissionError):
        call("sgd", dict(ids=["t/1", "gate/9"], steps=10, lr=1e-4, modules=["mlp"]))
    with pytest.raises(PermissionError):
        call("grad", dict(modules=["layers.3.mlp.down_proj"], ids=["signal/1"]))
    out = call("sgd", dict(ids=ws.t_train_ids(), steps=20, lr=2e-4, rank=16, modules=["layers.40.mlp.down_proj"]))
    lo = ws._get(out["lora"])
    assert lo.modules() == ["layers.40.mlp.down_proj"] and lo.rank("layers.40.mlp.down_proj") == 16
    t = ws._get(ws.t_truncate_rank(lora=out["lora"], k=4)["lora"])
    assert t.rank("layers.40.mlp.down_proj") == 4 and t.damage < lo.damage
    s = ws._get(ws.t_scale(lora=out["lora"], s=0.5)["lora"])
    assert abs(s.modules_["layers.40.mlp.down_proj"]["rel"] - 0.5 * lo.modules_["layers.40.mlp.down_proj"]["rel"]) < 1e-9
    assert ws.spent["gpu_seconds"] > 0 and ws.api.spent is ws.spent
    with pytest.raises(Exception) as ei:                 # gpu budget exhausted
        for _ in range(50):
            call("sgd", dict(steps=200, lr=1e-4, modules=["mlp"]))
    assert type(ei.value).__name__ == "BudgetExceeded"
    assert ws.spent["gpu_seconds"] <= ws.budget["gpu_seconds"] + 1e-9   # measured cost capped at the budget


def test_mock_program_through_sandbox(tmp_path):
    from lora_code.mock import make_mock_policy
    from lora_code.tools import program_build
    ws = _ws(tmp_path, {"evaluate": 2, "gpu_seconds": 5000})
    pol = make_mock_policy(42, skill=1.0)
    progs = {pol([{"role": "user", "content": f"[thread={t} step={s}]"}]) for t in range(3) for s in range(3)}
    assert len(progs) > 3                                     # programs vary across threads / steps
    prog = pol([{"role": "user", "content": "[thread=0 step=1]"}])
    art, err = program_build(ws, prog, "g00_t0_s01", timeout_s=120, mem_gb=4, workdir=tmp_path / "w")
    assert err is None, err
    assert art["strategy"] != "keep" or art["keep"]
    bad = "def main(api):\n    return {'lora': api.sgd(ids=['heldout/1'], steps=5, lr=1e-4)['lora']}\n"
    art, err = program_build(ws, bad, "g00_t1_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w2")
    assert art is None and "PermissionError" in err
    keep = "def main(api):\n    return {'lora': None}\n"
    art, err = program_build(ws, keep, "g00_t2_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w3")
    assert err is None and art["keep"]


def test_mock_run_end_to_end(tmp_path):
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--runs-dir", str(tmp_path), "--run-id", "B_test_mock",
           "--set", "e_t.branches=[parent,llm]", "--set", "nonllm.frontiers=false"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    run = tmp_path / "B_test_mock"
    state = json.loads((run / "state.json").read_text())
    for k in ("baseline/gate", "baseline/heldout", "e_ref", "gen00/sft", "gen00/search", "gen00/gate",
              "gen00/merge", "gen00/nonllm", "gen00/e_t/parent", "gen00/heldout", "gen00/summary"):
        assert k in state, k
    summ = json.loads((run / "summary.json").read_text())["gens"]["0"]
    assert summ["search"]["candidates"] >= 1 and summ["accepted"]["source"] in ("llm", "sft_fallback", "keep")
    assert "parent" in summ["E"] and summ["E_ref"]["sft_ref"] is not None
    assert (run / "gen_00" / "model.json").exists() and (run / "gen_00" / "diagnostics.json").exists()
    reg = json.loads((run / "mock_registry.json").read_text())
    assert "m0" in reg and "m1" in reg
    n_cand = len((run / "gen_00" / "candidates.jsonl").read_text().splitlines())
    t0 = time.time()
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert json.loads((run / "state.json").read_text()).keys() == state.keys()
    assert len((run / "gen_00" / "candidates.jsonl").read_text().splitlines()) == n_cand
    assert time.time() - t0 < 60          # resumed run: everything skipped, no server started
