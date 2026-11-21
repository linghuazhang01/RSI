"""Paper B orchestration in mock mode (no weights, no GPU), against the real eval/rsicore runtime.

  eval/.venv/bin/python -I -m pytest -q lora_code/tests/test_mock_flow.py

- test_mock_workspace_*: the sandbox tool surface in mock mode (train-id restriction, budgets, a program
  run through rsicore.sandbox.run_program, checks).
- test_mode_tool_surface / test_raw_recipe: method.mode (scratch default / weights / edit) tool sets, the raw LoRA handle,
  no training / gradient tool in any mode (2026-11-20), and the raw recipe (size) resolution.
- test_mock_run_end_to_end: lora_code/run.py --mock (default mode=edit) for 1 generation (reduced branches)
  in a temp runs dir, then a second invocation of the same run-id must skip everything.
- test_mock_run_scratch: mode=scratch for 1 generation (gate vs the parent, fallback keep, nothing trained,
  no sampled training answers).
- test_mock_data_roles_*: data_roles (2026-11-21): a gate dataset different from the train dataset (eval-set gate
  next to dapo_math draws; eval-set train next to dapo_math's gate split).
- editor tool loop (2026-11-21): test_explore_api_surface_and_budgets (Workspace.explore_api: no evaluate / raw_lora /
  GPU tools, no generation budget spent, handles freed, gpu_tools budget), test_editor_context_tokens_sets_main_max_model_len
  (editor.context_tokens -> main servers' --max-model-len), test_mock_tool_loop_scratch (one run_python round per
  proposal; same programs / scores / evaluations as editor.tool_rounds=0).
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


def _ws(tmp_path, budget=None, mode="scratch", raw=None):
    from lora_code.mock import MockStore
    from lora_code.primitives.train import TrainData
    from lora_code.tools import Workspace
    train = TrainData({f"t/{i}": {"messages": [{"role": "user", "content": "q"}], "target": "a", "source": "gold"}
                       for i in range(10)})
    ws = Workspace(mock=True, store=MockStore(), train=train, cfg={"lora": {"rank_cap": 64, "norm_cap_tau": 0.1}},
                   out_dir=tmp_path, budget=budget or {"evaluate": 2, "gpu_seconds": 100}, mode=mode, raw=raw,
                   raw_info={"n_train": 10, "rank": 16} if raw is not None else None)
    ws.build_api()
    return ws


def test_mode_tool_surface(tmp_path):
    from lora_code.mock import MockStore, m_sgd
    from lora_code.prompts import system_prompt
    from lora_code.tools import program_build
    store = MockStore()
    mods = [m for m in store.modules() if m.endswith("down_proj")][:8]
    raw = m_sgd(None, mods, [f"t/{i}" for i in range(10)], 40, 2e-4, 16)
    ed = _ws(tmp_path / "e", {"evaluate": 2, "gpu_seconds": 500}, mode="edit", raw=raw)
    names = set(ed.api.names())
    assert "raw_lora" in names and not names & {"sgd", "grad", "from_grad"}
    assert ed.api.call("raw_lora", {})["lora"] == "RAW"
    prog = ("def main(api):\n    lo = api.raw_lora()['lora']\n"
            "    lo = api.truncate_rank(lora=lo, k=4)['lora']\n    return {'lora': lo}\n")
    art, err = program_build(ed, prog, "g00_t0_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w1")
    assert err is None, err
    assert art["strategy"] == "raw+ops"
    assert ed.load_candidate(art["cid"]).damage < raw.damage
    bad = "def main(api):\n    return {'lora': api.sgd(steps=5, lr=1e-4)['lora']}\n"
    art, err = program_build(ed, bad, "g00_t1_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w2")
    assert art is None and "sgd" in err
    sc = _ws(tmp_path / "s", mode="scratch")
    names = set(sc.api.names())
    assert not names & {"sgd", "grad", "from_grad", "raw_lora"} and {"svd", "scale_singular", "acts"} <= names
    with pytest.raises(SystemExit):
        from lora_code.common import method_mode
        method_mode({"method": {"mode": "free"}})                   # free mode was removed
    cfg = {"lora": {}, "sandbox": {}}
    p_edit = system_prompt(cfg, budget={"evaluate": 4, "gpu_seconds": 10}, mode="edit")
    p_scr = system_prompt(cfg, budget={"evaluate": 4, "gpu_seconds": 10}, mode="scratch")
    assert "- sgd(" not in p_edit and "- raw_lora(" in p_edit and "RAW" in p_edit
    assert "- sgd(" not in p_scr and "- raw_lora(" not in p_scr and "from zero" in p_scr
    assert "SFT" not in p_edit + p_scr and "- grad(" not in p_edit + p_scr


def test_raw_recipe():
    from lora_code import baselines as BL
    from lora_code.common import gate_policy
    r, n, gb = BL.raw_recipe({})
    assert r == BL.DEFAULT_RAW and n is None and gb == 7200.0
    r, n, gb = BL.raw_recipe({"raw": {"n_train": 200, "rank": 8, "steps": 30, "gpu_seconds": 900}})
    assert n == 200 and r["rank"] == 8 and r["steps"] == 30 and gb == 900.0
    ids = [f"t/{i}" for i in range(500)]
    a, b = BL.subset_ids(ids, 50, 42), BL.subset_ids(list(reversed(ids)), 50, 42)
    assert a == b and len(a) == 50 and set(a) <= set(ids)
    assert BL.subset_ids(ids, None, 42) == sorted(ids)
    assert gate_policy({"method": {"mode": "edit"}}) == ("raw", "raw")
    assert gate_policy({"method": {"mode": "scratch"}}) == ("parent", "keep")
    for bad in ({"method": {"mode": "scratch"}, "gate": {"fallback": "raw"}},
                {"method": {"mode": "edit"}, "gate": {"reference": "sft"}}):
        with pytest.raises(SystemExit):
            gate_policy(bad)


def test_compare_and_train_loss_tools(tmp_path):
    """compare: ids / statistics of two already-evaluated models (never evaluates); train_loss: forward-only answer
    NLL with vs without the LoRA, charged in gpu_seconds; both switchable via cfg tools."""
    from lora_code.mock import MockStore, m_construct
    from lora_code.primitives.train import TrainData
    from lora_code.tools import Workspace, tools_for_mode

    class _Ev:                                      # minimal evaluator: cached results only
        def __init__(self):
            self.res = {}

        def cached(self, name, level):
            return self.res.get((name, level))
    ev = _Ev()
    train = TrainData({f"t/{i}": {"messages": [{"role": "user", "content": "q"}], "target": "a", "source": "gold"}
                       for i in range(10)})
    ws = Workspace(mock=True, store=MockStore(), train=train, cfg={"lora": {"rank_cap": 64, "norm_cap_tau": 0.1}},
                   out_dir=tmp_path, evaluator=ev, budget={"evaluate": 2, "gpu_seconds": 100}, mode="scratch")
    ws.build_api()
    assert {"compare", "train_loss"} <= set(ws.api.names())
    mod = [m for m in ws.store.modules() if m.endswith("down_proj")][0]
    lo = m_construct(ws._empty(), mod, 0.5, "rank1")
    ws.save_candidate("g00_t0_s00", lo)
    with pytest.raises(Exception):                  # not evaluated yet: compare never evaluates
        ws.api.call("compare", {"cid": "g00_t0_s00"})
    def recs(scores):
        return {"dapo_math": {"records": [{"id": f"x/{i}", "score": s, "len": 10 * (i + 1), "finish": "stop"}
                                          for i, s in enumerate(scores)]}}
    ev.res[(None, "screen")] = recs([1, 1, 0, 0])
    ev.res[(ws.lora_name(lo), "screen")] = recs([1, 0, 1, 1])
    c = ws.api.call("compare", {"cid": "g00_t0_s00"})["datasets"]["dapo_math"]
    assert (c["broken"], c["fixed"], c["net"], c["broken_ids"]) == (1, 2, 1, ["x/1"])
    assert "out" not in json.dumps(c) and "prompt" not in json.dumps(c)   # no item content
    h = ws._new("L", lo)
    r = ws.api.call("train_loss", {"lora": h, "n": 8})
    assert r["n"] == 8 and r["parent_loss"] == 1.2 and abs(r["delta"] - (r["loss"] - 1.2)) < 1e-9
    assert ws.spent["gpu_seconds"] > 0
    off = tools_for_mode("scratch", retention=False, cfg={"tools": {"compare": False, "train_loss": False}})
    assert "compare" not in off and "train_loss" not in off


def test_stall_counts_consecutive_keeps(tmp_path):
    from types import SimpleNamespace
    from lora_code.run import Run
    rc = SimpleNamespace(domain="math", run_dir=tmp_path, mock=True, seed=42)
    run = Run(rc, {})
    for g, src in enumerate(["llm", "keep", "keep", "keep"]):
        d = tmp_path / f"gen_{g:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "accepted.json").write_text(json.dumps({"source": src}))
    run.gdir = lambda g: tmp_path / f"gen_{g:02d}"
    assert [run.stall(g) for g in range(4)] == [0, 1, 2, 3]


def test_mock_workspace_restriction_and_budget(tmp_path):
    from lora_code.mock import MockStore, m_sgd
    store = MockStore()
    raw = m_sgd(None, ["layers.40.mlp.down_proj"], [f"t/{i}" for i in range(10)], 20, 2e-4, 16)
    ws = _ws(tmp_path, mode="edit", raw=raw)
    call = ws.api.call                                   # budgets are charged by the rsicore ToolAPI
    with pytest.raises(PermissionError):
        call("acts", dict(module="layers.3.mlp.down_proj", ids=["t/1", "gate/9"]))
    with pytest.raises(Exception):                       # no training / gradient tools are registered
        call("sgd", dict(steps=10, lr=1e-4, modules=["mlp"]))
    lo = ws._get("RAW")
    assert lo.modules() == ["layers.40.mlp.down_proj"] and lo.rank("layers.40.mlp.down_proj") == 16
    t = ws._get(ws.t_truncate_rank(lora="RAW", k=4)["lora"])
    assert t.rank("layers.40.mlp.down_proj") == 4 and t.damage < lo.damage
    s = ws._get(ws.t_scale(lora="RAW", s=0.5)["lora"])
    assert abs(s.modules_["layers.40.mlp.down_proj"]["rel"] - 0.5 * lo.modules_["layers.40.mlp.down_proj"]["rel"]) < 1e-9
    call("acts", dict(module="layers.3.mlp.down_proj", ids=ws.t_train_ids()[:4]))
    assert ws.spent["gpu_seconds"] > 0 and ws.api.spent is ws.spent
    with pytest.raises(Exception) as ei:                 # gpu budget exhausted
        for _ in range(200):
            call("acts", dict(module="layers.3.mlp.down_proj"))
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
    bad = ("def main(api):\n    api.acts(module='layers.3.mlp.down_proj', ids=['heldout/1'])\n"
           "    return {'lora': None}\n")
    art, err = program_build(ws, bad, "g00_t1_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w2")
    assert art is None and "PermissionError" in err
    keep = "def main(api):\n    return {'lora': None}\n"
    art, err = program_build(ws, keep, "g00_t2_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w3")
    assert err is None and art["keep"]


def test_mock_run_end_to_end(tmp_path):
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--runs-dir", str(tmp_path), "--run-id", "B_test_mock", "--set", "scope=all",
           "--set", "e_t.branches=[parent,llm]", "--set", "method.mode=edit"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    run = tmp_path / "B_test_mock"
    state = json.loads((run / "state.json").read_text())
    for k in ("baseline/gate", "baseline/heldout", "e_ref", "gen00/raw", "gen00/search", "gen00/gate",
              "gen00/merge", "gen00/nonllm", "gen00/e_t/parent", "gen00/e_t/raw", "gen00/heldout", "gen00/summary"):
        assert k in state, k
    assert not any(k.endswith("/sft") or "/e_t/sft" in k for k in state)   # no SFT baseline anywhere
    summ = json.loads((run / "summary.json").read_text())["gens"]["0"]
    assert summ["search"]["candidates"] >= 1 and summ["accepted"]["source"] in ("llm", "raw_fallback", "keep")
    assert summ["method"] == "edit" and summ["gate_reference"] == "raw" and summ["raw"]["source"] == "trained"
    assert {"parent", "raw"} <= set(summ["E"]) and summ["E_ref"]["raw_ref"] is not None
    assert summ["dE_reference_branch"] == "raw" and "sft" not in summ and "dE_train" not in summ
    assert not (run / "gen_00" / "baselines" / "sft.json").exists()
    assert "frontiers" not in json.loads((run / "gen_00" / "baselines" / "nonllm.json").read_text())
    assert (run / "gen_00" / "model.json").exists() and (run / "gen_00" / "diagnostics.json").exists()
    reg = json.loads((run / "mock_registry.json").read_text())
    model = json.loads((run / "gen_00" / "model.json").read_text())   # mode=edit: M_1 = M_0 + a LoRA, no merge
    assert "m0" in reg and (model["lora"] is None or model["lora"] in reg) and "m1" not in reg
    n_cand = len((run / "gen_00" / "candidates.jsonl").read_text().splitlines())
    t0 = time.time()
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert json.loads((run / "state.json").read_text()).keys() == state.keys()
    assert len((run / "gen_00" / "candidates.jsonl").read_text().splitlines()) == n_cand
    assert time.time() - t0 < 60          # resumed run: everything skipped, no server started


def test_mock_run_scratch(tmp_path):
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--runs-dir", str(tmp_path), "--run-id", "B_test_scratch", "--set", "method.mode=scratch",
           "--set", "scope=all", "--set", "e_t.branches=[parent,llm]"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    run = tmp_path / "B_test_scratch"
    state = json.loads((run / "state.json").read_text())
    assert "gen00/raw" not in state and "gen00/gate" in state
    assert not any(k.endswith("/sft") or "/e_t/sft" in k or "/e_t/raw" in k for k in state)
    assert not (run / "gen_00" / "train_samples").exists()
    assert not (run / "gen_00" / "baselines" / "sft.json").exists()
    assert "raw" not in json.loads((run / "e_ref" / "result.json").read_text())   # E task: nothing trained
    summ = json.loads((run / "summary.json").read_text())["gens"]["0"]
    assert summ["method"] == "scratch" and summ["gate_reference"] == "parent"
    assert summ["accepted"]["source"] in ("llm", "keep")          # never a raw fallback
    assert summ["dE_reference_branch"] == "parent"
    # only candidates that beat the current model on the signal items reach the gate; paired-CI gate
    cands = {c["cid"]: c for c in (json.loads(x) for x in (run / "gen_00" / "candidates.jsonl").read_text().splitlines())}
    gate = json.loads((run / "gen_00" / "gate.json").read_text())
    for s in gate["submissions"]:
        assert cands[s["cid"]]["score"] > 0
        assert "paired_gain" in s["info"] and "paired_ci" in s["info"]
    if summ["accepted"]["source"] == "keep":                          # 2026-11-21: nothing accepted -> control skipped
        assert summ["nonllm"]["kind"] is None and "skip_if_rejected" in summ["nonllm"]["skipped"]
    else:
        assert summ["nonllm"]["kind"] == "random_construct_search"
    assert all(k in ("pure_construct", "construct+ops", "keep", "error", "none") for k in summ["search"]["strategies"])
    assert (run / "gen_00" / "parent_gate.json").exists()
    # E_t editor runs the tool loop too (run_python on the E task workspace's exploration api)
    for b in ("parent", "llm"):
        tr = run / "gen_00" / "e_t" / b / "editor" / "thread_0.jsonl"
        if tr.exists():
            lines = [json.loads(x) for x in tr.read_text().splitlines() if x.strip()]
            assert [x for x in lines if x.get("role") == "tool" and x.get("ok")], b
            assert list((run / "gen_00" / "e_t" / b / "explore").glob("*/r01_0/stdout.txt")), b
    assert (run / "gen_00" / "e_t" / "parent" / "editor" / "thread_0.jsonl").exists()


def test_mock_run_single_domain(tmp_path):
    """scope=single: only the target dataset is evaluated / gated, held-out only on the target domain."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--runs-dir", str(tmp_path), "--run-id", "B_test_single"]   # scope single is the default
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    run = tmp_path / "B_test_single"
    state = json.loads((run / "state.json").read_text())
    assert "e_ref" not in state and not any("/e_t/" in k for k in state)
    gate = json.loads((run / "baseline" / "gate_scores.json").read_text())
    assert list(gate) == ["dapo_math"]
    summ = json.loads((run / "summary.json").read_text())["gens"]["0"]
    assert summ["scope"] == "single" and summ["retention_domains"] == []
    assert set(summ["heldout"]) <= {"math500", "gsm8k", "aime24", "aime25"} and summ["heldout"]
    assert set((summ["accepted"].get("gate_scores") or {"dapo_math": 0})) == {"dapo_math"}
    evals = {p.name for p in (run / "gen_00" / "evals").glob("*/*/*.json")}
    assert evals <= {"dapo_math.json"}, evals
    prompt = (run / "gen_00" / "editor" / "thread_0.jsonl").read_text()
    assert "Single-domain run" in prompt and "- regmean_preserve(" not in prompt


def test_mock_scratch_four_gpus(tmp_path):
    """4 GPUs (2026-11-20): main = 3 replicas + train 1; every scoring job split over the replicas, held-out tasks
    in parallel, gate submissions at once."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--gpus", "0,1,2,3", "--runs-dir", str(tmp_path), "--run-id", "B_4gpu", "--set", "method.mode=scratch",
           "--set", "hardware=h20"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    assert "roles {'main': [0, 1, 2], 'train': [3]}" in r.stdout
    run = tmp_path / "B_4gpu"
    assert len(list((run / "baseline" / "gate" / "dapo_math").glob("_part*"))) == 3
    assert len(list((run / "gen_00" / "evals" / "__parent__" / "screen" / "dapo_math_raw").glob("_part*"))) == 3
    assert {p.name for p in (run / "gen_00" / "heldout" / "_par").iterdir()} == {"math500", "gsm8k", "aime24", "aime25"}
    assert json.loads((run / "summary.json").read_text())["gens"]["0"]["search"]["candidates"] >= 1


def test_mock_eval_overfit_scratch(tmp_path):
    """objective eval_overfit + scratch: target = math_eval (MATH-500 + GSM8K test); train / signal / gate are all of
    it (gold answers visible to the editor's tools); no draws; run id _evalfit."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--runs-dir", str(tmp_path), "--set", "method.mode=scratch", "--set", "objective=eval_overfit",
           "--set", "heldout.limit=2"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    run = tmp_path / "B_math_s42_single_evalfit_mock"
    assert run.exists() and not (run / "draws").exists()
    td = json.loads((run / "gen_00" / "train_data.json").read_text())
    assert len(td) == 1819 and all(v["target"] for v in td.values())                 # every eval item, gold visible
    assert list(json.loads((run / "baseline" / "gate_scores.json").read_text())) == ["math_eval"]
    summ = json.loads((run / "summary.json").read_text())["gens"]["0"]
    assert summ["accepted"]["source"] in ("llm", "keep")
    assert set(json.loads((run / "baseline" / "heldout.json").read_text())) == {"math_eval"}   # same 1819 items
    assert set(summ["heldout"]) == {"math_eval"}


def _run_b(tmp_path, *sets):
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--runs-dir", str(tmp_path)]
    for s in sets:
        cmd += ["--set", s]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    return r


def test_mock_data_roles_eval_set_gate(tmp_path):
    """data_roles (2026-11-21): train = dapo_math draws, gate = all of math500_eval, held-out = math500 alone."""
    _run_b(tmp_path, "method.mode=scratch", "data_roles.gate=math500", "data_roles.eval=math500")
    run = tmp_path / "B_math_s42_single_gate-math500_mock"
    assert run.exists() and (run / "draws" / "gen_00" / "ids" / "dapo_math.train.json").exists()
    train_ids = json.loads((run / "draws" / "gen_00" / "ids" / "dapo_math.train.json").read_text())
    td = json.loads((run / "gen_00" / "train_data.json").read_text())
    assert sorted(td) == sorted(train_ids)                                        # the editor works on dapo_math
    gate = json.loads((run / "baseline" / "gate_scores.json").read_text())
    assert list(gate) == ["math500_eval"] and gate["math500_eval"]["n"] == 30     # gate_limit 30 in mock
    assert set(json.loads((run / "baseline" / "heldout.json").read_text())) == {"math500"}
    summ = json.loads((run / "summary.json").read_text())["gens"]["0"]
    assert summ["data_roles"] == {"train": "dapo_math", "gate": "math500_eval"}
    assert set(summ["heldout"]) == {"math500"} and summ["accepted"]["source"] in ("llm", "keep")
    assert set(summ["accepted"].get("gate_scores") or {"math500_eval": 0}) == {"math500_eval"}
    assert set(json.loads((run / "gen_00" / "parent_gate.json").read_text())) == {"math500_eval"}
    evals = run / "gen_00" / "evals"
    assert {p.name for p in evals.glob("*/screen/*.json")} <= {"dapo_math.json"}
    assert {p.name for p in evals.glob("*/gate/*.json")} <= {"math500_eval.json"}
    for s in json.loads((run / "gen_00" / "gate.json").read_text())["submissions"]:
        assert set(s["info"]["scores"]) == {"math500_eval"} and "paired_gain" in s["info"]
    prompt = (run / "gen_00" / "editor" / "thread_0.jsonl").read_text()
    assert "The gate accepts a candidate on math500_eval" in prompt


def test_mock_data_roles_eval_set_train(tmp_path):
    """data_roles: train = all 500 items of math500_eval (visible to the editor), gate = dapo_math's fixed split."""
    _run_b(tmp_path, "method.mode=scratch", "data_roles.train=math500", "data_roles.gate=dapo_math")
    run = tmp_path / "B_math_s42_single_train-math500_mock"
    td = json.loads((run / "gen_00" / "train_data.json").read_text())
    pool = json.loads((REPO / "data" / "splits" / "math500_eval" / "pool.json").read_text())
    assert len(td) == 500 and sorted(td) == sorted(pool) and all(v["target"] for v in td.values())
    assert json.loads((run / "state.json").read_text())["gen00/draw"]["train"] == 500
    gate = json.loads((run / "baseline" / "gate_scores.json").read_text())
    assert list(gate) == ["dapo_math"]
    assert (run / "draws" / "gate" / "dapo_math.gate.json").exists()
    summ = json.loads((run / "summary.json").read_text())["gens"]["0"]
    assert summ["data_roles"] == {"train": "math500_eval", "gate": "dapo_math"}
    assert set(summ["accepted"].get("gate_scores") or {"dapo_math": 0}) == {"dapo_math"}
    evals = run / "gen_00" / "evals"
    assert {p.name for p in evals.glob("*/screen/*.json")} <= {"math500_eval.json"}
    assert {p.name for p in evals.glob("*/gate/*.json")} <= {"dapo_math.json"}


def test_mock_weights_mode(tmp_path):
    """method.mode weights (2026-11-21): the editor edits the weights directly (no training); 4 GPUs = editor [0],
    candidate slots [1], [2] (+ the editor's GPU while it sleeps), train [3]; an accepted edit is written into the
    weights; the non-LLM branch makes random weight edits; the default mode is scratch."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "2", "--mock",
           "--gpus", "0,1,2,3", "--runs-dir", str(tmp_path), "--run-id", "B_weights", "--set", "method.mode=weights",
           "--set", "hardware=h20"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    assert "method=weights gate reference=parent fallback=keep" in r.stdout
    run = tmp_path / "B_weights"
    summ = json.loads((run / "summary.json").read_text())["gens"]
    for g in ("0", "1"):
        s = summ[g]
        assert s["method"] == "weights" and s["accepted"]["source"] in ("llm", "keep")
        if s["accepted"]["source"] == "keep":                           # 2026-11-21: nothing accepted -> skipped
            assert s["nonllm"]["kind"] is None and "skip_if_rejected" in s["nonllm"]["skipped"]
        else:
            assert s["nonllm"]["kind"] == "random_weight_search"
        assert all(k.startswith("weights") or k in ("keep", "error", "none") for k in s["search"]["strategies"]), \
            s["search"]["strategies"]
    msgs = [json.loads(x) for x in (run / "gen_00" / "editor" / "thread_0.jsonl").read_text().splitlines() if x.strip()]
    text = "\n".join(str(m.get("content", "")) for m in msgs)
    assert "w_scale_rows" in text and '{"edit"' in text                          # weights prompt + programs
    assert list((run / "gen_00" / "loras").glob("*.wedit.json"))                 # saved weight edits
    cmd_default = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
                   "--runs-dir", str(tmp_path), "--run-id", "B_default"]
    r2 = subprocess.run(cmd_default, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r2.returncode == 0 and "method=scratch" in r2.stdout, r2.stdout[-2000:] + r2.stderr[-3000:]


def test_mock_edit_chain_trains_once(tmp_path):
    """mode=edit: only generation 0 trains (the raw LoRA); later generations edit the current LoRA on M_0
    (no training, no merge), the gate compares with the current model, a rejected generation keeps it."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "3", "--mock",
           "--runs-dir", str(tmp_path), "--run-id", "B_chain", "--set", "raw.n_train=64", "--set", "method.mode=edit"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    run = tmp_path / "B_chain"
    state = json.loads((run / "state.json").read_text())
    summ = json.loads((run / "summary.json").read_text())["gens"]
    assert state["gen00/raw"]["n_train"] == 64 and not any(k.endswith("/sft") for k in state)
    prev = None
    for g in range(3):
        gd = run / f"gen_{g:02d}"
        model = json.loads((gd / "model.json").read_text())
        raw = json.loads((gd / "baselines" / "raw.json").read_text())
        s = summ[str(g)]
        assert not (gd / "model").exists()                                          # nothing merged
        if g == 0:
            assert raw["source"] == "trained" and s["editor"] == "m0" and s["gate_reference"] == "raw"
        else:
            assert not (gd / "baselines" / "raw_lora.json").exists()                 # no training after gen 0
            assert not (gd / "train_samples").exists()
            assert s["gate_reference"] == "parent" and s["gate_fallback"] == "keep"
            if prev:
                assert raw["source"] == "current" and raw["name"] == prev and s["editor"] == prev
        if s["accepted"]["source"] == "keep" and prev:
            assert model["lora"] == prev                                            # the current LoRA is kept
        elif s["accepted"]["name"]:
            assert model["lora"] == s["accepted"]["name"]                           # replaced by the accepted one
        prev = model["lora"]
    assert len({json.loads((run / f"gen_{g:02d}" / "model.json").read_text())["path"] for g in range(3)}) == 1


def test_mock_data_sizes_and_fixed_noise(tmp_path):
    """data.* draw sizes, raw.n_train, baseline.noise_fixed (no re-runs) in one mock run."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--runs-dir", str(tmp_path), "--run-id", "B_sizes", "--set", "data.n=200", "--set", "data.gate_n=100",
           "--set", "data.screen_n=30", "--set", "raw.n_train=40",
           "--set", "baseline.noise_fixed=0.05", "--set", "method.mode=edit"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    run = tmp_path / "B_sizes"
    ids = run / "draws" / "gen_00" / "ids"
    assert len(json.loads((ids / "dapo_math.train.json").read_text())) == 100      # 200 x 0.5 (2026-11-20; was 0.8)
    assert len(json.loads((ids / "dapo_math.screen.json").read_text())) == 30
    assert len(json.loads((run / "draws" / "gate" / "dapo_math.gate.json").read_text())) == 100
    assert json.loads((run / "gen_00" / "baselines" / "raw.json").read_text())["train"]["n_train"] == 40
    gate = json.loads((run / "baseline" / "gate_scores.json").read_text())["dapo_math"]
    assert gate["noise_band"] == 0.05 and gate["noise_source"].startswith("fixed")
    assert not (run / "baseline" / "noise").exists()


def test_mock_simple_objective(tmp_path):
    """objective simple (2026-11-21): train = simple_data's 1,500 items (gold visible), gate = its 500, held-out = the
    same 500 (greedy)."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--runs-dir", str(tmp_path), "--set", "objective=simple"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    run = tmp_path / "B_math_s42_single_simple_mock"
    assert run.exists() and not (run / "draws").exists()
    td = json.loads((run / "gen_00" / "train_data.json").read_text())
    assert len(td) == 1500
    assert list(json.loads((run / "baseline" / "gate_scores.json").read_text())) == ["simple_math_eval"]
    assert set(json.loads((run / "baseline" / "heldout.json").read_text())) == {"simple_math_eval"}
    summ = json.loads((run / "summary.json").read_text())["gens"]["0"]
    assert summ["data_roles"] == {"train": "simple_math_train", "gate": "simple_math_eval"}


# ============================================================================ editor tool loop (2026-11-21)
def _explore(ws, code, tmp_path, cid="g00_t0_s00", r=1, i=0, cfg=None):
    from lora_code.tools import editor_chat_tools
    ct = editor_chat_tools(ws, cfg or {"editor": {"explore": {"timeout_s": 60, "mem_gb": 4}}})
    out = ct.call("run_python", {"code": code}, {"cid": cid, "gen_dir": str(tmp_path / "gd")}, r, i)
    return out, json.loads(out["content"])


def test_explore_api_surface_and_budgets(tmp_path):
    """Workspace.explore_api (2026-11-21): no evaluate / raw_lora / GPU tools, nothing charged to the generation's
    budgets, handles freed after each call, tool log under '~x' tags, in scratch / edit / weights mode; with
    editor.explore.gpu_tools the GPU tools are charged to their own exploration budget."""
    from lora_code.mock import MockStore, m_sgd
    from lora_code.tools import classify, editor_chat_tools
    store = MockStore()
    raw = m_sgd(None, [m for m in store.modules() if m.endswith("down_proj")][:4], [f"t/{i}" for i in range(10)],
                20, 2e-4, 16)
    for mode in ("scratch", "edit", "weights"):
        ws = _ws(tmp_path / mode, {"evaluate": 2, "gpu_seconds": 100}, mode=mode, raw=raw if mode == "edit" else None)
        api, cleanup = ws.explore_api("g00_t0_s00~x01_0")
        names = set(api.names())
        cleanup()
        assert not names & {"evaluate", "raw_lora", "acts", "train_loss", "regmean_preserve"}, (mode, names)
        assert {"svd", "budget", "modules", "candidate", "compare"} <= names
        assert ("new_edit" in names) == (mode == "weights")
        n_handles = len(ws._h)
        code = ("print(api.budget())\ns = api.svd(module='layers.3.mlp.down_proj')\nlo = api.new_lora()['lora']\n"
                + ("lo = api.truncate_rank(lora='RAW', k=4)['lora']\n" if mode == "edit" else "")
                + "for t in ('evaluate', 'raw_lora', 'acts'):\n"
                  "    try:\n        getattr(api, t)\n        print('HAS', t)\n    except AttributeError:\n        pass\n"
                  "result = {'lo': lo, 'S': s['S'][:2]}")
        out, res = _explore(ws, code, tmp_path / mode)
        assert out["ok"] and res["ok"], res
        assert "HAS" not in res.get("stdout", "") and res["result"]["lo"].startswith("L")
        assert ws.api.spent == {"evaluate": 0.0, "gpu_seconds": 0.0} and ws.gpu_measured == 0.0
        assert len(ws._h) == n_handles                                       # exploration handles freed
        assert list(ws.tool_log) == ["g00_t0_s00~x01_0"] and ws._cid() == "_"
        assert classify(ws.tool_log.get("g00_t0_s00", [])) == "keep"         # programs never see exploration calls
        assert (tmp_path / mode / "gd" / "explore" / "g00_t0_s00" / "r01_0" / "stdout.txt").exists()
    # gpu_tools: acts allowed, charged to the exploration budget only; exhausted -> refused in exploration
    ws = _ws(tmp_path / "gpu", {"evaluate": 2, "gpu_seconds": 100})
    cfg = {"editor": {"explore": {"timeout_s": 60, "mem_gb": 4, "gpu_tools": True, "gpu_seconds": 10}}}
    assert "separate exploration budget" in editor_chat_tools(ws, cfg).run_python_doc
    out, res = _explore(ws, "result = api.acts(module='layers.3.mlp.down_proj', ids=api.train_ids()[:4])['in_norm']",
                        tmp_path / "gpu", cfg=cfg)
    assert res["ok"] and res["result"] == 12.0, res
    assert ws.api.spent["gpu_seconds"] == 0.0 and ws.gpu_measured == 0.0 and ws._xspent["gpu_seconds"] > 0
    out, res = _explore(ws, "for _ in range(20):\n    api.acts(module='layers.3.mlp.down_proj')\n", tmp_path / "gpu",
                        r=2, cfg=cfg)
    assert not res["ok"] and "BudgetExceeded" in res["error"] and ws.api.spent["gpu_seconds"] == 0.0
    ws.api.call("acts", {"module": "layers.3.mlp.down_proj", "ids": ws.t_train_ids()[:4]})   # program budget intact
    assert ws.api.spent["gpu_seconds"] > 0
    assert editor_chat_tools(ws, {"editor": {"tool_rounds": 0}}) is None


def test_editor_context_tokens_sets_main_max_model_len(tmp_path, monkeypatch):
    """editor.context_tokens is the one context knob: --max-model-len of every server started from server.main_args
    (main replicas / weights slots; overrides main_args), the Editor's context_tokens; theta0_args unchanged;
    hardware profiles b300 / h20 131072, h20_80 49152."""
    import types

    import rsicore.servers as S
    import yaml
    from rsicore.editor import editor_kwargs
    from rsicore.servers import max_model_len

    from lora_code.common import deep_merge, main_server_args
    base = yaml.safe_load((REPO / "lora_code" / "configs" / "base.yaml").read_text())
    for hw, n in (("b300", 131072), ("h20", 131072), ("h20_80", 49152)):
        cfg = deep_merge(base, base["hardware_profiles"][hw] or {})
        args = main_server_args(cfg)
        assert max_model_len(args) == n and args.count("--max-model-len") == 1, (hw, args)
        assert cfg["server"]["theta0_args"] == (base["hardware_profiles"][hw].get("server") or base["server"])["theta0_args"]
        assert editor_kwargs(cfg, server_args=args)["context_tokens"] == n
        assert editor_kwargs(cfg)["max_tokens"] == 16384 and editor_kwargs(cfg)["tool_rounds"] == 50
        assert editor_kwargs(cfg)["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    cfg = {"server": {"main_args": ["--max-model-len", "20480", "--max-logprobs", "20"]},
           "editor": {"context_tokens": 65536}}
    assert main_server_args(cfg) == ["--max-model-len", "65536", "--max-logprobs", "20"]
    cfg["editor"]["context_tokens"] = None                                   # null: main_args as written
    assert main_server_args(cfg) == ["--max-model-len", "20480", "--max-logprobs", "20"]
    assert editor_kwargs(cfg, server_args=main_server_args(cfg))["context_tokens"] == 20480
    # the main replicas get it (real path, servers stubbed), the theta0 server does not
    made = []

    class FakeServer:
        def __init__(self, model, served, gpus, **kw):
            made.append((served, kw["extra_args"]))

        def alive(self):
            return True
    monkeypatch.setattr(S, "VLLMServer", FakeServer)
    monkeypatch.setattr(S, "start_all", lambda srvs: None)
    from lora_code.run import Servers
    cfg = deep_merge(base, base["hardware_profiles"]["h20"])
    cfg = deep_merge(cfg, {"editor": {"context_tokens": 98304}})
    rc = types.SimpleNamespace(gpus=[0, 1, 2, 3], mock=False, run_dir=tmp_path, seed=42)
    srv = Servers(rc, cfg)
    srv.replicas("main", "/m", "m0")
    srv.replicas("theta0", "/m", "theta0")
    main = [a for s, a in made if s == "m0"]
    th0 = [a for s, a in made if s == "theta0"]
    assert main and all(max_model_len(a) == 98304 for a in main)
    assert th0 and all(max_model_len(a) == 20480 for a in th0)


def _transcript(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def test_mock_tool_loop_scratch(tmp_path):
    """Default editor = thinking + tool loop: one mock run_python round per proposal on the exploration api
    (transcript format 2, editor_rounds in the candidate costs, explore output on disk); the same programs, scores
    and evaluations as with editor.tool_rounds=0 (one call per candidate, no tool lines)."""
    _run_b(tmp_path / "loop", "method.mode=scratch")
    _run_b(tmp_path / "single", "method.mode=scratch", "editor.tool_rounds=0")
    runs = {k: next((tmp_path / k).iterdir()) for k in ("loop", "single")}
    for k in ("loop", "single"):
        assert runs[k].name == runs["loop"].name
    gl, gs = runs["loop"] / "gen_00", runs["single"] / "gen_00"
    tr = _transcript(gl / "editor" / "thread_0.jsonl")
    assert tr[0]["role"] == "system" and tr[0].get("format") == 2
    assert "Before answering you may call functions for up to 50 rounds" in tr[0]["content"]
    tools = [l for l in tr if l.get("role") == "tool"]
    finals = [l for l in tr if l.get("final")]
    assert tools and all(l["name"] == "run_python" and l["ok"] for l in tools), tools[:1]
    assert len(tools) == len(finals) and all(f["stats"]["rounds"] == 1 and f["stats"]["stop"] == "answer" for f in finals)
    assert "'spent'" in json.loads(tools[0]["content"])["stdout"]
    assert list((gl / "explore").glob("*/r01_0/stdout.txt"))
    tr0 = _transcript(gs / "editor" / "thread_0.jsonl")
    assert "Before answering" not in tr0[0]["content"] and "Each turn, reply with a short plan" in tr0[0]["content"]
    assert not [l for l in tr0 if l.get("role") == "tool"] and all(l["stats"]["rounds"] == 0 for l in tr0 if l.get("final"))
    assert not (gs / "explore").exists()
    cl = {c["cid"]: c for c in (json.loads(x) for x in (gl / "candidates.jsonl").read_text().splitlines())}
    cs = {c["cid"]: c for c in (json.loads(x) for x in (gs / "candidates.jsonl").read_text().splitlines())}
    assert cl.keys() == cs.keys() and cl
    for cid in cl:
        assert cl[cid]["cost"]["editor_rounds"] == 1 and cl[cid]["cost"]["editor_tool_calls"] == 1
        assert cs[cid]["cost"]["editor_rounds"] == 0
        assert (gl / "programs" / f"{cid}.py").read_text() == (gs / "programs" / f"{cid}.py").read_text()
        assert cl[cid]["score"] == cs[cid]["score"]                          # exploration spends no evaluate
    ev = sorted(p.relative_to(gl / "evals") for p in (gl / "evals").rglob("*.json"))
    assert ev == sorted(p.relative_to(gs / "evals") for p in (gs / "evals").rglob("*.json"))
