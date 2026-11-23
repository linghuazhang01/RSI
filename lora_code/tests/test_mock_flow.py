"""Paper B orchestration in mock mode (no weights, no GPU), against the real eval/rsicore runtime.

  eval/.venv/bin/python -I -m pytest -q lora_code/tests/test_mock_flow.py

- test_mock_workspace_*: the sandbox tool surface in mock mode (train-id restriction, budgets, a program
  run through rsicore.sandbox.run_program, checks).
- test_mode_tool_surface / test_raw_recipe: method.mode (scratch default / weights / edit) tool sets, the raw LoRA handle,
  no training / gradient tool in any mode (2026-11-20), and the raw recipe (size) resolution.
- test_mock_run_end_to_end: lora_code/run.py --mock (mode=edit) for 1 generation in a temp runs dir (only the
  run's domain: simple_math_pool / simple_math_eval / simple_math_test; screen size, raw size; no noise band),
  then a second invocation of the same run-id must skip everything.
- test_mock_simple_flow: the default (simple) flow on 4 GPUs, scratch; weights / edit chain / merge.mode adapter runs.
- editor tool loop (2026-11-21): test_explore_api_surface_and_budgets (Workspace.explore_api: no raw_lora /
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
                   out_dir=tmp_path, budget=budget or {"gpu_seconds": 100}, mode=mode, raw=raw,
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
    ed = _ws(tmp_path / "e", {"gpu_seconds": 500}, mode="edit", raw=raw)
    names = set(ed.api.names())
    assert "raw_lora" in names and not names & {"sgd", "grad", "from_grad"}
    assert ed.api.call("raw_lora", {})["lora"] == "RAW"
    prog = ("def main(api):\n    lo = api.raw_lora()['lora']\n"
            "    lo = api.truncate_rank(lora=lo, k=4)['lora']\n    return {'lora': lo}\n")
    art, err = program_build(ed, prog, "g00_t0_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w1")
    assert err is None, err
    assert art["strategy"] == "raw+ops"
    # (2026-11-22) the history's account of what changed: module types, per type layers / count / rank / rel norm
    e = art["edit"]
    assert e["types"] == ["mlp.down_proj"] and e["n"] == art["modules"]
    assert e["groups"][0]["type"] == "mlp.down_proj" and e["groups"][0]["n"] == art["modules"]
    assert e["groups"][0]["rank"] == "4" and "layers" in e["groups"][0] and "rel_norm" in e["groups"][0]
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
    p_edit = system_prompt(cfg, budget={"gpu_seconds": 10}, mode="edit")
    p_scr = system_prompt(cfg, budget={"gpu_seconds": 10}, mode="scratch")
    assert "- sgd(" not in p_edit and "- raw_lora(" in p_edit and "RAW" in p_edit
    assert "- sgd(" not in p_scr and "- raw_lora(" not in p_scr and "from zero" in p_scr
    assert "SFT" not in p_edit + p_scr and "- grad(" not in p_edit + p_scr


def test_raw_recipe():
    from lora_code import baselines as BL
    from lora_code.common import gate_fallback
    r, n, gb = BL.raw_recipe({})
    assert r == BL.DEFAULT_RAW and n is None and gb == 7200.0
    r, n, gb = BL.raw_recipe({"raw": {"n_train": 200, "rank": 8, "steps": 30, "gpu_seconds": 900}})
    assert n == 200 and r["rank"] == 8 and r["steps"] == 30 and gb == 900.0
    ids = [f"t/{i}" for i in range(500)]
    a, b = BL.subset_ids(ids, 50, 42), BL.subset_ids(list(reversed(ids)), 50, 42)
    assert a == b and len(a) == 50 and set(a) <= set(ids)
    assert BL.subset_ids(ids, None, 42) == sorted(ids)
    assert gate_fallback({"method": {"mode": "edit"}}) == "raw"         # the gate itself: always vs the current model
    assert gate_fallback({"method": {"mode": "scratch"}}) == "keep"
    for bad in ({"method": {"mode": "scratch"}, "gate": {"fallback": "raw"}},
                {"method": {"mode": "edit"}, "gate": {"fallback": "sft"}}):
        with pytest.raises(SystemExit):
            gate_fallback(bad)


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
                   out_dir=tmp_path, evaluator=ev, budget={"gpu_seconds": 100}, mode="scratch")
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
    ev.res[(None, "gate")] = recs([1, 1, 0, 0])          # (2026-11-23) candidates are only gate-evaluated: compare
    ev.res[(ws.lora_name(lo), "gate")] = recs([1, 0, 1, 1])   # without level reads the gate results
    c = ws.api.call("compare", {"cid": "g00_t0_s00"})["datasets"]["dapo_math"]
    assert (c["broken"], c["fixed"], c["net"], c["broken_ids"]) == (1, 2, 1, ["x/1"])
    assert "out" not in json.dumps(c) and "prompt" not in json.dumps(c)   # no item content
    h = ws._new("L", lo)
    r = ws.api.call("train_loss", {"lora": h, "n": 8})
    assert r["n"] == 8 and r["parent_loss"] == 1.2 and abs(r["delta"] - (r["loss"] - 1.2)) < 1e-9
    assert ws.spent["gpu_seconds"] > 0
    off = tools_for_mode("scratch", cfg={"tools": {"compare": False, "train_loss": False}})
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
    ws = _ws(tmp_path, {"gpu_seconds": 5000})
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
    assert err is None and art["keep"] and "note" not in art
    # (2026-11-22) a returned note is recorded in the artifact, whitespace collapsed and cut to 600 characters
    noted = ("def main(api):\n    return {'lora': None, 'note': 'rejected: scale x1.1 (train_loss +0.03)\\n\\n'"
             " + 'x' * 900}\n")
    art, err = program_build(ws, noted, "g00_t3_s00", timeout_s=120, mem_gb=4, workdir=tmp_path / "w4")
    assert err is None and art["keep"] and art["note"].startswith("rejected: scale x1.1 (train_loss +0.03) xxx")
    assert len(art["note"]) == 600 and art["note"].endswith("...")


def test_mock_run_end_to_end(tmp_path):
    """mode=edit, 1 generation, the one (simple) flow: the raw LoRA is trained in generation 0 (raw.n_train), every
    candidate is gated during the search (reference = the current model, so the summary says 'parent'), only the
    run's domain is drawn / scored (simple.screen_n screen items, the simple_math_eval gate, the simple_math_test
    held-out), M_0's gate score without a noise band; no E_t / theta0 / retention anywhere; the generation-0 prompt
    describes the real gate (per candidate vs the current model, RAW as the fallback); a rerun skips everything."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--runs-dir", str(tmp_path), "--run-id", "B_test_mock", "--set", "method.mode=edit",
           "--set", "simple.screen_n=30", "--set", "raw.n_train=40", "--set", "heldout.limit=4"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    assert "roles {'main': [0], 'train': [1]}" in r.stdout and "theta0" not in r.stdout
    run = tmp_path / "B_test_mock"
    state = json.loads((run / "state.json").read_text())
    for k in ("baseline/gate", "baseline/heldout", "gen00/raw", "gen00/search", "gen00/gate",
              "gen00/merge", "gen00/nonllm", "gen00/heldout", "gen00/summary"):
        assert k in state, k
    assert not any(k == "e_ref" or "/e_t/" in k or k.endswith("/sft") for k in state)
    assert not (run / "e_ref").exists() and not (run / "gen_00" / "e_t").exists()
    assert state["gen00/raw"]["n_train"] == 40
    assert len(json.loads((run / "draws" / "gen_00" / "ids" / "simple_math_pool.screen.json").read_text())) == 30
    gate = json.loads((run / "baseline" / "gate_scores.json").read_text())
    assert list(gate) == ["simple_math_eval"] and gate["simple_math_eval"]["n"] == 30          # gate_limit 30 (mock)
    assert set(gate["simple_math_eval"]) == {"score", "n", "n_scored", "se", "records"}           # no noise band
    assert not (run / "baseline" / "noise").exists() and set(state["baseline/gate"]["simple_math_eval"]) == {"score"}
    summ = json.loads((run / "summary.json").read_text())["gens"]["0"]
    assert summ["search"]["candidates"] >= 1 and summ["accepted"]["source"] in ("llm", "raw_fallback", "keep")
    assert summ["method"] == "edit" and summ["gate_reference"] == "parent" and summ["raw"]["source"] == "trained"
    assert summ["data_roles"] == {"train": "simple_math_pool", "gate": "simple_math_eval"}
    for gone in ("E", "E_ref", "dE_write", "dE_nonLLM", "dE_reference_branch", "scope", "retention_domains", "sft"):
        assert gone not in summ, gone
    assert set(summ["heldout"]) == {"simple_math_test"}
    assert set(summ["accepted"].get("gate_scores") or {"simple_math_eval": 0}) == {"simple_math_eval"}
    evals = {p.name for p in (run / "gen_00" / "evals").glob("*/*/*.json")}
    assert evals <= {"simple_math_pool.json", "simple_math_eval.json"}, evals
    raw = json.loads((run / "gen_00" / "baselines" / "raw.json").read_text())
    assert not {"worst_retention_delta", "kl"} & set(raw["points"]["screen"])
    non = json.loads((run / "gen_00" / "baselines" / "nonllm.json").read_text())
    for p in (non.get("random_search") or {}).get("points") or []:
        assert "regmean" not in p and "worst_retention_delta" not in p
    assert not (run / "gen_00" / "baselines" / "sft.json").exists()
    assert (run / "gen_00" / "model.json").exists() and (run / "gen_00" / "diagnostics.json").exists()
    reg = json.loads((run / "mock_registry.json").read_text())
    model = json.loads((run / "gen_00" / "model.json").read_text())   # mode=edit: M_1 = M_0 + a LoRA, no merge
    assert "m0" in reg and "theta0" not in reg and (model["lora"] is None or model["lora"] in reg) and "m1" not in reg
    tr = [json.loads(x) for x in (run / "gen_00" / "editor" / "thread_0.jsonl").read_text().splitlines() if x.strip()]
    prompt = "\n".join(str(x.get("content")) for x in tr if x.get("role") in ("system", "user"))   # what it reads
    assert "Single-domain run" in prompt and "regmean_preserve" not in prompt and "retention" not in prompt
    assert "The gate accepts a candidate on simple_math_eval" in prompt
    assert "evaluated ONCE on a fixed hidden set of 30 items" in prompt and "raw LoRA's by more than" not in prompt
    assert "RAW itself is accepted if it passes the same gate" in prompt and "token-KL" not in prompt
    assert "act ratio" not in prompt and "after the search, not visible" not in prompt
    n_cand = len((run / "gen_00" / "candidates.jsonl").read_text().splitlines())
    t0 = time.time()
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert json.loads((run / "state.json").read_text()).keys() == state.keys()
    assert len((run / "gen_00" / "candidates.jsonl").read_text().splitlines()) == n_cand
    assert time.time() - t0 < 60          # resumed run: everything skipped, no server started


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
            # gate per candidate (the simple flow): every candidate is gated against the current model
            assert raw["source"] == "trained" and s["editor"] == "m0" and s["gate_reference"] == "parent"
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


def test_mock_scratch_adapter(tmp_path):
    """scratch + merge.mode adapter (2026-11-22): nothing is merged; M_g = M_0 + C_g on the one m0 server, C_{g+1} =
    C_g (+) the accepted delta kept under the accepted candidate's adapter name (its registry quality = the gated
    one), a rejected generation keeps C_g; nothing is trained; the editor of g >= 1 is C_g's adapter."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "3", "--mock",
           "--runs-dir", str(tmp_path), "--run-id", "B_adapter", "--set", "method.mode=scratch",
           "--set", "merge.mode=adapter", "--set", "gate.min_gain=0", "--set", "gate.require_significant=false"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    assert "merge=adapter max_lora_rank=256" in r.stdout
    run = tmp_path / "B_adapter"
    state = json.loads((run / "state.json").read_text())
    summ = json.loads((run / "summary.json").read_text())["gens"]
    reg = json.loads((run / "mock_registry.json").read_text())
    assert not any(k.endswith("/raw") or k.endswith("/sft") for k in state)              # nothing trained
    assert not any(k in reg for k in ("m1", "m2", "m3"))                                  # no merged successor
    prev, accepted = None, 0
    for g in range(3):
        gd = run / f"gen_{g:02d}"
        model = json.loads((gd / "model.json").read_text())
        s = summ[str(g)]
        assert not (gd / "model").exists() and model["merge_mode"] == "adapter"         # nothing merged
        assert s["method"] == "scratch" and s["merge_mode"] == "adapter" and s["gate_reference"] == "parent"
        assert s["editor"] == (prev or "m0") and s["successor"] == model["lora"]
        assert state[f"gen{g:02d}/merge"]["merged"] is False
        if s["accepted"]["source"] == "llm":
            accepted += 1
            assert model["lora"] == s["accepted"]["name"]
            C = json.loads(Path(model["lora_file"]).read_text())
            assert C["meta"]["cumulative"] and model["components"][-1] == g
            dW = json.loads((gd / "accepted_lora.json").read_text())
            for m, v in dW["modules"].items():                                            # the delta is inside C
                assert C["modules"][m]["rank"] >= v["rank"]
            assert reg[model["lora"]] == model["quality"]                                 # gated quality kept
        else:
            assert model["lora"] == prev                                                   # keep C_g
        prev = model["lora"]
    assert accepted >= 1, "no generation accepted a candidate; the adapter path was not exercised"
    assert len({json.loads((run / f"gen_{g:02d}" / "model.json").read_text())["path"] for g in range(3)}) == 1


def test_mock_simple_flow(tmp_path):
    """The simple flow v2 (2026-11-21; the only flow since 2026-11-22), 4 GPUs (3 main replicas): a fresh 500-item screen of the clean
    pool per generation (draws/gen_XX), train = the rest of the pool, gate = simple_math_eval, held-out =
    simple_math_test (avg@4). The CURRENT model answers the screen once (n 4, thinking text kept: reference_screen.json,
    api.screen_results); candidates are NOT screened; each of the k x 1 candidates gets ONE gate evaluation paired vs
    the parent's gate records (scored once), recorded in gate.json; the largest passing gain is accepted; no
    api.evaluate. Thread 1 runs the paper-style recipe (screen tools, acts on own outputs, direction / unit,
    tensor_compute(vectors), project_out, save_edit / load_edit)."""
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "1", "--mock",
           "--gpus", "0,1,2,3", "--runs-dir", str(tmp_path), "--set", "hardware=h20",
           "--set", "simple.screen_n=20", "--set", "heldout.limit=4"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    assert "roles {'main': [0, 1, 2], 'train': [3]}" in r.stdout
    run = tmp_path / "B_math_s42_single_simple_mock"                             # the default run id
    # every scoring job is split over the 3 replicas
    assert len(list((run / "baseline" / "gate" / "simple_math_eval").glob("_part*"))) == 3
    ids = run / "draws" / "gen_00" / "ids"
    screen = json.loads((ids / "simple_math_pool.screen.json").read_text())
    gate = json.loads((REPO / "data" / "splits" / "simple_math_eval" / "pool.json").read_text())
    test = json.loads((REPO / "data" / "splits" / "simple_math_test" / "pool.json").read_text())
    assert len(screen) == 20 and not set(screen) & (set(gate) | set(test))
    held = json.loads((run / "baseline" / "heldout.json").read_text())
    assert set(held) == {"simple_math_test"} and held["simple_math_test"].get("avg_at_n") is not None   # avg@4
    gd = run / "gen_00"
    par = sorted(p.name for p in (gd / "evals" / "__parent__").iterdir())
    scr = [p for p in par if p.startswith("screen@")]
    gat = [p for p in par if p.startswith("gate@")]
    assert len(scr) == 1 and len(gat) == 1 and "screen" not in par and "gate" not in par, par
    pg = json.loads((gd / "evals" / "__parent__" / gat[0] / "simple_math_eval.json").read_text())
    assert pg["decoding"]["n"] == 4 and pg["decoding"]["thinking"] and all(x["ns"] == 4 for x in pg["records"])
    ps = json.loads((gd / "evals" / "__parent__" / scr[0] / "simple_math_pool.json").read_text())
    assert ps["decoding"]["n"] == 4 and ps["n"] == 20
    assert len(list((gd / "evals" / "__parent__" / scr[0] / "simple_math_pool_raw").glob("_part*"))) == 3
    rs = json.loads((gd / "reference_screen.json").read_text())["summary"]
    assert rs["n_items"] == 20 and rs["samples_per_item"] == 4 and sum(rs["correct_counts"].values()) == 20
    # candidates: k x 1, never screened (only gate@ caches), one gate result each, recorded in gate.json
    non = json.loads((gd / "baselines" / "nonllm.json").read_text())
    nonllm = {p.get("name") for p in (non.get("random_search") or {}).get("points") or []}
    cands = [json.loads(x) for x in (gd / "candidates.jsonl").read_text().splitlines() if x.strip()]
    assert sorted(c["cid"] for c in cands) == [f"g00_t{t}_s00" for t in range(3)]
    for c in cands:
        if not c["artifact"]:
            continue
        fb = c["screen"]["feedback"]
        assert "target" not in fb and fb["gate"]["n"] == 30 and c["screen"]["score"] == fb["gate"]["gain"]
        assert {"passed", "accuracy", "reference_accuracy", "gain", "ci", "broken", "fixed"} <= set(fb["gate"])
    for d in (gd / "evals").iterdir():
        if d.name != "__parent__" and d.name not in nonllm:            # the non-LLM branch screens its own points
            assert all(x.name.startswith("gate@") for x in d.iterdir()), list(d.iterdir())
    glog = json.loads((gd / "gate.json").read_text())
    assert glog["rule"].startswith("gate_every_candidate")
    assert sorted(x["cid"] for x in glog["submissions"]) == sorted(c["cid"] for c in cands if c["artifact"])
    acc = json.loads((gd / "accepted.json").read_text())
    passed = [x for x in glog["submissions"] if x["passed"]]
    if passed:
        best = max(passed, key=lambda x: x["info"]["gain"])
        assert acc["source"] == "llm" and acc["cid"] == glog["accepted"] == best["cid"]
    else:
        assert acc["source"] == "keep" and glog["accepted"] is None
    st = json.loads((run / "state.json").read_text())["gen00/search"]
    assert "evaluate_budget" not in st and "evaluate_spent" not in st
    t1 = next(c for c in cands if c["cid"] == "g00_t1_s00")
    tools = t1["artifact"]["tools_used"]
    assert "evaluate" not in tools and t1["artifact"]["strategy"] == "pure_construct", t1
    for k in ("screen_results", "screen_output", "output_matrices", "acts", "direction", "unit", "tensor_compute",
              "project_out", "save_edit", "drop_edit", "list_edits", "load_edit"):
        assert tools.get(k), (k, tools)
    assert (gd / "saved" / "g00_t1_s00" / "best.json").exists()
    tr = [json.loads(x) for x in (gd / "editor" / "thread_0.jsonl").read_text().splitlines() if x.strip()]
    sysmsg, user = tr[0]["content"], next(x for x in tr if x.get("role") == "user")["content"]
    assert "there is no api.evaluate" in sysmsg and "evaluated ONCE on a fixed hidden set of 30 items" in sysmsg
    assert "- evaluate(" not in sysmsg and "- screen_results(" in sysmsg and "api.project_out" in sysmsg
    assert "share ONE Python session" in sysmsg and "up to 20 rounds" in sysmsg
    assert "You are session 1 of 3 in this generation" in user and "Your own answers on this generation's 20" in user
    assert "No api.evaluate this generation" in user
    assert "worst_retention_delta" not in user and "full_score" not in user and "act ratio" not in user


# ============================================================================ editor tool loop (2026-11-21)
def _explore(ws, code, tmp_path, cid="g00_t0_s00", r=1, i=0, cfg=None):
    from lora_code.tools import editor_chat_tools
    ct = editor_chat_tools(ws, cfg or {"editor": {"explore": {"timeout_s": 60, "mem_gb": 4}}})
    out = ct.call("run_python", {"code": code}, {"cid": cid, "gen_dir": str(tmp_path / "gd")}, r, i)
    return out, json.loads(out["content"])


def test_explore_api_surface_and_budgets(tmp_path):
    """Workspace.explore_api (2026-11-21): no raw_lora / GPU tools, nothing charged to the generation's
    budgets, handles freed after each call, tool log under '~x' tags, in scratch / edit / weights mode; with
    editor.explore.gpu_tools the GPU tools are charged to their own exploration budget."""
    from lora_code.mock import MockStore, m_sgd
    from lora_code.tools import classify, editor_chat_tools
    store = MockStore()
    raw = m_sgd(None, [m for m in store.modules() if m.endswith("down_proj")][:4], [f"t/{i}" for i in range(10)],
                20, 2e-4, 16)
    for mode in ("scratch", "edit", "weights"):
        ws = _ws(tmp_path / mode, {"gpu_seconds": 100}, mode=mode, raw=raw if mode == "edit" else None)
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
        assert ws.api.spent == {"gpu_seconds": 0.0} and ws.gpu_measured == 0.0
        assert len(ws._h) == n_handles                                       # exploration handles freed
        assert list(ws.tool_log) == ["g00_t0_s00~x01_0"] and ws._cid() == "_"
        assert classify(ws.tool_log.get("g00_t0_s00", [])) == "keep"         # programs never see exploration calls
        assert (tmp_path / mode / "gd" / "explore" / "g00_t0_s00" / "r01_0" / "stdout.txt").exists()
    # gpu_tools: acts allowed, charged to the exploration budget only; exhausted -> refused in exploration
    ws = _ws(tmp_path / "gpu", {"gpu_seconds": 100})
    cfg = {"editor": {"explore": {"timeout_s": 60, "mem_gb": 4, "gpu_tools": True, "gpu_seconds": 10}}}
    assert "separate exploration budget" in editor_chat_tools(ws, cfg).run_python_doc
    out, res = _explore(ws, "result = api.acts(module='layers.3.mlp.down_proj', ids=api.train_ids()[:4])['in_norm']",
                        tmp_path / "gpu", cfg=cfg)
    assert res["ok"] and res["result"] == 12.0, res
    assert ws.api.spent["gpu_seconds"] == 0.0 and ws.gpu_measured == 0.0
    assert ws._xbudgets["g00_t0_s00"][1]["gpu_seconds"] > 0
    out, res = _explore(ws, "for _ in range(20):\n    api.acts(module='layers.3.mlp.down_proj')\n", tmp_path / "gpu",
                        r=2, cfg=cfg)
    assert not res["ok"] and "BudgetExceeded" in res["error"] and ws.api.spent["gpu_seconds"] == 0.0
    # (2026-11-22) the budget is per session: another candidate's session still has its full gpu_seconds
    out, res = _explore(ws, "result = api.acts(module='layers.3.mlp.down_proj', ids=api.train_ids()[:4])['in_norm']",
                        tmp_path / "gpu", cid="g00_t1_s00", cfg=cfg)
    assert res["ok"], res
    assert ws._xbudgets["g00_t1_s00"][0] == {"gpu_seconds": 10.0} and ws._xbudgets["g00_t1_s00"][1]["gpu_seconds"] < 10
    ws.api.call("acts", {"module": "layers.3.mlp.down_proj", "ids": ws.t_train_ids()[:4]})   # program budget intact
    assert ws.api.spent["gpu_seconds"] > 0
    assert editor_chat_tools(ws, {"editor": {"tool_rounds": 0}}) is None


def test_editor_context_tokens_sets_main_max_model_len(tmp_path, monkeypatch):
    """editor.context_tokens is the one context knob: --max-model-len of every server started from server.main_args
    (main replicas / weights slots; overrides main_args), the Editor's context_tokens; hardware profiles b300 / h20
    131072, h20_80 49152. 'main' is the only server slot (no theta0 since 2026-11-22)."""
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
        assert "theta0_args" not in cfg["server"]
        assert editor_kwargs(cfg, server_args=args)["context_tokens"] == n
        assert editor_kwargs(cfg)["max_tokens"] == 16384 and editor_kwargs(cfg)["tool_rounds"] == 20
        assert editor_kwargs(cfg)["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    cfg = {"server": {"main_args": ["--max-model-len", "20480", "--max-logprobs", "20"]},
           "editor": {"context_tokens": 65536}}
    assert main_server_args(cfg) == ["--max-model-len", "65536", "--max-logprobs", "20"]
    cfg["editor"]["context_tokens"] = None                                   # null: main_args as written
    assert main_server_args(cfg) == ["--max-model-len", "20480", "--max-logprobs", "20"]
    assert editor_kwargs(cfg, server_args=main_server_args(cfg))["context_tokens"] == 20480
    # the main replicas get it (real path, servers stubbed)
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
    main = [a for s, a in made if s == "m0"]
    assert len(main) == 3 and all(max_model_len(a) == 98304 for a in main)      # 4 GPUs: 3 replicas | train
    with pytest.raises(ValueError):
        srv.replicas("theta0", "/m", "theta0")


def _transcript(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def test_mock_tool_loop_scratch(tmp_path):
    """Default editor = thinking + tool loop: one mock run_python round per proposal on the exploration api
    (transcript format 2, editor_rounds in the candidate costs, explore output on disk); the same programs, scores
    and evaluations as with editor.tool_rounds=0 (one call per candidate, no tool lines)."""
    # history_tool off in the loop run: with one step per generation the sessions line names search_history when the
    # tool is on, and the mock editor's choices depend on the message length (tool_rounds 0 has no history tool)
    _run_b(tmp_path / "loop", "method.mode=scratch", "editor.history_tool=false")
    _run_b(tmp_path / "single", "method.mode=scratch", "editor.tool_rounds=0")
    runs = {k: next((tmp_path / k).iterdir()) for k in ("loop", "single")}
    for k in ("loop", "single"):
        assert runs[k].name == runs["loop"].name
    gl, gs = runs["loop"] / "gen_00", runs["single"] / "gen_00"
    tr = _transcript(gl / "editor" / "thread_0.jsonl")
    assert tr[0]["role"] == "system" and tr[0].get("format") == 2
    assert "Before answering you may call functions for up to 20 rounds" in tr[0]["content"]
    tools = [l for l in tr if l.get("role") == "tool"]
    finals = [l for l in tr if l.get("final")]
    assert tools and all(l["name"] == "run_python" and l["ok"] for l in tools), tools[:1]
    assert len(tools) == len(finals) and all(f["stats"]["rounds"] == 1 and f["stats"]["stop"] == "answer" for f in finals)
    assert "'spent'" in json.loads(tools[0]["content"])["stdout"]
    assert list((gl / "explore").glob("*/session/r01_0/*"))         # persistent session per candidate (base.yaml)
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


def test_search_history_tool_scratch(tmp_path):
    """search_history (2026-11-21): generation 1's editor gets the tool; it lists generation 0's candidates with
    Paper B's summary (module types, modules, target gain, ...) and same_as by digest; read_record opens them."""
    from rsicore.chat_tools import ChatTools
    from rsicore.sandbox import ToolAPI

    from lora_code.tools import search_history
    py = REPO / "eval" / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    cmd = [py, "-I", str(REPO / "lora_code" / "run.py"), "--domain", "math", "--gens", "2", "--mock",
           "--runs-dir", str(tmp_path), "--run-id", "B_test_hist", "--set", "method.mode=scratch"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    run = tmp_path / "B_test_hist"
    sysmsg = json.loads((run / "gen_01" / "editor" / "thread_0.jsonl").read_text().splitlines()[0])["content"]
    assert "search_history(gen, sort, offset)" in sysmsg
    g0 = [json.loads(x) for x in (run / "gen_00" / "candidates.jsonl").read_text().splitlines() if x.strip()]
    t = ChatTools(lambda tag, ctx: ToolAPI({}), history=search_history({0: run / "gen_00"}))
    ctx = {"cid": "g01_t0_s00", "gen_dir": str(run / "gen_01"), "visible_cids": []}
    rows, off = [], 0
    while off is not None:
        out = json.loads(t.call("search_history", {"offset": off}, ctx, 1, 0)["content"])
        rows += out["rows"]
        off = out.get("next_offset")
    assert len(rows) == len(g0) and all(x["gen"] == 0 for x in rows)
    ok = [x for x in rows if "error" not in x]
    assert ok and all("types" in x and "modules" in x and "strategy" not in x for x in ok if not x.get("keep"))
    digests = [(c.get("artifact") or {}).get("digest") for c in g0 if not c.get("error")]
    assert sum("same_as" in x for x in ok) == len(digests) - len(set(digests))
    rr = json.loads(t.call("read_record", {"cid": g0[0]["cid"], "part": "program"}, ctx, 1, 0)["content"])
    assert rr["ok"] and rr["text"] == (run / "gen_00" / "programs" / f"{g0[0]['cid']}.py").read_text()
    # (2026-11-22) generation 1's first message carries the structured archive: one row per generation-0
    # candidate, failures included, and full programs only through read_record (no quoted best program)
    tr = [json.loads(x) for x in (run / "gen_01" / "editor" / "thread_0.jsonl").read_text().splitlines()]
    first = next(x["content"] for x in tr if x.get("role") == "user")
    assert "Archive of earlier generations: EVERY candidate" in first and "read_record(cid, 'program'" in first
    assert "best program of generation 0" not in first
    arows = [json.loads(x) for x in first.splitlines() if x.startswith('  {"cid": "g00_')]
    assert sorted(x["cid"] for x in arows) == sorted(c["cid"] for c in g0)
    assert all(("error" in x) == bool(c.get("error")) for x, c in zip(sorted(arows, key=lambda x: x["cid"]),
                                                                       sorted(g0, key=lambda c: c["cid"])))


def test_tool_devices(tmp_path):
    """gpu_layout.tools_on_replicas (2026-11-22): session / candidate k uses tool device k mod n (its store / HF
    model / replica) and that device's lock; sessions on different devices run their tools at once, sessions on
    the same device one after the other; without devices the workspace lock and store as before."""
    import threading
    import time as _t

    from lora_code.mock import MockStore
    ws = _ws(tmp_path)
    s0 = ws.store
    assert ws.dev_index("g01_t3_s00") is None and ws._tool_lock() is ws._lock
    stores = [MockStore(), MockStore()]
    ws.set_devices([{"store": stores[0], "hf": None, "server": "R0"}, {"store": stores[1], "hf": None, "server": "R1"}])
    assert [ws.dev_index(c) for c in ("g01_t0_s00", "g01_t1_s00", "g01_t2_s00~x", "g01_t3_s00~x01_0", "x")] == \
        [0, 1, 0, 1, None]
    ws.begin_program("g01_t3_s00")
    assert ws.store is stores[1] and ws._tool_lock() is ws.devs[1]["lock"] and ws._loss_replica() == "R1"
    ws.end_program("g01_t3_s00")
    assert ws.store is s0 and ws._tool_lock() is ws._lock

    inside, peak, lk = [0], [0], threading.Lock()

    def slow(**kw):
        with lk:
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
        _t.sleep(0.2)
        with lk:
            inside[0] -= 1
        return {}
    fn = ws._wrap("modules", slow)

    def run(cid):
        ws.begin_program(cid)
        try:
            fn()
        finally:
            ws.end_program(cid)
    for cids, want in ((["g01_t0_s00", "g01_t1_s00"], 2), (["g01_t0_s00", "g01_t2_s00"], 1)):
        peak[0] = 0
        ts = [threading.Thread(target=run, args=(c,)) for c in cids]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert peak[0] == want, (cids, peak[0])


def test_vllm_answer_nll():
    """tools.vllm_answer_nll (2026-11-22): token-id prompts in batches, -logprob of each answer token (labels !=
    -100) read from prompt_logprobs at its own position, mean over all answer tokens; missing logprobs raise."""
    import math

    import pytest

    from lora_code.tools import vllm_answer_nll
    enc = [([5, 6, 7, 8], [-100, -100, 7, 8]), ([9, 3], [-100, 3]), ([4, 4, 2], [-100, -100, 2])]
    calls = []

    def post(url, payload, timeout=None):
        calls.append((url, payload))
        assert payload["prompt_logprobs"] == 1 and payload["max_tokens"] == 1
        chs = []
        for k, ids in enumerate(payload["prompt"]):
            plp = [None] + [{str(t): {"logprob": -0.1 * t, "rank": 1}, "99": {"logprob": -9.0, "rank": 2}}
                            for t in ids[1:]]
            chs.append({"index": k, "prompt_logprobs": plp})
        return {"choices": list(reversed(chs))}
    nll, secs = vllm_answer_nll("http://h:1/v1", "tl_x", enc, batch=2, post=post)
    assert math.isclose(nll, 0.1 * (7 + 8 + 3 + 2) / 4) and secs >= 0
    assert [c[0] for c in calls] == ["http://h:1/v1/completions"] * 2 and calls[0][1]["model"] == "tl_x"
    assert calls[0][1]["prompt"] == [[5, 6, 7, 8], [9, 3]] and calls[1][1]["prompt"] == [[4, 4, 2]]

    def bad(url, payload, timeout=None):
        return {"choices": [{"index": 0, "prompt_logprobs": [None, {"1": {"logprob": -1.0}}]}]}
    with pytest.raises(RuntimeError, match="no logprob for token 3"):
        vllm_answer_nll("http://h:1/v1", "m", [([9, 3], [-100, 3])], post=bad)
    import time as _t
    with pytest.raises(RuntimeError, match="budget or wall-clock limit ran out"):
        vllm_answer_nll("http://h:1/v1", "m", enc, post=post, deadline=_t.time() - 1)


def test_vllm_loss_cleans_up(tmp_path, monkeypatch):
    """Workspace._vllm_loss (2026-11-22): the adapter is unloaded and its directory deleted also when loading it on
    the replica fails; the replica's load time is charged to the call."""
    import pytest

    import lora_code.primitives.lora as L

    class Lo:
        def modules(self):
            return ["layers.1.mlp.down_proj"]

        def max_rank(self):
            return 4

    class Srv:
        endpoint, max_lora_rank = "http://h:1/v1", 64

        def __init__(self):
            self.unloaded = []

        def load_lora(self, name, path):
            assert Path(path).is_dir()
            raise RuntimeError("load_lora failed: rank too large")

        def unload_lora(self, name):
            self.unloaded.append(name)

    def fake_save(lo, path, store, **kw):
        Path(path).mkdir(parents=True)
        (Path(path) / "adapter_model.safetensors").write_text("x")
    monkeypatch.setattr(L, "save_peft", fake_save)
    ws = _ws(tmp_path)
    srv = Srv()
    ws.hf = type("H", (), {"encode": staticmethod(lambda e: ([1, 2, 3], [-100, 2, 3]))})()
    ws.evaluator = type("E", (), {"served": "m0", "replicas": [srv]})()
    ws.begin_program("g01_t0_s00")
    with pytest.raises(RuntimeError, match="rank too large"):
        ws._vllm_loss([{"messages": [], "target": "a"}], Lo())
    ws.end_program("g01_t0_s00")
    assert srv.unloaded == [f"tl_{ws.name_prefix}_1"]
    assert not any((tmp_path / "train_loss_adapters").glob("*"))


def test_warm_devices_raises_and_budget_shows_session_gpu(tmp_path):
    """warm_devices (2026-11-22) raises the first HF load failure with the device named; api.budget inside
    run_python reports the session's own exploration GPU seconds."""
    import pytest

    class Bad:
        device = "cuda:3"

        @property
        def model(self):
            raise MemoryError("CUDA out of memory")
    ws = _ws(tmp_path)
    ws.set_devices([{"hf": None, "store": None, "server": None}, {"hf": Bad(), "store": None, "server": None}])
    with pytest.raises(RuntimeError, match=r"tool device 1 \(cuda:3\).*MemoryError"):
        ws.warm_devices()
    cfg = {"editor": {"explore": {"timeout_s": 60, "mem_gb": 4, "gpu_tools": True, "gpu_seconds": 10}}}
    ws = _ws(tmp_path / "b")
    out, res = _explore(ws, "api.acts(module='layers.3.mlp.down_proj', ids=api.train_ids()[:4])\nresult = api.budget()",
                        tmp_path / "b", cfg=cfg)
    assert res["ok"], res
    assert 0 < res["result"]["explore_gpu_seconds_left"] < 10 and "explore_note" in res["result"]


def test_train_loss_backend_choice(tmp_path):
    """Workspace.loss_backend (2026-11-22): vllm only when tools.train_loss_backend is vllm, outside mock, with an
    evaluator, a LoRA (not a weight edit) and no parent LoRA; hf otherwise."""
    from lora_code.mock import MockLoRA, MockWeightEdit
    ws = _ws(tmp_path)
    lo = MockLoRA()
    assert ws.loss_backend(lo, None) == "hf"
    ws.cfg = {**ws.cfg, "tools": {"train_loss_backend": "vllm"}}
    assert ws.loss_backend(lo, None) == "hf"                 # mock
    srv = type("S", (), {"max_lora_rank": 64})()
    ws.mock, ws.evaluator = False, type("E", (), {"replicas": [srv], "served": "m"})()
    assert ws.loss_backend(lo, None) == "vllm"
    assert ws.loss_backend(lo, MockLoRA()) == "hf"           # mode=edit: the parent is a LoRA
    assert ws.loss_backend(MockWeightEdit(), None) == "hf"
    big = MockLoRA({"layers.1.mlp.down_proj": {"rank": 65, "rel": 0.01}})
    assert ws.loss_backend(big, None) == "hf"                # padded to 128 > the replica's --max-lora-rank 64
    ws.evaluator = object()
    assert ws.loss_backend(lo, None) == "hf"                 # no replica to score on


def test_replica_memory():
    """common.replica_memory (2026-11-22): with tools_on_replicas the main replicas get replica_vllm_gib (and a
    scaled utilization fallback); unchanged without it or in mode=weights; the b300 profile turns it on."""
    import yaml

    from lora_code.common import replica_memory, tools_on_replicas
    args = ["--rsi-gpu-memory-gib", "220", "--gpu-memory-utilization", "0.81", "--max-model-len", "131072"]
    on = {"gpu_layout": {"tools_on_replicas": True, "replica_vllm_gib": 140}}
    assert replica_memory(args, on) == ["--rsi-gpu-memory-gib", "140", "--gpu-memory-utilization", "0.515",
                                        "--max-model-len", "131072"]
    assert replica_memory(args, {"gpu_layout": {"replica_vllm_gib": 140}}) == args
    assert not tools_on_replicas({**on, "method": {"mode": "weights"}})
    base = yaml.safe_load((REPO / "lora_code" / "configs" / "base.yaml").read_text())
    assert base["hardware_profiles"]["b300"]["gpu_layout"] == {"tools_on_replicas": True, "replica_vllm_gib": 140}
    assert not base["gpu_layout"]["tools_on_replicas"]
    assert not (base["hardware_profiles"]["h20"].get("gpu_layout") or {}).get("tools_on_replicas")


def test_archive_row():
    """prompts.archive_row (2026-11-22): what the candidate changed (module types; per type layers, count, rank,
    rel norm, weight-edit ops), its gate result, error first line, kept, note."""
    from types import SimpleNamespace as NS

    from lora_code.prompts import archive_row
    edit = {"n": 7, "types": ["mlp.down_proj", "self_attn.o_proj"], "groups": [
        {"type": "mlp.down_proj", "n": 5, "layers": "10-14", "rank": "4", "rel_norm": "0.002-0.004"},
        {"type": "self_attn.o_proj", "n": 2, "layers": "3,7", "rank": "1-2", "rel_norm": "0.001"}]}
    ok = NS(cid="g00_t1_s00", error=None, score=0.0025, artifact={
        "strategy": "svd", "modules": 7, "max_rank": 4, "tools_used": {"svd": 3}, "edit": edit,
        "note": "down_proj 10-14: largest gradient norm on failed problems"})
    r = archive_row(ok, {"passed": False, "gain": 0.00251, "ci": [-0.0105, 0.016], "reason": "gain not positive"})
    assert r == {"cid": "g00_t1_s00", "types": ["mlp.down_proj", "self_attn.o_proj"],
                 "modules": ["mlp.down_proj L10-14 x5 rank 4 rel 0.002-0.004",
                             "self_attn.o_proj L3,7 x2 rank 1-2 rel 0.001"],
                 "gate": {"passed": False, "gain": 0.0025, "ci": [-0.0105, 0.016], "reason": "gain not positive"},
                 "note": "down_proj 10-14: largest gradient norm on failed problems"}
    wed = NS(cid="g00_t4_s00", error=None, score=None, artifact={"kind": "weights", "edit": {
        "n": 3, "types": ["mlp.up_proj"], "groups": [{"type": "mlp.up_proj", "n": 3, "layers": "20-22",
                                                     "rel_norm": "0.01", "ops": "scale s=1.02"}]}})
    assert archive_row(wed) == {"cid": "g00_t4_s00", "kind": "weights", "types": ["mlp.up_proj"],
                                "modules": ["mlp.up_proj L20-22 x3 rel 0.01 [scale s=1.02]"]}
    old = NS(cid="g00_t5_s00", error=None, score=None, artifact={"modules": 12, "max_rank": 4})
    assert archive_row(old) == {"cid": "g00_t5_s00", "modules": 12, "rank": 4}
    kept = NS(cid="g00_t2_s00", error=None, score=None, artifact={"keep": True, "note": "all 5 raised train_loss"})
    assert archive_row(kept) == {"cid": "g00_t2_s00", "kept": True, "note": "all 5 raised train_loss"}
    bad = NS(cid="g00_t3_s00", error="program failed: NameError: name 'x' is not defined\nTraceback ...",
             score=None, artifact=None)
    assert archive_row(bad) == {"cid": "g00_t3_s00", "error": "program failed: NameError: name 'x' is not defined"}


def test_edit_summary():
    """tools.edit_summary (2026-11-22): modules grouped by type (and by op setting for a weight edit) with layer
    ranges, rank and rel-norm spans; non-layer tensors keep their name; at most EDIT_GROUPS groups."""
    from types import SimpleNamespace as NS

    from lora_code.tools import EDIT_GROUPS, edit_summary
    rep = {"modules": {**{f"layers.{i}.mlp.down_proj": {"rank": 4, "rel_norm": 0.002 + 0.001 * (i - 10)}
                          for i in (10, 11, 12, 14)},
                       "layers.3.self_attn.o_proj": {"rank": 1, "rel_norm": 0.001},
                       "layers.7.self_attn.o_proj": {"rank": 2, "rel_norm": 0.001}}}
    s = edit_summary(NS(kind="lora"), rep)
    assert s == {"n": 6, "types": ["mlp.down_proj", "self_attn.o_proj"], "groups": [
        {"type": "mlp.down_proj", "n": 4, "layers": "10-12,14", "rank": "4", "rel_norm": "0.002-0.006"},
        {"type": "self_attn.o_proj", "n": 2, "layers": "3,7", "rank": "1-2", "rel_norm": "0.001"}]}

    class WE:
        kind = "weights"

        def summary(self):
            return {"layers.2.mlp.up_proj": [{"op": "scale", "s": 1.02}],
                    "layers.3.mlp.up_proj": [{"op": "scale", "s": 1.02}],
                    "layers.4.mlp.up_proj": [{"op": "rows", "idx": [1, 2, 3], "f": 0.9}, {"op": "lowrank", "rank": 2}],
                    "norm.weight": [{"op": "delta", "shape": [8], "norm": 0.1}]}
    rep = {"modules": {m: {"rel_norm": 0.01} for m in WE().summary()}}
    s = edit_summary(WE(), rep)
    assert s["types"] == ["mlp.up_proj", "norm.weight"]
    assert s["groups"][0] == {"type": "mlp.up_proj", "n": 2, "layers": "2-3", "rel_norm": "0.01", "ops": "scale s=1.02"}
    assert {"type": "mlp.up_proj", "n": 1, "layers": "4", "rel_norm": "0.01",
            "ops": "rows n=3 f=0.9; lowrank r=2"} in s["groups"]
    assert {"type": "norm.weight", "n": 1, "rel_norm": "0.01", "ops": "delta"} in s["groups"]
    many = {"modules": {f"layers.{i}.mlp.{leaf}": {"rank": 1, "rel_norm": 0.001}
                        for i in range(2) for leaf in ("gate_proj", "up_proj", "down_proj")} |
                       {f"layers.{i}.self_attn.{leaf}": {"rank": 1, "rel_norm": 0.001}
                        for i in range(2) for leaf in ("q_proj", "k_proj", "v_proj", "o_proj")} |
                       {f"layers.{i}.linear_attn.{leaf}": {"rank": 1, "rel_norm": 0.001}
                        for i in range(2) for leaf in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")}}
    s = edit_summary(NS(kind="lora"), many)
    assert len(s["groups"]) == EDIT_GROUPS and s["more_groups"] == 12 - EDIT_GROUPS and len(s["types"]) == 12


def test_prompt_gate_text_and_rules():
    """System prompt: the gate rule follows the config (min_gain, require_significant, gate item count; every
    candidate vs the current model, the only flow since 2026-11-23; mode=edit generation 0: RAW as the fallback), the
    api-scope / module-name / vector-side / robust-code rules, no api.evaluate / evaluate budget, none of the removed
    tools (regmean_preserve) or measurements (retention, KL); the tensor-tool rules only when the tool is on (absent
    key: on, 2026-11-22)."""
    from lora_code.prompts import gate_text, system_prompt
    base = {"gate": {"min_gain": 0.01, "require_significant": True}}
    sig = gate_text(base, gate_n=500)
    assert "statistically significant" in sig and "about 5 more correct items out of 500" in sig
    assert "evaluated ONCE on a fixed hidden set of 500 items" in sig and "none passes: the model stays as it is" in sig
    nosig = gate_text({"gate": {"min_gain": 0.01, "require_significant": False}}, gate_n=500)
    assert "no significance test" in nosig and "95% CI" not in nosig and "Retention" not in nosig
    rawfb = gate_text({"gate": {"min_gain": 0.0}}, gate_n=30, raw_fallback=True)
    assert "RAW itself is accepted if it passes the same gate" in rawfb and "must be positive" in rawfb
    assert "raw LoRA's by more than" not in rawfb and "epsilon" not in rawfb
    cfg = {"gate": {"min_gain": 0.01, "require_significant": False}, "sandbox": {"timeout_s": 1800},
           "tools": {"tensor_compute": False}}
    sp = system_prompt(cfg, budget={"gpu_seconds": 7200}, mode="scratch", gate_n=500)
    assert "no significance test" in sp and "95% CI" not in sp and "1800 s of wall clock" in sp
    assert "NOT a global there" in sp and "api.modules()" in sp and "OUTPUT-side" in sp and "try/except" in sp
    assert "there is no api.evaluate" in sp and "- evaluate(" not in sp and "evaluate calls" not in sp
    assert "api.evaluate" not in sp.replace("there is no api.evaluate", "")
    for gone in ("regmean_preserve", "retention", "KL", "program_evaluate_calls_left", "per program"):
        assert gone not in sp, gone
    assert "(acts and\n  train_loss run the model" in sp and "(acts / train_loss)" in sp
    assert "api.tensors()" not in sp and "T5.name" not in sp and "tensor_compute" not in sp   # tool switched off
    noscr = system_prompt(dict(cfg, tools={"tensor_compute": False, "screen_results": False}),
                          budget={"gpu_seconds": 7200}, mode="scratch")
    assert "you only see each\n  candidate's gate statistics" in noscr and "- screen_results(" not in noscr
    on = dict(cfg, tools={})                                     # absent keys: on
    fp = system_prompt(on, budget={"gpu_seconds": 7200}, mode="scratch", gate_n=500)
    assert "api.tensors()" in fp and "T5.name" in fp and "lora = {module: (A [r, in], B [out, r])}" in fp
    assert "at most ~300 s" in fp and "writes = " not in fp and "- screen_results(" in fp
    ep = system_prompt(on, budget={"gpu_seconds": 7200}, mode="edit", gate_n=30, raw_fallback=True)
    assert "RAW itself is accepted" in ep and "raise the target score." in ep and "token-KL" not in ep
    assert "relchange_prune, orthogonalize" in ep
    wp = system_prompt(dict(on, weights={"vector_cap_tau": 0.25}), budget={"gpu_seconds": 7200}, mode="weights")
    assert "rms(W'-W) <= 0.25 * max(rms(W), 1)" in wp and "zero-centred" in wp and "use w_scale" in wp
    assert "dense additions computed by api.tensor_compute" in wp and "writes = {tensor: new full tensor}" in wp
    assert "lm_head are never editable" in wp and "evaluate calls" not in wp
    off = system_prompt(dict(on, tools={"tensor_compute": False}), budget={"gpu_seconds": 7200}, mode="weights")
    assert "rms(W'-W)" not in off and "api.tensors()" not in off and "tensor_compute" not in off


class _CountingEv:
    """Minimal evaluator for Workspace.evaluate_lora: counts scorings (feedback calls)."""

    def __init__(self):
        self.names, self.n = set(), 0

    def known(self, name):
        return name in self.names

    def register(self, lo, name):
        self.names.add(name)

    def feedback(self, name, level, lora_report=None, cost=None):
        self.n += 1
        return 0.1, {"score": 0.1, "level": level, "lora": name, "cost": cost}


def test_no_evaluate_tool_and_budget(tmp_path):
    """2026-11-23: there is no evaluate tool or evaluate budget; Workspace.evaluate_lora (the raw LoRA's / non-LLM
    branch's points) scores once per (LoRA, level), never charged; api.budget() shows the program's time left and the
    note that every returned candidate gets one gate evaluation."""
    from lora_code.mock import MockStore, m_sgd
    from lora_code.primitives.train import TrainData
    from lora_code.tools import TOOL_DOCS, WEIGHTS_DOC_OVERRIDES, Workspace, tools_for_mode
    assert "evaluate" not in TOOL_DOCS and "evaluate" not in WEIGHTS_DOC_OVERRIDES
    assert Workspace.EXPLORE_NEVER == ("raw_lora",)
    for mode in ("scratch", "edit", "weights"):
        assert "evaluate" not in tools_for_mode(mode, cfg={})
        assert "level='gate'" in tools_for_mode(mode, cfg={})["compare"]
    train = TrainData({f"t/{i}": {"messages": [{"role": "user", "content": "q"}], "target": "a", "source": "gold"}
                       for i in range(10)})
    ev = _CountingEv()
    ws = Workspace(mock=True, store=MockStore(), train=train, out_dir=tmp_path, evaluator=ev,
                   budget={"gpu_seconds": 100}, cfg={"lora": {"rank_cap": 64, "norm_cap_tau": 0.1}})
    ws.build_api()
    assert "evaluate" not in ws.api.names() and not hasattr(ws, "t_evaluate") and not hasattr(ws, "screen_candidate")
    lo = m_sgd(None, ["layers.3.mlp.down_proj"], ["t/1"], 5, 1e-4, 4)
    sc, fb = ws.evaluate_lora(lo, "screen")
    assert sc == 0.1 and ev.n == 1 and "evaluate_left" not in fb["cost"] and fb["cost"]["gpu_seconds_left"] == 100
    assert ws.evaluate_lora(lo, "screen") == (sc, fb) and ev.n == 1           # cached
    ws.evaluate_lora(lo, "full")
    assert ev.n == 2 and ws.spent.get("evaluate") is None
    ws.begin_program("g00_t0_s00")
    b = ws.t_budget()
    assert 1790 < b["program_seconds_left"] <= 1800 and "program_evaluate_calls_left" not in b
    assert b["spent"] == {"gpu_seconds": 0} or "evaluate" not in b["spent"]
    assert b["note"].startswith("no api.evaluate in this run")
    ws.end_program("g00_t0_s00")
    assert "program_seconds_left" not in ws.t_budget()
    assert ws.budget_status() == {"gpu_seconds_left": 100.0, "gpu_seconds": 100}


def test_context_budget_line():
    """Step context (2026-11-21): no api.evaluate (one gate evaluation per candidate), GPU budget left, program time
    limit and observed timings; candidate rows without the removed keys; one step: session k of N."""
    from types import SimpleNamespace
    from lora_code.prompts import ContextRenderer
    st = {"gpu_seconds_left": 50.0, "gpu_seconds": 7200}
    r = ContextRenderer(task_text="T", diag_text="D", budget_fn=lambda: st, timeout_s=1800)
    hist = [SimpleNamespace(cid="g00_t0_s00", thread=0, step=0, artifact={"seconds": 600.0}, cost={"screen_s": 90.0},
                            error=None, screen={"score": 0.1, "feedback": {"gate": {"passed": True, "gain": 0.1}}},
                            full=None, score=0.1, program="x"),
            SimpleNamespace(cid="g00_t1_s00", thread=1, step=0, artifact=None, cost={}, score=None, screen=None,
                            full=None, program="y", error="program failed: timeout after 1800 s")]
    msg = r(hist, 0)
    assert ("No api.evaluate this generation: every returned candidate gets one gate evaluation automatically, 50 of "
            "7200 GPU-seconds.") in msg
    assert "600 s (median)" in msg and "1 program(s) were killed" in msg and "about 90 s" in msg
    assert "score = the paired gain in gate accuracy vs the current model" in msg
    for gone in ("evaluate calls", "Few evaluations", "vs_raw", "worst_retention_delta", "\"kl\"", "full_score"):
        assert gone not in msg, gone
    assert "No api.evaluate" not in ContextRenderer(task_text="T", diag_text="D")([], 0)
    assert "You are session" not in msg
    r = ContextRenderer(task_text="T", diag_text="D", archive_text="Archive of earlier generations:",
                        budget_fn=lambda: st, timeout_s=1800, sessions=8, history_tool=True)
    msg = r([], 2)
    assert "You are session 3 of 8 in this generation" in msg and "search_history and the archive below" in msg
    r2 = ContextRenderer(task_text="T", diag_text="D", sessions=4)
    assert "You are session 4 of 4" in r2([], 3) and "search_history" not in r2([], 3)
