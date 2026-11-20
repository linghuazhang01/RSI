"""E_t: the editor-capability measurement on a FIXED task (RSI_LOOP_DESIGN §3.5).

A fresh editor session (no archive) run by a branch model (parent M_t, M_t ⊕ raw LoRA in mode=edit,
M_t ⊕ self-written LoRA, M_t ⊕ non-LLM LoRA) writes a LoRA for the ORIGINAL base model θ0 on Seal-Tools (sealtool_train train
items; screen feedback on a fixed sealtool_train signal slice), with 16 evaluate calls and a fixed GPU
budget, single seed. The best LoRA is scored on held-out Seal-Tools test + BFCL; when the run's target
domain is tool_calling the code task is used instead (taco_verified -> LiveCodeBench).
E-task items are a fixed deterministic subset of the dataset's pool minus the run's gate (independent of
generation and branch). E-ref = θ0 itself on the E test (computed once per run; no SFT baseline).
method.mode applies here too: edit = the session edits a raw LoRA for θ0 (the `raw` recipe on the E task,
trained once in e_ref; the only training of the E task); scratch = from zero. No session trains.
"""
from __future__ import annotations

import random
from pathlib import Path

from .common import main_server_args, pool_ids, read_json, write_json

E_TASKS = {
    "tool_calling": {"dataset": "sealtool_train", "domain": "tool_calling", "test": ["sealtool", "bfcl"]},
    "code": {"dataset": "taco_verified", "domain": "code", "test": ["livecodebench"]},
}


def e_task_for(domain: str, cfg: dict) -> dict:
    ec = cfg.get("e_t", {})
    key = ec.get("task_when_target_is_tool_calling", "code") if domain == "tool_calling" else ec.get("task", "tool_calling")
    t = dict(E_TASKS[key])
    if ec.get("test_tasks", {}).get(key):
        t["test"] = list(ec["test_tasks"][key])
    t["key"] = key
    return t


def e_items(rc, task: dict, cfg: dict) -> dict:
    """Fixed E-task ids: {train: [...], signal: [...]} from pool \\ gate, seeded by the data seed only."""
    from rsicore.draws import gate_ids

    ec = cfg.get("e_t", {})
    ds = task["dataset"]
    gate = set(gate_ids(rc, ds))
    pool = sorted(i for i in pool_ids(ds) if i not in gate)
    rng = random.Random(f"{ec.get('items_seed', 20261118)}:etask:{ds}")
    rng.shuffle(pool)
    n_tr, n_sig = int(ec.get("n_train", 1600)), int(ec.get("n_signal", 200))
    return {"train": sorted(pool[:n_tr]), "signal": sorted(pool[n_tr:n_tr + n_sig])}


def e_test(endpoint: str, served: str, task: dict, out_dir: Path, *, lora: str | None, mock: bool,
           limit: int | None, decoding: dict | None = None) -> dict:
    """Score θ0 (+lora) on the held-out E-test tasks. Records get ids '<task>:<id>' for pairing.
    decoding: heldout.decoding (sampled avg@n, like every held-out evaluation); None = tasks.yaml decoding."""
    from rsicore.evaluate import score
    from rsicore.heldout import decoding_args

    cache = Path(out_dir) / "e_test.json"
    got = read_json(cache)
    if got is not None:
        return got
    recs, per = [], {}
    for t in task["test"]:
        dec = decoding_args(decoding, t)
        extra = (["--limit", str(limit)] if limit else []) + dec["extra"]
        er = score(endpoint, served, t, pool="test", ids=None, out_dir=Path(out_dir) / f"test_{t}", lora=lora,
                   extra=extra, mock=mock, **dec["kw"])
        rr = [{"id": f"{t}:{r.get('id')}", "score": r.get("score")} for r in er.records]
        sc = [r["score"] for r in rr if r["score"] is not None]
        per[t] = {"score": sum(sc) / len(sc) if sc else None, "n": len(rr), "n_scored": len(sc)}
        recs += rr
    allsc = [r["score"] for r in recs if r["score"] is not None]
    got = {"score": sum(allsc) / len(allsc) if allsc else None, "per_task": per, "records": recs}
    write_json(cache, got)
    return got


def run_etask(rc, cfg: dict, *, out_dir: Path, branch: str, editor_endpoint: str, editor_served: str,
              editor_skill: float | None, theta0, task: dict, items: dict, make_workspace, mode: str = "scratch",
              raw=None) -> dict:
    """One E_t measurement. `theta0` = dict(endpoint, served, server, store, hf, quality);
    make_workspace(out_dir, evaluator, budget, prefix) builds the θ0 workspace. Returns the result dict."""
    from rsicore.editor import Editor, editor_kwargs
    from rsicore.search import inner_search

    from . import mock as M
    from .evaluator import Evaluator
    from .prompts import ContextRenderer, system_prompt, task_text
    from .tools import editor_chat_tools, editor_mock_policy, program_build

    out_dir = Path(out_dir)
    res_path = out_dir / "result.json"
    got = read_json(res_path)
    if got is not None:
        return got
    ec = cfg.get("e_t", {})
    budget = {"evaluate": int(ec.get("evaluate", 16)), "gpu_seconds": float(ec.get("gpu_seconds", 3600))}
    ds = task["dataset"]
    ev = Evaluator(rc, cfg, out_dir=out_dir, endpoint=theta0["endpoint"], served=theta0["served"],
                   server=theta0["server"], ids={"screen": {ds: items["signal"]}, "full": {ds: items["signal"]}},
                   target_dataset=ds, kl_texts=[], store=theta0["store"], mock=rc.mock,
                   registry=rc.run_dir / "mock_registry.json", parent_quality=theta0.get("quality"),
                   tag=f"E-{branch}")
    ev.mark_shown_examples = False
    raw_info = None
    if raw is not None:   # (LoRA, name): register the raw LoRA on this evaluator so feedback has vs_raw
        ev.register(raw[0], raw[1])
        ev.raw_name = raw[1]
    ws = make_workspace(out_dir, ev, budget, f"e{branch[:3]}")
    ws.build_api()
    if raw is not None:
        raw_info = dict(ws.raw_info or {})
    sysmsg = system_prompt(cfg, budget=budget, e_task=True, mode=mode, retention=bool(ws.retention_texts))
    ctx = ContextRenderer(task_text=task_text(gen=0, target=task["domain"], dataset=ds, retention=[],
                                              n_train=len(items["train"]), e_task=True,
                                              raw_info=raw_info),
                          diag_text="(Diagnostics are not provided for the E task; use api.modules(), api.svd(), "
                                    "api.acts().)")
    seed = rc.seed
    policy = M.make_mock_policy(seed, skill=editor_skill if editor_skill is not None else 0.7,
                                tag=f"E{branch}", mode=mode, retention=bool(ws.retention_texts)) if rc.mock else None

    # 2026-11-21: thinking + tool loop as in the main search, run_python on the E task workspace's exploration api
    policy, tools = editor_mock_policy(policy, cfg), editor_chat_tools(ws, cfg)
    ekw = editor_kwargs(cfg, server_args=main_server_args(cfg))   # the editor runs on the main server

    def editor_factory(th):
        return Editor(editor_endpoint, editor_served, system_prompt=sysmsg, seed=seed + th,
                      transcript=out_dir / "editor" / f"thread_{th}.jsonl", mock_policy=policy, tools=tools, **ekw)

    sb = cfg.get("sandbox", {})

    def build(program, cid):
        return program_build(ws, program, cid, timeout_s=int(sb.get("timeout_s", 1800)),
                             mem_gb=int(sb.get("mem_gb", 16)), workdir=out_dir / "work" / cid)

    def screen(art, cid):
        return ws.screen_candidate(art["cid"])   # charged on the workspace's ToolAPI (identical LoRA: free)

    def full(art, cid):
        return ws.evaluate_lora(ws.load_candidate(art["cid"]), "full", charge=False)

    # records (candidates.jsonl, programs/, feedback/) go straight into out_dir = gen_XX/e_t/<branch>/
    ranked = inner_search(rc, 0, editor_factory=editor_factory, render_context=ctx, build_fn=build,
                          screen_fn=screen, full_fn=full, threads=int(ec.get("threads", 4)),
                          steps=int(ec.get("steps", 4)), max_candidates=budget["evaluate"],
                          top_full=int(ec.get("top_full", 1)), out_dir=out_dir)
    best = next((c for c in ranked if c.artifact is not None and c.score is not None), None)
    lo = ws.load_candidate(best.cid) if best else None
    name = None
    if lo is not None and lo.modules():
        name = f"E-{rc.run_id}-{out_dir.parent.parent.name}-{branch}"
        ev.register(lo, name)
        ev.ensure_loaded(name)
    test = e_test(theta0["endpoint"], theta0["served"], task, out_dir, lora=name, mock=rc.mock,
                  limit=ec.get("test_limit"), decoding=(cfg.get("heldout") or {}).get("decoding"))
    res = {"branch": branch, "editor": editor_served, "best_cid": best.cid if best else None,
           "best_screen": best.score if best else None, "lora": name, "score": test["score"],
           "per_task": test["per_task"], "records": test["records"],
           "strategies": [c.artifact.get("strategy") for c in ranked if c.artifact],
           "n_candidates": len(ranked), "n_failed": sum(1 for c in ranked if c.error),
           "spent": dict(ws.spent)}
    write_json(res_path, res)
    return res
