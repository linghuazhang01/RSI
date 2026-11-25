"""Toy "paper" that drives the whole rsicore loop in mock mode, without edit_code/ or lora_code/.

  eval/.venv/bin/python -I eval/tests/rsicore/toy_paper.py --runs-dir /tmp/toy_runs --gens 2 --limit 40

Per generation g (Paper-B-like: candidates are LoRA-style JSON artifacts hot-swapped on the server; the simple
flow): draw of simple_math_pool -> mock server serving M_g -> M_g on the gate (simple_math_eval) -> inner_search
with a mock editor whose programs call sandboxed tools -> builders register each candidate's quality in the mock
registry -> gate per candidate (screen_fn = the paired gate gain over M_g) -> accept the largest positive gain ->
summary (no held-out test since 2026-11-25). The accepted candidate's quality becomes M_{g+1}.
A second run with the same arguments must skip every step (prints "ALL STEPS SKIPPED").
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from rsicore import config, draws, editor, evaluate, records, sandbox, search, servers, state  # noqa: E402

SYSTEM = "You write programs `def main(api)` that choose an edit strength in [0, 1]. Reply with one ```python block."


def toy_quality(base_q: float, s: float) -> float:
    return round(max(0.05, min(0.95, base_q + 0.25 * s - 0.15 * s * s)), 4)


def mock_policy(thread):
    def policy(history):
        step = sum(1 for m in history if m["role"] == "assistant")
        if thread == 1 and step == 0:   # a buggy first program: out-of-range argument -> tool error
            return "def main(api):\n    return api.estimate(strength=5.0)\n"
        s = min(1.0, 0.2 + 0.3 * step + 0.1 * thread)
        return (f"def main(api):\n    est = api.estimate(strength={s})\n"
                f"    return {{'strength': {s}, 'estimate': est}}\n")
    return policy


def main():
    ap = config.build_arg_parser("B")
    ap.add_argument("--limit", type=int, default=40, help="items per evaluation (keeps the toy fast)")
    ap.set_defaults(domain="math", mock=True, run_id="toy_B_math")
    for act in ap._actions:            # --domain is required for real papers; the toy defaults it
        if act.dest == "domain":
            act.required = False
    a = ap.parse_args()
    if not a.paper_dir:
        pdir = Path(a.runs_dir) / "_toy_paper_dir"
        (pdir / "configs" / "domains").mkdir(parents=True, exist_ok=True)
        (pdir / "configs" / "base.yaml").write_text("search: {threads: 2, steps: 2, max_candidates: 4}\n"
                                                    "simple: {screen_n: 40}\nbase_quality: 0.55\n")
        (pdir / "configs" / "domains" / "math.yaml").write_text("target: math\n")
        a.paper_dir = str(pdir)
    rc = config.load_run_config("B", a)
    gds = draws.gate_dataset(rc)                  # simple_math_eval (the train role simple_math_pool is drawn)
    st = state.RunState(rc.run_dir)
    reg = rc.run_dir / "mock_registry.json"
    S, L = rc.paper_cfg["search"], a.limit
    work = 0

    def run(key):
        nonlocal work
        if not st.done(key):
            work += 1
            print(f"[toy] running {key}", flush=True)
        return st.step(key)

    with run("gate_split") as todo:
        if todo:
            draws.ensure_gate(rc)
    cur_q = (st.get("model") or {}).get("quality", rc.paper_cfg["base_quality"])
    for g in range(rc.gens):
        if st.done(f"gen{g:02d}/done"):
            cur_q = st.get(f"gen{g:02d}/done")["next_quality"]
            continue
        gd = records.gen_dir(rc, g)
        with run(f"gen{g:02d}/draw") as todo:
            if todo:
                draws.draw_gen(rc, g, target="math")      # (the screen is unused: candidates are gated directly)
        served = f"toy-m{g:02d}"
        regd = records.read_json(reg, {})
        regd[served] = cur_q
        records.write_json(reg, regd)
        with servers.VLLMServer(rc.model_path, served, rc.gpus, enable_lora=True, mock=True, mock_registry=reg,
                                seed=rc.seed, log_dir=rc.run_dir / "logs") as srv:
            gate_ids = draws.gate_ids(rc, gds)
            ref = evaluate.score(srv.endpoint, served, gds, pool="pool", ids=gate_ids, out_dir=gd / "eval" / "ref",
                                 extra=["--limit", str(L)], mock=True)       # M_g on the gate

            def build_fn(program, cid):
                api = sandbox.ToolAPI({"estimate": 3})

                def estimate(strength: float):
                    if not 0 <= strength <= 1:
                        raise ValueError("strength must be in [0, 1]")
                    return {"predicted_gain": round(0.25 * strength - 0.15 * strength ** 2, 4)}
                api.register("estimate", estimate, cost={"estimate": 1}, doc="cheap analytic estimate of the gain")
                res = sandbox.run_program(program, api, timeout_s=60, mem_gb=4, workdir=gd / "work" / cid)
                if not res.ok:
                    return None, res.error
                s = float(res.result["strength"])
                art = {"name": cid, "strength": s, "quality": toy_quality(cur_q, s), "path": str(gd / "loras" / cid),
                       "cost": {"tool_calls": res.info["tool_calls"]}}
                Path(art["path"]).mkdir(parents=True, exist_ok=True)
                records.write_json(Path(art["path"]) / "adapter_config.json", art)
                r = records.read_json(reg, {})
                r[cid] = art["quality"]
                records.write_json(reg, r)
                return art, None

            def gate_candidate(art, cid):            # gate per candidate: the paired gain over M_g on the gate
                srv.load_lora(art["name"], art["path"])
                e = evaluate.score(srv.endpoint, served, gds, pool="pool", ids=gate_ids, out_dir=gd / "eval" / "gate" / cid,
                                   lora=art["name"], extra=["--limit", str(L)], mock=True)
                d = evaluate.paired_delta(ref.records, e.records)
                return d["delta"], {"accuracy": e.accuracy, "n": e.summary["n"], "gain": d["delta"], "ci": d["ci"]}

            def render(history, thread):
                lines = [f"{c.cid}: score={c.score} error={(c.error or '')[:80]!r}" for c in history]
                return "History:\n" + ("\n".join(lines) or "(none)") + f"\nYou are thread {thread}."

            with run(f"gen{g:02d}/search") as todo:
                if todo:
                    ranked = search.inner_search(
                        rc, g, render_context=render, build_fn=build_fn,
                        editor_factory=lambda th: editor.Editor(srv.endpoint, served, system_prompt=SYSTEM,
                                                                seed=rc.seed + th, mock_policy=mock_policy(th),
                                                                transcript=gd / "editor" / f"thread_{th}.jsonl"),
                        screen_fn=gate_candidate, threads=S["threads"], steps=S["steps"],
                        max_candidates=S["max_candidates"])
                    st.mark(f"gen{g:02d}/search", {"n": len(ranked), "best": ranked[0].cid})

            with run(f"gen{g:02d}/accept") as todo:      # the largest positive paired gate gain
                if todo:
                    ranked = search.ranked(search.load_candidates(rc, g).values())
                    acc = next((c for c in ranked if c.artifact and c.score is not None and c.score > 0), None)
                    records.write_json(gd / "accepted.json", acc.to_record() if acc else
                                       {"accepted": None, "reason": "no candidate with a positive gate gain"})
            accepted = records.read_json(gd / "accepted.json")
            nxt = accepted.get("artifact") if accepted.get("cid") else None
        next_q = nxt["quality"] if nxt else cur_q
        info = {"accepted": accepted.get("cid"), "quality": cur_q, "next_quality": next_q}
        records.update_summary(rc, g, info)
        st.mark(f"gen{g:02d}/done", info)
        print(f"[toy] gen {g}: accepted={info['accepted']} quality {cur_q} -> {next_q}", flush=True)
        cur_q = next_q
        work += 1
    print("[toy] ALL STEPS SKIPPED" if work == 0 else f"[toy] done ({work} steps) -> {rc.run_dir}", flush=True)


if __name__ == "__main__":
    main()
