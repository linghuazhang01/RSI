"""Self-test of eval/rsicore in mock mode (no GPU, no network except localhost).

  eval/.venv/bin/python -I eval/tests/rsicore/test_core.py [-k substring] [--skip-toy]

Covers: RunState resume, ToolAPI budgets (fixed, measured cost_fn, dedupe), sandbox (tool round trip, timeout kill, budget exceeded,
error propagation, environment isolation, file/subprocess/network denial), Editor with mock_policy
(transcript + replay), inner_search with toy functions (resume, out_dir), the per-generation draws
of simple_<d>_pool in a temp run dir, evaluate.score / paired_delta / run_heldout (simple_<d>_test) and the mock
server's prompt_logprobs (Paper B's train_loss) against the mock server, per-domain mock quality (target up, other
domain down), sampled mock answers depending on the seed, gate helpers, and the end-to-end toy paper (twice: the
second run must skip everything).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from rsicore import config, draws, editor, evaluate, gate, heldout, records, sandbox, search, servers, state  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="rsicore_test_"))
RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ----------------------------------------------------------------------------- helpers
def make_rc(name, paper="B", domain="math", **kw):
    pdir = TMP / f"paper_{name}"
    (pdir / "configs" / "domains").mkdir(parents=True, exist_ok=True)
    (pdir / "configs" / "base.yaml").write_text("search: {threads: 2, steps: 2}\nlimit: 50\n")
    (pdir / "configs" / "domains" / f"{domain}.yaml").write_text("search: {steps: 3}\n")
    ap = config.build_arg_parser(paper)
    argv = ["--domain", domain, "--mock", "--run-id", name, "--runs-dir", str(TMP / "runs"),
            "--paper-dir", str(pdir), "--gens", "2"] + kw.pop("argv", [])
    return config.load_run_config(paper, ap.parse_args(argv))


# ----------------------------------------------------------------------------- tests
def test_config_and_state():
    rc = make_rc("cfg")
    check(rc.paper_cfg["search"] == {"threads": 2, "steps": 3}, f"merge failed: {rc.paper_cfg}")
    check((rc.run_dir / "run_config.yaml").exists(), "run_config.yaml missing")
    rc2 = make_rc("cfg", argv=["--set", "search.threads=2"])            # same effective config: ok
    check(rc2.paper_cfg == rc.paper_cfg, "--set changed config unexpectedly")
    try:
        make_rc("cfg", argv=["--seed", "7"])
        raise AssertionError("lock did not reject a different seed")
    except SystemExit as e:
        check("locked" in str(e), str(e))
    st = state.RunState(rc.run_dir)
    with st.step("a") as todo:
        check(todo, "fresh step should run")
    try:
        with st.step("b") as todo:
            raise RuntimeError("crash")
    except RuntimeError:
        pass
    st.mark("c", {"x": 1})
    st2 = state.RunState(rc.run_dir)                                     # resume from disk
    check(st2.done("a") and not st2.done("b") and st2.get("c")["x"] == 1, f"state not resumed: {st2.keys()}")
    with st2.step("a") as todo:
        check(not todo, "done step should be skipped")
    records.update_summary(rc, 0, {"acc": 0.5})
    records.update_summary(rc, 1, {"acc": 0.6})
    s = json.loads((rc.run_dir / "summary.json").read_text())
    check(set(s["gens"]) == {"0", "1"}, f"summary: {s}")
    return "config merge + lock, RunState resume, summary ok"


def test_toolapi_budget():
    api = sandbox.ToolAPI({"evaluate": 2, "gpu_seconds": 10})
    api.register("evaluate", lambda config: {"ok": config}, cost={"evaluate": 1, "gpu_seconds": 4}, doc="eval it")
    api.call("evaluate", {"config": 1})
    api.call("evaluate", {"config": 2})
    try:
        api.call("evaluate", {"config": 3})
        raise AssertionError("budget not enforced")
    except sandbox.BudgetExceeded:
        pass
    check(api.spent == {"evaluate": 2, "gpu_seconds": 8}, f"spent {api.spent}")
    check("evaluate(config)" in api.describe() and "Budget remaining" in api.describe(), api.describe())
    return f"spent={api.spent}"


def _api():
    api = sandbox.ToolAPI({"evaluate": 2})
    api.register("evaluate", lambda bits: {"quality": 1 - 1 / bits}, cost={"evaluate": 1})
    api.register("fail", lambda: 1 / 0)
    api.register("bad_result", lambda: object())
    return api


def test_sandbox():
    os.environ["RSI_TEST_SECRET"] = "s3cr3t"
    code = f'''
import os, json
def main(api):
    out = {{"q": api.evaluate(bits=4), "env": dict(os.environ)}}
    try:
        api.fail()
    except api.ToolError as e:
        out["fail"] = str(e)
    try:
        api.bad_result()
    except api.ToolError as e:
        out["bad"] = str(e)
    api.evaluate(bits=8)
    try:
        api.evaluate(bits=16)
    except BudgetExceeded as e:
        out["budget"] = str(e)
    for name, fn in [("read_repo", lambda: open({str(ROOT / "data" / "README.md")!r}).read()),
                     ("write_out", lambda: open("/tmp/rsicore_should_not_exist", "w")),
                     ("write_in", lambda: open("ok.txt", "w").write("x")),
                     ("subprocess", lambda: __import__("subprocess").run(["true"])),
                     ("socket", lambda: __import__("socket").create_connection(("127.0.0.1", 9), timeout=1))]:
        try:
            fn(); out[name] = "allowed"
        except Exception as e:
            out[name] = type(e).__name__
    print("child says hi")
    return out
'''
    api = _api()
    r = sandbox.run_program(code, api, timeout_s=60, mem_gb=4, workdir=TMP / "sbx1")
    check(r.ok, f"program failed: {r.error}\n{r.stderr}")
    res = r.result
    check(res["q"] == {"quality": 0.75}, f"round trip: {res['q']}")
    check("RSI_TEST_SECRET" not in res["env"] and "s3cr3t" not in json.dumps(res["env"]), "parent env leaked")
    check("ZeroDivisionError" in res["fail"], f"error propagation: {res['fail']}")
    check("JSON" in res["bad"], f"non-JSON tool result: {res['bad']}")
    check("exhausted" in res["budget"] and api.spent["evaluate"] == 2, f"budget: {res.get('budget')} {api.spent}")
    check(res["read_repo"] == "PermissionError" and res["write_out"] == "PermissionError", f"fs policy: {res}")
    check(res["write_in"] == "allowed" and res["subprocess"] == "PermissionError", f"policy: {res}")
    check(res["socket"] in ("PermissionError", "OSError"), f"socket: {res['socket']}")
    check("child says hi" in r.stdout, "stdout not captured")
    check(not Path("/tmp/rsicore_should_not_exist").exists(), "wrote outside workdir")
    t0 = time.time()
    r2 = sandbox.run_program("def main(api):\n    while True:\n        pass\n", _api(), timeout_s=3, mem_gb=4,
                             workdir=TMP / "sbx2")
    check(not r2.ok and "timeout" in r2.error and time.time() - t0 < 15, f"timeout: {r2.error}")
    r3 = sandbox.run_program("def main(api):\n    raise ValueError('nope')\n", _api(), timeout_s=30, mem_gb=4,
                             workdir=TMP / "sbx3")
    check(not r3.ok and "ValueError: nope" in r3.error, f"exception: {r3.error}")
    r4 = sandbox.run_program("def main(api):\n    return {1, 2}\n", _api(), timeout_s=30, mem_gb=4, workdir=TMP / "sbx4")
    check(not r4.ok and "JSON-serializable" in r4.error, f"non-JSON return: {r4.error}")
    r5 = sandbox.run_program("x = 1\n", _api(), timeout_s=30, mem_gb=4, workdir=TMP / "sbx5")
    check(not r5.ok and "main" in r5.error, f"missing main: {r5.error}")
    r6 = sandbox.run_program("def main(api):\n    return api.nope()\n", _api(), timeout_s=30, mem_gb=4,
                             workdir=TMP / "sbx6")
    check(not r6.ok and "AttributeError" in r6.error, f"unknown tool: {r6.error}")
    return f"network={r.info['network_isolation']}; limits={r.info['limits']}"


def _policy(history):
    n = sum(1 for m in history if m["role"] == "assistant")
    return f"def main(api):\n    return {{'turn': {n}}}\n"


def test_editor_mock():
    tr = TMP / "editor" / "thread_0.jsonl"
    ed = editor.Editor("http://unused", "m", system_prompt="SYS", seed=1, transcript=tr, mock_policy=_policy)
    c0 = ed.propose("first")
    c1 = ed.propose("second")
    check("'turn': 0" in c0 and "'turn': 1" in c1, f"codes: {c0!r} {c1!r}")
    check(len(records.read_jsonl(tr)) == 5, "transcript lines")
    ed2 = editor.Editor("http://unused", "m", system_prompt="SYS", seed=1, transcript=tr, mock_policy=_policy)
    check(ed2.propose("first") == c0 and ed2.propose("second") == c1 and ed2.n_calls == 0, "replay failed")
    c2 = ed2.propose("third")
    check(ed2.n_calls == 1 and "'turn': 2" in c2, "continue after replay")
    ed3 = editor.Editor("http://unused", "m", system_prompt="SYS", seed=1, transcript=tr, mock_policy=_policy)
    ed3.propose("first")
    ed3.propose("DIFFERENT")
    check(ed3.n_calls == 1 and len(records.read_jsonl(tr)) == 5, "divergence did not truncate")
    check(editor.extract_python("<think>```python\nx=1\n```</think>text ```python\ny=2\n``` end") == "y=2\n", "extract")
    check(editor.extract_python("no code") == "", "extract empty")
    return "propose/transcript/replay/truncate ok"


def test_inner_search_resume():
    rc = make_rc("search")
    calls = {"build": 0, "screen": 0}
    crash = {"at": "g00_t1_s01"}

    def factory(th):
        return editor.Editor("http://unused", "m", system_prompt="S", seed=th,
                             transcript=records.gen_dir(rc, 0) / "editor" / f"thread_{th}.jsonl",
                             mock_policy=lambda h, th=th: f"def main(api):\n    return {th * 10 + len(h)}\n")

    def render(history, th):
        return f"thread {th}; seen " + ",".join(f"{c.cid}:{c.score}" for c in history)

    def build(program, cid):
        calls["build"] += 1
        if cid.endswith("t0_s01"):
            return None, "deliberate build error"
        return {"name": cid, "q": len(program) % 7 / 7}, None

    def screen(art, cid):
        calls["screen"] += 1
        if cid == crash["at"]:
            raise RuntimeError("simulated crash")
        return art["q"], {"note": "screen " + cid}

    kw = dict(editor_factory=factory, render_context=render, build_fn=build, screen_fn=screen,
              threads=2, steps=3, max_candidates=5)
    try:
        search.inner_search(rc, 0, **kw)
        raise AssertionError("crash not propagated")
    except RuntimeError:
        pass
    n_before = len(records.read_jsonl(records.gen_dir(rc, 0) / "candidates.jsonl"))
    check(n_before == 3, f"expected 3 recorded before crash, got {n_before}")
    crash["at"] = None
    calls.update(build=0, screen=0)
    out = search.inner_search(rc, 0, **kw)
    check(len(out) == 5 and calls["build"] == 2 and calls["screen"] == 2, f"resume rebuilt: {calls}, n={len(out)}")
    gd = records.gen_dir(rc, 0)
    check(all((gd / "programs" / f"{c.cid}.py").exists() and (gd / "feedback" / f"{c.cid}.json").exists() for c in out),
          "programs/feedback files")
    check(all(c.full is None and c.score == (c.screen or {}).get("score") for c in out), "score = screen score")
    check([c.cid for c in out] == [c.cid for c in search.ranked(out)] and out[0].score == max(
        c.score for c in out if c.score is not None), "ranking by score")
    rec = records.read_jsonl(gd / "candidates.jsonl")[0]
    check(set(rec) == {"cid", "thread", "step", "program", "artifact", "error", "screen", "full", "score", "cost"},
          f"record keys {sorted(rec)}")
    import inspect
    check(not {"full_fn", "top_full", "rerun_seeds"} & set(inspect.signature(search.inner_search).parameters)
          and not hasattr(search, "submit_to_gate"), "removed search API")
    err = [c for c in out if c.error]
    check(len(err) == 1 and err[0].screen is None, "error candidate")
    calls.update(build=0, screen=0)
    out2 = search.inner_search(rc, 0, **kw)
    check(sum(calls.values()) == 0 and [c.cid for c in out2] == [c.cid for c in out], f"third run did work: {calls}")
    calls.update(build=0, screen=0)
    rc2 = make_rc("search_default_max")
    out3 = search.inner_search(rc2, 0, **dict(kw, max_candidates=None, steps=2))     # None = threads x steps
    check(len(out3) == 4, f"max_candidates None: {len(out3)} candidates")
    return f"5 candidates, resume skipped {n_before}, top={out[0].cid} score={out[0].score:.3f}"


def test_draws():
    rc = make_rc("draws", argv=["--set", "simple.screen_n=40"])
    g = draws.ensure_gate(rc)
    check(list(g) == ["math"] and len(g["math"]["simple_math_eval"]) == 500, "gate size")
    m = draws.draw_gen(rc, 0, target="math")
    check(m["target"] == "math" and m["simple"] and m["dataset"] == "simple_math_pool", "manifest")
    tr, sig, scr, kl = (draws.ids(rc, 0, "simple_math_pool", r) for r in ("train", "signal", "screen", "kl"))
    check(len(scr) == 40 and sig == scr and len(kl) == draws.KL_N and set(kl) <= set(tr), "role sizes")
    test = set(draws.build_eval_set("simple_math_test"))
    check(not (set(tr) | set(scr)) & (set(draws.gate_ids(rc, "simple_math_eval")) | test), "train / screen overlap gate")
    check(draws.draw_gen(rc, 0, target="math") == m, "redraw changed manifest")
    m1 = draws.draw_gen(rc, 1, target="math")
    check(m1["gen"] == 1 and draws.ids(rc, 1, "simple_math_pool", "screen") != scr, "gen 1 draws anew")
    for bad in (lambda: draws.ids(rc, 0, "dapo_math", "screen"), lambda: draws.gate_ids(rc, "dapo_math")):
        try:
            bad()
            raise AssertionError("a non-simple dataset was accepted")
        except ValueError:
            pass
    return f"gate 500, gen0 train/screen/kl = {len(tr)}/{len(scr)}/{len(kl)}"


def test_evaluate_mock():
    rc = make_rc("evalmock", argv=["--set", "simple.screen_n=400"])
    draws.draw_gen(rc, 0, target="math")
    reg = rc.run_dir / "mock_registry.json"
    records.write_json(reg, {"base": 0.8, "weak": 0.3})
    with servers.VLLMServer(rc.model_path, "base", [0], mock=True, mock_registry=reg, enable_lora=True,
                            seed=rc.seed, log_dir=rc.run_dir / "logs") as srv:
        ids = draws.ids(rc, 0, "simple_math_pool", "screen")
        a = evaluate.score(srv.endpoint, "base", "simple_math_pool", pool="pool", ids=ids,
                           out_dir=rc.run_dir / "ev" / "base", mock=True)
        check(a.summary["n"] == 400 and 0.7 <= a.accuracy <= 0.9, f"base accuracy {a.accuracy}")
        (rc.run_dir / "weak_lora").mkdir(exist_ok=True)
        srv.load_lora("weak", rc.run_dir / "weak_lora")
        b = evaluate.score(srv.endpoint, "base", "simple_math_pool", pool="pool", ids=ids,
                           out_dir=rc.run_dir / "ev" / "weak", lora="weak", mock=True)
        check(0.2 <= b.accuracy <= 0.4, f"weak accuracy {b.accuracy}")
        d = evaluate.paired_delta(a.records, b.records)
        check(d["n"] == 400 and d["ci"][0] <= d["delta"] <= d["ci"][1] < 0, f"paired delta {d}")
        # prompt_logprobs on /v1/completions (Paper B's train_loss reads them; rsicore has no token KL since 2026-11-23)
        from rsicore import _net
        plp = {}
        for name in ("base", "weak"):
            r = _net.post_json(srv.endpoint.rstrip("/") + "/completions", {"model": name, "prompt": "Compute 2+2 now.",
                               "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1}, timeout=60)
            plp[name] = r["choices"][0]["prompt_logprobs"]
        check(plp["base"][0] is None and len(plp["base"]) > 3 and all(isinstance(x, dict) and x for x in plp["base"][1:])
              and plp["base"] != plp["weak"], "mock prompt_logprobs")
        dom = heldout.heldout_domains(rc.paper_cfg, rc.domain)
        check(dom == "eval:simple_math_test", dom)
        h = heldout.run_heldout(srv.endpoint, "base", rc.run_dir / "heldout", domains=dom, mock=True, limit=5)
        check(set(h) == {"simple_math_test"} and h["simple_math_test"]["n"] == 5, f"heldout {h}")
        t0 = time.time()
        heldout.run_heldout(srv.endpoint, "base", rc.run_dir / "heldout", domains=dom, mock=True, limit=5)
        check(time.time() - t0 < 2, "heldout rerun not skipped")
        hc = heldout.run_heldout(srv.endpoint, "base", rc.run_dir / "heldout_code", domains="eval:simple_code_test",
                                 mock=True, limit=3)
        check(hc["simple_code_test"]["n_pending"] == 3, f"code pending {hc}")
    check(gate.relative_pass(0.8, 1.0, 0.8) and not gate.relative_pass(0.79, 1.0, 0.8), "relative_pass")
    return (f"simple_math_pool screen acc base={a.accuracy:.3f} weak={b.accuracy:.3f} delta={d['delta']:.3f} "
            f"ci=[{d['ci'][0]:.3f},{d['ci'][1]:.3f}]; heldout acc={ {k: v['accuracy'] for k, v in h.items()} }")


def test_toolapi_cost_fn_dedupe():
    api = sandbox.ToolAPI({"evaluate": 3, "gpu_seconds": 10})
    runs = {"evaluate": 0, "train": 0}

    def ev(lora):
        runs["evaluate"] += 1
        return {"lora": lora, "n": runs["evaluate"]}

    def train(seconds):
        runs["train"] += 1
        return {"seconds": seconds}

    api.register("evaluate", ev, cost={"evaluate": 1}, dedupe_key_fn=lambda kw: kw.get("lora"))
    api.register("train", train, cost={"gpu_seconds": 0}, cost_fn=lambda kw, res: {"gpu_seconds": res["seconds"]})
    a = api.call("evaluate", {"lora": "L1"})
    b = api.call("evaluate", {"lora": "L1"})                    # identical: cached, free
    check(a == b and runs["evaluate"] == 1 and api.spent["evaluate"] == 1, f"dedupe {api.spent} {runs}")
    api.call("evaluate", {"lora": "L2"})
    api.call("evaluate", {"lora": "L3"})
    check(api.call("evaluate", {"lora": "L2"})["n"] == 2, "dedupe after budget exhausted")   # still free
    try:
        api.call("evaluate", {"lora": "L4"})
        raise AssertionError("fixed budget not enforced")
    except sandbox.BudgetExceeded:
        pass
    api.call("train", {"seconds": 4})
    check(api.spent["gpu_seconds"] == 4, f"measured cost {api.spent}")
    try:
        api.call("train", {"seconds": 7})                         # runs, then exceeds -> raises after
        raise AssertionError("measured budget not enforced")
    except sandbox.BudgetExceeded:
        pass
    check(runs["train"] == 2 and api.spent["gpu_seconds"] == 11, f"measured overrun {api.spent}")
    try:
        api.call("train", {"seconds": 0})                         # exhausted: refused before running
        raise AssertionError("exhausted measured budget not refused")
    except sandbox.BudgetExceeded:
        pass
    check(runs["train"] == 2, "tool ran although its budget was exhausted")
    d = api.describe()
    check("measured: gpu_seconds" in d and "repeated identical calls are free" in d, d)
    api.charge("evaluate", 0.5)                                   # old interface still there
    check(api.spent["evaluate"] == 3.5 and api.calls["evaluate"] == 5, f"{api.spent} {api.calls}")
    return f"spent={api.spent} calls={api.calls}"


def test_search_out_dir():
    rc = make_rc("search_out")
    out = rc.run_dir / "gen_01" / "extra" / "llm"
    calls = {"build": 0}

    def factory(th):
        return editor.Editor("http://unused", "m", system_prompt="S", seed=th,
                             transcript=out / "editor" / f"thread_{th}.jsonl",
                             mock_policy=lambda h, th=th: f"def main(api):\n    return {th}\n")

    def build(program, cid):
        calls["build"] += 1
        return {"q": 0.1 * (1 + int(cid[-1]))}, None

    kw = dict(editor_factory=factory, render_context=lambda h, th: f"t{th} n{len(h)}", build_fn=build,
              screen_fn=lambda a, cid: (a["q"], {}), threads=1, steps=2, max_candidates=2)
    r1 = search.inner_search(rc, 1, out_dir=out, **kw)
    check((out / "candidates.jsonl").exists() and (out / "programs").is_dir(), "records not in out_dir")
    check(not (rc.run_dir / "gen_01" / "candidates.jsonl").exists(), "main gen dir written")
    check(not any("gen_" in p.name for p in out.rglob("*")), f"nested gen dir: {list(out.rglob('gen_*'))}")
    r2 = search.inner_search(rc, 1, out_dir=out, **kw)
    check(calls["build"] == 2 and [c.cid for c in r2] == [c.cid for c in r1], "out_dir resume")
    check(set(search.load_candidates(rc, 1, out)) == {c.cid for c in r1}, "load_candidates(out_dir)")
    check(search.load_candidates(rc, 1) == {}, "default dir should be empty")
    return f"records under {out.relative_to(rc.run_dir)}: {sorted(p.name for p in out.iterdir())}"


def test_mock_domain_quality_and_seeds():
    from rsicore import mock_quality as MQ
    check(MQ.resolve({"default": 0.5, "math": 0.8, "dapo_math": 0.9}, task="dapo_math", domain="math") == 0.9, "task key")
    check(MQ.resolve({"default": 0.5, "math": 0.8}, task="gsm8k", domain="math") == 0.8, "domain key")
    check(MQ.resolve({"quality": {"default": 0.5}}, task="x", domain="code") == 0.5, "wrapped default")
    check(MQ.resolve(0.3, task="x") == 0.3 and abs(MQ.scalar({"default": 0.5, "math": 1.0}) - 0.6) < 1e-12, "scalar")
    rc = make_rc("domainq")
    reg = rc.run_dir / "mock_registry.json"
    records.write_json(reg, {"base": 0.6, "cand": {"default": 0.3, "math": 0.9}})
    tasks = ["simple_math_test", "simple_knowledge_test"]          # indexed through their source training sets
    with servers.VLLMServer(rc.model_path, "base", [0], mock=True, mock_registry=reg, enable_lora=True, seed=rc.seed,
                            log_dir=rc.run_dir / "logs", extra_args=["--tasks", "dapo_math,triviaqa_train"]) as srv:
        (rc.run_dir / "cand_lora").mkdir(exist_ok=True)
        srv.load_lora("cand", rc.run_dir / "cand_lora")
        acc = {}
        for name, lora in (("base", None), ("cand", "cand")):
            for t in tasks:
                ids = draws.build_eval_set(t)[:300]
                acc[(name, t)] = evaluate.score(srv.endpoint, "base", t, pool="pool", ids=ids, lora=lora, mock=True,
                                                out_dir=rc.run_dir / "ev" / name).accuracy
        check(acc[("cand", "simple_math_test")] > acc[("base", "simple_math_test")] + 0.15, f"target did not rise {acc}")
        check(acc[("cand", "simple_knowledge_test")] < acc[("base", "simple_knowledge_test")] - 0.15,
              f"other domain did not fall {acc}")
        # sampled mock answers depend on the request seed (deterministic per seed); greedy ones do not
        ds, ids = "simple_math_test", draws.build_eval_set("simple_math_test")[:300]

        def run(tag, seed, temperature):
            return evaluate.score(srv.endpoint, "base", ds, pool="pool", ids=ids, mock=True, seed=seed,
                                  temperature=temperature, out_dir=rc.run_dir / "seeds" / tag).accuracy
        greedy = [run(f"g{k}", rc.seed + k, 0.0) for k in range(2)]
        sampled = [run(f"t{k}", rc.seed + k, 0.6) for k in range(3)]
        again = run("t1b", rc.seed + 1, 0.6)
    check(greedy[0] == greedy[1], f"greedy repeats differ {greedy}")
    check(len(set(sampled)) > 1 and again == sampled[1], f"sampled repeats {sampled} / {again}")
    s = json.loads((rc.run_dir / "seeds" / "t1" / "summary.json").read_text())[ds]
    check(s["decoding"]["seed"] == rc.seed + 1 and s["decoding"]["temperature"] == 0.6, f"decoding {s['decoding']}")
    return (f"acc {{{', '.join(f'{k[0]}/{k[1]}={v:.2f}' for k, v in acc.items())}}}; greedy={greedy} "
            f"t=0.6 scores={[round(x, 3) for x in sampled]}")


def test_vllm_command():
    srv = servers.VLLMServer("/models/qwen3.8-27b", "qwen-g01", [2, 3], enable_lora=True, max_loras=4,
                             extra_args=["--gpu-memory-utilization", "0.85"], seed=42, port=8123)
    cmd = srv.command()
    s = " ".join(cmd)
    for frag in ["serve /models/qwen3.8-27b", "--served-model-name qwen-g01", "--port 8123", "--seed 42",
                 "--enable-auto-tool-choice", "--tool-call-parser hermes", "--reasoning-parser qwen3",
                 "--tensor-parallel-size 2", "--enable-lora --max-loras 4 --max-lora-rank 64",
                 "--max-model-len 40960", "--gpu-memory-utilization 0.85"]:
        check(frag in s, f"missing {frag!r} in {s}")
    check(srv.endpoint == "http://127.0.0.1:8123/v1", srv.endpoint)
    m = servers.VLLMServer("/x", "base", [0], mock=True, mock_registry=Path("/r.json"), extra_args=["--tasks", "gsm8k"])
    check("mock_server.py" in " ".join(m.command()) and m.command()[-2:] == ["--tasks", "gsm8k"], m.command())
    return s[s.index("serve"):]


def test_toy_paper():
    py = sys.executable
    cmd = [py, "-I", str(Path(__file__).resolve().parent / "toy_paper.py"), "--runs-dir", str(TMP / "toy_runs"),
           "--gens", "2", "--limit", "40"]
    t0 = time.time()
    r1 = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time() - t0
    check(r1.returncode == 0, f"toy run 1 failed:\n{r1.stdout[-3000:]}\n{r1.stderr[-3000:]}")
    t0 = time.time()
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    t2 = time.time() - t0
    check(r2.returncode == 0, f"toy run 2 failed:\n{r2.stdout[-3000:]}\n{r2.stderr[-3000:]}")
    check("ALL STEPS SKIPPED" in r2.stdout, f"second run did work:\n{r2.stdout[-2000:]}")
    s = json.loads((TMP / "toy_runs" / "toy_B_math" / "summary.json").read_text())
    check(set(s["gens"]) == {"0", "1"}, f"summary gens {s['gens'].keys()}")
    last = [l for l in r1.stdout.splitlines() if l.startswith("[toy]")][-3:]
    return f"run1 {t1:.0f}s, run2 {t2:.1f}s (skipped); " + " | ".join(last)


TESTS = [test_config_and_state, test_toolapi_budget, test_toolapi_cost_fn_dedupe, test_sandbox, test_editor_mock,
         test_inner_search_resume, test_search_out_dir, test_draws, test_evaluate_mock, test_mock_domain_quality_and_seeds,
         test_vllm_command, test_toy_paper]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    ap.add_argument("--skip-toy", action="store_true")
    a = ap.parse_args()
    ok = True
    for t in TESTS:
        if a.k not in t.__name__ or (a.skip_toy and t is test_toy_paper):
            continue
        t0 = time.time()
        try:
            msg = t()
            print(f"PASS {t.__name__:28s} {time.time() - t0:6.1f}s  {msg}", flush=True)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"FAIL {t.__name__:28s} {time.time() - t0:6.1f}s  {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"\nRESULT: {'ALL PASSED' if ok else 'FAILED'}  (temp dir {TMP})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
