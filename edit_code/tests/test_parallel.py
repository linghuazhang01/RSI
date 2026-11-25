"""Parallel candidate evaluation (2026-11-19): editor on one GPU, one candidate server per candidate-GPU group.

    <venv>/bin/python -I -m pytest -q edit_code/tests/test_parallel.py

- the Paper A GPU layout (editor 1 GPU, candidate groups);
- rsicore.search.inner_search(parallel=N) really runs a step's threads at the same time, records them in
  thread order, and parallel=1 keeps the old sequential behaviour;
- QuantProblem.serve() hands out distinct slots (evaluate's thread safety: test_kl_flow).
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "eval")]


def test_layout_editor_one_gpu_and_groups():
    from edit_code.common import candidate_groups, gpu_layout

    def lay(n, **cfg):
        rc = SimpleNamespace(gpus=[10 + i for i in range(n)])
        return gpu_layout(rc, cfg), candidate_groups(rc, cfg)
    assert lay(2) == (({"editor": [10], "candidate": [11]}, []), [[11]])
    assert lay(4)[1] == [[11], [12], [13]]
    assert lay(8)[1] == [[11], [12], [13], [14], [15], [16], [17]]
    (roles, unused), groups = lay(4, gpu_layout={"candidate_group_size": 2})
    assert groups == [[11, 12]] and unused == [13]
    assert lay(5, gpu_layout={"candidate_group_size": 2})[1] == [[11, 12], [13, 14]]
    (roles, _), groups = lay(5, gpu_roles={"editor": [0, 1]})
    assert roles["editor"] == [10, 11] and groups == [[12], [13], [14]]
    with pytest.raises(ValueError):
        lay(1, gpu_layout={"share_single_gpu": False})
    assert lay(1)[1] == [[10]]                                     # B300 / mock: shared single GPU
    with pytest.raises(ValueError):
        lay(4, gpu_layout={"candidate_group_size": 3})


class _Editor:
    def __init__(self, th):
        self.th = th

    def propose(self, msg):
        return f"# thread {self.th}\nprogram"


def _search(tmp, parallel, delay=0.2):
    from rsicore.search import inner_search
    live, peak, lock = [0], [0], threading.Lock()

    def screen(art, cid):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(delay)
        with lock:
            live[0] -= 1
        return float(art["v"]), {"cid": cid}

    seen = {}

    def render(history, th):
        seen.setdefault(th, []).append(sorted(c.cid for c in history))
        return f"thread {th}"

    rc = SimpleNamespace(run_dir=tmp, run_id="t")
    t0 = time.time()
    out = inner_search(rc, 1, editor_factory=_Editor, render_context=render,
                       build_fn=lambda prog, cid: ({"v": int(cid[5]) / 10}, None), screen_fn=screen,
                       threads=4, steps=2, max_candidates=8, out_dir=tmp, parallel=parallel)
    return out, peak[0], time.time() - t0, seen


def test_inner_search_parallel(tmp_path):
    out, peak, secs, seen = _search(tmp_path / "par", parallel=4)
    assert len(out) == 8 and peak == 4                             # 4 screens at the same time
    assert secs < 8 * 0.2 + 0.6                                    # well under the sequential time
    assert seen[3][1] == seen[0][1]                                # step 1: every thread sees the same history
    lines = (tmp_path / "par" / "candidates.jsonl").read_text().splitlines()
    assert [l.split('"cid": "')[1][:10] for l in lines] == [f"g01_t{t}_s{s:02d}" for s in range(2) for t in range(4)]
    out1, peak1, _, seen1 = _search(tmp_path / "seq", parallel=1, delay=0.01)
    assert peak1 == 1 and len(out1) == 8
    assert seen1[1][0] == ["g01_t0_s00"]                           # sequential: thread 1 sees thread 0
    assert [c.cid for c in out] == [c.cid for c in out1]           # same ranking either way


def _problem(tmp, n_gpus=4):
    from edit_code.problem import QuantProblem
    rc = SimpleNamespace(mock=True, run_dir=tmp, run_id="t", seed=42, gpus=list(range(n_gpus)))
    cfg = {"gpu_layout": {"candidate_group_size": 1}, "candidates": {"keep_checkpoints": 1}}
    return QuantProblem(rc, cfg, tag="t", gen=1, out_dir=tmp / "p", prep_dir=tmp / "prep", source_dir=None,
                        source_state=None, budget_gib=10.0, datasets=["simple_math_pool"], kl_texts=[])


def test_serve_pool_distinct_slots(tmp_path, monkeypatch):
    import edit_code.problem as P
    prob = _problem(tmp_path)
    used, live, peak, lock = [], [0], [0], threading.Lock()

    class FakeServer:
        def __init__(self, path, name, gpus, **kw):
            self.gpus = gpus

        def __enter__(self):
            with lock:
                used.append(tuple(self.gpus))
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            return self

        def __exit__(self, *a):
            with lock:
                live[0] -= 1

    import rsicore.servers
    monkeypatch.setattr(rsicore.servers, "VLLMServer", FakeServer)
    monkeypatch.setattr(P.M, "register", lambda *a, **k: None)
    holding = []

    def go(i):
        with prob.serve({"mock_quality": 0.5, "config_hash": str(i)}, f"n{i}") as srv:
            with lock:
                holding.append(tuple(srv.gpus))
                assert len(set(holding)) == len(holding)           # no slot handed out twice at once
            time.sleep(0.15)
            with lock:
                holding.remove(tuple(srv.gpus))
    ts = [threading.Thread(target=go, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 3 and set(used) == {(1,), (2,), (3,)} and len(used) == 6


def test_memory_budget_in_gib(monkeypatch):
    """h20 plans 110 GB per card: --rsi-gpu-memory-gib 110 becomes --gpu-memory-utilization 110 / total."""
    import rsicore.servers as S
    args = ["--rsi-gpu-memory-gib", "110", "--gpu-memory-utilization", "0.78", "--max-num-seqs", "128"]
    out, info = S.apply_memory_budget(args, [0], total_gib=140.4)
    assert S.MEM_FLAG not in out and out.count("--gpu-memory-utilization") == 1
    assert out[out.index("--gpu-memory-utilization") + 1] == "0.783" and info["utilization"] == 0.783
    out, info = S.apply_memory_budget(args, [0], total_gib=96.0)       # smaller card: 110/96 capped at 0.95
    assert out[out.index("--gpu-memory-utilization") + 1] == "0.95"
    monkeypatch.setattr(S, "gpu_total_gib", lambda ids: None)           # no nvidia-smi: configured 0.78 kept
    out, info = S.apply_memory_budget(args, [0])
    assert out[out.index("--gpu-memory-utilization") + 1] == "0.78" and info["total_gib"] is None
    monkeypatch.setattr(S, "gpu_total_gib", lambda ids: 140.0)
    srv = S.VLLMServer("/m", "x", [3, 4], extra_args=args)
    cmd = srv.command()
    assert S.MEM_FLAG not in cmd and cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.786"
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    mock = S.VLLMServer("/m", "x", [0], extra_args=args, mock=True)
    assert S.MEM_FLAG not in mock.command()


def test_h20_profiles(tmp_path):
    from rsicore.config import build_arg_parser, load_run_config

    def load(paper, hw):
        a = build_arg_parser(paper).parse_args(["--domain", "math", "--mock", "--runs-dir", str(tmp_path / hw / paper),
                                                "--set", f"hardware={hw}"])
        return load_run_config(paper, a).paper_cfg
    a, b = load("A", "h20"), load("B", "h20")
    for args in (a["vllm"]["editor_args"], a["vllm"]["candidate_args"], b["server"]["main_args"]):
        assert args[args.index("--rsi-gpu-memory-gib") + 1] == "110" and "128" in args
    assert a["calibration"]["layers_per_pass"] == 16 and b["train"]["micro_batch"] == 2 and b["server"]["max_loras"] == 8
    a80, b80 = load("A", "h20_80"), load("B", "h20_80")
    assert a80["calibration"]["layers_per_pass"] == 4 and b80["train"]["micro_batch"] == 1
    assert "--rsi-gpu-memory-gib" not in a80["vllm"]["editor_args"]
