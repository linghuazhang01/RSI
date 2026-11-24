"""8 GPUs, 4 proposals per round (2026-11-24): a candidate's gate runs on ceil(slots / proposals per step) slots.

    python3 -I -m pytest -q edit_code/tests/test_gate_split.py

- QuantProblem.gate_slots(): 8 slots / 4 proposals -> 2, / 1 -> 8, / 8 -> 1, unset or gpu_layout.gate_split false -> 1;
- a candidate gate with gate_slots() > 1 is served by serve_all(most=k, keep=True) and split over the replicas;
- Runner.run_search sets gate_parallel to the proposals of one step;
- the knob is not part of the run lock; plan/configs/run/A_math_r30_4x1.yaml is a 4 x 1 search.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_simple_v2 as V  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _prob(tmp_path, n_slots=8, cfg=None):
    prob = V._problem(tmp_path, budget=0, n_gpus=n_slots + 1, cfg=cfg)
    assert prob.n_slots == n_slots
    return prob


@pytest.mark.parametrize("parallel,want", [(None, 1), (1, 8), (2, 4), (3, 3), (4, 2), (8, 1), (16, 1)])
def test_gate_slots(tmp_path, parallel, want):
    prob = _prob(tmp_path)
    prob.gate_parallel = parallel
    assert prob.gate_slots() == want


def test_gate_split_off_and_seven_slots(tmp_path):
    prob = _prob(tmp_path, cfg={"gpu_layout": {"candidate_group_size": 1, "gate_split": False}})
    prob.gate_parallel = 4
    assert prob.gate_slots() == 1
    prob = _prob(tmp_path / "b", n_slots=7)     # editor awake: 7 slots, 4 proposals -> 2, 2, 2 and 1 (greedy)
    prob.gate_parallel = 4
    assert prob.gate_slots() == 2


def test_serve_all_most_caps_slots(tmp_path, monkeypatch):
    import rsicore.servers as SV
    prob = _prob(tmp_path)

    class FakeServer:
        def __init__(self, path, name, gpus, **k):
            self.gpus, self.endpoint = list(gpus), f"http://fake/{gpus[0]}/v1"

        def stop(self):
            pass
    monkeypatch.setattr(SV, "VLLMServer", FakeServer)
    monkeypatch.setattr(SV, "start_all", lambda srvs: srvs)
    with prob.serve_all({"config_hash": "a", "mock_quality": 0.5}, "a", most=2, keep=True) as a, \
            prob.serve_all({"config_hash": "b", "mock_quality": 0.5}, "b", most=2, keep=True) as b:
        assert len(a) == len(b) == 2 and prob._pool.qsize() == 4
        assert not {tuple(s.gpus) for s in a} & {tuple(s.gpus) for s in b}
    assert prob._pool.qsize() == 8


def test_candidate_gate_uses_shared_slots(tmp_path, monkeypatch):
    prob = _prob(tmp_path)
    used = []

    class Stop(Exception):
        pass

    @contextmanager
    def fake_all(art, name, most=None, keep=False):
        used.append(("all", most, keep))
        raise Stop
        yield  # pragma: no cover

    @contextmanager
    def fake_one(art, name):
        used.append(("one",))
        raise Stop
        yield  # pragma: no cover
    monkeypatch.setattr(prob, "serve_all", fake_all)
    monkeypatch.setattr(prob, "serve", fake_one)
    for parallel in (None, 8, 4, 1):
        prob.gate_parallel = parallel
        with pytest.raises(Stop):
            prob._evaluate({"config_hash": "h"}, "gate", "c", ("h", "gate"), "gate")
    with pytest.raises(Stop):                    # the incumbent's own gate keeps one slot
        prob._evaluate({"config_hash": "h"}, "gate", "c", ("h", "gate"), "incumbent")
    assert used == [("one",), ("one",), ("all", 2, True), ("all", 8, True), ("one",)]


def test_mock_gate_split_records_match(tmp_path):
    """Mock: the same candidate gated on 1 slot and split over 2 slots gives the same per-item records."""
    import json
    import rsicore.draws as DR
    ids = DR.gate_ids(None, "simple_math_eval")[:12]
    res = {}
    for parallel in (None, 4):
        prob = _prob(tmp_path / str(parallel))
        prob.gate_datasets = ["simple_math_eval"]
        prob.gate_parallel = parallel
        prob._ids = lambda ds, stage: ids
        prob._gate_decision = lambda a, p: (True, "ok")
        art = {"config_hash": "hx", "mock_quality": 0.6, "bytes_gib": 1.0, "budget_gib": 10.0, "linear_avg_bits": 4.0}
        r = prob._evaluate(art, "gate", "c", ("hx", "gate"), "gate")
        (p,) = r["feedback"]["probes"].values()
        assert p["n"] == len(ids)
        recs = [json.loads(x) for x in Path(p["records"]).read_text().splitlines()]
        res[parallel] = sorted(json.dumps({k: v for k, v in x.items() if k not in ("latency_s", "seconds", "t")},
                                          sort_keys=True) for x in recs)
        assert prob._pool.qsize() == 8
    assert len(res[None]) == len(ids) and res[None] == res[4]
    assert (tmp_path / "4" / "p" / "evals" / "c" / "gate" / "_part1").exists()     # really split over 2 replicas


def test_run_search_sets_gate_parallel(monkeypatch):
    from edit_code import run as RUN
    seen = {}
    monkeypatch.setattr(RUN.S, "inner_search", lambda rc, gen, **k: seen.setdefault("k", k) and [])
    for (th, mc), want in (((4, 4), 4), ((8, 8), 8), ((1, 8), 1), ((8, 3), 3)):
        prob = SimpleNamespace(n_slots=8, gate_parallel=None)
        runner = SimpleNamespace(cfg={"search": {"threads": th, "steps": 8 // th or 1, "max_candidates": mc}},
                                 rc=None, gpu_editor=[0], editor_sleep=True, log=print,
                                 search_fns=lambda *a, **k: {})
        RUN.Runner.run_search(runner, prob, 1, Path("."), None, archive="")
        assert prob.gate_parallel == want


def test_lock_ignores_gate_split_and_4x1_config():
    import yaml
    from rsicore.config import lock_view
    a = {"gpu_layout": {"editor_sleep": True, "gate_split": True}}
    b = {"gpu_layout": {"editor_sleep": True}}
    assert lock_view(a, "A") == lock_view(b, "A")
    cfg = yaml.safe_load((REPO / "edit_code" / "configs" / "base.yaml").read_text())
    assert cfg["gpu_layout"]["gate_split"] is True
    run = yaml.safe_load((REPO / "plan" / "configs" / "run" / "A_math_r30_4x1.yaml").read_text())
    assert (run["search"]["threads"], run["search"]["steps"], run["search"]["max_candidates"]) == (4, 1, 4)
