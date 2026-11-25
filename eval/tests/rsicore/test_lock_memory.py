"""Run lock vs the memory plan (2026-11-25): the h20 profile was re-planned at 100 GiB per card after an OOM in the
generation-0 profile table, and an OOMed / interrupted run must resume with the new plan. Memory and batching knobs
are not part of the lock (like scoring.concurrency / gpu_layout.gate_split): kl_eval.max_num_batched_tokens,
calibration.layers_per_pass, and the memory-only flags of the vLLM arg lists (--rsi-gpu-memory-gib,
--gpu-memory-utilization, --max-num-batched-tokens); every other flag and key stays locked.
    python -I -m pytest -q eval/tests/rsicore/test_lock_memory.py
"""
import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from rsicore import config as C  # noqa: E402
from rsicore.config import build_arg_parser, load_run_config, lock_vllm_args  # noqa: E402

# the commit whose edit_code/configs/base.yaml still planned the h20 profile at 110 GiB per card (no calibration /
# kl_eval block under h20): the locks of the OOMed runs were written from it (HEAD when the plan changed)
OLD_PLAN_COMMIT = "aff3dde"


def _git_show(path: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_PLAN_COMMIT}:{path}"], check=True,
                              capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:       # e.g. a shallow clone on the GPU machine
        pytest.skip(f"git show {OLD_PLAN_COMMIT}:{path} unavailable: {e}")


def _old_paper_dir(tmp_path: Path) -> Path:
    """A paper dir holding the previous code's Paper A configs (base.yaml + domains/math.yaml)."""
    pdir = tmp_path / "old_edit_code"
    (pdir / "configs" / "domains").mkdir(parents=True)
    (pdir / "configs" / "base.yaml").write_text(_git_show("edit_code/configs/base.yaml"))
    (pdir / "configs" / "domains" / "math.yaml").write_text(_git_show("edit_code/configs/domains/math.yaml"))
    return pdir


def _load_a(runs, run_id, *extra):
    ap = build_arg_parser("A")
    ap.add_argument("--budget-gib", type=float, default=None)
    return load_run_config("A", ap.parse_args(["--domain", "math", "--runs-dir", str(runs), "--mock",
                                               "--run-id", run_id, *extra]))


def test_lock_vllm_args_drops_memory_flags_only():
    args = ["--rsi-gpu-memory-gib", "110", "--gpu-memory-utilization", "0.78", "--max-model-len", "131072",
            "--max-num-seqs", "128", "--max-logprobs", "20", "--max-num-batched-tokens", "8192"]
    assert lock_vllm_args(args) == ["--max-model-len", "131072", "--max-num-seqs", "128", "--max-logprobs", "20"]
    eq = ["--rsi-gpu-memory-gib=90", "--max-model-len", "131072", "--gpu-memory-utilization=0.64",
          "--max-num-batched-tokens=2048", "--enable-prefix-caching"]
    assert lock_vllm_args(eq) == ["--max-model-len", "131072", "--enable-prefix-caching"]
    assert lock_vllm_args(["--gpu-memory-utilization", 0.9, "--max-logprobs", 20]) == ["--max-logprobs", 20]
    assert lock_vllm_args(["--max-model-len", "20480", "--rsi-gpu-memory-gib"]) == ["--max-model-len", "20480"]
    assert lock_vllm_args(["--max-num-batched-tokens-x", "1"]) == ["--max-num-batched-tokens-x", "1"]   # not a prefix
    assert lock_vllm_args(None) is None and lock_vllm_args("x") == "x" and lock_vllm_args([]) == []
    a = ["--gpu-memory-utilization", "0.5", "--max-model-len", "8"]
    lock_vllm_args(a)
    assert a == ["--gpu-memory-utilization", "0.5", "--max-model-len", "8"]                  # input untouched


def test_lock_view_normalizes_arg_lists_at_any_depth():
    def cfg(gib, util, mml="131072"):
        args = ["--rsi-gpu-memory-gib", gib, "--gpu-memory-utilization", util, "--max-model-len", mml]
        return {"vllm": {"editor_args": list(args), "candidate_args": list(args)},
                "server": {"main_args": list(args)},                                         # Paper B
                "hardware_profiles": {"h20": {"vllm": {"editor_args": list(args)}}},
                "mock_overrides": {"vllm": {"candidate_args": list(args)}}}
    for paper in ("A", "B"):
        assert C.lock_view(cfg("110", "0.78"), paper) == C.lock_view(cfg("90", "0.64"), paper)
        assert C.lock_view(cfg("110", "0.78"), paper) != C.lock_view(cfg("90", "0.64", "49152"), paper)
    a = cfg("110", "0.78")
    b = copy.deepcopy(a)
    b["hardware_profiles"]["h20"]["vllm"]["editor_args"][-1] = "65536"        # a locked flag deep in a profile
    assert C.lock_view(a) != C.lock_view(b)
    b = copy.deepcopy(a)
    b["vllm"]["editor_args"] += ["--max-num-seqs", "32"]                       # a flag added
    assert C.lock_view(a) != C.lock_view(b)
    c = {"vllm": {"other_args": ["--gpu-memory-utilization", "0.5"]}}         # only the named arg lists
    assert C.lock_view(c) != C.lock_view({"vllm": {"other_args": ["--gpu-memory-utilization", "0.6"]}})


def test_memory_knobs_ignored_empty_blocks_absent():
    base = {"calibration": {"max_texts": 256, "max_len": 2560, "layers_per_pass": 16},
            "kl_eval": {"n": 256, "max_num_batched_tokens": None},
            "hardware_profiles": {"h20": {"editor": {"context_tokens": 131072}}}}
    new = copy.deepcopy(base)
    new["calibration"]["layers_per_pass"] = 12
    new["kl_eval"]["max_num_batched_tokens"] = 2048
    new["hardware_profiles"]["h20"].update(calibration={"layers_per_pass": 12},       # blocks holding only ignored
                                           kl_eval={"max_num_batched_tokens": 2048})  # keys == absent
    new["mock_overrides"] = {"calibration": {"layers_per_pass": 4}}                    # also under mock_overrides
    assert C.lock_view(base, "A") == C.lock_view(new, "A")
    for path, v in (("calibration.max_len", 2048), ("kl_eval.n", 128), ("calibration.max_texts", 128)):
        ch = copy.deepcopy(new)                                                        # their neighbours stay locked
        blk, k = path.split(".")
        ch[blk][k] = v
        assert C.lock_view(ch, "A") != C.lock_view(new, "A"), path
    ch = copy.deepcopy(new)
    ch["hardware_profiles"]["h20"]["calibration"]["max_len"] = 2048
    assert C.lock_view(ch, "A") != C.lock_view(new, "A")


@pytest.mark.parametrize("hw", ["h20", "b300"])
def test_previous_plan_lock_resumes_with_the_new_base_yaml(tmp_path, hw):
    """A lock written by the previous code (h20 at 110 GiB, no calibration / kl_eval block under h20) compares equal to
    the new base.yaml, and the run resumes with the same command; the new plan's values are what the run now uses."""
    old_pdir = _old_paper_dir(tmp_path)
    runs = tmp_path / "runs"
    old = _load_a(runs, "oom", "--set", f"hardware={hw}", "--paper-dir", str(old_pdir))
    new_cfg = _load_a(tmp_path / "fresh", "x", "--set", f"hardware={hw}").paper_cfg
    assert old.paper_cfg != new_cfg                                       # the plan did change (not a no-op test)
    assert C.lock_view(old.paper_cfg, "A") == C.lock_view(new_cfg, "A")
    rc = _load_a(runs, "oom", "--set", f"hardware={hw}")                  # resume: same command, new base.yaml
    assert rc.paper_cfg == new_cfg
    if hw == "h20":
        c = rc.paper_cfg
        assert c["kl_eval"]["max_num_batched_tokens"] == 2048 and c["calibration"]["layers_per_pass"] == 12
        assert c["vllm"]["editor_args"][:2] == ["--rsi-gpu-memory-gib", "90"]
        assert c["vllm"]["candidate_args"][:2] == ["--rsi-gpu-memory-gib", "84"]
        assert old.paper_cfg["vllm"]["editor_args"][:2] == ["--rsi-gpu-memory-gib", "110"]
    lock = yaml.safe_load((runs / "oom" / "run_config.yaml").read_text())
    assert lock["paper_cfg"] == new_cfg                                   # the lock now records the new plan
    with pytest.raises(SystemExit, match="max-model-len|context_tokens"):  # a locked knob still refuses
        _load_a(runs, "oom", "--set", f"hardware={hw}", "--set", "editor.context_tokens=65536")
    with pytest.raises(SystemExit, match="calibration.max_len"):
        _load_a(runs, "oom", "--set", f"hardware={hw}", "--set", "calibration.max_len=2048")


def test_memory_knob_sets_resume(tmp_path):
    """--set of the memory / batching knobs on a resume is accepted (an OOMed run can be given a smaller plan)."""
    runs = tmp_path / "runs"
    _load_a(runs, "r", "--set", "hardware=h20")
    rc = _load_a(runs, "r", "--set", "hardware=h20", "--set", "calibration.layers_per_pass=8",
                 "--set", "kl_eval.max_num_batched_tokens=1024",
                 "--set", 'vllm.candidate_args=["--rsi-gpu-memory-gib","80","--gpu-memory-utilization","0.57",'
                          '"--max-model-len","20480","--max-num-seqs","128","--max-logprobs","20"]')
    assert rc.paper_cfg["calibration"]["layers_per_pass"] == 8
    with pytest.raises(SystemExit, match="candidate_args"):              # a non-memory flag of the list: refused
        _load_a(runs, "r", "--set", "hardware=h20",
                "--set", 'vllm.candidate_args=["--rsi-gpu-memory-gib","84","--max-model-len","20480",'
                         '"--max-num-seqs","64","--max-logprobs","20"]')
