"""Stopping vLLM servers and exit cleanup (2026-11-25), without vLLM or a GPU.

A real `vllm serve` runs in its own session and its GPU memory is held by EngineCore CHILDREN, so stop() must
wait for (and if needed SIGKILL) the whole process group, not just the leader; servers nobody stopped are
stopped at exit (atexit) and on SIGTERM (install_exit_handlers), and spawned / forked children never stop the
parent's servers. The "server" here is a fake process group: a shell leader with children that ignore SIGTERM
or take a while to exit.

  eval/.venv/bin/python -m pytest -q eval/tests/rsicore/test_server_stop.py
"""
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from rsicore import servers  # noqa: E402

PY = sys.executable


def _pid_alive(pid: int) -> bool:
    """True while pid is a live (non-zombie) process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def _wait_file(p: Path, timeout: float = 10.0) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.exists() and p.read_text().strip():
            return p.read_text().strip()
        time.sleep(0.05)
    raise AssertionError(f"{p} not written")


def _fake_group(tmp: Path, child: str, tag: str = "c"):
    """Popen of a shell leader in a new session whose background child runs `child` (a shell command);
    returns (proc, child pid)."""
    pidf = tmp / f"{tag}.pid"
    proc = subprocess.Popen(["sh", "-c", f"( {child} ) & echo $! > {pidf}; wait"], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, int(_wait_file(pidf))


def _fake_server(tmp: Path, proc, name: str = "fake") -> "servers.VLLMServer":
    """A mock VLLMServer object that owns `proc` (as start() would leave it), registered as live."""
    srv = servers.VLLMServer("mock-x", name, [0], mock=True, port=1, log_dir=tmp / "logs")
    srv.proc = proc
    servers._register(srv)
    return srv


IGNORE_TERM = "trap '' TERM; exec sleep 120"          # the ignored disposition survives exec
SLOW_EXIT = (f"exec {PY} -c \"import signal, sys, time; "
             f"signal.signal(signal.SIGTERM, lambda *a: (time.sleep(1.0), sys.exit(0))); time.sleep(120)\"")


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(servers, "STOP_GRACE_S", 1.0)
    monkeypatch.setattr(servers, "STOP_KILL_WAIT_S", 5.0)
    yield
    servers.stop_all_servers()


def test_stop_kills_child_that_ignores_sigterm(tmp_path, fast, capsys):
    proc, child = _fake_group(tmp_path, IGNORE_TERM)
    srv = _fake_server(tmp_path, proc)
    t0 = time.time()
    srv.stop()
    assert not _pid_alive(child), "the child that ignores SIGTERM survived stop()"
    assert proc.returncode is not None and srv.proc is None
    assert time.time() - t0 < 8
    err = capsys.readouterr().err
    assert "sending SIGKILL" in err and err.count("[servers]") == 1      # one line when it escalated
    assert srv not in servers.live_servers()
    srv.stop()                                                            # idempotent


def test_stop_waits_for_slow_child_not_just_leader(tmp_path, fast, capsys, monkeypatch):
    monkeypatch.setattr(servers, "STOP_GRACE_S", 10.0)
    proc, child = _fake_group(tmp_path, SLOW_EXIT)
    time.sleep(0.5)                                  # let python install its handler
    srv = _fake_server(tmp_path, proc)
    t0 = time.time()
    srv.stop()
    dt = time.time() - t0
    assert not _pid_alive(child)
    assert dt >= 0.8, f"stop() returned after {dt:.2f}s, before the child had exited"
    assert "SIGKILL" not in capsys.readouterr().err                       # no escalation needed


def test_stop_after_leader_exited_still_kills_group(tmp_path, fast):
    """vLLM's API server can die while EngineCore lives on: the old stop() skipped the kill then."""
    pidf = tmp_path / "c.pid"
    proc = subprocess.Popen(["sh", "-c", f"sleep 120 & echo $! > {pidf}; exit 0"], start_new_session=True)
    child = int(_wait_file(pidf))
    time.sleep(0.3)                                  # leader is a zombie now (not reaped yet)
    srv = _fake_server(tmp_path, proc)
    srv.stop()
    assert not _pid_alive(child)


def test_stop_servers_and_atexit_registry(tmp_path, fast):
    a_proc, a = _fake_group(tmp_path, IGNORE_TERM, "a")
    b_proc, b = _fake_group(tmp_path, "exec sleep 120", "b")
    sa, sb = _fake_server(tmp_path, a_proc, "a"), _fake_server(tmp_path, b_proc, "b")
    assert {id(sa), id(sb)} <= {id(s) for s in servers.live_servers()}
    t0 = time.time()
    servers.stop_all_servers()                       # the atexit handler
    assert not _pid_alive(a) and not _pid_alive(b)
    assert servers.live_servers() == []
    assert time.time() - t0 < 8                      # the SIGTERM grace periods overlap


def _script(tmp: Path, body: str) -> Path:
    p = tmp / "prog.py"
    p.write_text(textwrap.dedent(f"""
        import os, subprocess, sys, time
        sys.path.insert(0, {str(ROOT / 'eval')!r})
        from rsicore import servers
        servers.STOP_GRACE_S, servers.STOP_KILL_WAIT_S = 1.0, 5.0
        tmp = {str(tmp)!r}
        proc = subprocess.Popen(["sh", "-c", "(trap '' TERM; exec sleep 120) & echo $! > " + tmp + "/c.pid; wait"],
                                start_new_session=True)
        srv = servers.VLLMServer("mock-x", "fake", [0], mock=True, port=1, log_dir=tmp + "/logs")
        srv.proc = proc
        servers._register(srv)
        while not os.path.exists(tmp + "/c.pid") or not open(tmp + "/c.pid").read().strip():
            time.sleep(0.05)
    """) + textwrap.dedent(body))
    return p


def test_atexit_stops_live_server_at_normal_exit(tmp_path):
    prog = _script(tmp_path, "print('bye')\n")
    r = subprocess.run([PY, str(prog)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert not _pid_alive(int((tmp_path / "c.pid").read_text()))


def test_sigterm_runs_finally_and_stops_servers(tmp_path):
    prog = _script(tmp_path, """
        print(servers.install_exit_handlers(), flush=True)
        open(tmp + "/ready", "w").write("1")
        try:
            time.sleep(120)
        finally:
            open(tmp + "/finally", "w").write("1")
    """)
    p = subprocess.Popen([PY, str(prog)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _wait_file(tmp_path / "ready", 30)
    child = int((tmp_path / "c.pid").read_text())
    p.send_signal(signal.SIGTERM)
    out, err = p.communicate(timeout=60)
    assert p.returncode == 128 + signal.SIGTERM, err
    assert "SIGTERM" in out                         # SIGTERM converted
    assert (tmp_path / "finally").exists()          # try/finally ran
    assert not _pid_alive(child)
    assert "stopping 1 server(s)" in err


def test_sighup_left_alone_when_ignored(tmp_path):
    """launch.sh runs under nohup: SIGHUP is ignored and must stay ignored."""
    prog = _script(tmp_path, """
        import signal
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        print(servers.install_exit_handlers(), flush=True)
        srv.stop()
    """)
    r = subprocess.run([PY, str(prog)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "['SIGTERM']"


def test_install_exit_handlers_idempotent_main_thread_only():
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    saved_reg = dict(servers._EXIT_HANDLERS)
    try:
        servers._EXIT_HANDLERS.clear()
        for s in saved:
            signal.signal(s, signal.SIG_DFL)
        got = []
        t = threading.Thread(target=lambda: got.append(servers.install_exit_handlers()))
        t.start()
        t.join()
        assert got == [[]] and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL   # not from a thread
        first = servers.install_exit_handlers()
        assert first == ["SIGTERM", "SIGHUP"]
        assert servers.install_exit_handlers() == first                             # idempotent
        assert signal.getsignal(signal.SIGTERM) is servers._exit_on_signal
        assert servers.live_servers() == []
        with pytest.raises(SystemExit) as ei:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)                            # the handler runs at the next bytecode
        assert ei.value.code == 128 + signal.SIGTERM
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
        servers._EXIT_HANDLERS.clear()
        servers._EXIT_HANDLERS.update(saved_reg)


def _child_view(q):
    """Runs in a spawned / forked child: what the child's registry holds, then its atexit-style cleanup."""
    sys.path.insert(0, str(ROOT / "eval"))
    from rsicore import servers as S
    q.put(len(S.live_servers()))
    S.stop_all_servers()


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_children_do_not_stop_parent_servers(tmp_path, fast, method):
    if method not in mp.get_all_start_methods():
        pytest.skip(f"no {method} start method")
    proc, child = _fake_group(tmp_path, "exec sleep 120")
    srv = _fake_server(tmp_path, proc)
    ctx = mp.get_context(method)
    q = ctx.Queue()
    p = ctx.Process(target=_child_view, args=(q,))
    p.start()
    n = q.get(timeout=60)
    p.join(60)
    assert p.exitcode == 0
    assert n == 0, f"{method} child saw the parent's servers"
    assert _pid_alive(child) and srv.alive()        # the parent's server is untouched
    srv.stop()
    assert not _pid_alive(child)


def test_mock_server_stop_is_fast(tmp_path):
    """A real mock server (rsicore/mock_server.py) still starts and stops quickly."""
    srv = servers.VLLMServer("mock-x", "m", [0], mock=True, log_dir=tmp_path / "logs")
    with srv:
        assert srv in servers.live_servers() and srv.alive()
        pid = srv.proc.pid
        t0 = time.time()
    assert time.time() - t0 < 5
    assert not _pid_alive(pid) and srv not in servers.live_servers()


# ---------------------------------------------------------------------------- review regressions (2026-11-26)
def test_sigterm_during_stop_wait_kills_the_group(tmp_path):
    """A kill that arrives while stop() is still in its SIGTERM grace must not orphan the group: the server stays
    registered until its group is gone and the handler escalates it to SIGKILL."""
    prog = _script(tmp_path, """
        servers.STOP_GRACE_S = 30.0
        servers.install_exit_handlers()
        import threading, os, signal
        threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        srv.stop()                      # the child ignores SIGTERM: this waits in the grace when the kill lands
    """)
    t0 = time.time()
    r = subprocess.run([PY, str(prog)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 128 + signal.SIGTERM, r.stderr
    assert "stopping 1 server(s)" in r.stderr
    assert time.time() - t0 < 25, "the kill waited for the whole grace"
    assert not _pid_alive(int((tmp_path / "c.pid").read_text())), "the group survived the kill"


def test_sigterm_while_registry_lock_held_does_not_deadlock(tmp_path):
    prog = _script(tmp_path, """
        servers.install_exit_handlers()
        import os, signal
        with servers._LIVE_LOCK:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)
    """)
    r = subprocess.run([PY, str(prog)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 128 + signal.SIGTERM, r.stderr
    assert not _pid_alive(int((tmp_path / "c.pid").read_text()))


def test_handler_survives_dead_stderr(tmp_path):
    """After a terminal hang-up stderr raises EIO: the handler must still stop the servers and exit."""
    prog = _script(tmp_path, """
        servers.install_exit_handlers()
        import os, signal
        class Dead:
            def write(self, *a):
                raise OSError(5, "Input/output error")
            def flush(self):
                raise OSError(5, "Input/output error")
        sys.stderr = Dead()
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)
        except OSError:
            open(tmp + "/swallowed", "w").write("1")
        finally:
            sys.stderr = sys.__stderr__     # else the interpreter's own exit flush fails (exit code 120)
    """)
    r = subprocess.run([PY, str(prog)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 128 + signal.SIGTERM
    assert not (tmp_path / "swallowed").exists()
    assert not _pid_alive(int((tmp_path / "c.pid").read_text()))


def test_stop_skips_group_when_reaped_leader_pid_is_reused(tmp_path, fast, monkeypatch):
    """A leader reaped before stop() frees its pid; if that pid is in use again the group id is not ours."""
    proc, child = _fake_group(tmp_path, "exec sleep 120")
    srv = _fake_server(tmp_path, proc)
    os.killpg(proc.pid, signal.SIGKILL)             # simulate: our group died earlier ...
    proc.wait()
    sent = []
    monkeypatch.setattr(servers, "_pid_exists", lambda pid: True)            # ... and the pid was reused
    monkeypatch.setattr(servers, "_signal_group", lambda *a, **k: sent.append(a))
    srv.stop()
    assert sent == [] and srv not in servers.live_servers()
