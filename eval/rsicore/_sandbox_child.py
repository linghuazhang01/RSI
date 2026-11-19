"""Child side of rsicore.sandbox.run_program. Started as
    python -I _sandbox_child.py <config.json> <req_fd> <resp_fd>
Uses only the standard library. Order: apply rlimits -> install the audit hook (file writes only in
the workdir; no reads of denied roots; no subprocess / socket connect / os.kill) -> read program.py ->
exec it -> call main(api) -> send the JSON result. `api.<tool>(**kwargs)` is a JSON-lines RPC to the
parent's ToolAPI; tool errors come back as exceptions (api.BudgetExceeded / api.ToolError).
"""
import json
import os
import sys
import traceback

CFG = json.load(open(sys.argv[1]))
REQ = os.fdopen(int(sys.argv[2]), "w", buffering=1)
RESP = os.fdopen(int(sys.argv[3]), "r")


def _limits():
    import resource
    applied = {}
    lim = {"RLIMIT_AS": CFG["mem_bytes"], "RLIMIT_CPU": CFG["cpu_s"], "RLIMIT_FSIZE": CFG["fsize_bytes"],
           "RLIMIT_NOFILE": 256, "RLIMIT_CORE": 0}
    for name, v in lim.items():
        r = getattr(resource, name, None)
        if r is None:
            applied[name] = "unavailable"
            continue
        try:
            soft, hard = resource.getrlimit(r)
            v2 = v if hard == resource.RLIM_INFINITY else min(v, hard)
            resource.setrlimit(r, (v2, v2))
            applied[name] = v2
        except (ValueError, OSError) as e:
            applied[name] = f"unsupported: {e}"
    return applied


LIMITS = _limits()


def _real(p):
    return os.path.realpath(os.fsdecode(p))


WORK = _real(CFG["workdir"])
ALLOW_READ = [_real(p) for p in CFG["allow_read"]]
DENY_READ = [_real(p) for p in CFG["deny_read"]]
BLOCK = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
         "os.kill", "os.killpg", "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg",
         "socket.getaddrinfo", "pty.spawn", "os.chroot", "os.setuid", "winreg.OpenKey", "webbrowser.open"}
WRITE_EVENTS = {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.symlink",
                "os.link", "os.truncate", "os.utime", "shutil.rmtree", "shutil.move", "shutil.copyfile"}
_IN = [False]


def _under(p, root):
    return p == root or p.startswith(root.rstrip("/") + "/")


def _check_read(p):
    if _under(p, WORK) or any(_under(p, a) for a in ALLOW_READ):
        return
    if any(_under(p, d) for d in DENY_READ):
        raise PermissionError(f"sandbox: reading {p} is not allowed")


def _check_write(p):
    if not (_under(p, WORK) or p == "/dev/null"):
        raise PermissionError(f"sandbox: writing outside the workdir is not allowed ({p})")


def _hook(event, args):
    if _IN[0]:
        return
    if event in BLOCK:
        raise PermissionError(f"sandbox: {event} is not allowed")
    if event not in ("open", "os.listdir", "os.scandir", "os.chdir") and event not in WRITE_EVENTS:
        return
    _IN[0] = True
    try:
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int) or path is None:
                return
            p = _real(path)
            write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                    (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
            (_check_write if write else _check_read)(p)
        elif event in ("os.listdir", "os.scandir"):
            if args and isinstance(args[0], (str, bytes, os.PathLike)):
                _check_read(_real(args[0]))
        elif event == "os.chdir":
            if not _under(_real(args[0]), WORK):
                raise PermissionError("sandbox: chdir outside the workdir is not allowed")
        else:
            for a in args[:2]:
                if isinstance(a, (str, bytes, os.PathLike)):
                    _check_write(_real(a))
    finally:
        _IN[0] = False


class BudgetExceeded(Exception):
    pass


class ToolError(Exception):
    pass


class _API:
    def __init__(self, tools, docs):
        self._tools, self._docs, self._n = set(tools), docs, 0
        self.BudgetExceeded, self.ToolError = BudgetExceeded, ToolError

    def describe(self):
        return self._docs

    def tools(self):
        return sorted(self._tools)

    def __getattr__(self, name):
        if name.startswith("_") or name not in self._tools:
            raise AttributeError(f"api has no tool {name!r}; available: {sorted(self._tools)}")

        def call(*args, **kwargs):
            if args:
                raise TypeError(f"api.{name}: pass arguments by keyword")
            self._n += 1
            REQ.write(json.dumps({"type": "call", "id": self._n, "tool": name, "kwargs": kwargs}) + "\n")
            line = RESP.readline()
            if not line:
                raise ToolError("parent closed the channel")
            r = json.loads(line)
            if r.get("ok"):
                return r.get("result")
            if r.get("error_type") == "BudgetExceeded":
                raise BudgetExceeded(r.get("error"))
            raise ToolError(f"{r.get('error_type')}: {r.get('error')}")
        return call


def main():
    hello = json.loads(RESP.readline())
    api = _API(hello["tools"], hello["docs"])
    REQ.write(json.dumps({"type": "hello", "limits": LIMITS}) + "\n")
    os.chdir(WORK)
    sys.addaudithook(_hook)
    out = {"type": "final", "ok": False, "result": None, "error": None}
    try:
        src = open(os.path.join(WORK, CFG["program"])).read()
        g = {"__name__": "__rsi_program__", "__file__": CFG["program"], "BudgetExceeded": BudgetExceeded,
             "ToolError": ToolError}
        exec(compile(src, CFG["program"], "exec"), g)
        if not callable(g.get("main")):
            raise NameError("the program must define `def main(api): ...`")
        res = g["main"](api)
        try:
            enc = json.dumps(res, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"main(api) must return a JSON-serializable value: {e}") from None
        out.update(ok=True, result=json.loads(enc))
    except BaseException as e:  # noqa: BLE001 - report everything, including SystemExit
        tb = traceback.format_exc(limit=20)
        out["error"] = f"{type(e).__name__}: {e}\n{tb[-6000:]}"
    sys.stdout.flush()
    REQ.write(json.dumps(out) + "\n")
    REQ.flush()


main()
