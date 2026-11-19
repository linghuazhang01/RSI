"""Tiny stdlib HTTP helpers (no proxies, JSON in/out). Used for localhost servers only."""
import json
import socket
import threading
import urllib.error
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never use env proxies


class HTTPError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status, self.body = status, body


def request(method: str, url: str, obj=None, timeout: float = 600, headers: dict | None = None):
    """Return (status, text). Raises HTTPError for non-2xx, OSError for connection problems."""
    data = None if obj is None else json.dumps(obj).encode()
    h = {"Content-Type": "application/json"} if data is not None else {}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise HTTPError(e.code, e.read().decode("utf-8", "replace")) from None


def post_json(url: str, obj, timeout: float = 600, headers: dict | None = None):
    _, text = request("POST", url, obj, timeout, headers)
    return json.loads(text) if text.strip().startswith(("{", "[")) else text


def get_json(url: str, timeout: float = 10):
    _, text = request("GET", url, None, timeout)
    return json.loads(text)


_PORT_LOCK = threading.Lock()
_GIVEN: set = set()


def free_port() -> int:
    """A free local port, never the same one twice in this process (servers may start concurrently)."""
    with _PORT_LOCK:
        for _ in range(100):
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
            s.close()
            if p not in _GIVEN:
                _GIVEN.add(p)
                return p
        return p
