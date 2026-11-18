"""Minimal async client for OpenAI-compatible /v1/chat/completions endpoints (vLLM, oracle server).

* concurrency via asyncio.Semaphore, retries with exponential backoff (429 / 5xx / network errors)
* on-disk cache: eval/.cache/gen/<sha256>.json, keyed on (model, lora, cache_tag, request payload)
  -> give every checkpoint a unique served model name or --cache-tag, otherwise generations collide.
* the API key (if any) is read from an environment variable whose NAME is configurable; it is never
  written to disk or to results.
"""
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

import httpx

from .common import CACHE_DIR


def cache_key(payload: dict, tag: str) -> str:
    blob = json.dumps({"payload": payload, "tag": tag}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class ChatClient:
    def __init__(self, endpoint: str, concurrency: int = 32, retries: int = 6, timeout: float = 3600,
                 api_key_env: str | None = "OPENAI_API_KEY", use_cache: bool = True, cache_tag: str = "",
                 cache_dir: Path = CACHE_DIR / "gen"):
        ep = endpoint.rstrip("/")
        if not ep.endswith("/chat/completions"):
            ep = ep + ("/chat/completions" if ep.endswith("/v1") else "/v1/chat/completions")
        self.url = ep
        self.concurrency = concurrency
        self.sem = None  # created inside the running event loop (run_all)
        self.retries = retries
        self.timeout = timeout
        key = os.environ.get(api_key_env) if api_key_env else None
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.use_cache = use_cache
        self.cache_tag = cache_tag
        self.cache_dir = cache_dir
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0, "failures": 0}

    def _cache_path(self, k):
        return self.cache_dir / k[:2] / f"{k}.json"

    async def chat(self, client: httpx.AsyncClient, payload: dict) -> dict:
        k = cache_key(payload, self.cache_tag)
        cp = self._cache_path(k)
        if self.use_cache and cp.exists():
            self.stats["cache_hits"] += 1
            return json.loads(cp.read_text())
        async with self.sem:
            delay = 1.0
            last_err = None
            for attempt in range(self.retries + 1):
                try:
                    t0 = time.time()
                    r = await client.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
                    if r.status_code == 200:
                        resp = r.json()
                        resp["_latency_s"] = round(time.time() - t0, 3)
                        self.stats["requests"] += 1
                        if self.use_cache:
                            cp.parent.mkdir(parents=True, exist_ok=True)
                            tmp = cp.with_suffix(".tmp")
                            tmp.write_text(json.dumps(resp, ensure_ascii=False))
                            tmp.replace(cp)
                        return resp
                    last_err = f"HTTP {r.status_code}: {r.text[:500]}"
                    if r.status_code < 500 and r.status_code not in (408, 409, 429):
                        break  # client error: do not retry
                except (httpx.HTTPError, json.JSONDecodeError) as e:
                    last_err = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    self.stats["retries"] += 1
                    await asyncio.sleep(delay + random.random())
                    delay = min(delay * 2, 60)
            self.stats["failures"] += 1
            return {"_error": last_err}

    async def run_all(self, payloads: list[dict], progress: bool = True) -> list[dict]:
        self.sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=64)
        async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
            tasks = [self.chat(client, p) for p in payloads]
            if not progress:
                return await asyncio.gather(*tasks)
            from tqdm.asyncio import tqdm_asyncio
            return await tqdm_asyncio.gather(*tasks, mininterval=2)


def choice_view(resp: dict) -> list[dict]:
    """Normalize a chat completion into a list of {content, reasoning, tool_calls, finish_reason}."""
    if "_error" in resp:
        return []
    out = []
    for ch in resp.get("choices", []):
        m = ch.get("message") or {}
        out.append({
            "content": m.get("content") or "",
            "reasoning": m.get("reasoning_content") or m.get("reasoning") or None,
            "tool_calls": m.get("tool_calls") or [],
            "finish_reason": ch.get("finish_reason"),
        })
    return out
