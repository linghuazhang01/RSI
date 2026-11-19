#!/usr/bin/env python
"""Unified evaluation client for OpenAI-compatible endpoints (vLLM incl. multi-LoRA).

Example:
  eval/.venv/bin/python -I eval/run_eval.py --task gsm8k --pool gate \
      --endpoint http://NODE:8000/v1 --model qwen3.8-27b-r0 [--lora math_lora_v3] \
      --out eval/results/r0_gate

  --task may be a comma-separated list. --lora sends the LoRA name as the request `model`
  (vLLM multi-LoRA addressing); --model is still recorded. Results:
  <out>/<task>.jsonl (one record per unit/item, with per-sample raw outputs, parsed answers, scores)
  <out>/summary.json (accuracy, n, task-specific breakdowns, decoding config, provenance)
"""
import argparse
import asyncio
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # our own package (works under python -I)

import yaml  # noqa: E402

from evallib.client import ChatClient, choice_view  # noqa: E402
from evallib.common import EVAL_DIR, load_items, strip_thinking  # noqa: E402
from evallib.summary import summarize, update_summary  # noqa: E402
from evallib.tasks import load_tasks  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="task name(s), comma-separated (see configs/tasks.yaml)")
    ap.add_argument("--pool", required=True, choices=["pool", "gate", "test", "search", "reserve"])
    ap.add_argument("--endpoint", required=True, help="e.g. http://host:8000/v1")
    ap.add_argument("--model", required=True, help="served base model name")
    ap.add_argument("--lora", default=None, help="served LoRA adapter name (vLLM --lora-modules name)")
    ap.add_argument("--out", default=None, help="run directory (default eval/results/<run_id>)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=str(EVAL_DIR / "configs" / "tasks.yaml"))
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--seed", type=int, default=None,
                    help="override the task/default sampling seed sent to the server (only sent when temperature > 0)")
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--n", type=int)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--thinking", choices=["on", "off"], default=None)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the endpoint key (optional)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-tag", default="", help="extra cache key (use when a model name is reused)")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N items (sorted by id)")
    ap.add_argument("--ids-file", default=None, help="JSON list of item ids to restrict to")
    ap.add_argument("--no-sandbox", action="store_true", help="never launch the Docker code sandbox")
    ap.add_argument("--sandbox-parallel", type=int, default=4)
    ap.add_argument("--judge-endpoint", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def decoding_for(task_cfg, defaults, a):
    d = {k: task_cfg.get(k, defaults[k]) for k in
         ("temperature", "top_p", "n", "max_tokens", "max_tokens_thinking", "thinking", "seed",
          "send_chat_template_kwargs")}
    if a.temperature is not None:
        d["temperature"] = a.temperature
    if a.seed is not None:
        d["seed"] = a.seed
    if a.top_p is not None:
        d["top_p"] = a.top_p
    if a.n is not None:
        d["n"] = a.n
    if a.thinking is not None:
        d["thinking"] = a.thinking == "on"
    if a.max_tokens is not None:
        d["max_tokens"] = d["max_tokens_thinking"] = a.max_tokens
    d["effective_max_tokens"] = d["max_tokens_thinking"] if d["thinking"] else d["max_tokens"]
    return d


def build_payload(req, dec, served_name):
    p = {"model": served_name, "messages": req["messages"], "temperature": dec["temperature"],
         "top_p": dec["top_p"], "max_tokens": dec["effective_max_tokens"], "n": dec["n"]}
    if dec["temperature"] > 0 and dec.get("seed") is not None:
        p["seed"] = dec["seed"]
    if req.get("tools"):
        p["tools"] = req["tools"]
        p["tool_choice"] = "auto"
    if dec.get("send_chat_template_kwargs", True):
        p["chat_template_kwargs"] = {"enable_thinking": bool(dec["thinking"])}
    return p


def run_judge(task, records, items_by_id, jcfg, a, out_dir):
    """Optional judge pass for mt_bench / alpaca_eval."""
    from evallib.graders import judge as J
    endpoint = a.judge_endpoint or jcfg.get("endpoint")
    if not endpoint:
        return {"judge": "skipped (no --judge-endpoint); scores are null"}
    model = a.judge_model or jcfg["model"]
    jc = ChatClient(endpoint, concurrency=a.concurrency, api_key_env=jcfg.get("api_key_env"),
                    use_cache=not a.no_cache, cache_tag="judge")
    payloads, slots = [], []
    if task.cfg["grader"] == "mt_bench_judge":
        by_item = {}
        for r in records:
            by_item.setdefault(r["id"], {})[r["unit_id"]] = r
        for iid, units in by_item.items():
            ans = [strip_thinking(units[u]["raw_output"] or "") for u in ("turn1", "turn2")]
            for t, msgs in enumerate(J.mt_bench_judge_messages(items_by_id[iid], ans)):
                payloads.append({"model": model, "messages": msgs, "temperature": jcfg["temperature"],
                                 "max_tokens": jcfg["max_tokens"]})
                slots.append(units["turn1" if t == 0 else "turn2"])
        resps = asyncio.run(jc.run_all(payloads, progress=not a.quiet))
        for r, resp in zip(slots, resps):
            ch = choice_view(resp)
            rating = J.parse_rating(ch[0]["content"]) if ch else None
            r["score"], r["sample_scores"], r["status"] = rating, [rating], "judged" if rating is not None else "judge_failed"
            r["judge_output"] = ch[0]["content"] if ch else resp.get("_error")
    else:
        flags = []
        for r in records:
            msgs, model_is_m = J.alpaca_judge_messages(items_by_id[r["id"]], strip_thinking(r["raw_output"] or ""))
            payloads.append({"model": model, "messages": msgs, "temperature": 1.0, "max_tokens": 1,
                             "logprobs": True, "top_logprobs": 5})
            slots.append(r)
            flags.append(model_is_m)
        resps = asyncio.run(jc.run_all(payloads, progress=not a.quiet))
        for r, resp, f in zip(slots, resps, flags):
            ch = choice_view(resp)
            p = J.alpaca_preference(ch[0], resp, f) if ch else None
            r["score"], r["sample_scores"], r["status"] = p, [p], "judged" if p is not None else "judge_failed"
    return {"judge": {"endpoint": endpoint, "model": model, "client_stats": jc.stats}}


def main(argv=None):
    a = parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    tasks = load_tasks(cfg)
    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S") + f"_{(a.lora or a.model).replace('/', '_')}"
    out_dir = Path(a.out) if a.out else EVAL_DIR / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    served = a.lora or a.model
    ids = json.load(open(a.ids_file)) if a.ids_file else None

    for tname in [t.strip() for t in a.task.split(",") if t.strip()]:
        if tname not in tasks:
            sys.exit(f"unknown task {tname}; available: {sorted(tasks)}")
        task = tasks[tname]
        allowed = task.cfg.get("allowed_pools")
        if allowed and a.pool not in allowed:
            sys.exit(f"task {tname} only supports pools {allowed}")
        items = sorted(load_items(task.dataset, a.pool, ids), key=lambda r: r["id"])
        if a.limit:
            items = items[: a.limit]
        if not items:
            print(f"[{tname}] no items in pool {a.pool}; skipping")
            continue
        dec = decoding_for(task.cfg, cfg["defaults"], a)
        client = ChatClient(a.endpoint, concurrency=a.concurrency, retries=a.retries, timeout=a.timeout,
                            api_key_env=a.api_key_env, use_cache=not a.no_cache, cache_tag=a.cache_tag)
        t0 = time.time()
        outputs = {it["id"]: {} for it in items}
        errors = []
        for stage in range(task.stages):
            reqs = []
            for it in items:
                for uid, req in task.units(it, stage, outputs[it["id"]]):
                    reqs.append((it["id"], uid, build_payload(req, dec, served)))
            if not a.quiet:
                print(f"[{tname}] pool={a.pool} items={len(items)} stage={stage} requests={len(reqs)} model={served}")
            resps = asyncio.run(client.run_all([p for _, _, p in reqs], progress=not a.quiet))
            for (iid, uid, _), resp in zip(reqs, resps):
                if "_error" in resp:
                    errors.append({"id": iid, "unit": uid, "error": resp["_error"]})
                outputs[iid][uid] = choice_view(resp)
        gen_s = time.time() - t0

        records = []
        for it in items:
            for r in task.grade(it, outputs[it["id"]]):
                g = task.cfg["grader"]
                if g in ("teval", "metatool", "apibank"):
                    r["group"] = it["metadata"][{"teval": "subset", "metatool": "task", "apibank": "level"}[g]]
                records.append(r)

        extra = {}
        if task.cfg["grader"] in ("humanevalplus", "mbppplus", "livecodebench", "apps"):
            from evallib.graders import code as C
            C.write_samples(task.cfg["grader"], out_dir / "sandbox", records)
            per = None if a.no_sandbox else C.run_sandbox(task.cfg["grader"], out_dir / "sandbox", a.sandbox_parallel)
            if per is None:
                extra["sandbox"] = {"status": "PENDING (Docker unavailable or --no-sandbox)",
                                    "command": C.pending_command(task.name, out_dir)}
                print(f"[{tname}] code execution PENDING. Run later:\n  {extra['sandbox']['command']}")
            else:
                key = (lambda r: r["id"].split("/", 1)[1]) if task.cfg["grader"] == "livecodebench" else (lambda r: r["id"])
                for r in records:
                    res = per.get(key(r), [])
                    r["sample_scores"] = [float(x["plus"]) for x in res]
                    r["metrics"] = res
                    r["score"] = sum(r["sample_scores"]) / len(res) if res else 0.0
                    r["status"] = "ok"
                extra["sandbox"] = {"status": "done"}
            for r in records:
                r.pop("_codes", None)
        if task.cfg.get("optional_judge"):
            extra.update(run_judge(task, records, {it["id"]: it for it in items}, cfg["judge"], a, out_dir))

        with open(out_dir / f"{tname}.jsonl", "w") as f:
            for r in records:
                r["prompt_id"] = r["id"]
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        extra.update({
            "pool": a.pool, "model": a.model, "lora": a.lora, "served_model_name": served, "endpoint": a.endpoint,
            "decoding": dec, "generation_seconds": round(gen_s, 1), "client_stats": client.stats,
            "n_generation_errors": len(errors), "generation_errors_sample": errors[:5],
            "limit": a.limit, "ids_file": a.ids_file, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "versions": {p: _ver(p) for p in ("math-verify", "bfcl-eval", "ifbench", "httpx")},
        })
        summ = summarize(task, records, extra)
        update_summary(out_dir, tname, summ)
        acc = summ["accuracy"]
        print(f"[{tname}] accuracy={'NA' if acc is None else f'{acc:.4f}'} n={summ['n']} "
              f"pending={summ['n_pending']} errors={len(errors)} -> {out_dir / (tname + '.jsonl')}")
    return 0


def _ver(p):
    try:
        return md.version(p)
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
