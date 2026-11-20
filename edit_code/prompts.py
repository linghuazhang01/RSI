"""Editor system prompt and per-step context rendering for Paper A.

The editor (the original model M_0, frozen; or the incumbent with loop.editor=current) must answer with ONE ```python block defining
`def main(api): ... return config`. render_context(history, thread) shows: budget and remaining
evaluates, a compact diagnostics summary (first step), the table of all candidates so far (all
threads), the best candidate's config, this thread's last feedback with 2-3 regression examples,
and the archive of earlier generations (experience channel; switchable).
"""
from __future__ import annotations

import json

from .diagnostics import summarize_profile

SYSTEM_PROMPT = """You are quantizing a large language model to a fixed memory budget. The model being quantized is a copy of YOURSELF: every config is applied to the original BF16 weights. The budget is the same in every round; each round tries to beat the best quantization accepted so far (the incumbent), which you can see together with the experience of earlier rounds.

Write a Python program `def main(api): ... return config` that returns a quantization config. The program runs in a sandbox (no network, no files outside its working dir); it can only use the tools on `api`, always called with KEYWORD arguments (e.g. `api.bytes(config=cfg)`, `api.evaluate(config=cfg)`):
{tools}

Config schema (JSON object):
{{
  "default": {{"bits": 4, "group_size": 128, "quantizer": "gptq"}},        # required
  "rules": [{{"layers": "56-63", "groups": ["down", "gate_up"], "bits": 8}}],  # optional, applied in order
  "groups": {{"17.down": {{"bits": 3}}, "gdn_ba": {{"bits": 16}}}},             # optional, applied last
  "embed": "bf16",      # "bf16" | 8 | 4
  "lm_head": 8          # "bf16" | 8 | 4
}}
- bits in {bits} (16 = keep BF16); group_size in {group_sizes}; quantizer in {quantizers}.
- Module groups per layer. Full-attention layers: qk (q_proj+k_proj), v, o. Linear-attention (Gated DeltaNet) layers: gdn_qkv (in_proj_qkv), gdn_z (in_proj_z), gdn_ba (in_proj_b+in_proj_a, tiny gates), gdn_out (out_proj). Every layer: gate_up (gate_proj+up_proj), down (down_proj). Group keys look like "17.down".
- Layer selectors: int (negative = from the end), "a-b", "all", or a list. Group selectors: name, "all" or list. "groups" keys may be "17.down", "down", "56-63.down" or "*.down".
- The packed size (codes + fp16 scales + zero points + unquantized parts) must be <= the budget; check with api.bytes(config).
- profile_table() gives, per group key and bit width, the whole-model KL increase when only that group is RTN-quantized (forward pass on the original model, 64 texts). hessian_table() gives, per group, quantizer (rtn / gptq / awq) and bit width, the Hessian-weighted output error of that group alone (every calibration token, no forward pass): lower noise, local. Sensitivities are roughly additive at >= 3.5 average bits and underestimate damage at 2-3 bits; evaluate(config) measures the real thing.
- Analysis (free, no evaluate): group_error(config) shows where a config's quantization error concentrates; compare(cid) compares an evaluated candidate with the incumbent (or another candidate) item by item: which items broke / got fixed (ids only), by output length, truncation.
- Each program's returned config is evaluated on held-back training items and compared with the INCUMBENT on the same items (retention = your accuracy / the incumbent's; paired delta; KL vs the original BF16 model); you then see the feedback and may improve. api.evaluate(config) inside a program spends the same per-round evaluate budget.
- Acceptance (after the search, not visible to you): size <= budget and a higher score than the incumbent on a fixed hidden test set (paired per item: mean gain >= {min_delta}, a statistically significant gain and no significant loss on any domain). Otherwise the incumbent stays.

Answer with a short plan, then exactly one ```python code block containing the full program."""


FINAL_RULE = "Answer with a short plan, then exactly one ```python code block containing the full program."

# editor.tool_rounds > 0 (2026-11-21): the editor may call chat tools before answering; replaces FINAL_RULE
TOOL_PROTOCOL = """Before you answer you may call functions, for up to {rounds} rounds:
- run_python(code) runs exploratory code in the same sandbox with the same `api` (keyword arguments as above), so you can inspect and filter the tables and check configs before you commit to a program. api.evaluate is NOT available there (only your final program may evaluate){group_error}. print() what you need or set `result = ...`; output is cut to a few thousand characters, so filter big tables in the code.
- read_record(cid, part) shows the full program ("program") or the full feedback ("feedback") of a candidate in the table.
Nothing persists between run_python calls or into your final program: it must be self-contained.
When you are done, reply WITHOUT a function call: a short plan, then exactly one ```python code block containing the full program."""


def _tool_rounds(cfg: dict) -> tuple[int, bool]:
    """(editor.tool_rounds, editor.explore.gpu_tools) with the rsicore.editor defaults."""
    try:
        from rsicore.editor import editor_settings
        s = editor_settings(cfg)
        return int(s["tool_rounds"] or 0), bool(s["explore"].get("gpu_tools"))
    except ImportError:                      # pragma: no cover - rsicore not on sys.path
        e = cfg.get("editor") or {}
        return int(e.get("tool_rounds", 50) or 0), bool((e.get("explore") or {}).get("gpu_tools"))


def system_prompt(problem, tool_doc: str, cfg: dict) -> str:
    sp = problem.space
    g = cfg["gate"]
    text = SYSTEM_PROMPT.format(tools=tool_doc, bits=sp["bits"], group_sizes=sp["group_sizes"],
                                quantizers=sp["quantizers"], min_delta=g.get("min_delta", 0.0))
    rounds, gpu_tools = _tool_rounds(cfg)
    if rounds > 0:                           # tool_rounds 0: byte-identical to the single-call prompt
        ge = ("" if gpu_tools or not (cfg.get("tools") or {}).get("group_error", True)
              else "; group_error there only accepts settings already cached (those of hessian_table)")
        if text.endswith(FINAL_RULE):
            text = text[:-len(FINAL_RULE)]
        text += TOOL_PROTOCOL.format(rounds=rounds, group_error=ge)
    return text


def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _feedback(c) -> dict | None:
    fb = (c.full or {}).get("feedback") if c.full else None
    fb = fb or (c.screen or {}).get("feedback")
    if isinstance(fb, dict) and "feedback" in fb and "probes" not in fb:
        fb = fb["feedback"]
    return fb


def candidate_table(history, domains: list[str]) -> str:
    head = "| cid | thread | step | GiB | avg bits | KL | " + " | ".join(domains) + " | score | status |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for c in history:
        fb = _feedback(c) or {}
        art = c.artifact or {}
        kl = (fb.get("kl") or {}).get("overall")
        cells = []
        for d in domains:
            p = (fb.get("probes") or {}).get(d) or {}
            s = _fmt(p.get("score"))
            if p.get("vs_incumbent") is not None:
                s += f" ({p['vs_incumbent']:+.3f}{'*' if p.get('significant') else ''})"
            cells.append(s)
        status = "error: " + c.error.splitlines()[0][:80] if c.error else ("full" if c.full else "screen")
        lines.append(f"| {c.cid} | {c.thread} | {c.step} | {_fmt(art.get('bytes_gib'))} | {_fmt(art.get('linear_avg_bits'), 2)} "
                     f"| {_fmt(kl, 4)} | " + " | ".join(cells) + f" | {_fmt(c.score, 4)} | {status} |")
    return "\n".join(lines)


def make_render_context(problem, cfg: dict, *, task: str = "main", archive: str = ""):
    """Returns render_context(history, thread) -> str for rsicore.search.inner_search."""
    n_examples = cfg["feedback"].get("examples", 2)

    def render(history, thread: int) -> str:
        step = sum(1 for c in history if c.thread == thread)
        parts = [f"[thread={thread} step={step} task={task} gen={problem.gen}]",
                 f"Budget (fixed for every round): {problem.budget_gib:.3f} GiB packed. "
                 f"Evaluates left this round: {problem.evals_left} (incl. api.evaluate calls)."]
        if step == 0:
            info = problem.spec.summary()
            parts.append("Model: " + json.dumps({"num_layers": info["num_layers"], "layer_types": info["layer_types"],
                                                 "linear_params": info["linear_params"],
                                                 "params_by_group": info["params_by_group"],
                                                 "lm_head_tied": info["lm_head_tied"]}))
            inc = problem.incumbent
            if inc:
                parts.append(f"Incumbent to beat (accepted in round {inc.get('gen')}, {inc.get('bytes_gib')} GiB, "
                             f"linear avg bits {inc.get('linear_avg_bits')}); its score on this round's items: "
                             + json.dumps(problem.incumbent_scores) + "; config (compressed form): "
                             + json.dumps(inc.get("config"))[:3000])
            if problem.noise and any(v for v in problem.noise.values()):   # greedy decoding: band 0, not shown
                parts.append("Noise band (max-min over repeated sampled runs of the original model): " + json.dumps(problem.noise))
            parts.append("Profile summary (KL increase if only that group is RTN-quantized):\n"
                         + json.dumps(summarize_profile(problem.tables["profile"], top=8))[:4000])
            red = [f"{r['layer']}.{r['module']}" for r in problem.tables["activation"] if r.get("red_flag")]
            if red:
                parts.append(f"Activation red flags (heavy outliers, {len(red)}): " + ", ".join(red[:30]))
            if archive:
                parts.append("Archive of earlier generations:\n" + archive)
        if history:
            parts.append("All candidates so far (score = retention vs the incumbent - KL penalty; "
                         "(+/-delta vs the incumbent on the same items, * = significant)):\n"
                         + candidate_table(history, problem.domains))
            ok = [c for c in history if c.score is not None]
            if ok:
                best = max(ok, key=lambda c: c.score)
                parts.append(f"Best so far: {best.cid} (score {best.score:.4f}), config: "
                             + json.dumps((best.artifact or {}).get("config"))[:2500])
            mine = [c for c in history if c.thread == thread]
            if mine:
                last = mine[-1]
                if last.error:
                    parts.append(f"Your last program ({last.cid}) failed:\n{last.error[:2500]}")
                else:
                    fb = dict(_feedback(last) or {})
                    regs = fb.pop("regressions", [])[: n_examples * max(1, len(problem.datasets))]
                    parts.append(f"Feedback for your last candidate {last.cid}:\n" + json.dumps(fb)[:3500])
                    if regs:
                        parts.append("Items the incumbent got right and your candidate got wrong:\n"
                                     + json.dumps(regs, ensure_ascii=False)[:3000])
        parts.append("Beat the incumbent: improve on the best candidate (yours or another thread's) or try a different idea. "
                     "Return one ```python block with def main(api): ... return config.")
        return "\n\n".join(parts)

    return render


def archive_text(entries: list[dict]) -> str:
    """entries: one per earlier round {gen, accepted, gate_reason, incumbent_score, top: [{cid, score, config}]}.
    Scores of different rounds are on different training items (compare within a round)."""
    lines = []
    for e in entries:
        lines.append(f"- round {e['gen']}: new incumbent accepted={e.get('accepted')}; gate: {str(e.get('gate_reason'))[:300]}; "
                     f"incumbent score on that round's items {e.get('incumbent_score')}")
        for c in e.get("top") or []:
            lines.append(f"    {c['cid']}: score {c.get('score')}, {c.get('bytes_gib')} GiB, config {json.dumps(c.get('config'))[:600]}")
    return "\n".join(lines)
