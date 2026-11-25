"""Editor system prompt and per-step context rendering for Paper A (KL only, 2026-11-25).

The editor (the original model M_0, frozen; or the incumbent with loop.editor=current) must answer with ONE ```python
block defining `def main(api): ... return config`. render_context(history, thread) shows: the budget and the search
shape, a compact diagnostics summary, the incumbent (its KL to the original model) and the archive of earlier rounds
on the first step, and this round's candidates with their KL results.

There is no screen and no accuracy gate: every returned config gets ONE KL evaluation (per-trace token KL(M_0 ||
config) on the fixed KL set, M_0's own answers to held-back training items) and later steps / rounds see it
(candidate table, own feedback, search_history, archive); the lowest KL of the round is accepted when it is below the
incumbent's. With steps 1 each thread is told that it is session k of N working independently this round and should
use search_history / the archive to try something different from earlier rounds.
"""
from __future__ import annotations

import json

from .common import error_line, kl_eval_cfg
from .diagnostics import summarize_profile

SYSTEM_BODY = """You are quantizing a large language model to a fixed memory budget. The model being quantized is a copy of YOURSELF: every config is applied to the original BF16 weights. The budget is the same in every round; each round tries to beat the lowest-KL quantization accepted so far (the incumbent), which you can see together with the experience of earlier rounds.

Write a Python program `def main(api): ... return config` that returns a quantization config. The program runs in a sandbox (no network, no files outside its working dir); it can only use the tools on `api`, always called with KEYWORD arguments (e.g. `api.bytes(config=cfg)`):
{tools}

Config schema (JSON object):
{{
  "default": {{"bits": 4, "group_size": 128, "quantizer": "gptq"}},        # required
  "rules": [{{"layers": "56-63", "groups": ["down", "gate_up"], "bits": 8}}],  # optional, applied in order
  "groups": {{"17.down": {{"bits": 3}}, "gdn_ba": {{"bits": 16}}}},             # optional, applied last
  "embed": "bf16",      # "bf16" | 8 | 4
  "lm_head": 8,         # "bf16" | 8 | 4
  "note": "..."         # optional: why these groups, what you tried and rejected (see below)
}}
- bits in {bits} (16 = keep BF16); group_size in {group_sizes}; quantizer in {quantizers}.
- Module groups per layer. Full-attention layers: qk (q_proj+k_proj), v, o. Linear-attention (Gated DeltaNet) layers: gdn_qkv (in_proj_qkv), gdn_z (in_proj_z), gdn_ba (in_proj_b+in_proj_a, tiny gates), gdn_out (out_proj). Every layer: gate_up (gate_proj+up_proj), down (down_proj). Group keys look like "17.down".
- Layer selectors: int (negative = from the end), "a-b", "all", or a list. Group selectors: name, "all" or list. "groups" keys may be "17.down", "down", "56-63.down" or "*.down".
- The packed size (codes + fp16 scales + zero points + unquantized parts) must be <= the budget; check with api.bytes(config).
- profile_table() gives, per group key and bit width, the whole-model KL increase when only that group is RTN-quantized (forward pass on the original model, 64 texts). hessian_table() gives, per group, quantizer (rtn / gptq / awq) and bit width, the Hessian-weighted output error of that group alone (every calibration token, no forward pass): lower noise, local. Sensitivities are roughly additive at >= 3.5 average bits and underestimate damage at 2-3 bits; only the KL evaluation measures the real thing.
- Analysis (free): group_error(config) shows where a config's quantization error concentrates.
- {kl_desc} There is no api.evaluate and no accuracy score: KL to the original model is the only number.
- {acceptance}
- In your program `api` exists only as the argument of main(api) (it is not a global there): pass it to helper functions explicitly, e.g. def pick(api, budget): ...
- Time: your program is killed after {timeout_s} s of wall clock, every tool call included (group_error / hessian_table on settings that are not cached yet quantize on the GPU and take real time). A killed program is a lost candidate.
- note (optional, at most 600 characters, plain text, the "note" key of the returned config): which groups you changed and WHY those (the evidence that picked them: a profile_table / hessian_table / group_error number, the KL of earlier candidates), then what you tried and rejected, with the numbers that decided it (e.g. "8 bit for down 0-3 and v 40-47: largest gptq 4-bit herr_rel in hessian_table; paid for with gate_up 20-35 at 3 bit; rejected: awq on gdn_qkv, group_error total +9% vs gptq"). The history already lists the config; the note is for the reasons. Every later session sees it, so a good note keeps them from repeating your failed ideas.

"""

PERSIST_NONE = "Nothing persists between run_python calls or into your final program: it must be self-contained."
PERSIST_SESSION = ("Variables, imports and functions persist between your run_python calls while you write this program "
                   "(one session; a timeout or crash restarts it and loses them), but not into your final program: it "
                   "must be self-contained.")

# the body ends the system prompt with FINAL_RULE (tool_rounds 0) or TOOL_PROTOCOL (tool_rounds > 0)
FINAL_RULE = "Answer with a short plan, then exactly one ```python code block containing the full program."
SYSTEM_PROMPT = SYSTEM_BODY + FINAL_RULE

# editor.tool_rounds > 0 (2026-11-21): the editor may call chat tools before answering; replaces FINAL_RULE
TOOL_PROTOCOL = """Before you answer you may call functions, for up to {rounds} rounds (one round = one reply with function calls; at most {calls} calls in all):
- run_python(code) runs exploratory code in the same sandbox with the same `api` (keyword arguments as above), so you can inspect and filter the tables and check configs before you commit to a program.{group_error}. print() what you need or set `result = ...`; output is cut to a few thousand characters, so filter big tables in the code.
- read_record(cid, part) shows the full program ("program") or the full feedback ("feedback", with the full config) of a candidate in the table{history_read}.{history}
{persist}
When you are done, reply WITHOUT a function call: a short plan, then exactly one ```python code block containing the full program."""


def _tool_rounds(cfg: dict) -> tuple[int, bool, bool, int]:
    """(editor.tool_rounds, editor.explore.gpu_tools, editor.explore.persistent, editor.max_tool_calls) as
    rsicore.editor.editor_settings merges them with its defaults."""
    from rsicore.editor import editor_settings
    s = editor_settings(cfg)
    ex = s["explore"]
    return (int(s["tool_rounds"] or 0), bool(ex.get("gpu_tools")), bool(ex.get("persistent")),
            int(s.get("max_tool_calls") or 0))


KL_DESC = ("Every config your program returns gets ONE evaluation that samples nothing: the model it quantizes reads "
           "{n} fixed reference traces (the original BF16 model's own thinking-mode answers to held-back training "
           "items, written once before round 1, never shown to you) and the evaluation measures the token-level KL "
           "divergence KL(original || quantized) on every answer position (top-{k} approximation), averaged per trace "
           "and then over the traces; lower = closer to the original. Its result (KL, the incumbent's KL, KL reduction "
           "= the incumbent's KL minus yours, per-trace quantiles) is shown to later sessions and rounds.")

FIRST_ROUND = ("No quantization has been accepted yet (loop.start bf16): the incumbent is the original BF16 model "
               "itself (KL 0 to itself), which does not fit the budget. ")


def acceptance_text(cfg: dict, first: bool = False) -> str:
    """The acceptance rule: the lowest KL of the round wins when it is below the incumbent's. first (loop.start bf16,
    the incumbent is still the BF16 original): the lowest KL among the configs within the budget is accepted."""
    if first:
        return (FIRST_ROUND + "Acceptance (end of this round): among the configs that fit the budget, the one with "
                "the LOWEST KL to the original model becomes the first incumbent, and later rounds must beat it. None "
                "fits: the incumbent stays the original and the next round uses the same rule.")
    return ("Acceptance (end of the round): among the configs that fit the budget, the one with the LOWEST KL to the "
            "original model becomes the new incumbent if its KL is below the incumbent's (no significance test). "
            "Otherwise the incumbent stays.")


def _n_traces(problem):
    try:
        return len(problem.kl_items())
    except Exception:  # noqa: BLE001 - informative only (traces not written yet)
        return None


def _kl_desc(problem, cfg: dict) -> str:
    n = _n_traces(problem)
    return KL_DESC.format(n=n if n else "the", k=int(kl_eval_cfg(cfg)["top_k"]))


HISTORY_READ = " or of any earlier candidate that search_history lists"
HISTORY_LINE = ("\n- search_history(gen, sort, offset) lists every candidate already tried (this generation's and the "
                "earlier generations') with a one-line description of its config, its size and its KL "
                "result; same_as marks a config identical to an earlier one. Check it before you write a program, "
                "so you do not repeat a config that was already tried.")


def system_prompt(problem, tool_doc: str, cfg: dict, history: bool = False) -> str:
    sp = problem.space
    first = bool(getattr(problem, "incumbent_is_original", False))
    text = SYSTEM_BODY.format(tools=tool_doc, bits=sp["bits"], group_sizes=sp["group_sizes"],
                              quantizers=sp["quantizers"], timeout_s=int((cfg.get("sandbox") or {}).get("timeout_s", 1800)),
                              kl_desc=_kl_desc(problem, cfg), acceptance=acceptance_text(cfg, first=first))
    if first:   # loop.start bf16, nothing accepted yet
        text = text.replace("the lowest-KL quantization accepted so far (the incumbent)",
                            "the lowest-KL quantization accepted so far (the incumbent; none yet: this round picks the "
                            "first one)")
    rounds, gpu_tools, persistent, calls = _tool_rounds(cfg)
    if rounds <= 0:                          # tool_rounds 0: the single-call prompt
        return text + FINAL_RULE
    ge = ("" if gpu_tools or not (cfg.get("tools") or {}).get("group_error", True)
          else "; group_error there only accepts settings already cached (those of hessian_table)")
    return text + TOOL_PROTOCOL.format(rounds=rounds, calls=calls, group_error=ge, history_read=HISTORY_READ if history else "",
                                       history=HISTORY_LINE if history else "",
                                       persist=PERSIST_SESSION if persistent else PERSIST_NONE)


def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _eval_fb(screen: dict | None) -> dict | None:
    """The KL feedback (run.kl_feedback) kept in a candidate's screen slot (rsicore.search's name for it)."""
    return (screen or {}).get("feedback")


def _feedback(c) -> dict | None:
    """A candidate's feedback: its KL feedback."""
    return _eval_fb(c.screen)


def _setting(x: dict) -> str:
    out = f"{x.get('bits')}b"
    if x.get("bits") != 16:
        out += "".join(f"/{p}{x[k]}" for k, p in (("group_size", "g"), ("quantizer", "")) if x.get(k) is not None)
    return out


def _layer_ranges(xs, chars: int = 40) -> str:
    """[1, 2, 3, 5] -> '1-3,5' (cut to about `chars` characters with '...')."""
    xs = sorted(set(xs))
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    t = ",".join(out)
    return t if len(t) <= chars else t[:chars].rsplit(",", 1)[0] + ",..."


def config_brief(config: dict | None) -> str | None:
    """One line describing a (compressed) config: the default setting and the overrides grouped by setting, with
    the module kinds they hit and (2026-11-22, as Paper B's history rows) their layers, e.g.
    "default 4b/g128/gptq; 16b: gdn_ba L0-11 x12; 8b/g128/gptq: v L40-47 x8, gdn_ba L0-5 x6"."""
    if not isinstance(config, dict):
        return None
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(list))
    for k, v in (config.get("groups") or {}).items():
        layer, _, kind = str(k).partition(".")
        if not kind:
            layer, kind = "", layer
        by[_setting(v or {})][kind].append(int(layer) if layer.lstrip("-").isdigit() else None)
    parts = ["default " + _setting(config.get("default") or {})]
    for st, kinds in sorted(by.items(), key=lambda kv: -sum(len(x) for x in kv[1].values())):
        top = sorted(kinds.items(), key=lambda kv: -len(kv[1]))[:4]
        cells = []
        for m, ls in top:
            lay = [x for x in ls if x is not None]
            cells.append(f"{m} " + (f"L{_layer_ranges(lay)} " if lay else "") + f"x{len(ls)}")
        parts.append(f"{st}: " + ", ".join(cells) + (f" (+{len(kinds) - 4} kinds)" if len(kinds) > 4 else ""))
    return "; ".join(parts)


def history_row(rec: dict) -> dict:
    """search_history summary of one candidates.jsonl record: what the config is (config_brief), its size and its KL
    (the KL reduction vs that generation's incumbent is in the row's gate statistics)."""
    art = rec.get("artifact") or {}
    fb = _eval_fb(rec.get("screen"))
    g = (fb if isinstance(fb, dict) else {}).get("kl_eval") or {}
    return {"config": config_brief(art.get("config")), "GiB": art.get("bytes_gib"),
            "avg_bits": None if art.get("linear_avg_bits") is None else round(float(art["linear_avg_bits"]), 2),
            "kl": g.get("kl"), "note": art.get("note") or None}


def kl_cell(g: dict | None) -> str:
    """One-line KL result: KL reduction vs the incumbent (or why it was not evaluated)."""
    if not isinstance(g, dict):
        return "-"
    if g.get("kl") is None or g.get("gain") is None:
        return str(g.get("reason") or "-").split(" (")[0][:80]
    return f"{g['gain']:+.6f}" + ("" if g.get("passed") else f" ({str(g.get('reason') or '')[:60]})")


def kl_table(history) -> str:
    """Candidate table: size, KL to the original model and the KL reduction vs the incumbent."""
    head = "| cid | thread | step | GiB | avg bits | KL to original | KL reduction vs incumbent |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for c in history:
        fb = _feedback(c) or {}
        art = c.artifact or {}
        g = fb.get("kl_eval") if isinstance(fb.get("kl_eval"), dict) else None
        cell = "error: " + c.error.splitlines()[0][:80] if c.error else kl_cell(g)
        lines.append(f"| {c.cid} | {c.thread} | {c.step} | {_fmt(art.get('bytes_gib'))} | "
                     f"{_fmt(art.get('linear_avg_bits'), 2)} | {_fmt((g or {}).get('kl'), 6)} | {cell} |")
    return "\n".join(lines)


def make_render_context(problem, cfg: dict, *, search: dict, archive: str = "", history_tool: bool = False):
    """Returns render_context(history, thread) -> str for rsicore.search.inner_search. search: the search shape
    {threads, steps} (edit_code.common.search_settings); history_tool: the editor has the search_history chat tool."""
    threads = int(search.get("threads", 1))
    steps = int(search.get("steps", 1))
    one_step = steps == 1
    first = bool(getattr(problem, "incumbent_is_original", False))   # loop.start bf16, nothing accepted yet
    goal = "Find the lowest-KL config within the budget" if first else "Beat the incumbent (lower KL)"

    def render(history, thread: int) -> str:
        step = sum(1 for c in history if c.thread == thread)
        budget = (f"Budget (fixed for every round): {problem.budget_gib:.3f} GiB packed. "
                  "The config your program returns gets one KL evaluation (KL to the original model on the fixed "
                  f"reference traces); this round: {threads} session(s) x {steps} step(s).")
        parts = [f"[thread={thread} step={step} task=main gen={problem.gen}]", budget]   # (marker parsed by mock._MARK)
        if one_step and step == 0 and threads > 1:
            src = "search_history and the archive of earlier rounds" if history_tool else "the archive of earlier rounds"
            parts.append(f"You are session {thread + 1} of {threads} in this round. The other sessions work "
                         "independently at the same time; you do not see their candidates and they do not see yours. "
                         f"Use {src} to avoid repeating what earlier rounds already tried, and try something "
                         "different from them.")
        if step == 0:
            info = problem.spec.summary()
            parts.append("Model: " + json.dumps({"num_layers": info["num_layers"], "layer_types": info["layer_types"],
                                                 "linear_params": info["linear_params"],
                                                 "params_by_group": info["params_by_group"],
                                                 "lm_head_tied": info["lm_head_tied"]}))
            inc = problem.incumbent
            if first:
                parts.append(f"{FIRST_ROUND}Its size: {inc.get('bytes_gib')} GiB (16 bits) vs the budget "
                             f"{problem.budget_gib:.3f} GiB. Every quantized config has a positive KL. Write the best "
                             "config you can within the budget: the lowest KL of this round becomes the first "
                             "incumbent.")
            elif inc:
                parts.append(f"Incumbent to beat (accepted in round {inc.get('gen')}, {inc.get('bytes_gib')} GiB, "
                             f"linear avg bits {inc.get('linear_avg_bits')}): KL to the original model "
                             f"{_fmt(inc.get('kl'), 6)} (per-trace quantiles {json.dumps(inc.get('kl_quantiles'))}); "
                             "config (compressed form): " + json.dumps(inc.get("config"))[:3000])
            parts.append("Profile summary (KL increase if only that group is RTN-quantized):\n"
                         + json.dumps(summarize_profile(problem.tables["profile"], top=8))[:4000])
            red = [f"{r['layer']}.{r['module']}" for r in problem.tables["activation"] if r.get("red_flag")]
            if red:
                parts.append(f"Activation red flags (heavy outliers, {len(red)}): " + ", ".join(red[:30]))
            if archive:                     # archive_text starts with its own header (ARCHIVE_HEAD)
                parts.append(archive)
        if history:
            parts.append("Candidates of this round so far (KL = mean token KL from the original model on the reference "
                         "traces, lower is better; KL reduction = the incumbent's KL minus the candidate's):\n"
                         + kl_table(history))
            ok = [c for c in history if c.score is not None]
            if ok:
                best = max(ok, key=lambda c: c.score)       # score = -KL
                parts.append(f"Lowest KL so far: {best.cid} ({-best.score:.6f}), config: "
                             + json.dumps((best.artifact or {}).get("config"))[:2500])
            mine = [c for c in history if c.thread == thread]
            if mine:
                last = mine[-1]
                if last.error:
                    parts.append(f"Your last program ({last.cid}) failed:\n{last.error[:2500]}")
                else:
                    parts.append(f"KL result of your last candidate {last.cid}:\n" + json.dumps(_feedback(last))[:3500])
        if history and steps > 1:
            parts.append(f"{goal}: improve this thread's best config using the KL results above, borrow "
                         "from other threads, or try a different idea. Return one ```python block with def main(api): "
                         "... return config.")
        elif not history:                   # nothing of this round is visible
            parts.append(f"{goal}: try an idea that differs from what earlier rounds tried. "
                         "Return one ```python block with def main(api): ... return config.")
        else:
            parts.append(f"{goal}: improve on the best candidate (yours or another thread's) or try a "
                         "different idea. Return one ```python block with def main(api): ... return config.")
        return "\n\n".join(parts)

    return render


ARCHIVE_HEAD = ("Archive of earlier rounds: EVERY candidate tried, one row each, failures included. config = the config "
                "in one line (default setting; overrides grouped by setting: module kind, layers L, count x), GiB / "
                "avg_bits = its size, kl = its KL to the original model on the fixed reference traces (the same traces "
                "in every round, so KLs are comparable across rounds), error = why it was not built or evaluated, "
                "note = its own account of what it tried and why.")
ARCHIVE_READ = ("Full program or feedback (with the full config) of any candidate above: read_record(cid, 'program' | "
                "'feedback'); search_history lists them with more statistics. Do not repeat what failed: try something "
                "different or fix the reason it failed.")


def _num(x, nd: int = 4):
    try:
        return None if x is None else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def archive_row(c, gate: dict | None = None) -> dict:
    """One archive row of an earlier round's candidate (rsicore.search.Candidate): what the config is, its size, its
    KL (gate = its run.gate_outcomes row), the first line of its error, its note."""
    art = c.artifact or {}
    row = {"cid": c.cid}
    if c.error:
        row["error"] = error_line(c.error)
    elif art:
        row["config"] = config_brief(art.get("config"))
        row["GiB"] = art.get("bytes_gib")
        if art.get("linear_avg_bits") is not None:
            row["avg_bits"] = round(float(art["linear_avg_bits"]), 2)
    if isinstance(gate, dict):
        if gate.get("kl") is not None:
            row["kl"] = _num(gate["kl"], 6)
        elif gate.get("reason") and not c.error:
            row["error"] = str(gate["reason"])[:120]
    if art.get("note"):
        row["note"] = art["note"]
    return row


def archive_text(entries: list[dict], *, read_tools: bool = True, max_chars: int = 30000) -> str:
    """The experience channel. entries: one per earlier round {gen, accepted, cid, reason, incumbent_kl, n, best:
    [(cid, KL)], first_quantization, rows: [archive_row], configs: [(cid, full config)] (quoted only without
    read_tools)}. max_chars: the rows of the oldest rounds are dropped first to fit."""
    blocks = []
    for e in entries:
        acc = f"new incumbent {e.get('cid')}" if e.get("accepted") else \
            f"incumbent kept ({str(e.get('reason'))[:300]})"
        best = ", ".join(f"{cid}={v:.6f}" for cid, v in (e.get("best") or []) if v is not None)
        ik = e.get("incumbent_kl")
        head = [f"- round {e['gen']}: {acc}; {e.get('n', 0)} candidates"
                + (f"; incumbent KL {ik:.6f}" if isinstance(ik, (int, float)) else "")
                + (f"; lowest KLs {best}" if best else "")]
        if e.get("first_quantization"):      # loop.start bf16: the incumbent was the BF16 original
            head.append("  first-quantization round: the incumbent was the original BF16 model (KL 0), so the lowest "
                        "KL within the budget was accepted")
        rows = ["  " + json.dumps(r, ensure_ascii=False, default=str) for r in e.get("rows") or []]
        if not read_tools:
            rows += [f"  full config of {cid}: {json.dumps(cfg)[:2500]}" for cid, cfg in e.get("configs") or []]
        blocks.append((e["gen"], head, rows))
    if not blocks:
        return ""
    tail = [ARCHIVE_READ] if read_tools else []

    def text(drop: int) -> str:
        out = [ARCHIVE_HEAD]
        for i, (gen, head, rows) in enumerate(blocks):
            out += head + (rows if i >= drop else
                           [f"  ({len(rows)} rows omitted for length" +
                            (f"; search_history(gen={gen}) lists them)" if read_tools else ")")])
        return "\n".join(out + tail)
    drop = 0
    while drop < len(blocks) - 1 and len(text(drop)) > max_chars:
        drop += 1
    return text(drop)
