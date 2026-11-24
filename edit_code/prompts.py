"""Editor system prompt and per-step context rendering for Paper A.

The editor (the original model M_0, frozen; or the incumbent with loop.editor=current) must answer with ONE ```python block defining
`def main(api): ... return config`. render_context(history, thread) shows: the budget and the search shape, a compact
diagnostics summary and the archive of earlier generations (experience channel; switchable) on the first step, the
incumbent's (latest model's) answers to this round's screen items, and this round's candidates with their gate
results.

The flow is hard-wired (2026-11-23; formerly the base.yaml defaults search.screen_candidates false,
gate_every_candidate true, program_evaluate_calls 0): no api.evaluate anywhere; the editor sees the incumbent's answers
to this round's screen items (summary in the context, full data through screen_results / screen_output); every
returned config gets ONE gate evaluation and later steps / rounds see its gate statistics (candidate table, own
feedback, search_history, archive); the acceptance text names the end-of-round rule (largest passing gain). With
steps 1 each thread is told that it is session k of N working independently this round and should use
search_history / the archive to try something different from earlier rounds. Thread / step / tool-round numbers
always come from the config. editor.explore.persistent: the protocol says that run_python variables persist within a
proposal.
"""
from __future__ import annotations

import json

from .common import error_line
from .diagnostics import summarize_profile

SYSTEM_PROMPT = """You are quantizing a large language model to a fixed memory budget. The model being quantized is a copy of YOURSELF: every config is applied to the original BF16 weights. The budget is the same in every round; each round tries to beat the best quantization accepted so far (the incumbent), which you can see together with the experience of earlier rounds.

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
- profile_table() gives, per group key and bit width, the whole-model KL increase when only that group is RTN-quantized (forward pass on the original model, 64 texts). hessian_table() gives, per group, quantizer (rtn / gptq / awq) and bit width, the Hessian-weighted output error of that group alone (every calibration token, no forward pass): lower noise, local. Sensitivities are roughly additive at >= 3.5 average bits and underestimate damage at 2-3 bits; only a real evaluation measures the real thing.
- Analysis (free): group_error(config) shows where a config's quantization error concentrates.
- The incumbent answered this round's screen items ({screen}, held-back TRAINING items; thinking mode): a summary is in your context and api.screen_results / api.screen_output give every item with its prompt, gold answer, extracted answers and the full outputs incl. the thinking, so you can see what the incumbent gets wrong, gets right only sometimes, or truncates. Your returned config is NOT scored on these items: {gate_desc} There is no api.evaluate.
- {acceptance}
- In your program `api` exists only as the argument of main(api) (it is not a global there): pass it to helper functions explicitly, e.g. def pick(api, budget): ...
- Time: your program is killed after {timeout_s} s of wall clock, every tool call included (group_error / hessian_table on settings that are not cached yet quantize on the GPU and take real time). A killed program is a lost candidate.
- note (optional, at most 600 characters, plain text, the "note" key of the returned config): which groups you changed and WHY those (the evidence that picked them: a profile_table / hessian_table / group_error number, the incumbent's failures), then what you tried and rejected, with the numbers that decided it (e.g. "8 bit for down 0-3 and v 40-47: largest gptq 4-bit herr_rel in hessian_table; paid for with gate_up 20-35 at 3 bit; rejected: awq on gdn_qkv, group_error total +9% vs gptq"). The history already lists the config; the note is for the reasons. Every later session sees it, so a good note keeps them from repeating your failed ideas.

Answer with a short plan, then exactly one ```python code block containing the full program."""


PERSIST_NONE = "Nothing persists between run_python calls or into your final program: it must be self-contained."
PERSIST_SESSION = ("Variables, imports and functions persist between your run_python calls while you write this program "
                   "(one session; a timeout or crash restarts it and loses them), but not into your final program: it "
                   "must be self-contained.")

FINAL_RULE = "Answer with a short plan, then exactly one ```python code block containing the full program."

# editor.tool_rounds > 0 (2026-11-21): the editor may call chat tools before answering; replaces FINAL_RULE
TOOL_PROTOCOL = """Before you answer you may call functions, for up to {rounds} rounds (one round = one reply with function calls; at most {calls} calls in all):
- run_python(code) runs exploratory code in the same sandbox with the same `api` (keyword arguments as above), so you can inspect and filter the tables and check configs before you commit to a program.{group_error}. print() what you need or set `result = ...`; output is cut to a few thousand characters, so filter big tables in the code.
- read_record(cid, part) shows the full program ("program") or the full feedback ("feedback", with the full config) of a candidate in the table{history_read}.{history}
{persist}
When you are done, reply WITHOUT a function call: a short plan, then exactly one ```python code block containing the full program."""


def _tool_rounds(cfg: dict) -> tuple[int, bool, bool, int]:
    """(editor.tool_rounds, editor.explore.gpu_tools, editor.explore.persistent, editor.max_tool_calls) with the
    rsicore.editor defaults (the round count is always the configured one, e.g. 20 in the simple profile)."""
    try:
        from rsicore.editor import editor_settings
        s = editor_settings(cfg)
        ex = s.get("explore") or {}
    except ImportError:                      # pragma: no cover - rsicore not on sys.path
        s = cfg.get("editor") or {}
        s = {"tool_rounds": s.get("tool_rounds", 50), "max_tool_calls": s.get("max_tool_calls", 100)}
        ex = (cfg.get("editor") or {}).get("explore") or {}
    if "persistent" not in ex:               # rsicore.editor_settings may not know the key yet
        ex = dict(ex, persistent=((cfg.get("editor") or {}).get("explore") or {}).get("persistent", False))
    return (int(s["tool_rounds"] or 0), bool(ex.get("gpu_tools")), bool(ex.get("persistent")),
            int(s.get("max_tool_calls") or 0))


GATE_DESC_ACC = ("it gets ONE evaluation on the fixed gate set ({gate}), paired per item with the incumbent; its gate "
                 "result (accuracy, paired gain, CI, passed or not and why, ids of broken / fixed gate items, never their "
                 "content) is shown to later sessions and rounds.")
GATE_DESC_KL = ("it gets ONE gate evaluation that samples nothing: the model it quantizes reads {gate} (fixed answers "
                "written once by the original BF16 model to held-back gate items, never shown to you) and the gate "
                "measures the token-level KL divergence KL(original || quantized) on every position (top-{k} "
                "approximation), averaged per trace; lower is closer to the original. Paired per trace with the "
                "incumbent; gain = the incumbent's KL minus yours (positive = closer to the original). Its gate result "
                "(KL, the incumbent's KL, gain, CI, passed or not and why) is shown to later sessions and rounds.")


def is_kl_gate(cfg: dict) -> bool:
    return ((cfg or {}).get("gate") or {}).get("metric", "accuracy") == "kl"


def acceptance_text(cfg: dict) -> str:
    """The acceptance rule as configured (2026-11-21; gate.min_delta / require_significant), not a fixed text: every
    candidate is gated; the largest passing gain is accepted."""
    g = cfg.get("gate") or {}
    md = float(g.get("min_delta", 0.0))
    if is_kl_gate(cfg):                  # (2026-11-24) gain = KL reduction vs the incumbent
        rule = (f"mean KL reduction >= {md:g}" if md > 0 else "a positive mean KL reduction")
        rule += (", a statistically significant reduction" if g.get("require_significant", True)
                 else "; there is no significance test")
        return ("Acceptance (end of the round): a config passes its gate evaluation when size <= budget and its KL to "
                f"the original model is lower than the incumbent's on the fixed reference traces (paired per trace: "
                f"{rule}); among the passing configs of the round the one with the LARGEST KL reduction becomes the new "
                "incumbent. None passes: the incumbent stays.")
    rule = (f"mean gain >= {md:g} ({md * 100:g} accuracy points)" if md > 0
            else "a positive mean gain")       # (2026-11-21) threshold 0: any strictly positive gain
    rule += (", a statistically significant gain and no significant loss on any domain"
             if g.get("require_significant", True) else "; there is no significance test")
    return ("Acceptance (end of the round): a config passes its gate evaluation when size <= budget and it beats "
            f"the incumbent on the fixed gate set (paired per item: {rule}); among the passing configs of the "
            "round the one with the LARGEST gain becomes the new incumbent. None passes: the incumbent stays.")


def _sizes(problem) -> dict:
    """Screen / gate sizes for the prompt texts: items and samples per item (decoding n; 1 = greedy)."""
    dec = getattr(problem, "decoding", None) or {}
    out = {"screen_k": int((dec.get("screen") or {}).get("n") or 1), "gate_k": int((dec.get("gate") or {}).get("n") or 1)}
    try:
        out["screen_n"] = sum(len(problem.data[ds]["screen"]) for ds in problem.datasets)
    except (AttributeError, KeyError, TypeError):
        out["screen_n"] = None
    try:
        out["gate_n"] = sum(len(problem._ids(ds, "gate")) for ds in problem.gate_datasets)
    except Exception:  # noqa: BLE001 - informative only
        out["gate_n"] = None
    if getattr(problem, "kl_gate", False):     # KL gate: the reference traces, one each, nothing sampled
        try:
            out["gate_n"], out["gate_k"] = len(problem.kl_items()), 1
        except Exception:  # noqa: BLE001 - informative only (traces not written yet)
            out["gate_k"] = 1
    return out


def _gate_desc(cfg: dict, sz: dict) -> str:
    if is_kl_gate(cfg):
        n = sz.get("gate_n")
        return GATE_DESC_KL.format(gate=(f"{n} reference traces" if n else "the reference traces"),
                                   k=int(((cfg.get("gate") or {}).get("kl") or {}).get("top_k", 20)))
    return GATE_DESC_ACC.format(gate=_items(sz["gate_n"], sz["gate_k"]))


def _items(n, k) -> str:
    return (f"{n} items" if n else "the items") + (f" x {k} sampled answers" if k > 1 else "")


HISTORY_READ = " or of any earlier candidate that search_history lists"
HISTORY_LINE = ("\n- search_history(gen, sort, offset) lists every candidate already tried (this generation's and the "
                "earlier generations') with a one-line description of its config, its size, scores and gate "
                "outcome; same_as marks a config identical to an earlier one. Check it before you write a program, "
                "so you do not repeat a config that was already tried.")


def system_prompt(problem, tool_doc: str, cfg: dict, history: bool = False) -> str:
    sp = problem.space
    sz = _sizes(problem)
    text = SYSTEM_PROMPT.format(tools=tool_doc, bits=sp["bits"], group_sizes=sp["group_sizes"],
                                quantizers=sp["quantizers"], timeout_s=int((cfg.get("sandbox") or {}).get("timeout_s", 1800)),
                                screen=_items(sz["screen_n"], sz["screen_k"]), gate_desc=_gate_desc(cfg, sz),
                                acceptance=acceptance_text(cfg))
    rounds, gpu_tools, persistent, calls = _tool_rounds(cfg)
    if rounds > 0:                           # tool_rounds 0: the single-call prompt
        ge = ("" if gpu_tools or not (cfg.get("tools") or {}).get("group_error", True)
              else "; group_error there only accepts settings already cached (those of hessian_table)")
        if text.endswith(FINAL_RULE):
            text = text[:-len(FINAL_RULE)]
        text += TOOL_PROTOCOL.format(rounds=rounds, calls=calls, group_error=ge, history_read=HISTORY_READ if history else "",
                                     history=HISTORY_LINE if history else "",
                                     persist=PERSIST_SESSION if persistent else PERSIST_NONE)
    return text


def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _feedback(c) -> dict | None:
    """A candidate's feedback: its gate feedback (run.gate_feedback), kept in the candidate's screen slot."""
    fb = (c.screen or {}).get("feedback")
    if isinstance(fb, dict) and "feedback" in fb and "probes" not in fb:
        fb = fb["feedback"]
    return fb


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
    for k in ("embed", "lm_head"):
        v = config.get(k)
        if isinstance(v, dict) and v:
            parts.append(f"{k} {_setting(v)}")
    return "; ".join(parts)


def history_row(rec: dict) -> dict:
    """search_history summary of one candidates.jsonl record (2026-11-21): what the config is (config_brief),
    its size and per-domain gate delta vs that generation's incumbent (* = significant)."""
    art = rec.get("artifact") or {}
    fb = (rec.get("screen") or {}).get("feedback") or {}
    if isinstance(fb, dict) and "feedback" in fb and "probes" not in fb:
        fb = fb["feedback"]
    fb = fb if isinstance(fb, dict) else {}
    deltas = {}
    for d, p in (fb.get("probes") or {}).items():
        if isinstance(p, dict) and p.get("vs_incumbent") is not None:
            deltas[d] = f"{p['vs_incumbent']:+.3f}" + ("*" if p.get("significant") else "")
    return {"config": config_brief(art.get("config")), "GiB": art.get("bytes_gib"),
            "avg_bits": None if art.get("linear_avg_bits") is None else round(float(art["linear_avg_bits"]), 2),
            "vs_incumbent": deltas or None,
            "note": art.get("note") or None}


def gate_cell(g: dict | None) -> str:
    """One-line gate result (2026-11-21, gate per candidate): passed / failed, gain [CI], broken / fixed counts."""
    if not isinstance(g, dict):
        return "-"
    ci = g.get("ci")
    ci = f" [{ci[0]:+.4f}, {ci[1]:+.4f}]" if isinstance(ci, list) and len(ci) == 2 and None not in ci else ""
    gain = "-" if g.get("gain") is None else f"{g['gain']:+.4f}"
    if g.get("metric") == "kl":           # KL gate: gain = KL reduction, no broken / fixed items
        out = f"{'PASSED' if g.get('passed') else 'failed'} KL reduction {gain}{ci}"
    else:
        out = f"{'PASSED' if g.get('passed') else 'failed'} gain {gain}{ci}, broken {g.get('broken')} / fixed {g.get('fixed')}"
    if not g.get("passed") and g.get("reason"):
        out += f" ({str(g['reason']).split(' (')[0][:80]})"
    return out


def gate_table(history, kl: bool = False) -> str:
    """Candidate table: size and the gate result (kl: the KL gate's KL and KL reduction)."""
    head = ("| cid | thread | step | GiB | avg bits | gate KL to original | KL reduction vs incumbent [95% CI] |" if kl
            else "| cid | thread | step | GiB | avg bits | gate accuracy | gain vs incumbent [95% CI] / broken / fixed |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for c in history:
        fb = _feedback(c) or {}
        art = c.artifact or {}
        g = fb.get("gate") if isinstance(fb.get("gate"), dict) else None
        cell = "error: " + c.error.splitlines()[0][:80] if c.error else gate_cell(g)
        lines.append(f"| {c.cid} | {c.thread} | {c.step} | {_fmt(art.get('bytes_gib'))} | "
                     f"{_fmt(art.get('linear_avg_bits'), 2)} | "
                     f"{_fmt((g or {}).get('kl' if kl else 'accuracy'), 6 if kl else 4)} | {cell} |")
    return "\n".join(lines)


def screen_summary_text(summary: dict, chars: int = 3500) -> str:
    """The incumbent's screen run for the context (rsicore.screen_view.summary)."""
    if not summary or not summary.get("n_items"):
        return "The incumbent's screen results are not available."
    k = summary.get("samples_per_item")
    head = (f"The incumbent on this round's {summary['n_items']} screen items ({k} sampled answers each, thinking): "
            f"accuracy avg@{k} {summary.get('accuracy')}; items by number of correct samples "
            f"{json.dumps(summary.get('correct_counts'))}; truncated samples {summary.get('truncation_rate')}"
            f" (cut while thinking: {summary.get('truncated_thinking')}). Failing examples (all of them: "
            "api.screen_results(filter='wrong' | 'unstable'); full outputs: api.screen_output(id, sample)):\n")
    return head + json.dumps(summary.get("failing") or [], ensure_ascii=False)[:chars]


def make_render_context(problem, cfg: dict, *, archive: str = "", search: dict | None = None,
                        history_tool: bool = False):
    """Returns render_context(history, thread) -> str for rsicore.search.inner_search. search: the search shape
    {threads, steps} (edit_code.common.search_settings; None = cfg["search"]); history_tool: the editor has the
    search_history chat tool."""
    n_examples = cfg["feedback"].get("examples", 2)
    ss = search if search is not None else (cfg.get("search") or {})
    threads = int(ss.get("threads", 1))
    steps = int(ss.get("steps", 1))
    one_step = steps == 1
    summary_cache = {}

    def screen_summary() -> dict:
        if "s" not in summary_cache:           # the incumbent's screen run is fixed for the whole search
            try:
                summary_cache["s"] = problem.screen_summary(examples=max(3, int(n_examples)))
            except Exception as e:  # noqa: BLE001 - informative only
                summary_cache["s"] = {"n_items": 0, "error": str(e)[:200]}
        return summary_cache["s"]

    kl = bool(getattr(problem, "kl_gate", False))

    def render(history, thread: int) -> str:
        step = sum(1 for c in history if c.thread == thread)
        budget = (f"Budget (fixed for every round): {problem.budget_gib:.3f} GiB packed. "
                  "The config your program returns gets one gate evaluation (paired vs the incumbent); "
                  f"this round: {threads} session(s) x {steps} step(s).")
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
            if inc:
                gs = inc.get("gate_scores") or {}
                what = ("its gate KL to the original model (per dataset, mean over the traces): "
                        + json.dumps({d: (None if v is None else round(-v, 6)) for d, v in gs.items()})
                        if kl else "its gate accuracy: " + json.dumps(gs))
                parts.append(f"Incumbent to beat (accepted in round {inc.get('gen')}, {inc.get('bytes_gib')} GiB, "
                             f"linear avg bits {inc.get('linear_avg_bits')}); {what}; config (compressed form): "
                             + json.dumps(inc.get("config"))[:3000])
            parts.append("Profile summary (KL increase if only that group is RTN-quantized):\n"
                         + json.dumps(summarize_profile(problem.tables["profile"], top=8))[:4000])
            red = [f"{r['layer']}.{r['module']}" for r in problem.tables["activation"] if r.get("red_flag")]
            if red:
                parts.append(f"Activation red flags (heavy outliers, {len(red)}): " + ", ".join(red[:30]))
            if archive:                     # archive_text starts with its own header (ARCHIVE_HEAD)
                parts.append(archive)
        parts.append(screen_summary_text(screen_summary()))
        if history:
            if kl:
                parts.append("Candidates of this round so far, with their gate results (KL = mean token KL from the "
                             "original model on the reference traces, lower is better; KL reduction = the incumbent's KL "
                             "minus the candidate's, paired per trace):\n" + gate_table(history, kl=True))
            else:
                parts.append("Candidates of this round so far, with their gate results (gain = paired accuracy gain over "
                             "the incumbent on the gate set; broken / fixed = gate items the incumbent got right and the "
                             "candidate wrong / the reverse; read_record(cid, 'feedback') has the ids):\n"
                             + gate_table(history))
            ok = [c for c in history if c.score is not None]
            if ok:
                best = max(ok, key=lambda c: c.score)
                parts.append(f"Largest gate gain so far: {best.cid} ({best.score:+.4f}), config: "
                             + json.dumps((best.artifact or {}).get("config"))[:2500])
            mine = [c for c in history if c.thread == thread]
            if mine:
                last = mine[-1]
                if last.error:
                    parts.append(f"Your last program ({last.cid}) failed:\n{last.error[:2500]}")
                else:
                    parts.append(f"Gate result of your last candidate {last.cid}:\n" + json.dumps(_feedback(last))[:3500])
        if history and steps > 1:
            parts.append("Beat the incumbent: improve this thread's best config using the gate results above, borrow "
                         "from other threads, or try a different idea. Return one ```python block with def main(api): "
                         "... return config.")
        elif not history:                   # nothing of this round is visible (2026-11-21)
            parts.append("Beat the incumbent: try an idea that differs from what earlier rounds tried. "
                         "Return one ```python block with def main(api): ... return config.")
        else:
            parts.append("Beat the incumbent: improve on the best candidate (yours or another thread's) or try a "
                         "different idea. Return one ```python block with def main(api): ... return config.")
        return "\n\n".join(parts)

    return render


ARCHIVE_HEAD = ("Archive of earlier rounds: EVERY candidate tried, one row each, failures included (2026-11-22, as in "
                "Paper B). config = the config in one line (default setting; overrides grouped by setting: module kind, "
                "layers L, count x), GiB / avg_bits = its size, gate = its gate result (gain = paired gain in gate "
                "accuracy, or the KL reduction under the KL gate, vs that round's incumbent, ci = its 95% interval, reason when it failed), error = why it was not built, "
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
    """One archive row of an earlier round's candidate (rsicore.search.Candidate; 2026-11-22): what the config is,
    its size, its gate result (gate = run.gate_outcomes row) or score, the first line of its error, its note."""
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
        g = {"passed": bool(gate.get("passed"))}
        if gate.get("gain") is not None:
            g["gain"] = _num(gate["gain"])
        ci = gate.get("ci")
        if isinstance(ci, (list, tuple)):
            g["ci"] = [_num(x) for x in ci]
        elif isinstance(ci, dict):           # several gate datasets: {dataset: [lo, hi]}
            g["ci"] = {ds: [_num(x) for x in v] for ds, v in ci.items() if isinstance(v, (list, tuple))}
        if not g["passed"] and gate.get("reason") and not c.error:
            g["reason"] = str(gate["reason"])[:120]
        row["gate"] = g
    elif c.score is not None:
        row["score"] = _num(c.score)
    if art.get("note"):
        row["note"] = art["note"]
    return row


def archive_text(entries: list[dict], *, read_tools: bool = True, max_chars: int = 30000) -> str:
    """The experience channel (2026-11-22, structured as Paper B's archive). entries: one per earlier round {gen,
    accepted, cid, gate_reason, n, best: [(cid, paired gate gain)], reference_screen (flat summary), incumbent_score,
    rows: [archive_row], configs: [(cid, full config)] (quoted only without read_tools)}. max_chars: the rows of the
    oldest rounds are dropped first to fit. Gate gains are all on the same fixed gate set."""
    blocks = []
    for e in entries:
        acc = f"new incumbent {e.get('cid')}" if e.get("accepted") else \
            f"incumbent kept (gate: {str(e.get('gate_reason'))[:300]})"
        best = ", ".join(f"{cid}={v:+.4f}" for cid, v in (e.get("best") or []) if v is not None)
        head = [f"- round {e['gen']}: {acc}; {e.get('n', 0)} candidates"
                + (f"; largest paired gate gains {best}" if best else "")]
        rs = e.get("reference_screen")
        if rs and (rs.get("n_items") or rs.get("accuracy") is not None):
            size = (f" over {rs['n_items']} items x {rs.get('samples_per_item')} samples" if rs.get("n_items") else "")
            head.append(f"  incumbent on that round's screen items: accuracy {rs.get('accuracy')}{size}, items by correct "
                        f"samples {json.dumps(rs.get('correct_counts'))}, truncated {rs.get('truncation_rate')} "
                        f"(screen_results(gen={e['gen']}))")
        elif e.get("incumbent_score"):
            head.append(f"  incumbent score on that round's items {e.get('incumbent_score')}")
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
