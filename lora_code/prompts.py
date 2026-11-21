"""Editor system prompt and per-step context (RSI_LOOP_DESIGN §1.2, §3.1, §3.4).

The editor is the current model M_t. Each step it writes ONE program `def main(api): ...` that builds a
LoRA for ITSELF through the tools and returns {"lora": <handle>} ({"lora": None} keeps the model
unchanged). The rules, goal and tool list depend on method.mode (MODE_TEXT): edit = edit the raw LoRA
trained once (generation 0) with the fixed recipe; scratch = build from zero; weights = edit the dense weights
directly with structured, any-rank operations and return {"edit": <handle>}. The editor never trains and has
no gradient tools in any mode. The context shows the task, the fixed diagnostics (first step of a thread), every candidate
so far (all threads), this thread's best program and its latest feedback.
"""
from __future__ import annotations

import json

from .tools import program_evaluate_cap, tools_for_mode

SYSTEM_PROMPT = """You are {who}. You improve a language model by writing Python programs that {action} \
({what}). {whose}

Each turn, reply with a short plan and exactly ONE ```python code block defining

    def main(api):
        ...
        {return_line}

Rules
- Call tools only as api.<tool>(keyword=value, ...). Results are JSON (dicts / lists / numbers / strings).
  {objects} stay on the server and are referred to by string handles.
- In your program `api` exists only as the argument of main(api) (it is NOT a global there, unlike in run_python):
  pass it to every helper function explicitly, e.g. def build(api, module): ...
- Data access: activation tools accept ONLY the ids returned by api.train_ids() (the target task's
  training items of this generation). Signal, gate and held-out items are never readable; their scores come
  only from api.evaluate and from the feedback you see after each program.
- {limits}
- Time: a program is killed after {timeout_s} s of wall clock, every tool call included (evaluate, acts,
  train_loss and regmean_preserve run the model and take real time). Screen a few constructions, not dozens,
  and keep time to return; api.budget() shows program_seconds_left. A killed program is a lost candidate.
- Budgets for this generation, shared by all threads: {evaluate} evaluate calls (each candidate you return is
  evaluated automatically and costs one; every api.evaluate inside a program also costs one{per_program};
  identical {unit}s are evaluated once). When they run out, later candidates are not scored at all: look at
  the "left" numbers in the latest message and in api.budget() before calling api.evaluate.
  {gpu_seconds:.0f} GPU-seconds of forward passes (acts / regmean_preserve / train_loss). There is no training
  and no gradient tool. {goal}
- {gate}
- Module names: use only names returned by api.modules() (rows with module, layer, layer_type, out, in). The
  layer types are irregular (the diagnostics table lists each layer's type): MLP 'layers.12.mlp.down_proj'
  exists everywhere, 'self_attn.*' only in full-attention layers, 'linear_attn.*' only in Gated-DeltaNet layers.
  Never build a name from a pattern without checking it is in api.modules().
- Vector references for rank1 / from_direction live on one side of a module's W [out, in]: u and direction are
  OUTPUT-side vectors (length out: "S3.u0" of an svd handle, "A2.out_mean" of an acts handle); v and key are
  INPUT-side vectors (length in: "S3.v1", "A2.in_mean", "A2.pc0"). Take them from the SAME module's svd / acts
  (e.g. down_proj: in = the MLP width, out = the hidden size).
- Robust code: tool results can be empty (e.g. api.diagnostics for a module without a row) and one uncaught
  error fails the whole candidate. Check lengths before indexing and the structure before unpacking; wrap
  optional steps in try/except and fall back to what you already built.
- {strategies}

Tools
{tools}
"""

_LORA_TEXT = dict(
    what="a low-rank weight update dW = B A per linear module",
    return_line='return {"lora": handle}      # or {"lora": None} to leave the model unchanged',
    objects="LoRAs, SVDs and activation statistics",
    limits="Limits checked on every LoRA: rank <= {rank_cap} per module, ||dW||_F / ||W0||_F <= {tau} per module,\n"
           "  finite values, existing modules only.",
    unit="LoRA")

MODE_TEXT = {
    "edit": dict(
        **_LORA_TEXT,
        action="edit a LoRA that was trained for it",
        goal="One LoRA is trained once (generation 0) on the target task with a fixed recipe and then improved by "
             "you in every generation (handle 'RAW', api.raw_lora(): the LoRA you start from). Your LoRA replaces RAW (the next model is the original "
             "weights + your LoRA): raise the target score and lose less on the other domains / token-KL if "
             "they are measured.",
        strategies="Useful edits of RAW: per-module / per-layer scaling or dropping (api.scale with modules, "
                   "api.drop),\n  truncate_rank, intruder_scale, relchange_prune, regmean_preserve, orthogonalize "
                   "against earlier adapters\n  (api.stack()), singular-direction scaling or rank-1 / activation-"
                   "direction additions combined with RAW,\n  interpolating with earlier candidates (api.candidate(cid)). "
                   "Returning RAW unchanged is allowed but is no improvement."),
    "scratch": dict(
        **_LORA_TEXT,
        action="construct a LoRA from scratch without any training",
        goal="Build the LoRA from zero "
             "with the weight-space and activation tools. The gate compares your LoRA with the current model: "
             "the target score must rise and the retention domains must not get worse.",
        strategies="Useful constructions: singular-direction scaling of the current weights (api.svd + "
                   "api.scale_singular),\n  activation directions on train items (api.acts + api.from_direction), "
                   "rank-1 edits, combining / scaling\n  constructions, protecting retention with regmean_preserve, "
                   "orthogonalize or a global scale. Screen constructions cheaply with api.train_loss (answer NLL on "
                   "train items, no\n  evaluate) before spending evaluate calls; api.compare(cid) shows which items a "
                   "candidate broke or fixed vs the current model."),
    "weights": dict(
        action="edit the model's weights directly with structured operations of any rank, without any training",
        what="an ordered list of operations on the dense weight matrices W [out, in] of the linear modules: "
             "channel / module scaling, dense additions of LoRA-built updates, moves toward the original weights",
        return_line='return {"edit": handle}      # or {"edit": None} to leave the model unchanged',
        objects="Weight edits, LoRAs, SVDs and activation statistics",
        limits="Limits checked on every edit: ||W'-W||_F / ||W||_F <= {tau_w} per module (W = your current weights,\n"
               "  W' = after the edit){max_modules_text}, finite values, existing modules only. There is no rank limit.",
        unit="edit",
        goal="Start from api.new_edit() and append operations; they apply in order to the CURRENT weights. The gate "
             "compares the edited model with the current model: the target score must rise and the retention domains "
             "must not get worse.",
        strategies="Useful edits: row / column scaling of output / input channels (api.w_scale_rows / "
                   "api.w_scale_cols) found with\n  api.acts, api.svd or api.diagnostics; singular-direction or "
                   "activation-direction updates built with the LoRA tools\n  (api.svd + api.scale_singular, api.acts + "
                   "api.from_direction, rank1) and added as a dense dW with api.w_add_lora; whole-module\n  scaling "
                   "near 1 (api.w_scale); moving some modules back toward the original weights (api.w_toward_original, "
                   "which\n  also undoes earlier merges there). Check the limits with api.inspect_edit and screen edits "
                   "cheaply with\n  api.train_loss(edit=...) (answer NLL on train items, no evaluate) before spending "
                   "evaluate calls;\n  api.compare(cid) shows which items a candidate broke or fixed vs the current model."),
}


def _points(x: float) -> str:
    p = round(float(x) * 100, 4)
    return f"{p:g} accuracy point" + ("" if p == 1 else "s")


def gate_text(cfg: dict, *, kind: str = "parent", gate_n: int | None = None, retention: bool = True,
              e_task: bool = False) -> str:
    """The gate rule as configured (2026-11-21; gate.min_gain / require_significant / epsilon), not a fixed text."""
    if e_task:
        return ("Scoring (not visible to you): your LoRA is scored on hidden items of the target task against the "
                "unmodified base model.")
    g = cfg.get("gate") or {}
    if kind == "raw":
        eps = float(g.get("epsilon", 0.01))
        rule = (f"the target accuracy must not be worse than the raw LoRA's by more than {eps:g} "
                f"({_points(eps)})")
    else:
        mg = float(g.get("min_gain", 0.01))
        items = f", i.e. about {max(1, round(mg * gate_n))} more correct items out of {gate_n}" if gate_n else ""
        rule = (f"item by item against the current model, the mean gain in target accuracy must be at least {mg:g} "
                f"({_points(mg)}{items})")
        rule += (" and statistically significant (paired 95% CI lower bound above 0)"
                 if g.get("require_significant", True) else "; there is no significance test")
    ret = (" Retention domains must not drop by more than the allowed tolerance." if retention else "")
    return (f"Gate (after the search, not visible to you): on a fixed hidden item set, {rule}.{ret} Only candidates "
            "whose score beats the current model on your feedback items are submitted; the feedback items are a "
            "different sample, so aim clearly above the margin.")


def system_prompt(cfg: dict, *, budget: dict, e_task: bool = False, mode: str = "scratch", retention: bool = True,
                  gate_kind: str = "parent", gate_n: int | None = None) -> str:
    who = ("an AI model asked to write a LoRA for a FIXED base model on a tool-calling / coding task"
           if e_task else "the model being edited: the LoRA you write will be merged into YOUR OWN weights "
                          "if it passes a fixed gate, and the merged model becomes the next editor")
    whose = ("The model you edit is the original base model, not yourself." if e_task else
             "You are editing yourself.")
    if mode == "weights" and e_task:
        who = "an AI model asked to edit the weights of a FIXED base model on a tool-calling / coding task"
    elif mode == "weights":
        who = ("the model being edited: the weight edit you write will be applied to YOUR OWN weights if it passes a "
               "fixed gate, and the edited model becomes the next editor")
    tools = "\n".join(f"- {d}" for d in tools_for_mode(mode, retention=retention, cfg=cfg).values())
    wc = cfg.get("weights", {}) or {}
    mm = wc.get("max_modules")
    lim = dict(rank_cap=(cfg.get("lora", {}) or {}).get("rank_cap", 64),
               tau=(cfg.get("lora", {}) or {}).get("norm_cap_tau", 0.1),
               tau_w=wc.get("norm_cap_tau") if wc.get("norm_cap_tau") is not None else 0.05,
               max_modules_text=f", at most {int(mm)} modules per edit" if mm is not None else "")
    text = dict(MODE_TEXT[mode])
    text["limits"] = text["limits"].format(**lim)
    cap = program_evaluate_cap(cfg)
    text["per_program"] = f", at most {cap} per program" if cap is not None else ""
    text["gate"] = gate_text(cfg, kind=gate_kind, gate_n=gate_n, retention=retention, e_task=e_task)
    if not retention:                        # single-domain run: no retention domain is measured or gated
        text["goal"] = text["goal"].replace(" and the retention domains must not get worse", "")
    out = SYSTEM_PROMPT.format(who=who, whose=whose, timeout_s=cfg.get("sandbox", {}).get("timeout_s", 1800),
                               evaluate=int(budget.get("evaluate", 32)),
                               gpu_seconds=float(budget.get("gpu_seconds", 7200)), tools=tools, **text)
    return _with_tool_protocol(out, cfg, mode=mode, retention=retention)


REPLY_LINE = "Each turn, reply with a short plan and exactly ONE ```python code block defining"
TOOL_PROTOCOL = """Before answering you may call functions for up to {rounds} rounds (one round = one reply with \
function calls; at most {calls} calls in all):
- run_python(code) runs exploratory code in the program sandbox with the same `api` (look at modules, diagnostics, \
SVDs, train items, earlier candidates, compare, try constructions and inspect them). Not available there: \
{missing}; only your final program's result is evaluated, and exploring spends none of the budgets below{gpu_note}.
- read_record(cid, part) returns the full program or the full feedback of a candidate listed in the latest \
message{history_read}.{history}
Nothing persists between run_python calls or into your final program (handles made there are freed): the final \
program must be self-contained.

When you are done, reply WITHOUT a function call: a short plan and exactly ONE ```python code block defining"""


HISTORY_READ = " or of any earlier candidate that search_history lists"
HISTORY_LINE = ("\n- search_history(gen, sort, offset) lists every candidate already tried (this generation's and the "
                "earlier generations') with what it is (strategy, modules, rank, tools used), its target gain, worst "
                "retention change, KL, score and gate outcome; same_as marks a result identical to an earlier one. "
                "Check it before you write a program, so you do not spend an evaluation repeating an idea that was "
                "already tried.")


def history_row(rec: dict) -> dict:
    """search_history summary of one candidates.jsonl record (2026-11-21): what the candidate is and how it scored
    against that generation's current model (the summarize_candidate fields)."""
    from types import SimpleNamespace
    c = SimpleNamespace(**{k: rec.get(k) for k in ("cid", "thread", "step", "artifact", "screen", "full", "error")})
    row = summarize_candidate(c)
    art = rec.get("artifact") or {}
    tools = art.get("tools_used") if isinstance(art.get("tools_used"), dict) else {}
    top = sorted(tools.items(), key=lambda kv: -kv[1])[:6]
    out = {k: row.get(k) for k in ("strategy", "target_gain", "vs_raw", "worst_retention_delta", "kl")}
    out.update({"kind": art.get("kind") if art.get("kind") not in (None, "lora") else None,
                "modules": art.get("modules"), "max_rank": art.get("max_rank"),
                "keep": True if art.get("keep") else None,
                "tools": ", ".join(f"{k} x{v}" for k, v in top) or None})
    return out


def _with_tool_protocol(text: str, cfg: dict, *, mode: str, retention: bool) -> str:
    """The system prompt with the tool-loop protocol (2026-11-21) when editor.tool_rounds > 0; unchanged
    (byte-identical to the prompt before the tool loop) when it is 0."""
    from rsicore.editor import editor_settings
    s = editor_settings(cfg)
    rounds = int(s.get("tool_rounds") or 0)
    if rounds <= 0:
        return text
    from .tools import Workspace
    gpu = bool(s["explore"].get("gpu_tools"))
    have = set(tools_for_mode(mode, retention=retention, cfg=cfg))
    missing = [n for n in Workspace.EXPLORE_NEVER + (() if gpu else Workspace.GPU_TOOLS) if n in have]
    miss = (", ".join(f"api.{n}" for n in missing) or "(none)") + \
        (" (the handle 'RAW' still works)" if "raw_lora" in missing else "")
    note = (f" (the GPU tools there have their own exploration budget of {float(s['explore'].get('gpu_seconds') or 0):.0f}"
            " GPU-seconds)") if gpu else ""
    hist = bool(s.get("history_tool", True))
    para = TOOL_PROTOCOL.format(rounds=rounds, calls=int(s.get("max_tool_calls") or 0), missing=miss, gpu_note=note,
                                history_read=HISTORY_READ if hist else "", history=HISTORY_LINE if hist else "")
    assert REPLY_LINE in text
    return text.replace(REPLY_LINE, para, 1)


def _num(x, nd=4):
    return None if x is None else round(float(x), nd)


def summarize_candidate(c) -> dict:
    fb = (c.screen or {}).get("feedback") or {}
    tgt = fb.get("target") or {}
    ret = fb.get("retention") or {}
    drops = [(v.get("vs_parent") or {}).get("delta") for v in ret.values()]
    drops = [d for d in drops if d is not None]
    art = c.artifact or {}
    return {"cid": c.cid, "thread": c.thread, "step": c.step, "strategy": art.get("strategy"),
            **({"kind": "weights", "modules": art.get("modules")} if art.get("kind") == "weights" else {}),
            "screen_score": _num((c.screen or {}).get("score")), "full_score": _num((c.full or {}).get("score")),
            "target_gain": _num((tgt.get("vs_parent") or {}).get("delta")),
            "vs_raw": _num((tgt.get("vs_raw") or {}).get("delta")),
            "worst_retention_delta": _num(min(drops)) if drops else None, "kl": _num(fb.get("kl_vs_parent")),
            "error": (c.error or "")[:160] or None}


class ContextRenderer:
    """render_context(history, thread) for rsicore.search.inner_search."""

    def __init__(self, *, task_text: str, diag_text: str, archive_text: str = "", max_feedback_chars: int = 6000,
                 budget_fn=None, timeout_s: int | None = None):
        self.task_text = task_text
        self.diag_text = diag_text
        self.archive_text = archive_text
        self.max_fb = max_feedback_chars
        self.budget_fn = budget_fn          # () -> Workspace.budget_status() (2026-11-21)
        self.timeout_s = timeout_s

    def budget_line(self, history) -> str | None:
        """What is left this generation and how long things took so far (2026-11-21)."""
        if self.budget_fn is None:
            return None
        try:
            b = self.budget_fn() or {}
        except Exception:  # noqa: BLE001
            return None
        s = f"Budget left this generation (shared by all threads): {b.get('evaluate_left'):g} of {b.get('evaluate'):g} evaluate calls"
        if b.get("program_evaluate_calls") is not None:
            s += f" (api.evaluate inside a program: at most {b['program_evaluate_calls']} new ones)"
        if b.get("gpu_seconds") is not None:
            s += f", {b.get('gpu_seconds_left'):g} of {b['gpu_seconds']:g} GPU-seconds"
        s += "."
        prog = sorted(float((c.artifact or {}).get("seconds")) for c in history if (c.artifact or {}).get("seconds") is not None)
        scr = sorted(float(c.cost["screen_s"]) for c in history if isinstance(c.cost, dict) and c.cost.get("screen_s"))
        timeouts = sum(1 for c in history if c.error and "timeout" in c.error.lower())
        if self.timeout_s:
            t = f" Program time limit {self.timeout_s} s"
            if prog:
                t += f"; programs so far ran {prog[len(prog) // 2]:.0f} s (median), {prog[-1]:.0f} s at most"
            if timeouts:
                t += f"; {timeouts} program(s) were killed at the limit"
            s += t + "."
        if scr:
            s += f" One evaluation of a candidate took about {scr[len(scr) // 2]:.0f} s."
        if b.get("evaluate_left") is not None and b["evaluate_left"] <= 4:
            s += " Few evaluations are left: do not call api.evaluate in your program; return your best LoRA."
        return s

    def __call__(self, history, thread: int) -> str:
        mine = sorted([c for c in history if c.thread == thread], key=lambda c: c.step)
        step = len(mine)
        parts = [f"[thread={thread} step={step}]", self.task_text]
        bl = self.budget_line(history)
        if bl:
            parts.append(bl)
        if step == 0:
            parts.append(self.diag_text)
            if self.archive_text:
                parts.append(self.archive_text)
        if history:
            rows = [summarize_candidate(c) for c in sorted(history, key=lambda c: (c.step, c.thread))]
            parts.append("Candidates so far (all threads; score = target_gain - lambda*retention_drop - "
                         "lambda_kl*KL, vs the current model):\n" + "\n".join(json.dumps(r) for r in rows))
        if mine:
            scored = [c for c in mine if c.score is not None]
            best = max(scored, key=lambda c: c.score) if scored else None
            last = mine[-1]
            if best is not None:
                parts.append(f"Best program of this thread so far ({best.cid}, score {best.score:.4f}):\n"
                             f"```python\n{best.program}```")
            if last.error:
                parts.append(f"Your last program ({last.cid}) failed:\n{last.error[:2500]}")
            else:
                fb = json.dumps((last.screen or {}).get("feedback"), default=str)
                parts.append(f"Feedback on your last program ({last.cid}):\n{fb[: self.max_fb]}")
            parts.append("Improve this thread's best program, or borrow an idea from another thread. "
                         "Reply with one ```python block.")
        else:
            parts.append("Write your first program. Reply with one ```python block.")
        return "\n\n".join(parts)


def task_text(*, gen: int, target: str, dataset: str, retention: list[str], n_train: int,
              e_task: bool = False, raw_info: dict | None = None, gate_dataset: str | None = None) -> str:
    if e_task:
        return (f"Task (fixed editing-skill measurement): write a LoRA for the ORIGINAL base model that improves "
                f"{target} ({dataset}); {n_train} training items are available through api.train_ids(). "
                "Feedback compares against the unmodified base model." + _raw_line(raw_info))
    s = (f"Generation {gen}. Target domain: {target} (training set {dataset}; {n_train} train items via "
         f"api.train_ids()). " + (f"Retention domains (must not get worse): {', '.join(retention)}." if retention
                                  else "Single-domain run: only the target domain is evaluated and gated."))
    if gate_dataset and gate_dataset != dataset:     # data_roles.gate (2026-11-21): a different gate set
        s += f" The gate accepts a candidate on {gate_dataset} (scores only; its items are not readable)."
    return s + _raw_line(raw_info)


def _raw_line(raw_info: dict | None) -> str:
    if not raw_info:
        return ""
    recipe = (f"{raw_info.get('n_train')} train items, rank {raw_info.get('rank')}, {raw_info.get('steps')} steps")
    if raw_info.get("source") == "current":
        return (f"\nRAW = the LoRA currently applied to you (you are the original weights + RAW). It was trained "
                f"once, in generation 0, with a fixed recipe ({recipe}), and has been edited since; there is no new "
                "training in this generation. Your LoRA REPLACES RAW, so start from api.raw_lora(). The gate "
                "requires a clear improvement over the current model (= RAW).")
    gain = (f" It changes the target screen score by {raw_info.get('target_gain')} and the worst retention screen "
            f"score by {raw_info.get('worst_retention')}." if raw_info.get("target_gain") is not None else "")
    return (f"\nRAW = the LoRA trained with the fixed recipe ({recipe}); this is the only training of the run, later "
            f"generations edit the LoRA you produce now.{gain} Your LoRA replaces RAW (the next model is the original "
            "weights + your LoRA). Start from api.raw_lora().")
