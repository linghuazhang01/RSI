"""Editor system prompt and per-step context (RSI_LOOP_DESIGN §1.2, §3.1, §3.4).

The editor is the current model M_t. Each step it writes ONE program `def main(api): ...` that builds a
LoRA for ITSELF through the tools and returns {"lora": <handle>} ({"lora": None} keeps the model
unchanged). The rules, goal and tool list depend on method.mode (MODE_TEXT): edit = edit the raw LoRA
trained once (generation 0) with the fixed recipe; scratch = build from zero; weights = edit the dense weights
directly with structured, any-rank operations and return {"edit": <handle>}. The editor never trains and has
no gradient tools in any mode. The context shows the task, the fixed diagnostics (first step of a thread), every candidate
so far (all threads), this thread's best program and its latest feedback.
search.charge_screens false (2026-11-21, objective simple): the budget text says that returned candidates are scored
automatically at no cost and that api.evaluate draws on a separate per-generation pool; with one step per generation
each thread is told it is session k of N working independently (ContextRenderer sessions). The tensor tools
(api.tensors / api.tensor_compute, when on) are mentioned in the vector-reference, module-name and strategy rules.
"""
from __future__ import annotations

import json

from .tools import program_evaluate_cap, search_settings, tensor_compute_enabled, tools_for_mode

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
- {budget_text}
  {gpu_seconds:.0f} GPU-seconds of forward passes (acts / regmean_preserve / train_loss). There is no training
  and no gradient tool. {goal}
- {gate}
- Module names: use only names returned by api.modules() (rows with module, layer, layer_type, out, in). The
  layer types are irregular (the diagnostics table lists each layer's type): MLP 'layers.12.mlp.down_proj'
  exists everywhere, 'self_attn.*' only in full-attention layers, 'linear_attn.*' only in Gated-DeltaNet layers.
  Never build a name from a pattern without checking it is in api.modules().{tensor_names}
- Vector references for rank1 / from_direction live on one side of a module's W [out, in]: u and direction are
  OUTPUT-side vectors (length out: "S3.u0" of an svd handle, "A2.out_mean" of an acts handle); v and key are
  INPUT-side vectors (length in: "S3.v1", "A2.in_mean", "A2.pc0"). Take them from the SAME module's svd / acts
  (e.g. down_proj: in = the MLP width, out = the hidden size).{tensor_vectors}
- Robust code: tool results can be empty (e.g. api.diagnostics for a module without a row) and one uncaught
  error fails the whole candidate. Check lengths before indexing and the structure before unpacking; wrap
  optional steps in try/except and fall back to what you already built.
- note (optional, at most 600 characters, plain text): which modules this candidate changes and WHY those (the
  evidence that picked them: a diagnostic, an activation or gradient statistic, a probe result), then what you tried
  and rejected, with the numbers that decided it (e.g. "down_proj layers 10-14, rank 4: largest gradient norm on the
  failed problems; rejected: o_proj layers 20-24, train_loss +0.03 at scale 0.5 and +0.01 at -0.5"). The history
  table already lists the modules and parameters; the note is for the reasons. Every later session sees it, so a
  good note keeps them from repeating your failed ideas.
- {strategies}

Tools
{tools}
"""

DATA_ACCESS = """- Data access: activation tools accept ONLY the ids returned by api.train_ids() (the target task's
  training items of this generation). Signal, gate and held-out items are never readable; their scores come
  only from api.evaluate and from the feedback you see after each program."""
DATA_ACCESS_SCREEN = """- Data access: activation tools accept the ids returned by api.train_ids() (the target task's
  training items of this generation) and this generation's screen ids (api.screen_results: training-pool items you
  answered yourself). Gate and held-out items are never readable: you only see each candidate's gate statistics."""

_LORA_TEXT = dict(
    what="a low-rank weight update dW = B A per linear module",
    return_line='return {"lora": handle, "note": "..."}      # or {"lora": None, "note": "..."} to leave the model unchanged',
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
             "channel / module scaling, dense additions of LoRA-built updates, moves toward the original weights"
             "{tensor_what}",
        return_line='return {"edit": handle, "note": "..."}      # or {"edit": None, "note": "..."} to leave the model unchanged',
        objects="Weight edits, LoRAs, SVDs and activation statistics",
        limits="Limits checked on every edit: ||W'-W||_F / ||W||_F <= {tau_w} per module (W = your current weights,\n"
               "  W' = after the edit){max_modules_text}, finite values, existing modules only. There is no rank limit."
               "{vector_limits}",
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


BUDGET_CHARGED = """Budgets for this generation, shared by all threads: {evaluate} evaluate calls (each candidate you return is
  evaluated automatically and costs one; every api.evaluate inside a program also costs one{per_program};
  identical {unit}s are evaluated once). When they run out, later candidates are not scored at all: look at
  the "left" numbers in the latest message and in api.budget() before calling api.evaluate."""

# search.charge_screens false (2026-11-21, objective simple)
BUDGET_FREE = """Budgets for this generation: every candidate you return is scored automatically on the feedback items at
  no cost. api.evaluate inside a program draws on a separate pool of {evaluate} calls shared by all sessions of
  this generation{per_program}. Identical {unit}s are evaluated once, and an api.evaluate of a {unit} that
  was already scored is free. When the pool is used up api.evaluate raises an error (your returned candidate
  is still scored): look at the "left" numbers in the latest message and in api.budget() before calling it."""

TENSOR_NAMES = """
- api.modules() lists only the linear modules. Every other tensor (input_layernorm / post_attention_layernorm,
  self_attn.q_norm / k_norm, linear_attn.dt_bias / A_log / conv1d / norm, ...) is listed by api.tensors(); read any
  tensor with api.tensor_compute{tensor_write}. embed_tokens, the final norm and lm_head are never editable."""

TENSOR_VECTORS = """ Vectors returned by
  api.tensor_compute ("T5.name") work in the same places; they must have the right length (its vector_sizes):
  u / direction the output side, v / key the input side."""

TENSOR_STRATEGY = (" api.tensor_compute(code, tensors) runs your own torch code on CPU copies of up to\n  {max_tensors} "
                   "current tensors (custom statistics, directions, exact factors; prefer torch.svd_lowrank on large "
                   "matrices;\n  a few threads, at most ~{timeout_s} s and never longer than your program has left): "
                   "return vectors for rank1 /\n  from_direction or exact LoRA factors (lora = {{module: (A [r, in], B "
                   "[out, r])}}; the rank and norm caps apply as for\n  any LoRA).")

TENSOR_STRATEGY_W = (" api.tensor_compute(code, tensors, edit=None) runs your own torch code on CPU copies of up to\n  "
                     "{max_tensors} current tensors (custom statistics, directions and writes; prefer "
                     "torch.svd_lowrank on large matrices;\n  a few threads, at most ~{timeout_s} s and never longer "
                     "than your program has left): writes = {{tensor: new full tensor}}\n  become a dense addition of "
                     "any rank on any editable tensor, also the 1-D norms / dt_bias / A_log.")


# gate per candidate (2026-11-21, objective simple: search.gate_every_candidate, no api.evaluate)
BUDGET_GATE = """Budgets for this generation: there is no api.evaluate. Every {unit} you return gets ONE gate evaluation
  automatically (see Gate below); nothing else is scored, so judge your constructions with the free tools."""

SCREEN_STRATEGY = (" This generation's screen items were answered by YOU (the current model) {n} time(s) each with\n  "
                   "the scoring decoding (the summary is in every message): api.screen_results(filter='wrong' | "
                   "'unstable' | 'right')\n  lists every item with its prompt, gold answer, your extracted answers "
                   "and scores, and api.screen_output(id, sample)\n  pages one full answer, thinking included (both "
                   "free). Find what you get wrong and why before you edit.")

PAPER_RECIPE = (" A recipe from the model-editing literature: collect activations on your OWN answers for two groups\n  "
                "of screen items (e.g. a = api.acts(module=m, ids=wrong_ids, source='own'), b = the same on "
                "right_ids),\n  take d = api.direction(a=a['handle'] + '.out_mean', b=b['handle'] + '.out_mean') and "
                "remove it from a layer's\n  output matrices with api.project_out(target=lora, module=m, "
                "u=d['vector'], alpha=0.5) "
                "(api.output_matrices(layer) lists them);\n  api.unit / api.direction{tc} "
                "work on any vector refs. Save what you build with\n  api.save_edit(handle=h, name='best') (it "
                "survives between run_python calls and into your final program: return\n  {{'{key}': "
                "api.load_edit(name='best')['{key}']}}).")


def screen_text(summary: dict | None) -> str:
    """The per-step context block about the current model's own screen answers (2026-11-21): accuracy avg@n, the
    distribution of correct samples per item, truncation and a few failing items (screen_view.summary)."""
    sm = summary or {}
    if not sm.get("n_items"):
        return ""
    n = sm.get("samples_per_item") or 1
    out = (f"Your own answers on this generation's {sm['n_items']} screen items (training-pool items; you = the current "
           f"model, {n} sampled answer(s) per item with the scoring decoding): accuracy avg@{n} {sm.get('accuracy')}; "
           f"items by number of correct samples {json.dumps(sm.get('correct_counts'))}; truncation rate "
           f"{sm.get('truncation_rate')} (thinking cut at the token limit: {sm.get('truncated_thinking')} samples).")
    if sm.get("failing"):
        out += "\nA few items you do not always get right:\n" + "\n".join(
            json.dumps(r, ensure_ascii=False, default=str)[:1500] for r in sm["failing"])
    return out + ("\nEvery item and your full outputs: api.screen_results(filter=...) / api.screen_output(id, sample) "
                  "(free, also in run_python).")


def _points(x: float) -> str:
    p = round(float(x) * 100, 4)
    return f"{p:g} accuracy point" + ("" if p == 1 else "s")


def gate_text(cfg: dict, *, kind: str = "parent", gate_n: int | None = None, retention: bool = True,
              e_task: bool = False, every: bool = False) -> str:
    """The gate rule as configured (2026-11-21; gate.min_gain / require_significant / epsilon), not a fixed text.
    every (search.gate_every_candidate): each candidate is gated during the search and the largest passing gain wins."""
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
        if mg <= 0:                              # (2026-11-21) threshold 0: any strictly positive gain
            rule = "item by item against the current model, the mean gain in target accuracy must be positive"
        else:
            items = f", i.e. about {max(1, round(mg * gate_n))} more correct items out of {gate_n}" if gate_n else ""
            rule = (f"item by item against the current model, the mean gain in target accuracy must be at least {mg:g} "
                    f"({_points(mg)}{items})")
        rule += (" and statistically significant (paired 95% CI lower bound above 0)"
                 if g.get("require_significant", True) else "; there is no significance test")
    ret = (" Retention domains must not drop by more than the allowed tolerance." if retention else "")
    if every and kind != "raw":
        from rsicore.config import stage_decoding
        ns = int((stage_decoding(cfg, "gate") or {}).get("n") or 1)
        smp = f", {ns} sampled answers per item, the current model answered the same items with the same seed" \
            if ns > 1 else ""
        return (f"Gate: every candidate you return is evaluated ONCE on a fixed hidden set of {gate_n or 'the gate'} "
                f"items{smp}; {rule}.{ret} At the end of the generation the passing candidate with the LARGEST gain "
                "is accepted (none passes: the model stays as it is). You see each earlier candidate's gate result "
                "(accuracy, paired gain, CI, passed, broken / fixed item ids; never the items) in later steps and "
                "generations (search_history), not your own before you return it.")
    return (f"Gate (after the search, not visible to you): on a fixed hidden item set, {rule}.{ret} Only candidates "
            "whose score beats the current model on your feedback items are submitted; the feedback items are a "
            "different sample, so aim clearly above the margin.")


def system_prompt(cfg: dict, *, budget: dict, e_task: bool = False, mode: str = "scratch", retention: bool = True,
                  gate_kind: str = "parent", gate_n: int | None = None, free_screens: bool | None = None) -> str:
    """free_screens (2026-11-21): returned candidates are scored for free and budget["evaluate"] is the programs'
    api.evaluate pool; None = from search.charge_screens (false -> free), never for the E task."""
    who = ("an AI model asked to write a LoRA for a FIXED base model on a tool-calling / coding task"
           if e_task else "the model being edited: the LoRA you write will be merged into YOUR OWN weights "
                          "if it passes a fixed gate, and the merged model becomes the next editor")
    whose = ("The model you edit is the original base model, not yourself." if e_task else
             "You are editing yourself.")
    if mode == "scratch" and not e_task and str((cfg.get("merge") or {}).get("mode") or "merge") == "adapter":
        # merge.mode adapter (2026-11-22): accepted LoRAs are kept as one cumulative adapter, never merged
        who = ("the model being edited: you are the original weights plus every LoRA accepted so far, kept as one "
               "adapter and never merged; the LoRA you write is added on top of YOU (limits apply to your LoRA, not "
               "the accumulated adapter) if it passes a fixed gate, and that model becomes the next editor")
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
    have = tools_for_mode(mode, retention=retention, cfg=cfg)
    tc_on = "tensor_compute" in have and tensor_compute_enabled(cfg)
    lim["vector_limits"] = ""
    if tc_on and mode == "weights":
        vt = wc.get("vector_cap_tau") if wc.get("vector_cap_tau") is not None else 0.5
        lim["vector_limits"] = ("\n  1-D tensors (norms, dt_bias, A_log): rms(W'-W) <= " + f"{vt:g}" + " * max(rms(W), 1) "
                                "(the RMSNorm weights are zero-centred: the layer\n  multiplies by 1 + w); a "
                                "tensor_compute write that is (almost) a uniform rescale of W is refused (use w_scale);\n"
                                "  embed_tokens, the final norm and lm_head are never editable.")
    text = dict(MODE_TEXT[mode])
    text["limits"] = text["limits"].format(**lim)
    if "{tensor_what}" in text["what"]:
        text["what"] = text["what"].replace("{tensor_what}", ", dense additions computed by api.tensor_compute (any "
                                            "tensor, any rank)" if tc_on else "")
    cap = program_evaluate_cap(cfg)
    if free_screens is None:                 # search.charge_screens false (objective simple); never the E task
        free_screens = not e_task and not search_settings(cfg)["charge_screens"]
    text["per_program"] = (f", at most {cap} " + ("new ones " if free_screens else "") + "per program") \
        if cap is not None else ""
    text["budget_text"] = (BUDGET_FREE if free_screens else BUDGET_CHARGED).format(
        evaluate=int(budget.get("evaluate", 32)), per_program=text["per_program"], unit=text["unit"])
    ss = search_settings(cfg)
    gate_mode = ss["gate_every_candidate"] and not e_task
    text["gate"] = gate_text(cfg, kind=gate_kind, gate_n=gate_n, retention=retention, e_task=e_task, every=gate_mode)
    if gate_mode and "evaluate" not in have:   # 2026-11-21: no api.evaluate, one gate evaluation per candidate
        text["budget_text"] = BUDGET_GATE.format(unit=text["unit"])
    from .tools import paper_tools_enabled, screen_tools_enabled
    if not e_task and screen_tools_enabled(cfg) and "screen_results" in have:
        from rsicore.config import stage_decoding
        text["strategies"] += SCREEN_STRATEGY.format(n=int((stage_decoding(cfg, "screen") or {}).get("n") or 1))
    if not e_task and paper_tools_enabled(cfg) and "project_out" in have:
        text["strategies"] += PAPER_RECIPE.format(key="edit" if mode == "weights" else "lora",
                                                  tc=" / tensor_compute(vectors=[...])" if tc_on else "")
    if not retention:                        # single-domain run: no retention domain is measured or gated
        text["goal"] = text["goal"].replace(" and the retention domains must not get worse", "")
    text["tensor_names"] = text["tensor_vectors"] = ""
    if tc_on:                                # api.tensors / api.tensor_compute (2026-11-21)
        from .tools import tensor_compute_settings
        tcs = tensor_compute_settings(cfg)
        text["tensor_names"] = TENSOR_NAMES.format(
            tensor_write=" and change it only through its writes (mode weights)" if mode == "weights" else
            " (changes to non-linear tensors need method.mode weights)")
        text["tensor_vectors"] = TENSOR_VECTORS
        text["strategies"] += (TENSOR_STRATEGY_W if mode == "weights" else TENSOR_STRATEGY).format(
            max_tensors=int(tcs["max_tensors"]), timeout_s=int(tcs["timeout_s"]))
    out = SYSTEM_PROMPT.format(who=who, whose=whose, timeout_s=cfg.get("sandbox", {}).get("timeout_s", 1800),
                               gpu_seconds=float(budget.get("gpu_seconds", 7200)), tools=tools, **text)
    if not e_task and "screen_results" in have:  # 2026-11-21: screen items readable, gate items statistics only
        out = out.replace(DATA_ACCESS, DATA_ACCESS_SCREEN)
    if "evaluate" not in have:                   # no api.evaluate (simple v2): drop the old flow's evaluate wording
        for a, b in NO_EVALUATE_EDITS:
            out = out.replace(a, b)
    return _with_tool_protocol(out, cfg, mode=mode, retention=retention)


# (2026-11-21) wording of the evaluate flow that does not apply when programs have no api.evaluate and candidates are
# only gate-evaluated (compare needs screened candidates; gate statistics are in search_history / read_record)
NO_EVALUATE_EDITS = (
    ("every tool call included (evaluate, acts,", "every tool call included (acts,"),
    ("(answer NLL on train items, no\n  evaluate) before spending evaluate calls; api.compare(cid) shows which items a "
     "candidate broke or fixed vs the current model.",
     "(answer NLL on train items) before you return one; earlier candidates' gate results (broken / fixed ids) are in "
     "search_history / read_record."),
    ("(answer NLL on train items, no evaluate) before spending evaluate calls;\n  api.compare(cid) shows which items "
     "a candidate broke or fixed vs the current model.",
     "(answer NLL on train items) before you return one;\n  earlier candidates' gate results (broken / fixed ids) are "
     "in search_history / read_record."),
)


REPLY_LINE = "Each turn, reply with a short plan and exactly ONE ```python code block defining"
TOOL_PROTOCOL = """Before answering you may call functions for up to {rounds} rounds (one round = one reply with \
function calls; at most {calls} calls in all):
- run_python(code) runs exploratory code in the program sandbox with the same `api` (look at modules, diagnostics, \
SVDs, train items, earlier candidates, compare, try constructions and inspect them). Not available there: \
{missing}; only your final program's result is evaluated, and exploring spends none of the budgets below{gpu_note}.
- read_record(cid, part) returns the full program or the full feedback of a candidate listed in the latest \
message{history_read}.{history}
{persist}

When you are done, reply WITHOUT a function call: a short plan and exactly ONE ```python code block defining"""


PERSIST_NONE = ("Nothing persists between run_python calls or into your final program (handles made there are freed): "
                "the final program must be self-contained.")
PERSIST_SESSION = ("Your run_python calls share ONE Python session for this candidate: variables, imports and handles "
                   "persist from call to call (a timeout or crash restarts the session and loses them). Your final "
                   "program runs separately: it must build what it returns{saved}.")
HISTORY_READ = " or of any earlier candidate that search_history lists"
HISTORY_LINE = ("\n- search_history(gen, sort, offset) lists every candidate already tried (this generation's and the "
                "earlier generations') with what it changed (module types; per type the layers, rank and ||dW||/||W0||, or the weight-edit ops), its note, its target gain, worst "
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
    out = {k: row.get(k) for k in ("target_gain", "vs_raw", "worst_retention_delta", "kl")}
    out.update({"kind": art.get("kind") if art.get("kind") not in (None, "lora") else None,
                "keep": True if art.get("keep") else None, **change_fields(art),
                "note": art.get("note") or None})
    return out


def change_fields(art: dict) -> dict:
    """What a candidate changed, for the history rows (2026-11-22): types = the module types it touched, modules = one
    line per type (per op setting for a weight edit): '<type> L<layers> x<count> rank <r> rel <||dW||/||W0||>
    [<ops>]'. Artifacts without the edit summary (before 2026-11-22) give the module count and largest rank."""
    ed = art.get("edit")
    if not isinstance(ed, dict) or not isinstance(ed.get("groups"), list):
        return {k: v for k, v in (("modules", art.get("modules")), ("rank", art.get("max_rank"))) if v}
    lines = []
    for g in ed["groups"]:
        parts = [str(g.get("type"))]
        if g.get("layers"):
            parts.append(f"L{g['layers']}")
        parts.append(f"x{g.get('n')}")
        if g.get("rank"):
            parts.append(f"rank {g['rank']}")
        if g.get("rel_norm"):
            parts.append(f"rel {g['rel_norm']}")
        if g.get("ops"):
            parts.append(f"[{g['ops']}]")
        lines.append(" ".join(parts))
    if ed.get("more_groups"):
        lines.append(f"+{ed['more_groups']} more groups ({ed.get('n')} modules in all)")
    return {"types": ed.get("types") or None, "modules": lines or None}


ARCHIVE_HEAD = ("Archive of earlier generations: EVERY candidate tried, one row each, failures included. types = the "
                "module types it changed, modules = one line per type: layers (L), module count (x), LoRA rank, "
                "rel = ||dW||/||W0|| (and the ops of a weight edit), kept = it returned None (no change), gate = its gate result (gain = paired gain in gate accuracy vs that "
                "generation's current model, ci = its 95% interval), note = its own account of what it tried.")
ARCHIVE_READ = ("Full program or feedback of any candidate above: read_record(cid, 'program' | 'feedback'); "
                "search_history lists them with more statistics. Do not repeat what failed: try something different "
                "or fix the reason it failed.")


def archive_row(c, gate: dict | None = None) -> dict:
    """One archive row of an earlier generation's candidate (2026-11-22): what it changed, its gate result (gate =
    rsicore.chat_tools._gate_outcomes row) or score, the first line of its error, its note."""
    art = c.artifact or {}
    row = {"cid": c.cid}
    if c.error:
        row["error"] = (c.error.strip().splitlines() or [""])[0][:200]
    elif art.get("keep"):
        row["kept"] = True
    else:
        if art.get("kind") not in (None, "lora"):
            row["kind"] = art["kind"]
        row.update({k: v for k, v in change_fields(art).items() if v is not None})
    if gate:
        g = {"passed": bool(gate.get("passed"))}
        if gate.get("gain") is not None:
            g["gain"] = _num(gate["gain"])
        if isinstance(gate.get("ci"), (list, tuple)):
            g["ci"] = [_num(x) for x in gate["ci"]]
        if not g["passed"] and gate.get("reason"):
            g["reason"] = str(gate["reason"])[:120]
        row["gate"] = g
    elif c.score is not None:
        row["score"] = _num(c.score)
    if art.get("note"):
        row["note"] = art["note"]
    return row


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
            " GPU-seconds per session)") if gpu else ""
    hist = bool(s.get("history_tool", True))
    from .tools import explore_persistent
    if explore_persistent(cfg):              # editor.explore.persistent (2026-11-21)
        persist = PERSIST_SESSION.format(saved=", or load what you saved with api.save_edit(handle, name) via "
                                         "api.load_edit(name)" if "save_edit" in have else "")
    else:
        persist = PERSIST_NONE
    para = TOOL_PROTOCOL.format(rounds=rounds, calls=int(s.get("max_tool_calls") or 0), missing=miss, gpu_note=note,
                                history_read=HISTORY_READ if hist else "", history=HISTORY_LINE if hist else "",
                                persist=persist)
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
            **({"gate": _gate_row(fb["gate"])} if isinstance(fb.get("gate"), dict) else {}),
            "error": (c.error or "")[:160] or None, **({"note": art["note"]} if art.get("note") else {})}


def _gate_row(gt: dict) -> dict:
    """Compact gate result of a candidate for the step table (2026-11-21; the ids are in its feedback)."""
    return {k: (_num(v) if isinstance(v, float) else v) for k, v in gt.items()
            if k in ("passed", "gain", "ci", "accuracy", "broken", "fixed")}


class ContextRenderer:
    """render_context(history, thread) for rsicore.search.inner_search."""

    def __init__(self, *, task_text: str, diag_text: str, archive_text: str = "", max_feedback_chars: int = 6000,
                 budget_fn=None, timeout_s: int | None = None, sessions: int | None = None,
                 history_tool: bool = False, gate_mode: bool = False, screen_text: str = ""):
        self.task_text = task_text
        self.diag_text = diag_text
        self.archive_text = archive_text
        self.max_fb = max_feedback_chars
        self.budget_fn = budget_fn          # () -> Workspace.budget_status() (2026-11-21)
        self.timeout_s = timeout_s
        # sessions (2026-11-21): one step per generation with this many threads: each is told it is session k of N
        self.sessions = sessions
        self.history_tool = history_tool    # the editor has search_history (mentioned in the sessions line)
        # gate per candidate (2026-11-21): scores are paired gate gains; screen_text = the current model's own screen
        # answers (prompts.screen_text), shown in every step's message
        self.gate_mode = gate_mode
        self.screen_text = screen_text

    def budget_line(self, history) -> str | None:
        """What is left this generation and how long things took so far (2026-11-21)."""
        if self.budget_fn is None:
            return None
        try:
            b = self.budget_fn() or {}
        except Exception:  # noqa: BLE001
            return None
        free = b.get("charge_screens") is False    # 2026-11-21: automatic screens free, evaluate = the program pool
        if b.get("evaluate_tool") is False:          # gate per candidate, no api.evaluate (2026-11-21)
            s = "No api.evaluate this generation: every returned candidate gets one gate evaluation automatically"
            free, b = False, dict(b, evaluate_left=None, program_evaluate_calls=None)
        elif free:
            s = ("Returned candidates are scored automatically at no cost. api.evaluate pool left this generation "
                 f"(shared by all sessions): {b.get('evaluate_left'):g} of {b.get('evaluate'):g} calls")
        else:
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
        left = b.get("evaluate_left")
        if free and left is not None:              # re-thresholded on the program pool
            if left <= 0:
                s += " The api.evaluate pool is used up: do not call api.evaluate; return your best LoRA."
            elif left <= max(1, int(b.get("program_evaluate_calls") or 2)):
                s += (f" Only {left:g} api.evaluate call(s) are left in the pool: keep them for your strongest "
                      "construction (your returned LoRA is scored anyway).")
        elif left is not None and left <= 4:
            s += " Few evaluations are left: do not call api.evaluate in your program; return your best LoRA."
        return s

    def sessions_line(self, thread: int) -> str | None:
        """One step per generation (2026-11-21): no session sees the others' candidates of this generation."""
        if not self.sessions:
            return None
        src = " and ".join(x for x in (("search_history" if self.history_tool else ""),
                                        ("the archive below" if self.archive_text else "")) if x)
        s = (f"You are session {thread + 1} of {self.sessions} in this generation. The other sessions work "
             "independently at the same time and you will not see their candidates, so try something of your own.")
        if src:
            s += (f" Use {src} to see what earlier generations already tried and how it scored, and do not repeat "
                  "an idea that failed: try something different.")
        return s

    def __call__(self, history, thread: int) -> str:
        mine = sorted([c for c in history if c.thread == thread], key=lambda c: c.step)
        step = len(mine)
        parts = [f"[thread={thread} step={step}]", self.task_text]
        sl = self.sessions_line(thread) if step == 0 else None
        if sl:
            parts.append(sl)
        bl = self.budget_line(history)
        if bl:
            parts.append(bl)
        if self.screen_text:
            parts.append(self.screen_text)
        if step == 0:
            parts.append(self.diag_text)
            if self.archive_text:
                parts.append(self.archive_text)
        if history:
            rows = [summarize_candidate(c) for c in sorted(history, key=lambda c: (c.step, c.thread))]
            head = ("Candidates so far (all threads; score = the paired gain in gate accuracy vs the current model; "
                    "gate = its gate result):\n" if self.gate_mode else
                    "Candidates so far (all threads; score = target_gain - lambda*retention_drop - "
                    "lambda_kl*KL, vs the current model):\n")
            parts.append(head + "\n".join(json.dumps(r) for r in rows))
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
