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

from .tools import tools_for_mode

SYSTEM_PROMPT = """You are {who}. You improve a language model by writing Python programs that {action} \
({what}). {whose}

Each turn, reply with a short plan and exactly ONE ```python code block defining

    def main(api):
        ...
        {return_line}

Rules
- Call tools only as api.<tool>(keyword=value, ...). Results are JSON (dicts / lists / numbers / strings).
  {objects} stay on the server and are referred to by string handles.
- Data access: activation tools accept ONLY the ids returned by api.train_ids() (the target task's
  training items of this generation). Signal, gate and held-out items are never readable; their scores come
  only from api.evaluate and from the feedback you see after each program.
- {limits} Programs time out after {timeout_s} s.
- Budgets for this session: {evaluate} evaluate calls (each candidate you return is evaluated
  automatically and counts as one; identical {unit}s are evaluated once) and {gpu_seconds:.0f} GPU-seconds of
  forward passes (acts / regmean_preserve / train_loss). There is no training and no gradient tool. {goal}
- Gate (after the search, not visible to you): on a fixed hidden item set, item by item against the current
  model, the target must improve significantly (95% CI above 0) and by at least the required margin (mode edit,
  generation 0: not worse than the raw LoRA). Only candidates that beat the current model on your feedback
  items are submitted.
- Module names look like 'layers.12.mlp.down_proj', 'layers.3.self_attn.q_proj' (full-attention layers) and
  'layers.0.linear_attn.in_proj_qkv' (Gated-DeltaNet layers); use api.modules() to list them.
- Vector references for rank1 / from_direction: "S3.u0", "S3.v1" (svd handle), "A2.in_mean", "A2.out_mean",
  "A2.pc0" (acts handle).
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


def system_prompt(cfg: dict, *, budget: dict, e_task: bool = False, mode: str = "scratch", retention: bool = True) -> str:
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
    return SYSTEM_PROMPT.format(who=who, whose=whose, timeout_s=cfg.get("sandbox", {}).get("timeout_s", 1800),
                                evaluate=int(budget.get("evaluate", 32)),
                                gpu_seconds=float(budget.get("gpu_seconds", 7200)), tools=tools, **text)


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

    def __init__(self, *, task_text: str, diag_text: str, archive_text: str = "", max_feedback_chars: int = 6000):
        self.task_text = task_text
        self.diag_text = diag_text
        self.archive_text = archive_text
        self.max_fb = max_feedback_chars

    def __call__(self, history, thread: int) -> str:
        mine = sorted([c for c in history if c.thread == thread], key=lambda c: c.step)
        step = len(mine)
        parts = [f"[thread={thread} step={step}]", self.task_text]
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
              e_task: bool = False, raw_info: dict | None = None) -> str:
    if e_task:
        return (f"Task (fixed editing-skill measurement): write a LoRA for the ORIGINAL base model that improves "
                f"{target} ({dataset}); {n_train} training items are available through api.train_ids(). "
                "Feedback compares against the unmodified base model." + _raw_line(raw_info))
    s = (f"Generation {gen}. Target domain: {target} (training set {dataset}; {n_train} train items via "
         f"api.train_ids()). " + (f"Retention domains (must not get worse): {', '.join(retention)}." if retention
                                  else "Single-domain run: only the target domain is evaluated and gated."))
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
