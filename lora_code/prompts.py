"""Editor system prompt and per-step context (RSI_LOOP_DESIGN §1.2, §3.1, §3.4).

The editor is the current model M_t. Each step it writes ONE program `def main(api): ...` that builds a
LoRA for ITSELF through the tools and returns {"lora": <handle>} ({"lora": None} keeps the model
unchanged). The context shows the task, the fixed diagnostics (first step of a thread), every candidate
so far (all threads), this thread's best program and its latest feedback.
"""
from __future__ import annotations

import json

from .tools import TOOL_DOCS

SYSTEM_PROMPT = """You are {who}. You improve a language model by writing Python programs that construct a LoRA \
(a low-rank weight update dW = B A per linear module) for it. {whose}

Each turn, reply with a short plan and exactly ONE ```python code block defining

    def main(api):
        ...
        return {{"lora": handle}}      # or {{"lora": None}} to leave the model unchanged

Rules
- Call tools only as api.<tool>(keyword=value, ...). Results are JSON (dicts / lists / numbers / strings).
  LoRAs, SVDs, activation statistics and gradients stay on the server and are referred to by string handles.
- Data access: activation / gradient tools accept ONLY the ids returned by api.train_ids() (the target task's
  training items of this generation). Signal, gate and held-out items are never readable; their scores come
  only from api.evaluate and from the feedback you see after each program.
- Limits checked on every LoRA: rank <= {rank_cap} per module, ||dW||_F / ||W0||_F <= {tau} per module,
  finite values, existing modules only. Programs time out after {timeout_s} s.
- Budgets for this session: {evaluate} evaluate calls (each candidate you return is evaluated
  automatically and counts as one; identical LoRAs are evaluated once) and {gpu_seconds:.0f} GPU-seconds of
  gradient work (grad / sgd / acts / regmean_preserve). A plain SFT run with the same GPU budget is the
  baseline you must beat: target score high, retention (other domains, token-KL) not lower.
- Module names look like 'layers.12.mlp.down_proj', 'layers.3.self_attn.q_proj' (full-attention layers) and
  'layers.0.linear_attn.in_proj_qkv' (Gated-DeltaNet layers); use api.modules() to list them.
- Vector references for rank1 / from_direction: "S3.u0", "S3.v1" (svd handle), "A2.in_mean", "A2.out_mean",
  "A2.pc0" (acts handle), "G4.<module>.u0" / ".v0" (grad handle).
- Useful strategies: short SFT (api.sgd) on a subset of modules / layers; gradient-free constructions
  (singular-direction scaling, activation directions, rank-1 edits); protecting retention with
  truncate_rank, intruder_scale, regmean_preserve, relchange_prune, orthogonalize (against earlier
  adapters) or a global scale; reusing an earlier candidate with api.candidate(cid) and interpolating.

Tools
{tools}
"""


def system_prompt(cfg: dict, *, budget: dict, e_task: bool = False) -> str:
    who = ("an AI model asked to write a LoRA for a FIXED base model on a tool-calling / coding task"
           if e_task else "the model being edited: the LoRA you write will be merged into YOUR OWN weights "
                          "if it passes a fixed gate, and the merged model becomes the next editor")
    whose = ("The model you edit is the original base model, not yourself." if e_task else
             "You are editing yourself.")
    tools = "\n".join(f"- {d}" for d in TOOL_DOCS.values())
    return SYSTEM_PROMPT.format(who=who, whose=whose, rank_cap=cfg.get("lora", {}).get("rank_cap", 64),
                                tau=cfg.get("lora", {}).get("norm_cap_tau", 0.1),
                                timeout_s=cfg.get("sandbox", {}).get("timeout_s", 1800),
                                evaluate=int(budget.get("evaluate", 32)),
                                gpu_seconds=float(budget.get("gpu_seconds", 7200)), tools=tools)


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
            "screen_score": _num((c.screen or {}).get("score")), "full_score": _num((c.full or {}).get("score")),
            "target_gain": _num((tgt.get("vs_parent") or {}).get("delta")),
            "vs_sft": _num((tgt.get("vs_sft") or {}).get("delta")),
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


def task_text(*, gen: int, target: str, dataset: str, retention: list[str], n_train: int, sft_info: dict | None,
              e_task: bool = False) -> str:
    if e_task:
        return (f"Task (fixed editing-skill measurement): write a LoRA for the ORIGINAL base model that improves "
                f"{target} ({dataset}); {n_train} training items are available through api.train_ids(). "
                "Feedback compares against the unmodified base model.")
    s = (f"Generation {gen}. Target domain: {target} (training set {dataset}; {n_train} train items via "
         f"api.train_ids()). Retention domains (must not get worse): {', '.join(retention)}.")
    if sft_info:
        s += (f"\nReference: the fixed-recipe SFT LoRA (rank {sft_info.get('rank')}, same GPU budget) changes the "
              f"target screen score by {sft_info.get('target_gain')} and the worst retention screen score by "
              f"{sft_info.get('worst_retention')}.")
    return s
