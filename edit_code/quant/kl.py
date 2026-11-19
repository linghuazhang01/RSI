"""Exact token-level KL(ref || cand) with a local transformers model (used by the profile table and
the CPU tests; candidate evaluation on the GPU uses vLLM prompt_logprobs via rsicore.evaluate)."""
from __future__ import annotations

import torch


def _logits(model, ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=ids, use_cache=False)
    return out.logits[0, :-1].float()   # predict tokens 1..T-1


@torch.no_grad()
def reference(model, batches: list[torch.Tensor], device=None, store_dtype=torch.bfloat16) -> list[torch.Tensor]:
    device = device or next(model.parameters()).device
    return [torch.log_softmax(_logits(model, b.to(device)), -1).to(store_dtype) for b in batches]


@torch.no_grad()
def kl_to_reference(model, batches: list[torch.Tensor], refs: list[torch.Tensor], device=None,
                    chunk: int = 512) -> dict:
    device = device or next(model.parameters()).device
    per_tok, agree = [], []
    for b, ref in zip(batches, refs):
        lp = torch.log_softmax(_logits(model, b.to(device)), -1)
        r = ref.to(lp.device).float()
        for s in range(0, lp.shape[0], chunk):
            rp, cp = r[s:s + chunk], lp[s:s + chunk]
            per_tok.append((rp.exp() * (rp - cp)).sum(-1).clamp_min(0))
            agree.append((rp.argmax(-1) == cp.argmax(-1)).float())
    kl = torch.cat(per_tok) if per_tok else torch.zeros(1)
    ag = torch.cat(agree) if agree else torch.ones(1)
    return {"mean": float(kl.mean()), "p99": float(torch.quantile(kl.float().cpu(), 0.99)) if kl.numel() > 1
            else float(kl.max()), "top1_agree": float(ag.mean()), "tokens": int(kl.numel())}
