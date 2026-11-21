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


# ----------------------------------------------------------------------------- batched (2026-11-21)
# The profile table used to run one context per forward pass (64 contexts x 3 bits per group); with the torch
# fallback of the Gated-DeltaNet layers that left the GPU ~90% idle. pack() right-pads contexts into batches of
# at most `max_tokens` tokens. Every layer is causal (attention and the GDN recurrence / causal conv), so the
# padding only follows a context's real tokens and never changes their logits; the padded positions are dropped.
# Log-probs are produced in chunks of `chunk` positions in the same order for the reference and the candidate,
# so the reference is stored as a list of chunks (same total memory as before: tokens x vocab in bf16).

def pack(batches: list[torch.Tensor], max_tokens: int = 8192, pad_id: int = 0) -> list[tuple[torch.Tensor, list[int]]]:
    """[1, T] tensors -> [(ids [B, T_max] right-padded, lengths)], longest first, B x T_max <= max_tokens (a
    single context longer than max_tokens gets its own batch)."""
    order = sorted(range(len(batches)), key=lambda i: -batches[i].shape[-1])
    groups, cur = [], []
    for i in order:
        t_max = batches[cur[0]].shape[-1] if cur else batches[i].shape[-1]
        if cur and t_max * (len(cur) + 1) > max_tokens:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        lens = [int(batches[i].shape[-1]) for i in g]
        ids = torch.full((len(g), max(lens)), pad_id, dtype=torch.long)
        for r, i in enumerate(g):
            ids[r, :lens[r]] = batches[i][0]
        out.append((ids, lens))
    return out


def _packed_logprobs(model, packed, device, chunk: int):
    for ids, lens in packed:
        logits = model(input_ids=ids.to(device), use_cache=False).logits
        for r, n in enumerate(lens):
            for s in range(0, n - 1, chunk):                       # predict tokens 1..n-1 of context r
                yield torch.log_softmax(logits[r, s:min(s + chunk, n - 1)].float(), -1)
        del logits


@torch.no_grad()
def reference_packed(model, packed, device=None, store_dtype=torch.bfloat16, chunk: int = 512) -> list[torch.Tensor]:
    device = device or next(model.parameters()).device
    return [lp.to(store_dtype) for lp in _packed_logprobs(model, packed, device, chunk)]


@torch.no_grad()
def kl_to_reference_packed(model, packed, refs: list[torch.Tensor], device=None, chunk: int = 512) -> dict:
    """Same result as kl_to_reference on the unpadded contexts (up to bf16 rounding of batched matmuls)."""
    device = device or next(model.parameters()).device
    per_tok, agree = [], []
    for lp, ref in zip(_packed_logprobs(model, packed, device, chunk), refs):
        r = ref.to(lp.device).float()
        per_tok.append((r.exp() * (r - lp)).sum(-1).clamp_min(0))
        agree.append((r.argmax(-1) == lp.argmax(-1)).float())
    kl = torch.cat(per_tok) if per_tok else torch.zeros(1)
    ag = torch.cat(agree) if agree else torch.ones(1)
    return {"mean": float(kl.mean()), "p99": float(torch.quantile(kl.float().cpu(), 0.99)) if kl.numel() > 1
            else float(kl.max()), "top1_agree": float(ag.mean()), "tokens": int(kl.numel())}
