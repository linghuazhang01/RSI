"""Exact token-level KL(ref || cand) with a local transformers model.

Packed path (pack / reference_packed / kl_to_reference_packed): the profile table (diagnostics.profile_rows); since
2026-11-25 it runs backbone + LM head separately and keeps the reference as final hidden states (see the
"hidden-state chunks" section). Unpacked path (reference / kl_to_reference): the reference implementation the CPU
tests compare against. A candidate's KL
evaluation is rsicore.kl (vLLM), not this module.
"""
from __future__ import annotations

import math

import torch

CHUNK = 512   # log-prob positions per chunk; reference and candidate are chunked alike


def _summary(per_tok: list[torch.Tensor], agree: list[torch.Tensor]) -> dict:
    kl = torch.cat(per_tok) if per_tok else torch.zeros(1)
    ag = torch.cat(agree) if agree else torch.ones(1)
    return {"mean": float(kl.mean()), "p99": float(torch.quantile(kl.float().cpu(), 0.99)) if kl.numel() > 1
            else float(kl.max()), "top1_agree": float(ag.mean()), "tokens": int(kl.numel())}


# ----------------------------------------------------------------------------- test oracle (unpacked)
def _logits(model, ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=ids, use_cache=False)
    return out.logits[0, :-1].float()   # predict tokens 1..T-1


@torch.no_grad()
def reference(model, batches: list[torch.Tensor], device=None, store_dtype=torch.bfloat16) -> list[torch.Tensor]:
    device = device or next(model.parameters()).device
    return [torch.log_softmax(_logits(model, b.to(device)), -1).to(store_dtype) for b in batches]


@torch.no_grad()
def kl_to_reference(model, batches: list[torch.Tensor], refs: list[torch.Tensor], device=None,
                    chunk: int = CHUNK) -> dict:
    device = device or next(model.parameters()).device
    per_tok, agree = [], []
    for b, ref in zip(batches, refs):
        lp = torch.log_softmax(_logits(model, b.to(device)), -1)
        r = ref.to(lp.device).float()
        for s in range(0, lp.shape[0], chunk):
            rp, cp = r[s:s + chunk], lp[s:s + chunk]
            per_tok.append((rp.exp() * (rp - cp)).sum(-1).clamp_min(0))
            agree.append((rp.argmax(-1) == cp.argmax(-1)).float())
    return _summary(per_tok, agree)


# ----------------------------------------------------------------------------- batched (2026-11-21)
# The profile table used to run one context per forward pass (64 contexts x 3 bits per group); with the torch
# fallback of the Gated-DeltaNet layers that left the GPU ~90% idle. pack() right-pads contexts into batches of
# at most `max_tokens` tokens. Every layer is causal (attention and the GDN recurrence / causal conv), so the
# padding only follows a context's real tokens and never changes their logits; the padded positions are dropped.
# Log-probs are produced in chunks of `chunk` positions in the same order for the reference and the candidate,
# so the reference is stored as a list of chunks (tokens x vocab in bf16; since 2026-11-25 final hidden states
# instead, see below).

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


def n_chunks(packed, chunk: int = CHUNK) -> list[int]:
    """Log-prob chunks of each packed batch, as reference_packed yields them (ceil((n - 1) / chunk) per context)."""
    return [sum(math.ceil(max(n - 1, 0) / chunk) for n in lens) for _, lens in packed]


# ----------------------------------------------------------------------------- hidden-state chunks (2026-11-25)
# The full-logits path above kept, per profile worker on Qwen3.8-27B, the reference as full-vocab bf16 log-prob chunks
# (64 contexts x 511 x 248320 x 2 B = 16.2 GB) plus the bf16 logits of a whole packed 8192-token batch (4.07 GB) on
# the GPU, next to the 54.7 GB model; with any foreign memory on the card that OOMed. Now the model is split into
# backbone + LM head: one backbone pass per packed batch gives the final hidden states (the final RMSNorm is inside
# the backbone), and full-vocab log-probs exist only for one CHUNK of positions at a time. The reference is stored as
# its final hidden states per chunk (bf16 [n, hidden], ~0.33 GB in total) and its log-probs are recomputed with the
# same head and slicing, rounded to bf16 and back exactly as the stored log-probs were before.
# Valid because the profile only perturbs decoder linear modules (spec.groups(): q/k/v/o, GDN projections,
# gate/up/down; never lm_head, embed_tokens or the final norm), so the head is M_0's in both passes; the reference's
# final norm is already applied in the stored hidden states. reference_packed records a fingerprint of the head
# weight and kl_to_reference_packed raises if it changed.
# Models without the split (no .model / .lm_head, or a final logit transform such as softcapping / scaling in the
# config, or a forward whose logits differ from lm_head(backbone(.)) on a short prefix) use the full-logits path.

KL_PATH = "hidden_chunks_v2"   # diagnostics.profile_meta records it, so rows of the old path are recomputed on resume
_LOGIT_TRANSFORMS = ("final_logit_softcapping", "attn_logit_softcapping_final", "logit_scale", "logits_scaling",
                     "logit_scaling", "output_logit_scale", "mup_logit_scale")


class PackedRefs(list):
    """Reference chunks of reference_packed, in yield order. kind "hidden": final hidden states (bf16 [n, hidden]);
    kind "logprobs": full-vocab log-probs (store_dtype [n, vocab]; the fallback path). head_fp: fingerprint of the
    LM head weight at reference time (hidden kind)."""
    kind = "logprobs"
    head_fp = None
    store_dtype = torch.bfloat16


def head_split(model):
    """(backbone, lm_head) when logits == lm_head(backbone(input_ids).last_hidden_state) can be assumed from the
    structure (HF *ForCausalLM / Qwen3_5ForConditionalGeneration: .model and an nn.Linear .lm_head, no final logit
    transform in the config), else None. reference_packed also checks it numerically on a short prefix."""
    body, head = getattr(model, "model", None), getattr(model, "lm_head", None)
    if not isinstance(body, torch.nn.Module) or not isinstance(head, torch.nn.Linear):
        return None
    cfg = getattr(model, "config", None)
    cfgs = [c for c in (cfg, getattr(cfg, "text_config", None)) if c is not None]
    for c in cfgs:
        for a in _LOGIT_TRANSFORMS:
            v = getattr(c, a, None)
            if v is not None and v is not False and v != 1:
                return None
    return body, head


def _hidden(body, ids: torch.Tensor) -> torch.Tensor:
    out = body(input_ids=ids, use_cache=False)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def _head_logprobs(head, h: torch.Tensor) -> torch.Tensor:
    """fp32 log-probs of hidden states h [n, hidden]; the bf16 logits and their fp32 copy are freed on the way."""
    x = head(h).float()
    lp = torch.log_softmax(x, -1)
    del x
    return lp


def _head_fingerprint(head) -> torch.Tensor:
    w = head.weight
    return w[:: max(1, w.shape[0] // 1024)].float().sum(-1)    # ~1024 rows; 21 MB fp32 on Qwen3.8-27B


def _split_ok(model, body, head, packed, device) -> bool:
    """The full forward's logits equal lm_head(backbone) on a short prefix of the first context (a few ms)."""
    ids, lens = packed[0]
    pre = ids[:1, :min(lens[0], 16)].to(device)
    a = model(input_ids=pre, use_cache=False).logits[0].float()
    b = head(_hidden(body, pre)[0]).float()
    return bool(torch.allclose(a, b, rtol=1e-2, atol=1e-2 * max(1.0, float(a.abs().max()))))


def _chunks(packed, chunk: int):
    """(batch index, row, s, e) of every log-prob chunk, in the order reference_packed stores them."""
    for bi, (_, lens) in enumerate(packed):
        for r, n in enumerate(lens):
            for s in range(0, n - 1, chunk):                       # predict tokens 1..n-1 of context r
                yield bi, r, s, min(s + chunk, n - 1)


def _packed_hidden(body, packed, device, chunk: int):
    """Final hidden states [e - s, hidden] of every chunk (views of one backbone pass per packed batch)."""
    cur, h = -1, None
    for bi, r, s, e in _chunks(packed, chunk):
        if bi != cur:
            h = None
            h = _hidden(body, packed[bi][0].to(device))
            cur = bi
        yield h[r, s:e]
    del h


def _packed_logprobs(model, packed, device, chunk: int):
    """Full-logits path: fp32 log-probs of every chunk from the model's own logits."""
    for ids, lens in packed:
        logits = model(input_ids=ids.to(device), use_cache=False).logits
        for r, n in enumerate(lens):
            for s in range(0, n - 1, chunk):
                yield torch.log_softmax(logits[r, s:min(s + chunk, n - 1)].float(), -1)
        del logits


@torch.no_grad()
def reference_packed(model, packed, device=None, store_dtype=torch.bfloat16, chunk: int = CHUNK,
                     split: bool = True) -> PackedRefs:
    """The reference chunks of `packed`: final hidden states when the backbone / head split is available (split
    True and head_split(model) and the prefix check pass), else full-vocab log-probs in store_dtype."""
    device = device or next(model.parameters()).device
    out = PackedRefs()
    bh = head_split(model) if split and packed else None
    if bh is not None and _split_ok(model, *bh, packed, device):
        out.kind, out.head_fp = "hidden", _head_fingerprint(bh[1])
        out.store_dtype = store_dtype
        out.extend(h.clone() for h in _packed_hidden(bh[0], packed, device, chunk))   # clone: own [n, hidden] only
        return out
    out.extend(lp.to(store_dtype) for lp in _packed_logprobs(model, packed, device, chunk))
    return out


def _kl_rows(r: torch.Tensor, lp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token KL(ref || cand) and top-1 agreement of fp32 log-probs r, lp [n, vocab]. Same elementwise ops as
    kl_to_reference ((exp(r) * (r - lp)).sum(-1).clamp_min(0)), done in place in r and lp (both are consumed): two
    [n, vocab] fp32 buffers instead of five."""
    agree = (r.argmax(-1) == lp.argmax(-1)).float()
    d = lp.neg_().add_(r)          # r - lp (exact: r + (-lp) == r - lp)
    p = r.exp_()
    kl = d.mul_(p).sum(-1).clamp_min(0)
    return kl, agree


@torch.no_grad()
def kl_to_reference_packed(model, packed, refs, device=None, chunk: int = CHUNK) -> dict:
    """Same result as kl_to_reference on the unpadded contexts (up to bf16 rounding of batched matmuls). refs from
    reference_packed with the same packed / chunk; kind "hidden" needs the model's LM head unchanged since then."""
    device = device or next(model.parameters()).device
    per_tok, agree = [], []
    if getattr(refs, "kind", "logprobs") == "hidden":
        bh = head_split(model)
        if bh is None:
            raise RuntimeError("hidden-state reference but the model has no backbone / lm_head split")
        body, head = bh
        if refs.head_fp is not None and not torch.equal(_head_fingerprint(head), refs.head_fp.to(device)):
            raise RuntimeError("lm_head changed since reference_packed: the hidden-state reference recomputes the "
                               "reference log-probs with the current head, so it needs M_0's head")
        sd = getattr(refs, "store_dtype", torch.bfloat16)
        for h, href in zip(_packed_hidden(body, packed, device, chunk), refs):
            lp = _head_logprobs(head, h)
            r = _head_logprobs(head, href.to(device)).to(sd).float()   # as the stored refs were: bf16 and back
            k, a = _kl_rows(r, lp)
            del r, lp
            per_tok.append(k)
            agree.append(a)
        return _summary(per_tok, agree)
    for lp, ref in zip(_packed_logprobs(model, packed, device, chunk), refs):
        r = ref.to(lp.device, torch.float32, copy=True)        # a copy: _kl_rows works in place
        k, a = _kl_rows(r, lp)
        del r, lp
        per_tok.append(k)
        agree.append(a)
    return _summary(per_tok, agree)
