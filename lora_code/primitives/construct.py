"""Gradient-free LoRA constructors (RSI_LOOP_DESIGN §3.1): rank-1 edits, activation-direction edits,
singular-direction scaling (Transformer²-style).

All functions append a low-rank term to `lora` in place and return the number of rank units added.
"""
from __future__ import annotations


def _unit(v):
    import torch
    v = v.float().flatten()
    n = v.norm()
    if not torch.isfinite(n) or n <= 0:
        raise ValueError("zero / non-finite direction")
    return v / n


def rank1(lora, short: str, u, v, s: float, *, shape=None) -> int:
    """ΔW += s · û v̂ᵀ   (û in the output space, v̂ in the input space; both normalised)."""
    u, v = _unit(u), _unit(v)
    if shape is not None and (u.numel() != shape[0] or v.numel() != shape[1]):
        raise ValueError(f"rank1 on {short}: u has {u.numel()} (need {shape[0]}), v has {v.numel()} (need {shape[1]})")
    lora.add(short, v[None, :] * 1.0, (u * float(s))[:, None])
    return 1


def from_direction(lora, short: str, direction, key, alpha: float, *, shape=None) -> int:
    """Activation-steering as a weight edit: ΔW = α · d̂ k̂ᵀ / ||k||, so that ΔW k = α d̂ for the typical
    input k (e.g. the mean input activation from acts()). `direction` lives in the output space."""
    d = _unit(direction)
    k = key.float().flatten()
    kn = float(k.norm())
    if kn <= 0:
        raise ValueError("zero key vector")
    if shape is not None and (d.numel() != shape[0] or k.numel() != shape[1]):
        raise ValueError(f"from_direction on {short}: direction {d.numel()} (need {shape[0]}), key {k.numel()} (need {shape[1]})")
    lora.add(short, (k / kn / kn)[None, :], (d * float(alpha))[:, None])
    return 1


def scale_singular(lora, short: str, U, S, V, idx, factors) -> int:
    """ΔW = Σ_i (f_i − 1) σ_i u_i v_iᵀ over the chosen singular directions of W (U,S,V from svd())."""
    import torch
    idx = list(idx)
    if isinstance(factors, (int, float)):
        factors = [float(factors)] * len(idx)
    if len(factors) != len(idx):
        raise ValueError("idx and factors must have the same length")
    if not idx:
        return 0
    if max(idx) >= S.numel() or min(idx) < 0:
        raise ValueError(f"singular index out of range (have {S.numel()})")
    f = torch.tensor([float(x) - 1.0 for x in factors])
    sig = S[idx] * f
    keep = sig != 0
    if not keep.any():
        return 0
    Ui, Vi, sig = U[:, idx][:, keep], V[:, idx][:, keep], sig[keep]
    lora.add(short, Vi.T.contiguous(), (Ui * sig).contiguous())
    return int(keep.sum())
