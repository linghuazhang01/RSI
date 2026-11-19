"""LoRA / adapter-stack operators (METHOD_DESIGN §3.3). All return a NEW LoRA (inputs untouched).

scale, interpolate, combine, drop, truncate_rank, intruder_scale, relchange_prune, regmean_preserve,
orthogonalize, ties; plus intruder_scores() used by diagnostics and feedback.
"""
from __future__ import annotations

from .lora import LoRA


def _sel(lora: LoRA, modules):
    if modules is None:
        return set(lora.factors)
    if isinstance(modules, str):
        modules = [modules]
    out = set()
    for pat in modules:
        for m in lora.factors:
            if m == pat or m.startswith(pat + ".") or pat in m:
                out.add(m)
    return out


def scale(lora: LoRA, s: float, modules=None) -> LoRA:
    """ΔW ← s·ΔW on the selected modules (all by default). `modules` entries are exact names or
    substrings (e.g. 'mlp', 'layers.40.')."""
    out = lora.copy()
    for m in _sel(out, modules):
        A, B = out.factors[m]
        out.factors[m] = (A, B * float(s))
    out.meta.setdefault("ops", []).append(["scale", s])
    return out


def combine(loras, weights=None, cap: int | None = None) -> LoRA:
    """Σ_i w_i ΔW_i (ranks concatenated, then optionally re-compressed to ≤ cap)."""
    weights = weights or [1.0] * len(loras)
    out = LoRA(meta={"ops": [["combine", list(map(float, weights))]]})
    for lo, w in zip(loras, weights):
        for m, (A, B) in lo.factors.items():
            out.add(m, A.clone(), B * float(w))
    if cap is not None:
        out.recompress(cap)
    return out


def interpolate(a: LoRA, b: LoRA, lam: float, cap: int | None = None) -> LoRA:
    """(1−λ)·ΔW_a + λ·ΔW_b (checkpoint interpolation, or blending with a historical adapter)."""
    out = combine([a, b], [1.0 - float(lam), float(lam)], cap=cap)
    out.meta["ops"] = [["interpolate", lam]]
    return out


def drop(lora: LoRA, modules) -> LoRA:
    out = lora.copy()
    for m in _sel(out, modules):
        out.factors.pop(m)
    return out


def truncate_rank(lora: LoRA, k: int, modules=None) -> LoRA:
    """Keep the top-k singular directions of ΔW per module (PARA: cutting the tail is safe)."""
    out = lora.copy()
    for m in _sel(out, modules):
        U, S, V = out.compact(m)
        out.set_from_svd(m, U, S, V, k=int(k), tol=1e-7)
    out.meta.setdefault("ops", []).append(["truncate_rank", k])
    return out


def intruder_scores(lora: LoRA, short: str, U0) -> tuple[list[float], list[float]]:
    """For each singular direction u_i of ΔW: max_j |<u_i, U0_j>| against the top-K left singular vectors
    of the base weight (high = aligned with W0's dominant subspace; low = 'intruder' direction).
    Returns (singular values of ΔW, alignment per direction)."""
    U, S, _ = lora.compact(short)
    if S.numel() == 0:
        return [], []
    align = (U.T @ U0.to(U.dtype)).abs().max(dim=1).values
    return S.tolist(), align.tolist()


def intruder_scale(lora: LoRA, store, lam: float = 0.5, eps: float = 0.3, k0: int = 32, modules=None) -> LoRA:
    """Scale the intruder directions of ΔW (alignment with W0's top-k0 left singular subspace < eps) by λ
    (Intruder-dimension paper: shrinking intruders reduces forgetting; shrinking aligned ones increases it)."""
    out = lora.copy()
    n_scaled = 0
    for m in _sel(out, modules):
        U0, _, _ = store.svd(m, k0)
        U, S, V = out.compact(m)
        if S.numel() == 0:
            continue
        align = (U.T @ U0).abs().max(dim=1).values
        f = (align < eps).float() * (float(lam) - 1.0) + 1.0
        n_scaled += int((align < eps).sum())
        out.set_from_svd(m, U, S * f, V, tol=1e-7)
    out.meta.setdefault("ops", []).append(["intruder_scale", lam, eps, k0, n_scaled])
    return out


def relchange_prune(lora: LoRA, store, frac: float = 0.03, modules=None, device: str = "cpu") -> LoRA:
    """FAPM-style: zero the `frac` entries of ΔW with the largest relative change |ΔW|/|W0| (per module),
    then re-factor the pruned dense ΔW back to the module's original rank (truncated SVD)."""
    import torch
    out = lora.copy()
    for m in _sel(out, modules):
        r = out.rank(m)
        W0 = store.weight(m, device=device)
        dW = out.delta(m).to(device)
        ratio = dW.abs() / (W0.abs() + 1e-8)
        n = ratio.numel()
        kth = max(1, int(round(n * (1.0 - float(frac)))))
        thr = ratio.flatten().kthvalue(kth).values
        dW = torch.where(ratio > thr, torch.zeros_like(dW), dW)
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        out.set_from_svd(m, U[:, :r].cpu(), S[:r].cpu(), Vh[:r].T.cpu(), tol=1e-7)
    out.meta.setdefault("ops", []).append(["relchange_prune", frac])
    return out


def regmean_preserve(lora: LoRA, grams_target: dict, grams_preserve: dict, alpha: float = 1.0,
                     modules=None, ridge: float = 1e-4) -> LoRA:
    """RegMean-style preservation with the base as a merge member:
        W* = argmin_W ||(W − W0 − ΔW) X_t||² + α ||(W − W0) X_r||²   ⇒   ΔW* = ΔW G_t (G_t + α G_r)^{-1}
    computed in factored form A' = A G_t (G_t + α G_r)^{-1} (input Grams G = Σ x xᵀ, from acts/gram)."""
    import torch
    out = lora.copy()
    for m in _sel(out, modules):
        if m not in grams_target or m not in grams_preserve:
            continue
        Gt, Gr = grams_target[m].float(), grams_preserve[m].float()
        Gt = Gt / max(float(torch.trace(Gt)), 1e-12)
        Gr = Gr / max(float(torch.trace(Gr)), 1e-12)
        M = Gt + float(alpha) * Gr
        M = M + ridge * torch.eye(M.shape[0]) * float(torch.trace(M)) / M.shape[0]
        A, B = out.factors[m]
        # A' = A Gt M^{-1}  <=>  M^T A'^T = (A Gt)^T
        At = torch.linalg.solve(M.T, (A @ Gt).T)
        out.factors[m] = (At.T.contiguous(), B)
    out.meta.setdefault("ops", []).append(["regmean_preserve", alpha])
    return out


def orthogonalize(lora: LoRA, others, modules=None) -> LoRA:
    """Project the output (column) space of ΔW off the column spaces of the given adapters
    (O-LoRA / TSV-style de-interference against the adapter stack)."""
    import torch
    out = lora.copy()
    for m in _sel(out, modules):
        cols = [o.factors[m][1] for o in others if m in o.factors]
        if not cols:
            continue
        Q, _ = torch.linalg.qr(torch.cat(cols, 1))
        A, B = out.factors[m]
        out.factors[m] = (A, B - Q @ (Q.T @ B))
    out.meta.setdefault("ops", []).append(["orthogonalize", len(others)])
    return out


def ties(loras, weights=None, density: float = 0.2, cap: int = 64, device: str = "cpu") -> LoRA:
    """TIES merge on dense ΔW per module: trim to top-density magnitudes, elect sign by summed mass,
    disjoint mean of agreeing entries, scaled by mean weight; re-factored to rank ≤ cap."""
    import torch
    weights = weights or [1.0] * len(loras)
    out = LoRA(meta={"ops": [["ties", density, list(map(float, weights))]]})
    mods = sorted(set().union(*[set(lo.factors) for lo in loras]))
    for m in mods:
        ds = []
        for lo, w in zip(loras, weights):
            if m in lo.factors:
                d = lo.delta(m).to(device) * float(w)
                k = max(1, int(round(d.numel() * float(density))))
                thr = d.abs().flatten().kthvalue(d.numel() - k + 1).values
                ds.append(torch.where(d.abs() >= thr, d, torch.zeros_like(d)))
        D = torch.stack(ds)
        sign = torch.sign(D.sum(0))
        agree = (torch.sign(D) == sign) & (D != 0)
        merged = (D * agree).sum(0) / agree.sum(0).clamp_min(1)
        U, S, Vh = torch.linalg.svd(merged, full_matrices=False)
        out.set_from_svd(m, U[:, :cap].cpu(), S[:cap].cpu(), Vh[:cap].T.cpu(), tol=1e-7)
    return out
