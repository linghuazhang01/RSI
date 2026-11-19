"""Weight quantizers (torch): RTN, GPTQ, AWQ-lite; bit packing; dequantization.

All quantizers are asymmetric min-max per group along the input dimension (GPTQ/AutoGPTQ style):
    scale = (max(w_g, 0) - min(w_g, 0)) / (2^b - 1)   (stored fp16)
    zero  = round(-min / scale)                        (b-bit int)
    code  = clamp(round(w / scale) + zero, 0, 2^b - 1)
    w'    = (code - zero) * scale
GPTQ (Frantar et al. 2023) uses the Hessian H = sum_x x x^T of the module's input activations
(collected from the CURRENT model on calibration texts, see quant/hessian.py); groups' scales are
computed on the error-compensated weights when the group is reached ("static groups" off).
AWQ-lite: per-input-channel scaling s = rms(x)^alpha (normalised), alpha chosen on a grid to
minimise the H-weighted output error tr((W - W') H (W - W')^T); the dequantized weight is
Q(W diag(s)) diag(s)^-1 (in a real deployment 1/s is folded into the previous op; here it is
folded into the fake-quant weight).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class QResult:
    codes: torch.Tensor          # uint8 [out, in]
    scale: torch.Tensor          # fp16  [out, ngroups]
    zero: torch.Tensor           # uint8 [out, ngroups]
    bits: int
    group_size: int
    col_scale: torch.Tensor | None = None   # fp16 [in] (awq)

    def dequant(self, dtype=torch.bfloat16, device=None) -> torch.Tensor:
        return dequant(self, dtype=dtype, device=device)


def _groups(in_f: int, gs: int) -> int:
    return math.ceil(in_f / gs)


def _qparams(wg: torch.Tensor, bits: int):
    """wg [..., gs] float -> (scale fp16-rounded float, zero float) per group."""
    maxq = 2 ** bits - 1
    xmin = torch.clamp(wg.amin(dim=-1), max=0)
    xmax = torch.clamp(wg.amax(dim=-1), min=0)
    both0 = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both0, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(both0, torch.full_like(xmax, 1.0), xmax)
    scale = ((xmax - xmin) / maxq).to(torch.float16).float()
    scale = torch.where(scale <= 0, torch.full_like(scale, 1e-8), scale)
    zero = torch.clamp(torch.round(-xmin / scale), 0, maxq)
    return scale, zero


def _pad_cols(W: torch.Tensor, gs: int) -> tuple[torch.Tensor, int]:
    in_f = W.shape[1]
    pad = _groups(in_f, gs) * gs - in_f
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    return W, pad


def rtn(W: torch.Tensor, bits: int, group_size: int) -> QResult:
    W = W.float()
    out_f, in_f = W.shape
    Wp, pad = _pad_cols(W, group_size)
    wg = Wp.reshape(out_f, -1, group_size)
    if pad:  # do not let zero padding change the last group's range: replicate its first column
        last = wg[:, -1, :]
        valid = group_size - pad
        last[:, valid:] = last[:, :1]
    scale, zero = _qparams(wg, bits)
    maxq = 2 ** bits - 1
    codes = torch.clamp(torch.round(wg / scale[..., None]) + zero[..., None], 0, maxq)
    codes = codes.reshape(out_f, -1)[:, :in_f].to(torch.uint8)
    return QResult(codes=codes, scale=scale.to(torch.float16), zero=zero.to(torch.uint8), bits=bits,
                   group_size=group_size)


def dequant(q: QResult, dtype=torch.bfloat16, device=None) -> torch.Tensor:
    codes = q.codes.to(device) if device is not None else q.codes
    out_f, in_f = codes.shape
    gs = q.group_size
    scale = q.scale.to(codes.device).float()
    zero = q.zero.to(codes.device).float()
    idx = torch.arange(in_f, device=codes.device) // gs
    w = (codes.float() - zero[:, idx]) * scale[:, idx]
    if q.col_scale is not None:
        w = w / q.col_scale.to(codes.device).float()[None, :]
    return w.to(dtype)


def h_error(W: torch.Tensor, Wq: torch.Tensor, H: torch.Tensor) -> float:
    """tr((W - Wq) H (W - Wq)^T): squared output error summed over the calibration tokens."""
    D = (W.float() - Wq.float())
    return float(((D @ H.float()) * D).sum())


def _prep_hessian(H: torch.Tensor, W: torch.Tensor, damp: float):
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W = W.clone()
    W[:, dead] = 0
    d = damp * torch.mean(torch.diag(H))
    idx = torch.arange(H.shape[0], device=H.device)
    H[idx, idx] += d
    return H, W


def gptq(W: torch.Tensor, H: torch.Tensor, bits: int, group_size: int, *, damp: float = 0.01,
         block_size: int = 128) -> QResult:
    """GPTQ with lazy batch updates. W [out, in] (any float dtype), H [in, in] (sum of x x^T)."""
    dev = W.device
    W = W.float()
    out_f, in_f = W.shape
    H = H.to(dev)
    H, W = _prep_hessian(H, W, damp)
    for attempt in range(5):
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            break
        except RuntimeError:  # not positive definite: add more damping
            idx = torch.arange(in_f, device=dev)
            H[idx, idx] += damp * torch.mean(torch.diag(H)) * (10 ** attempt)
    else:
        raise RuntimeError("GPTQ: Hessian not positive definite even after extra damping")
    maxq = 2 ** bits - 1
    ng = _groups(in_f, group_size)
    scales = torch.zeros(out_f, ng, device=dev)
    zeros = torch.zeros(out_f, ng, device=dev)
    codes = torch.zeros(out_f, in_f, device=dev)
    for i1 in range(0, in_f, block_size):
        i2 = min(i1 + block_size, in_f)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            col = i1 + i
            if col % group_size == 0:
                g = col // group_size
                gend = min(col + group_size, in_f)
                # current (error-compensated) weights of this group: part in W1, rest in W
                seg = torch.cat([W1[:, i:min(count, i + gend - col)], W[:, i2:gend]], dim=1) \
                    if gend > i2 else W1[:, i:i + (gend - col)]
                s, z = _qparams(seg[:, None, :], bits)
                scales[:, g], zeros[:, g] = s[:, 0], z[:, 0]
            g = col // group_size
            s, z = scales[:, g], zeros[:, g]
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / s) + z, 0, maxq)
            codes[:, col] = q
            wq = (q - z) * s
            err = (w - wq) / d
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return QResult(codes=codes.to(torch.uint8), scale=scales.to(torch.float16), zero=zeros.to(torch.uint8),
                   bits=bits, group_size=group_size)


def awq_lite(W: torch.Tensor, H: torch.Tensor, n_tokens: int, bits: int, group_size: int,
             grid: int = 11) -> QResult:
    dev = W.device
    W = W.float()
    H = H.to(dev).float()
    rms = torch.sqrt(torch.clamp(torch.diag(H) / max(n_tokens, 1), min=1e-12))
    best, best_err = None, float("inf")
    for k in range(grid):
        alpha = k / (grid - 1)
        s = rms ** alpha
        s = s / torch.sqrt(s.max() * s.min())
        s16 = s.to(torch.float16)
        s = s16.float()
        q = rtn(W * s[None, :], bits, group_size)
        q.col_scale = s16
        err = h_error(W, dequant(q, torch.float32), H)
        if err < best_err:
            best, best_err = q, err
    return best


def quantize(W: torch.Tensor, quantizer: str, bits: int, group_size: int, *, H: torch.Tensor | None = None,
             n_tokens: int = 0, damp: float = 0.01) -> QResult:
    if quantizer == "rtn":
        return rtn(W, bits, group_size)
    if H is None:
        raise ValueError(f"quantizer {quantizer!r} needs a Hessian (calibration statistics)")
    if quantizer == "gptq":
        return gptq(W, H, bits, group_size, damp=damp)
    if quantizer == "awq":
        return awq_lite(W, H, n_tokens, bits, group_size)
    raise ValueError(f"unknown quantizer {quantizer!r}")


# --------------------------------------------------------------------------------- packing
def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """uint8 codes (< 2^bits) -> packed uint8 bytes (8 codes per `bits` bytes, little endian)."""
    flat = codes.reshape(-1).to(torch.int64)
    n = flat.numel()
    pad = (-n) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    v = flat.reshape(-1, 8)
    word = torch.zeros(v.shape[0], dtype=torch.int64, device=v.device)
    for j in range(8):
        word |= v[:, j] << (bits * j)
    by = torch.stack([(word >> (8 * k)) & 0xFF for k in range(bits)], dim=1)
    return by.to(torch.uint8).reshape(-1)


def unpack(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    by = packed.reshape(-1, bits).to(torch.int64)
    word = torch.zeros(by.shape[0], dtype=torch.int64, device=by.device)
    for k in range(bits):
        word |= by[:, k] << (8 * k)
    mask = (1 << bits) - 1
    v = torch.stack([(word >> (bits * j)) & mask for j in range(8)], dim=1).reshape(-1)
    return v[:numel].to(torch.uint8)
