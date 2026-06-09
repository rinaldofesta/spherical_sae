"""Calibrated-intervention utilities for the dose-response pilot (PR #1 / M1).

Turning the knob means rewriting the sparse code so a target concept c expresses an
*exact commanded share* p, then decoding the rewritten code WITHOUT a second
normalisation (see SphericalSAE.decode_from_shares). Three operators, one per mode,
all hitting the same code-level target so the baseline is a steelman:

  l1      -- set_share: zero c, renormalise the rest to (1-p), write p.
             Returns a point ON the simplex (sums to 1).
  softmax -- set_share: closed-form logit so softmax(support u {c})[c] = p.
             Returns the resulting softmax distribution (sums to 1).
  none    -- set_share (steelman): pick the magnitude h[c] = p/(1-p) * sum_others so
             the IMPLIED L1 share of c equals p. Keeps magnitudes (decoded as-is).

All operators are batched: h is (B, n_latents), c an int, p a float in (0, 1).
"""
from __future__ import annotations

import torch

EPS = 1e-8


def _as_2d(h: torch.Tensor) -> torch.Tensor:
    return h.unsqueeze(0) if h.dim() == 1 else h


def _l1_set_share(h: torch.Tensor, c: int, p: float) -> torch.Tensor:
    h = _as_2d(h).clone().float()
    h[:, c] = 0.0
    denom = h.sum(dim=1, keepdim=True).clamp(min=EPS)
    s = h / denom * (1.0 - p)
    s[:, c] = p
    return s


def _softmax_set_share(h: torch.Tensor, c: int, p: float) -> torch.Tensor:
    h = _as_2d(h).clone().float()
    support = h != 0
    support[:, c] = False                       # treat c separately
    neg_inf = torch.finfo(h.dtype).min
    other_logits = torch.where(support, h, torch.full_like(h, neg_inf))
    lse_others = torch.logsumexp(other_logits, dim=1, keepdim=True)  # (B,1)
    # want exp(l_c)/(exp(l_c)+Z_others) = p  ->  l_c = logit(p) + log Z_others
    logit_p = torch.log(torch.tensor(p / (1.0 - p)))
    l_c = logit_p + lse_others                  # (B,1)
    logits = other_logits.clone()
    logits[:, c:c + 1] = l_c
    probs = torch.softmax(logits, dim=1)
    return probs


def _none_set_share(h: torch.Tensor, c: int, p: float) -> torch.Tensor:
    h = _as_2d(h).clone().float()
    others = h.clone()
    others[:, c] = 0.0
    S = others.sum(dim=1, keepdim=True)         # (B,1)
    h[:, c:c + 1] = (p / (1.0 - p)) * S
    return h


def set_share(h: torch.Tensor, c: int, p: float, mode: str) -> torch.Tensor:
    """Rewrite code(s) h so concept c expresses commanded share p (mode-aware)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"share p must be in (0,1), got {p}")
    if mode == "l1":
        return _l1_set_share(h, c, p)
    if mode == "softmax":
        return _softmax_set_share(h, c, p)
    if mode == "none":
        return _none_set_share(h, c, p)
    raise ValueError(f"set_share: unsupported mode {mode!r} (use l1/softmax/none)")


def _l1_set_mix(h: torch.Tensor, targets: dict[int, float]) -> torch.Tensor:
    h = _as_2d(h).clone().float()
    rem = 1.0 - sum(targets.values())
    for c in targets:
        h[:, c] = 0.0
    denom = h.sum(dim=1, keepdim=True).clamp(min=EPS)
    s = h / denom * rem
    for c, p in targets.items():
        s[:, c] = p
    return s


def _none_set_mix(h: torch.Tensor, targets: dict[int, float]) -> torch.Tensor:
    h = _as_2d(h).clone().float()
    sp = sum(targets.values())
    others = h.clone()
    for c in targets:
        others[:, c] = 0.0
    O = others.sum(dim=1, keepdim=True)         # (B,1)
    T = O / (1.0 - sp)                          # implied total so each share hits p
    for c, p in targets.items():
        h[:, c:c + 1] = p * T
    return h


def set_mix(h: torch.Tensor, targets: dict[int, float], mode: str) -> torch.Tensor:
    """Rewrite code(s) so several concepts simultaneously express commanded shares.

    targets maps concept index -> share; their sum must be < 1 (the remainder goes to
    the other active features, keeping their relative mix).
    """
    sp = sum(targets.values())
    if not 0.0 < sp < 1.0:
        raise ValueError(f"set_mix: target shares must sum to (0,1), got {sp}")
    if any(not 0.0 < p < 1.0 for p in targets.values()):
        raise ValueError("set_mix: each share must be in (0,1)")
    if mode == "l1":
        return _l1_set_mix(h, targets)
    if mode == "none":
        return _none_set_mix(h, targets)
    raise ValueError(f"set_mix: unsupported mode {mode!r} (use l1/none)")


__all__ = ["set_share", "set_mix"]
