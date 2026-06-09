"""Calibration-map math for the M2 harness (PR #2).

One shared monotone dose-response curve per arm: pool measurements from all
CALIBRATION concepts (knob -> realised share), group repeated knobs by mean,
smooth with pool-adjacent-violators. The inverse of that curve is the single
map g: p_requested -> knob, applied UNCHANGED to held-out concepts. Fitting is
symmetric: every arm (simplex, none-steelman, no-SAE interpolation) gets the
exact same treatment.

Pure numpy except interpolate_toward (the no-SAE kill-control operator).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pav_isotonic(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Non-decreasing least-squares fit via pool-adjacent-violators."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    # blocks of (mean, weight, count), merged while out of order
    means, weights, counts = [], [], []
    for yi, wi in zip(y, w):
        means.append(yi); weights.append(wi); counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2, c2 = means.pop(), weights.pop(), counts.pop()
            m1, w1, c1 = means.pop(), weights.pop(), counts.pop()
            wt = w1 + w2
            means.append((m1 * w1 + m2 * w2) / wt)
            weights.append(wt); counts.append(c1 + c2)
    out = np.empty_like(y)
    i = 0
    for m, c in zip(means, counts):
        out[i:i + c] = m
        i += c
    return out


def fit_shared_map(knobs: np.ndarray, realized: np.ndarray) -> dict:
    """Pooled measurements -> one monotone curve {knobs (sorted unique), r_hat}."""
    knobs = np.asarray(knobs, dtype=np.float64)
    realized = np.asarray(realized, dtype=np.float64)
    uq = np.unique(knobs)
    means = np.array([realized[knobs == q].mean() for q in uq])
    return {"knobs": uq, "r_hat": pav_isotonic(means)}


def map_command(curve: dict, p: float) -> float:
    """g(p): the smallest knob whose smoothed realised share reaches p.

    Linear interpolation between the bracketing grid points; clamped to the
    achievable range. On flat (pooled) segments this returns the smallest knob
    reaching p -- the conservative command.
    """
    knobs = np.asarray(curve["knobs"], dtype=np.float64)
    r = np.asarray(curve["r_hat"], dtype=np.float64)
    if p <= r[0]:
        return float(knobs[0])
    if p >= r[-1]:
        return float(knobs[-1])
    j = int(np.searchsorted(r, p, side="left"))   # first index with r[j] >= p
    if r[j] == p:
        return float(knobs[j])
    q0, q1, r0, r1 = knobs[j - 1], knobs[j], r[j - 1], r[j]
    return float(q0 + (p - r0) / (r1 - r0) * (q1 - q0))


def mace(realized: np.ndarray, targets: np.ndarray) -> float:
    """Mean absolute calibration error |p - realised share|."""
    return float(np.mean(np.abs(np.asarray(realized) - np.asarray(targets))))


def monotonicity_violation_rate(values: np.ndarray, tol: float = 0.0) -> float:
    """Fraction of adjacent steps where the realised share DECREASES."""
    d = np.diff(np.asarray(values, dtype=np.float64))
    if d.size == 0:
        return 0.0
    return float((d < -tol).mean())


def ols_slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS fit y ~ x -> (slope, intercept, R^2).

    A perfectly flat response gets R^2 = 0, not 1: flat realised share is the
    binary-switch null and must not score as a perfect dose-response.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), r2


def leak_metrics(target_ctrl: float, target_after: float,
                 off_ctrl: np.ndarray, off_after: np.ndarray):
    """Off-target leak of one intervention -> (raw, geometric-subtracted residual).

    raw = sum_j |Δshare_j| / Δshare_target  (j = other panel concepts). Part of
    that displacement is GEOMETRIC, not entanglement: gaining Δ on the target
    must squeeze the rest of the mix proportionally, geo_j = share_j(ctrl) *
    Δtarget / (1 - share_target(ctrl)) -- the retrieval-level mirror of the
    code-level s_j * p/(1-p_old) renormalisation (plan section 2). The residual
    counts only the displacement IN EXCESS of that. Returns (None, None) when
    the knob produced no target gain (leak undefined).
    """
    gain = target_after - target_ctrl
    if gain <= 1e-9:
        return None, None
    off_ctrl = np.asarray(off_ctrl, dtype=np.float64)
    off_after = np.asarray(off_after, dtype=np.float64)
    deltas = np.abs(off_after - off_ctrl)
    geo = off_ctrl * gain / max(1.0 - target_ctrl, 1e-9)
    raw = float(deltas.sum() / gain)
    resid = float(np.maximum(deltas - geo, 0.0).sum() / gain)
    return raw, resid


def interpolate_toward(Q: torch.Tensor, centroid: torch.Tensor, w: float) -> torch.Tensor:
    """No-SAE kill control: q' = normalize((1-w)*q + w*centroid), batched."""
    mixed = (1.0 - w) * Q + w * centroid.unsqueeze(0)
    return F.normalize(mixed, dim=-1)


__all__ = ["pav_isotonic", "fit_shared_map", "map_command", "mace",
           "monotonicity_violation_rate", "ols_slope_r2", "leak_metrics",
           "interpolate_toward"]
