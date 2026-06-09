"""Unit tests for the calibration-map math (PR #2 / M2).

Run:  .venv/bin/python -m pytest scripts/proof/test_calibmap.py -q

These pin the invariants the calibration harness relies on:
  - pav_isotonic is a correct pool-adjacent-violators fit (non-decreasing,
    pools violators by weighted mean, leaves monotone input untouched);
  - fit_shared_map groups repeated knob values (across concepts) by mean and
    smooths with PAV -> ONE monotone dose-response curve per arm;
  - map_command inverts that curve: commanding g(p) realises ~p, clamps
    outside the achievable range, and on flat segments returns the SMALLEST
    knob that reaches p (conservative);
  - mace / monotonicity_violation_rate / ols_slope_r2 match hand-computed values;
  - interpolate_toward is the no-SAE kill-control operator:
    normalize((1-w)*q + w*centroid), exact at both endpoints, always unit-norm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

PROOF = Path(__file__).resolve().parent
if str(PROOF) not in sys.path:
    sys.path.insert(0, str(PROOF))

import _calibmap  # noqa: E402

TOL = 1e-9


# ------------------------------------------------------------- pav_isotonic
def test_pav_isotonic_keeps_monotone_input():
    y = np.array([0.1, 0.2, 0.5, 0.9])
    out = _calibmap.pav_isotonic(y)
    assert np.allclose(out, y, atol=TOL)


def test_pav_isotonic_pools_adjacent_violators():
    y = np.array([1.0, 3.0, 2.0])
    out = _calibmap.pav_isotonic(y)
    assert np.allclose(out, [1.0, 2.5, 2.5], atol=TOL)


def test_pav_isotonic_weighted_pool():
    y = np.array([3.0, 1.0])
    w = np.array([1.0, 3.0])
    out = _calibmap.pav_isotonic(y, w)
    assert np.allclose(out, [1.5, 1.5], atol=TOL)


def test_pav_isotonic_output_is_nondecreasing_on_noise():
    rng = np.random.default_rng(0)
    y = np.sort(rng.random(50)) + 0.3 * rng.standard_normal(50)
    out = _calibmap.pav_isotonic(y)
    assert (np.diff(out) >= -TOL).all()


# ----------------------------------------------------------- fit_shared_map
def test_fit_shared_map_groups_repeated_knobs_by_mean():
    # two concepts measured at the same two knob values
    knobs = np.array([0.1, 0.2, 0.1, 0.2])
    realized = np.array([0.2, 0.25, 0.4, 0.35])
    curve = _calibmap.fit_shared_map(knobs, realized)
    assert np.allclose(curve["knobs"], [0.1, 0.2], atol=TOL)
    # group means: 0.3 at both knobs -> PAV leaves [0.3, 0.3]
    assert np.allclose(curve["r_hat"], [0.3, 0.3], atol=TOL)


def test_fit_shared_map_smooths_nonmonotone_group_means():
    knobs = np.array([0.1, 0.2, 0.3])
    realized = np.array([0.1, 0.5, 0.3])
    curve = _calibmap.fit_shared_map(knobs, realized)
    assert (np.diff(curve["r_hat"]) >= -TOL).all()
    assert np.allclose(curve["r_hat"], [0.1, 0.4, 0.4], atol=TOL)


# ------------------------------------------------------------- map_command
def _sigmoid_curve(n=25, lo=0.02, hi=0.6):
    knobs = np.linspace(lo, hi, n)
    r = 1.0 / (1.0 + np.exp(-(knobs - 0.2) / 0.04))
    return _calibmap.fit_shared_map(knobs, r)


def test_map_command_inverts_a_sigmoid():
    curve = _sigmoid_curve()
    for p in (0.15, 0.3, 0.5, 0.8):
        q = _calibmap.map_command(curve, p)
        realized = float(np.interp(q, curve["knobs"], curve["r_hat"]))
        assert abs(realized - p) < 1e-6


def test_map_command_clamps_to_achievable_range():
    curve = _sigmoid_curve()
    assert _calibmap.map_command(curve, 1e-9) == pytest.approx(curve["knobs"][0])
    assert _calibmap.map_command(curve, 1.0) == pytest.approx(curve["knobs"][-1])


def test_map_command_flat_segment_returns_smallest_knob():
    curve = {"knobs": np.array([1.0, 2.0, 3.0, 4.0]),
             "r_hat": np.array([0.1, 0.5, 0.5, 0.9])}
    assert _calibmap.map_command(curve, 0.5) == pytest.approx(2.0)


# ------------------------------------------------------------------ metrics
def test_mace_is_mean_absolute_error():
    realized = np.array([0.12, 0.18, 0.35])
    targets = np.array([0.10, 0.20, 0.30])
    assert _calibmap.mace(realized, targets) == pytest.approx((0.02 + 0.02 + 0.05) / 3)


def test_monotonicity_violation_rate_counts_decreases():
    vals = np.array([0.1, 0.2, 0.15, 0.3])  # one decrease over three steps
    assert _calibmap.monotonicity_violation_rate(vals) == pytest.approx(1 / 3)


def test_monotonicity_violation_rate_zero_for_monotone():
    assert _calibmap.monotonicity_violation_rate(np.array([0.1, 0.1, 0.2])) == 0.0


def test_ols_slope_r2_on_perfect_line():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = 2.0 * x + 0.05
    slope, intercept, r2 = _calibmap.ols_slope_r2(x, y)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(0.05)
    assert r2 == pytest.approx(1.0)


def test_ols_slope_r2_flat_response_scores_zero():
    """A flat dose-response is the binary-switch null; it must NOT get R2=1."""
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.full(4, 0.3)
    slope, _, r2 = _calibmap.ols_slope_r2(x, y)
    assert slope == pytest.approx(0.0)
    assert r2 == 0.0


# ------------------------------------------------------------ off-target leak
def test_leak_metrics_pure_proportional_displacement_has_zero_residual():
    """If off-target shares shrink exactly in proportion to the target's gain
    (the geometric component of the intervention), the residual leak is 0."""
    raw, resid = _calibmap.leak_metrics(
        target_ctrl=0.0, target_after=0.5,
        off_ctrl=np.array([0.4]), off_after=np.array([0.2]))
    assert raw == pytest.approx(0.4)        # |0.2-0.4| / 0.5
    assert resid == pytest.approx(0.0)


def test_leak_metrics_excess_displacement_is_residual():
    raw, resid = _calibmap.leak_metrics(
        target_ctrl=0.0, target_after=0.5,
        off_ctrl=np.array([0.4]), off_after=np.array([0.1]))
    assert raw == pytest.approx(0.6)        # |0.1-0.4| / 0.5
    # geometric part: 0.4 * 0.5 / (1-0) = 0.2; excess |Δ|-geo = 0.1 -> /0.5
    assert resid == pytest.approx(0.2)


def test_leak_metrics_no_target_gain_returns_none():
    raw, resid = _calibmap.leak_metrics(
        target_ctrl=0.3, target_after=0.3,
        off_ctrl=np.array([0.4]), off_after=np.array([0.2]))
    assert raw is None and resid is None


# ------------------------------------------------- interpolate_toward (no-SAE)
def test_interpolate_toward_is_unit_norm():
    torch.manual_seed(0)
    Q = F.normalize(torch.randn(7, 16), dim=-1)
    c = F.normalize(torch.randn(16), dim=-1)
    out = _calibmap.interpolate_toward(Q, c, 0.37)
    assert torch.allclose(out.norm(dim=-1), torch.ones(7), atol=1e-5)


def test_interpolate_toward_endpoints():
    torch.manual_seed(1)
    Q = F.normalize(torch.randn(3, 8), dim=-1)
    c = F.normalize(torch.randn(8), dim=-1)
    assert torch.allclose(_calibmap.interpolate_toward(Q, c, 0.0), Q, atol=1e-5)
    at1 = _calibmap.interpolate_toward(Q, c, 1.0)
    assert torch.allclose(at1, c.expand_as(Q), atol=1e-5)
