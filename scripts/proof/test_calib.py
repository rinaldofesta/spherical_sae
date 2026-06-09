"""Unit tests for the calibrated-intervention utilities (PR #1 / M1).

Run:  .venv/bin/python -m pytest scripts/proof/test_calib.py -q

These pin the *exact* invariants the dose-response pilot relies on:
  - decode_from_shares is a plain affine map (no latent re-normalisation);
  - the simplex operators (l1/softmax) return points that SUM TO 1 and hit the
    commanded share EXACTLY, while preserving the relative mix of the others;
  - the `none` steelman operator hits the same *implied L1 share* exactly,
    so the baseline has share-parity at the code level (apples-to-apples).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PROOF = Path(__file__).resolve().parent
for p in (str(ROOT), str(PROOF)):
    if p not in sys.path:
        sys.path.insert(0, p)

from spherical_sae.model import SphericalSAE  # noqa: E402
import _calib  # noqa: E402

TOL = 1e-5


def _toy_model(d_in=8, n_latents=16, k=4, mode="l1"):
    torch.manual_seed(0)
    return SphericalSAE(d_in, n_latents, k, latent_norm=mode)


def _toy_code(n_latents=16, support=(1, 3, 7, 11), vals=(2.0, 1.0, 4.0, 3.0)):
    """A single-row sparse non-negative code with a known active support."""
    h = torch.zeros(1, n_latents)
    for i, v in zip(support, vals):
        h[0, i] = v
    return h


# ----------------------------------------------------------- decode_from_shares
def test_decode_from_shares_is_plain_affine():
    m = _toy_model(mode="softmax")
    s = torch.rand(3, m.n_latents)
    expected = s @ m.W_dec.t() + m.b_dec
    out = m.decode_from_shares(s)
    assert torch.allclose(out, expected, atol=1e-6)


def test_decode_from_shares_skips_normalization():
    """For softmax mode, model.decode re-softmaxes; decode_from_shares must not."""
    m = _toy_model(mode="softmax")
    s = _calib.set_share(_toy_code(m.n_latents), c=5, p=0.3, mode="softmax")
    direct = m.decode_from_shares(s)
    renorm = m.decode(s)  # applies softmax again -> different
    assert not torch.allclose(direct, renorm, atol=1e-4)


# --------------------------------------------------------------- l1 set_share
def test_l1_set_share_sums_to_one():
    s = _calib.set_share(_toy_code(), c=5, p=0.25, mode="l1")
    assert s.sum(dim=1).allclose(torch.ones(1), atol=TOL)


def test_l1_set_share_hits_target_exactly():
    for p in (0.1, 0.3, 0.5, 0.9):
        s = _calib.set_share(_toy_code(), c=5, p=p, mode="l1")
        assert abs(s[0, 5].item() - p) < TOL


def test_l1_set_share_preserves_other_ratios():
    h = _toy_code()  # support 1,3,7,11 with vals 2,1,4,3
    s = _calib.set_share(h, c=5, p=0.4, mode="l1")
    # others should keep their relative proportions (2:1:4:3) and sum to 1-p
    others = s[0, [1, 3, 7, 11]]
    assert abs(others.sum().item() - 0.6) < TOL
    ref = torch.tensor([2.0, 1.0, 4.0, 3.0])
    ref = ref / ref.sum() * 0.6
    assert torch.allclose(others, ref, atol=TOL)


# --------------------------------------------------------------- softmax set_share
def test_softmax_set_share_sums_to_one():
    s = _calib.set_share(_toy_code(), c=5, p=0.3, mode="softmax")
    assert s.sum(dim=1).allclose(torch.ones(1), atol=TOL)


def test_softmax_set_share_hits_target_exactly():
    for p in (0.1, 0.3, 0.5, 0.9):
        s = _calib.set_share(_toy_code(), c=5, p=p, mode="softmax")
        assert abs(s[0, 5].item() - p) < TOL


# --------------------------------------------------------------- none steelman
def test_none_set_share_implied_l1_share_is_exact():
    """none keeps magnitudes; its implied L1 share of c must equal p exactly."""
    for p in (0.1, 0.3, 0.5, 0.9):
        h = _calib.set_share(_toy_code(), c=5, p=p, mode="none")
        implied = h[0, 5].item() / h[0].sum().item()
        assert abs(implied - p) < TOL


# ------------------------------------------------------- device preservation
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("mode", ["l1", "softmax", "none"])
def test_set_share_works_on_device_resident_codes(mode):
    """The M2 harness rewrites codes that live on the GPU (MPS); the operators
    must not silently create CPU-side scalars that break cross-device ops."""
    h = _toy_code().to("mps")
    s = _calib.set_share(h, c=5, p=0.3, mode=mode)
    assert s.device.type == "mps"


# --------------------------------------------------------------- set_mix (l1)
def test_l1_set_mix_hits_all_targets_and_sums_to_one():
    s = _calib.set_mix(_toy_code(), {2: 0.4, 9: 0.2}, mode="l1")
    assert s.sum(dim=1).allclose(torch.ones(1), atol=TOL)
    assert abs(s[0, 2].item() - 0.4) < TOL
    assert abs(s[0, 9].item() - 0.2) < TOL


def test_set_mix_rejects_shares_over_one():
    with pytest.raises(ValueError):
        _calib.set_mix(_toy_code(), {2: 0.7, 9: 0.5}, mode="l1")
