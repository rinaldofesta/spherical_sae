"""Shared utilities for the "demo -> proof" experiments.

The whole point: run on SYNTHETIC data where the ground-truth dictionary D is
known, so we can measure whether the SAE actually recovers the true concepts --
something impossible on real embeddings (where cos_sim ~0.95 can't tell a real
dictionary from a degenerate one). Reuses the repo's SphericalSAE + train().
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# make the repo root importable when run as `python scripts/proof/xxx.py`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spherical_sae.train import reconstruction_losses, train  # noqa: E402

MODES = ["none", "l2", "l1", "softmax"]


def make_synthetic(n_samples, d_in, n_features, active, seed=0, noise=0.02, D=None):
    """Same generative story as the repo's make_synthetic_spherical, but also
    returns a ground-truth class label per sample (the dominant atom), which we
    use as relevance ground truth for the retrieval experiment.

    Returns (X [n, d] unit-norm, D [d, n_features] unit-norm, label [n]).
    """
    rng = np.random.default_rng(seed)
    if D is None:
        D = rng.standard_normal((d_in, n_features))
        D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-8
    X = np.empty((n_samples, d_in), dtype=np.float32)
    label = np.empty(n_samples, dtype=np.int64)
    for s in range(n_samples):
        idx = rng.choice(n_features, size=active, replace=False)
        w = rng.random(active)
        w /= w.sum()  # distributional weights (simplex), as in the repo
        vec = D[:, idx] @ w + noise * rng.standard_normal(d_in)
        X[s] = vec / (np.linalg.norm(vec) + 1e-8)
        label[s] = int(idx[int(np.argmax(w))])
    return (
        torch.from_numpy(X),
        torch.from_numpy(D.astype(np.float32)),
        torch.from_numpy(label),
    )


def recovery_metrics(W_dec, D):
    """How well learned decoder atoms (d, n_latents) recover true atoms D (d, n_features).

    Both are unit-normalised; we use |cosine| since the code is non-negative.
      - mmcs_recall:    each TRUE atom -> its closest LEARNED atom (did we find it?)
      - mmcs_precision: each LEARNED atom -> its closest TRUE atom (are atoms real?)
      - matched_*:      one-to-one optimal assignment (Hungarian) quality
    """
    Wl = F.normalize(W_dec, dim=0)            # (d, n_lat)
    Dt = F.normalize(D, dim=0)               # (d, n_feat)
    C = (Dt.t() @ Wl).abs()                  # (n_feat, n_lat) = |cosine|
    recall = C.max(dim=1).values.mean().item()
    precision = C.max(dim=0).values.mean().item()
    ri, ci = linear_sum_assignment((-C).cpu().numpy())
    matched = C[ri, ci]
    return {
        "mmcs_recall": recall,
        "mmcs_precision": precision,
        "matched_mean": matched.mean().item(),
        "matched_at_0.9": int((matched > 0.9).sum().item()),
        "n_true": int(D.shape[1]),
    }


_TCRIT = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
          7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def ci95(vals):
    """Mean and 95% confidence half-width (Student-t) of a small sample."""
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    m = float(a.mean())
    if n < 2:
        return m, 0.0
    sd = a.std(ddof=1)
    t = _TCRIT.get(n, 1.96)
    return m, float(t * sd / math.sqrt(n))


def train_eval_split(X, seed, test_frac=0.2):
    """Held-out split so reconstruction is measured on data the SAE never trained on
    (fixes the 'evaluate on training data' issue)."""
    n = X.shape[0]
    n_test = int(n * test_frac)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return X[perm[n_test:]], X[perm[:n_test]]


__all__ = ["ROOT", "MODES", "make_synthetic", "recovery_metrics", "ci95",
           "train_eval_split", "train", "reconstruction_losses"]
