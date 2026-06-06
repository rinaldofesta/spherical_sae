"""Data utilities: a synthetic spherical-mixture generator for smoke tests,
plus a loader for real unit-norm embeddings stored as .npy."""

from __future__ import annotations

import numpy as np
import torch


def make_synthetic_spherical(
    n_samples: int,
    d_in: int,
    n_features: int,
    active_per_sample: int,
    seed: int = 0,
    noise: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate unit-norm embeddings as sparse combinations of a ground-truth
    dictionary of directions -- the generative story a spherical SAE assumes.

    Each sample = normalize( sum_{i in S} w_i * D[:, i] + noise ),  |S| = active_per_sample,
    with non-negative weights w_i summing in a distributional way.

    Returns (X, D) with X: (n_samples, d_in) unit-norm, D: (d_in, n_features) unit-norm.
    """
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((d_in, n_features))
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-8

    X = np.empty((n_samples, d_in), dtype=np.float32)
    for s in range(n_samples):
        idx = rng.choice(n_features, size=active_per_sample, replace=False)
        w = rng.random(active_per_sample)
        w /= w.sum()  # distributional weights
        vec = D[:, idx] @ w + noise * rng.standard_normal(d_in)
        X[s] = vec / (np.linalg.norm(vec) + 1e-8)

    return torch.from_numpy(X), torch.from_numpy(D.astype(np.float32))


def load_embeddings(path: str, normalize: bool = True) -> torch.Tensor:
    """Load real embeddings from a .npy file; optionally L2-normalise to the sphere."""
    arr = np.load(path).astype(np.float32)
    x = torch.from_numpy(arr)
    if normalize:
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    return x
