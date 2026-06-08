"""Experiment B (real data) -- scaling law on the actual 384-d arXiv embeddings.

The synthetic toy saturates at the sparsity/noise floor (64-d data is spanned by a
small dictionary), so the power law can't show. The paper's law L(n) = c*n^-alpha
is a high-dimensional, real-data phenomenon -- so we measure it on the real
embeddings, with steps proportional to n (compute-fair) and a held-out split.

Requires data/embeddings.npy (build_dataset.py).

    python scripts/proof/run_scaling_real.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _common import MODES, reconstruction_losses, train, train_eval_split
from spherical_sae.data import load_embeddings

ROOT = Path(__file__).resolve().parents[2]


def fit_power_law(ns, losses):
    x = np.log(np.asarray(ns, dtype=np.float64))
    y = np.log(np.asarray(losses, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r2 = 1 - ((y - yhat) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)
    return -slope, float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default=str(ROOT / "data" / "embeddings.npy"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--base-steps", type=int, default=800)
    ap.add_argument("--subsample", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.embeddings)
    if not path.exists():
        print(f"!! Embeddings not found at {path}. Run scripts/build_dataset.py first.")
        return

    X = load_embeddings(str(path))  # L2-normalised onto the sphere
    g = torch.Generator().manual_seed(args.seed)
    if X.shape[0] > args.subsample:
        X = X[torch.randperm(X.shape[0], generator=g)[:args.subsample]]
    Xtr, Xte = train_eval_split(X, args.seed)
    print(f"== Scaling (REAL 384-d) ==  sizes={args.sizes}  k={args.k}  "
          f"steps proportional to n (base {args.base_steps})  train={tuple(Xtr.shape)}\n")

    base = min(args.sizes)
    t0 = time.time()
    curve = {m: [] for m in MODES}
    for n in args.sizes:
        steps_n = int(round(args.base_steps * n / base))
        for mode in MODES:
            model, _ = train(Xtr, n_latents=n, k=args.k, latent_norm=mode,
                             loss="cos", steps=steps_n, seed=args.seed, verbose=False)
            with torch.no_grad():
                xhat, _ = model(Xte)
                cl = 1.0 - float(reconstruction_losses(Xte, xhat)["cos_sim"])
            curve[mode].append(cl)
        print(f"  n={n:>5} (steps {steps_n}) -> "
              + "  ".join(f"{m} {curve[m][-1]:.4f}" for m in MODES), flush=True)

    print("\n" + "=" * 70)
    print(f"{'mode':>8} | {'alpha (slope)':>16} | {'R^2':>8}   (paper: alpha ~ 0.12-0.18)")
    print("-" * 70)
    fits = {}
    for m in MODES:
        alpha, r2 = fit_power_law(args.sizes, curve[m])
        fits[m] = {"alpha": alpha, "r2": r2, "losses": curve[m]}
        print(f"{m:>8} | {alpha:>16.3f} | {r2:>8.3f}")
    print("=" * 70)
    print(f"(elapsed {time.time()-t0:.1f}s)")

    dest = ROOT / "results" / "proof" / "scaling_real.json"
    dest.write_text(json.dumps({"config": vars(args), "fits": fits}, indent=2))
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
