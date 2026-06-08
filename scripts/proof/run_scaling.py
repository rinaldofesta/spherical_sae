"""Experiment B -- Scaling law (roadmap step #6).

Does reconstruction improve predictably as the dictionary grows? The reference
paper reports L(n) = c * n^-alpha with alpha ~ 0.12-0.18. We sweep n_latents and
fit alpha per mode on a HELD-OUT split (so we measure generalisation, not
memorisation). cos_loss = 1 - cos_sim is the trained objective, so we fit that
(Euclidean MSE is meaningless for the spherical modes).

    python scripts/proof/run_scaling.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _common import (MODES, make_synthetic, reconstruction_losses, train,
                     train_eval_split)


def fit_power_law(ns, losses):
    """Fit log(loss) = log(c) - alpha*log(n). Returns (alpha, r2)."""
    x = np.log(np.asarray(ns, dtype=np.float64))
    y = np.log(np.asarray(losses, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return -slope, float(r2)  # alpha = -slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1500, help="base steps (at the smallest size)")
    ap.add_argument("--steps-mode", type=str, default="linear", choices=["fixed", "linear"],
                    help="linear = steps proportional to n_latents (compute-fair; bigger dicts "
                         "get proportionally more training so they aren't undertrained)")
    ap.add_argument("--n-features", type=int, default=1024,
                    help="true dictionary size; keep > max(sizes) so bigger dicts always help")
    ap.add_argument("--active", type=int, default=10)
    ap.add_argument("--d-in", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=10000)
    args = ap.parse_args()

    print(f"== Scaling law ==  sizes={args.sizes}  seeds={args.seeds}  "
          f"k={args.k} steps={args.steps}\n")
    t0 = time.time()
    # curve[mode][n] = list of test cos_loss over seeds
    curve = {m: {n: [] for n in args.sizes} for m in MODES}
    for seed in args.seeds:
        X, D, _ = make_synthetic(args.n_samples, args.d_in, args.n_features,
                                 args.active, seed=seed)
        Xtr, Xte = train_eval_split(X, seed)
        base = min(args.sizes)
        for n in args.sizes:
            steps_n = args.steps if args.steps_mode == "fixed" else int(round(args.steps * n / base))
            for mode in MODES:
                model, _ = train(Xtr, n_latents=n, k=args.k, latent_norm=mode,
                                 loss="cos", steps=steps_n, seed=seed, verbose=False)
                with torch.no_grad():
                    xhat, _ = model(Xte)
                    cl = 1.0 - float(reconstruction_losses(Xte, xhat)["cos_sim"])
                curve[mode][n].append(cl)
            print(f"  seed {seed} | n={n:>5} (steps {steps_n}) done", flush=True)

    print("\n" + "=" * 78)
    print(f"{'mode':>8} | " + " | ".join(f"n={n}" for n in args.sizes))
    print(f"{'':>8} |  (mean test cos_loss = 1 - cos_sim, lower is better)")
    print("-" * 78)
    fits = {}
    for m in MODES:
        means = [float(np.mean(curve[m][n])) for n in args.sizes]
        alpha, r2 = fit_power_law(args.sizes, means)
        fits[m] = {"alpha": alpha, "r2": r2, "means": means}
        cells = " | ".join(f"{v:.4f}" for v in means)
        print(f"{m:>8} | {cells}")
    print("-" * 78)
    print(f"{'mode':>8} | {'alpha (slope)':>16} | {'R^2':>8}   "
          f"(paper: alpha ~ 0.12-0.18)")
    for m in MODES:
        print(f"{m:>8} | {fits[m]['alpha']:>16.3f} | {fits[m]['r2']:>8.3f}")
    print("=" * 78)
    print(f"(elapsed {time.time()-t0:.1f}s)")

    out = {"config": vars(args), "curve": curve, "fits": fits}
    dest = Path(__file__).resolve().parents[2] / "results" / "proof" / "scaling.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
