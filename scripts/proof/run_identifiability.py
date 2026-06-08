"""Experiment A -- Identifiability on ground truth (roadmap steps #1, #2, #4-split, #5).

On synthetic data with a KNOWN dictionary, for each latent_norm mode
(none/l2/l1/softmax), over multiple seeds, measure:
  - how well the SAE recovers the true atoms (MMCS recall/precision, #matched@0.9)
  - reconstruction on a HELD-OUT test split (cos_sim, nmse)
and report mean +/- 95% CI so we can tell a real difference from noise.

    python scripts/proof/run_identifiability.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from _common import (MODES, ci95, make_synthetic, recovery_metrics,
                     reconstruction_losses, train, train_eval_split)


def run(seeds, noise, steps, n_latents, k, n_features, active, d_in, n_samples):
    keys = ["mmcs_recall", "mmcs_precision", "matched_at_0.9",
            "test_cos_sim", "test_nmse"]
    per = {m: {kk: [] for kk in keys} for m in MODES}
    for si, seed in enumerate(seeds):
        X, D, _ = make_synthetic(n_samples, d_in, n_features, active,
                                 seed=seed, noise=noise)
        Xtr, Xte = train_eval_split(X, seed)
        for mode in MODES:
            model, _ = train(Xtr, n_latents=n_latents, k=k, latent_norm=mode,
                             loss="cos", steps=steps, seed=seed, verbose=False)
            rec = recovery_metrics(model.W_dec.detach(), D)
            with torch.no_grad():
                xhat, _ = model(Xte)
                L = reconstruction_losses(Xte, xhat)
            per[mode]["mmcs_recall"].append(rec["mmcs_recall"])
            per[mode]["mmcs_precision"].append(rec["mmcs_precision"])
            per[mode]["matched_at_0.9"].append(rec["matched_at_0.9"])
            per[mode]["test_cos_sim"].append(float(L["cos_sim"]))
            per[mode]["test_nmse"].append(float(L["nmse"]))
            print(f"  seed {seed} | {mode:>7} | recall {rec['mmcs_recall']:.3f} "
                  f"| matched@.9 {rec['matched_at_0.9']:>3}/{rec['n_true']} "
                  f"| test_cos {float(L['cos_sim']):.3f}", flush=True)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-latents", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-features", type=int, default=256)
    ap.add_argument("--active", type=int, default=8)
    ap.add_argument("--d-in", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=8000)
    args = ap.parse_args()

    cfg = vars(args).copy()
    print(f"== Identifiability ==  modes={MODES}  seeds={args.seeds}  "
          f"n_latents={args.n_latents} k={args.k} steps={args.steps} noise={args.noise}\n")
    t0 = time.time()
    per = run(args.seeds, args.noise, args.steps, args.n_latents, args.k,
              args.n_features, args.active, args.d_in, args.n_samples)
    dt = time.time() - t0

    # aggregate
    agg = {}
    for m in MODES:
        agg[m] = {kk: ci95(per[m][kk]) for kk in per[m]}

    print("\n" + "=" * 88)
    print(f"{'mode':>8} | {'recall':>14} | {'precision':>14} | "
          f"{'matched@0.9':>14} | {'test_cos':>14}")
    print("-" * 88)
    for m in MODES:
        r = agg[m]
        def fmt(key, dec=3):
            mu, h = r[key]
            return f"{mu:.{dec}f}±{h:.{dec}f}"
        print(f"{m:>8} | {fmt('mmcs_recall'):>14} | {fmt('mmcs_precision'):>14} | "
              f"{fmt('matched_at_0.9',1):>14} | {fmt('test_cos_sim'):>14}")
    print("=" * 88)
    print(f"(elapsed {dt:.1f}s, {len(args.seeds)} seeds x {len(MODES)} modes)")

    out = {
        "config": cfg,
        "raw": per,
        "agg": {m: {kk: {"mean": v[0], "ci95": v[1]} for kk, v in agg[m].items()}
                for m in MODES},
    }
    dest = Path(__file__).resolve().parents[2] / "results" / "proof" / "identifiability.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
