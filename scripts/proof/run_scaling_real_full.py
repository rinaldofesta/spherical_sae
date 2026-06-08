"""Experiment B (full scale) -- the definitive scaling law on the real arXiv data.

The reduced run left the power law open for none/l2/l1 (loss flat/rising with n),
likely because of too little data + too few steps per large dictionary. Here we
go full scale: the FULL 117k embeddings, dictionaries up to 4096, and steps
proportional to n so even the biggest dictionary is properly trained.

Engineering for a long job:
  - uses MPS (Apple GPU) if available, else CPU
  - writes results incrementally after each size and RESUMES (skips done sizes),
    so a kill/timeout never loses progress -- just re-run to continue.

    python scripts/proof/run_scaling_real_full.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _common import MODES, train_eval_split
from spherical_sae.data import load_embeddings
from spherical_sae.model import SphericalSAE

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "proof" / "scaling_real_full.json"


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one(Xtr, n_latents, k, mode, steps, device, lr=1e-3, batch=512, seed=0):
    torch.manual_seed(seed)
    model = SphericalSAE(Xtr.shape[1], n_latents, k, latent_norm=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    for step in range(steps):
        b = Xtr[torch.randint(0, n, (batch,), device=device)]
        x_hat, _ = model(b)
        obj = (1.0 - F.cosine_similarity(b, x_hat, dim=-1)).mean()
        opt.zero_grad()
        obj.backward()
        opt.step()
        model.normalize_decoder()
        if device == "mps" and step % 500 == 499:
            torch.mps.empty_cache()  # MPS doesn't auto-free; bound memory over long runs
    return model


@torch.no_grad()
def eval_cosloss(model, Xte, batch=8192):
    tot, cnt = 0.0, 0
    for i in range(0, Xte.shape[0], batch):
        xb = Xte[i:i + batch]
        xh, _ = model(xb)
        tot += (1.0 - F.cosine_similarity(xb, xh, dim=-1)).sum().item()
        cnt += xb.shape[0]
    return tot / cnt


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
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--base-steps", type=int, default=800,
                    help="steps at the smallest size; steps scale linearly with n")
    ap.add_argument("--subsample", type=int, default=120000, help=">N keeps all 117k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    path = Path(args.embeddings)
    if not path.exists():
        print(f"!! Embeddings not found at {path}. Run scripts/build_dataset.py first.")
        return
    device = pick_device(args.device)

    X = load_embeddings(str(path))
    g = torch.Generator().manual_seed(args.seed)
    if X.shape[0] > args.subsample:
        X = X[torch.randperm(X.shape[0], generator=g)[:args.subsample]]
    Xtr, Xte = train_eval_split(X, args.seed)
    Xtr, Xte = Xtr.to(device), Xte.to(device)
    print(f"== Scaling FULL (real 384-d) ==  device={device}  sizes={args.sizes}  k={args.k}\n"
          f"   train={tuple(Xtr.shape)}  test={tuple(Xte.shape)}  steps = {args.base_steps} * n/{min(args.sizes)}\n",
          flush=True)

    # resume: load any prior progress
    state = {"config": vars(args), "curve": {m: {} for m in MODES}}
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
            for m in MODES:
                state["curve"][m].update(prev.get("curve", {}).get(m, {}))
            print(f"(resuming; already have: "
                  f"{sorted({int(n) for m in MODES for n in state['curve'][m]})})\n", flush=True)
        except Exception:
            pass

    base = min(args.sizes)
    t0 = time.time()
    for n in args.sizes:
        steps_n = int(round(args.base_steps * n / base))
        for mode in MODES:
            if str(n) in state["curve"][mode]:
                continue  # already done (resume)
            tt = time.time()
            model = train_one(Xtr, n, args.k, mode, steps_n, device, seed=args.seed)
            cl = eval_cosloss(model, Xte)
            state["curve"][mode][str(n)] = cl
            OUT.write_text(json.dumps(state, indent=2))  # incremental save
            print(f"  n={n:>5} ({steps_n:>5} steps) | {mode:>7} | cos_loss {cl:.4f} "
                  f"| {time.time()-tt:.0f}s  (total {time.time()-t0:.0f}s)", flush=True)
            del model
            if device == "mps":
                torch.mps.empty_cache()

    # final fit
    print("\n" + "=" * 70)
    print(f"{'mode':>8} | " + " | ".join(f"n={n}" for n in args.sizes) + " |   alpha |   R^2")
    print("-" * 70)
    fits = {}
    for m in MODES:
        losses = [state["curve"][m][str(n)] for n in args.sizes]
        alpha, r2 = fit_power_law(args.sizes, losses)
        fits[m] = {"alpha": alpha, "r2": r2, "losses": losses}
        cells = " | ".join(f"{v:.4f}" for v in losses)
        print(f"{m:>8} | {cells} | {alpha:>7.3f} | {r2:>5.3f}")
    print("=" * 70)
    print("(paper: alpha ~ 0.12-0.18)")
    state["fits"] = fits
    OUT.write_text(json.dumps(state, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
