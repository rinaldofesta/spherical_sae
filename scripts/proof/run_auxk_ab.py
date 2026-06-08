"""Experiment C -- AuxK geometry A/B (roadmap step #4, the 'same ruler' fix).

The repo revives dead latents with a EUCLIDEAN (MSE) residual target, bolted onto
a COSINE-trained model -- a geometry mismatch suspected of causing the spherical
model's leftover dead latents. Here we A/B the dead-latent revival objective:
  off    : no AuxK
  euclid : the repo's MSE residual revival
  cosine : a direction-matched (cosine) residual revival
on an OVERCOMPLETE dictionary (where dead latents actually appear), measuring how
many latents stay dead and how well the dictionary is recovered.

    python scripts/proof/run_auxk_ab.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import ci95, make_synthetic, recovery_metrics
from spherical_sae.model import SphericalSAE


def train_with_aux(X, mode, aux_mode, n_latents, k, k_aux, steps, seed,
                   dead_after, lr=1e-3, batch=512, aux_alpha=1.0 / 32.0):
    torch.manual_seed(seed)
    d = X.shape[1]
    model = SphericalSAE(d, n_latents, k, latent_norm=mode, k_aux=k_aux)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    use_aux = aux_mode != "off"
    for _ in range(steps):
        b = X[torch.randint(0, n, (batch,))]
        x_hat, h, e_hat = model.forward_train(b, dead_after if use_aux else None)
        cos = F.cosine_similarity(b, x_hat, dim=-1)
        obj = (1.0 - cos).mean()
        if e_hat is not None:
            resid = (b - x_hat).detach()
            if aux_mode == "euclid":
                aux = (resid - e_hat).pow(2).sum(-1).mean() / (resid.pow(2).sum(-1).mean() + 1e-8)
            else:  # cosine: revive on the DIRECTION of the residual
                aux = (1.0 - F.cosine_similarity(F.normalize(resid, dim=-1), e_hat, dim=-1)).mean()
            obj = obj + aux_alpha * aux
        opt.zero_grad()
        obj.backward()
        opt.step()
        model.normalize_decoder()
        model.update_dead_stats(h)
    with torch.no_grad():
        _, h = model(X)
        dead = int((h != 0).any(0).logical_not().sum().item())
    return model, dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=str, nargs="+", default=["none", "l2"])
    ap.add_argument("--aux", type=str, nargs="+", default=["off", "euclid", "cosine"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-latents", type=int, default=1024)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k-aux", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--dead-after", type=int, default=100)
    ap.add_argument("--n-features", type=int, default=256)
    ap.add_argument("--active", type=int, default=8)
    ap.add_argument("--d-in", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=8000)
    args = ap.parse_args()

    print(f"== AuxK geometry A/B ==  modes={args.modes}  aux={args.aux}  "
          f"n_latents={args.n_latents} (overcomplete vs {args.n_features} true)\n")
    t0 = time.time()
    cells = {}  # (mode, aux) -> {dead:[], recall:[], precision:[]}
    for seed in args.seeds:
        X, D, _ = make_synthetic(args.n_samples, args.d_in, args.n_features,
                                 args.active, seed=seed)
        for mode in args.modes:
            for aux in args.aux:
                model, dead = train_with_aux(
                    X, mode, aux, args.n_latents, args.k, args.k_aux,
                    args.steps, seed, args.dead_after)
                rec = recovery_metrics(model.W_dec.detach(), D)
                key = (mode, aux)
                cells.setdefault(key, {"dead": [], "recall": [], "precision": []})
                cells[key]["dead"].append(dead)
                cells[key]["recall"].append(rec["mmcs_recall"])
                cells[key]["precision"].append(rec["mmcs_precision"])
                print(f"  seed {seed} | {mode:>5} | aux={aux:>6} | "
                      f"dead {dead:>4}/{args.n_latents} | recall {rec['mmcs_recall']:.3f}",
                      flush=True)

    print("\n" + "=" * 80)
    print(f"{'mode':>6} | {'aux':>7} | {'dead latents':>16} | {'recall':>14} | {'precision':>14}")
    print("-" * 80)
    out = {}
    for mode in args.modes:
        for aux in args.aux:
            c = cells[(mode, aux)]
            dm, dh = ci95(c["dead"])
            rm, rh = ci95(c["recall"])
            pm, ph = ci95(c["precision"])
            out[f"{mode}/{aux}"] = {"dead": [dm, dh], "recall": [rm, rh], "precision": [pm, ph]}
            print(f"{mode:>6} | {aux:>7} | {dm:>7.1f}±{dh:<7.1f} | "
                  f"{rm:.3f}±{rh:.3f} | {pm:.3f}±{ph:.3f}")
    print("=" * 80)
    print(f"(elapsed {time.time()-t0:.1f}s)")

    dest = Path(__file__).resolve().parents[2] / "results" / "proof" / "auxk_ab.json"
    dest.write_text(json.dumps({"config": vars(args), "results": out}, indent=2))
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
