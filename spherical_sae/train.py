"""Minimal training loop + metrics for (Spherical) SAEs.

Run a smoke test comparing a standard top-k SAE against the spherical variant
on synthetic unit-norm data:

    python -m spherical_sae.train

Train on real embeddings (.npy of shape [N, d]):

    python -m spherical_sae.train --embeddings path/to/emb.npy --latent-norm l2
"""

from __future__ import annotations

import argparse

import torch

from .data import load_embeddings, make_synthetic_spherical
from .model import SphericalSAE


def reconstruction_losses(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return MSE, cosine loss, and normalised-MSE (vs predicting the mean)."""
    mse = (x - x_hat).pow(2).sum(-1).mean()
    cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
    cos_loss = (1.0 - cos).mean()
    baseline = (x - x.mean(0, keepdim=True)).pow(2).sum(-1).mean()
    nmse = mse / (baseline + 1e-8)
    return {"mse": mse, "cos_loss": cos_loss, "nmse": nmse, "cos_sim": cos.mean()}


def train(
    x: torch.Tensor,
    *,
    n_latents: int,
    k: int,
    latent_norm: str,
    loss: str = "cos",
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
    k_aux: int = 0,
    aux_alpha: float = 1.0 / 32.0,
    dead_after_steps: int | None = None,
    log_every: int = 250,
    verbose: bool = True,
) -> tuple[SphericalSAE, dict[str, float]]:
    torch.manual_seed(seed)
    d_in = x.shape[1]
    model = SphericalSAE(d_in=d_in, n_latents=n_latents, k=k,
                         latent_norm=latent_norm, k_aux=k_aux)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]
    use_aux = k_aux > 0 and dead_after_steps is not None

    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        batch = x[idx]
        x_hat, h, e_hat = model.forward_train(batch, dead_after_steps if use_aux else None)
        losses = reconstruction_losses(batch, x_hat)
        obj = losses["cos_loss"] if loss == "cos" else losses["mse"]
        aux_val = 0.0
        if e_hat is not None:
            # revive dead latents by having them model the (detached) residual.
            resid = (batch - x_hat).detach()
            aux_loss = (resid - e_hat).pow(2).sum(-1).mean()
            # normalise by residual energy so alpha is scale-free.
            aux_loss = aux_loss / (resid.pow(2).sum(-1).mean() + 1e-8)
            obj = obj + aux_alpha * aux_loss
            aux_val = float(aux_loss.detach())
        opt.zero_grad()
        obj.backward()
        opt.step()
        model.normalize_decoder()
        model.update_dead_stats(h)
        if verbose and (step % log_every == 0 or step == steps - 1):
            n_dead = int(model.dead_mask(dead_after_steps).sum()) if use_aux else 0
            print(
                f"  step {step:5d} | nmse {losses['nmse']:.4f} "
                f"| cos_sim {losses['cos_sim']:.4f} | aux {aux_val:.4f} "
                f"| dead(now) {n_dead}"
            )

    with torch.no_grad():
        x_hat, h = model(x)
        final = reconstruction_losses(x, x_hat)
        active = (h != 0).float().sum(-1).mean().item()
        dead = (h != 0).any(0).logical_not().float().sum().item()
    metrics = {kk: float(v) for kk, v in final.items()}
    metrics.update(mean_active=active, dead_latents=dead)
    return model, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default=None, help="path to .npy [N, d]")
    ap.add_argument("--n-latents", type=int, default=512)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--latent-norm", type=str, default=None,
                    choices=["none", "l2", "l1", "softmax"])
    ap.add_argument("--loss", type=str, default="cos", choices=["cos", "mse"])
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    if args.embeddings:
        x = load_embeddings(args.embeddings)
        print(f"Loaded embeddings: {tuple(x.shape)}")
    else:
        x, _ = make_synthetic_spherical(
            n_samples=8000, d_in=64, n_features=256, active_per_sample=8
        )
        print(f"Synthetic spherical data: {tuple(x.shape)} "
              f"(64 dims, 256 ground-truth features, 8 active/sample)")

    modes = [args.latent_norm] if args.latent_norm else ["none", "l2", "l1", "softmax"]
    print(f"\nComparing latent_norm modes: {modes}\n")
    results = {}
    for mode in modes:
        print(f"[latent_norm = {mode}]  loss = {args.loss}")
        _, m = train(x, n_latents=args.n_latents, k=args.k, latent_norm=mode,
                     loss=args.loss, steps=args.steps)
        results[mode] = m
        print(f"  -> final: nmse {m['nmse']:.4f} | cos_sim {m['cos_sim']:.4f} "
              f"| mean_active {m['mean_active']:.1f} | dead {int(m['dead_latents'])}\n")

    print("=" * 64)
    print(f"{'mode':>8} | {'nmse':>8} | {'cos_sim':>8} | {'dead':>5}")
    print("-" * 64)
    for mode, m in results.items():
        print(f"{mode:>8} | {m['nmse']:>8.4f} | {m['cos_sim']:>8.4f} | {int(m['dead_latents']):>5}")


if __name__ == "__main__":
    main()
