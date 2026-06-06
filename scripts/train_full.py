"""Full-dataset training of (Spherical) SAEs with the AuxK dead-latent revival.

Trains one SAE per latent-normalisation mode on the real ArXiv-ML embeddings,
with the auxiliary loss that revives dead latents, then evaluates over the full
dataset in batches. Saves checkpoints (gitignored) and a metrics JSON (versioned).

    python scripts/train_full.py --modes none l2 --steps 8000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spherical_sae.data import load_embeddings
from spherical_sae.model import SphericalSAE
from spherical_sae.train import reconstruction_losses

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "data" / "checkpoints"
RESULTS_DIR = ROOT / "results"


@torch.no_grad()
def evaluate(model: SphericalSAE, x: torch.Tensor, batch: int = 4096) -> dict[str, float]:
    """Batched evaluation over the full dataset (avoids a 117k x n_latents blowup)."""
    n = x.shape[0]
    tot = {"mse": 0.0, "cos_sum": 0.0, "nmse_num": 0.0}
    fired = torch.zeros(model.n_latents, device=x.device)
    active_sum = 0.0
    mean_x = x.mean(0, keepdim=True)
    baseline = (x - mean_x).pow(2).sum(-1).mean().item()
    for i in range(0, n, batch):
        b = x[i:i + batch]
        x_hat, h = model(b)
        tot["mse"] += (b - x_hat).pow(2).sum(-1).sum().item()
        tot["cos_sum"] += torch.nn.functional.cosine_similarity(b, x_hat, dim=-1).sum().item()
        fired += (h != 0).any(0).float()
        active_sum += (h != 0).float().sum(-1).sum().item()
    mse = tot["mse"] / n
    return {
        "cos_sim": tot["cos_sum"] / n,
        "mse": mse,
        "nmse": mse / (baseline + 1e-8),
        "mean_active": active_sum / n,
        "dead_latents": int((fired == 0).sum().item()),
        "dead_frac": float((fired == 0).float().mean().item()),
    }


def train_one(x, *, mode, n_latents, k, k_aux, steps, batch_size, lr,
              aux_alpha, dead_after_steps, loss, seed, log_every, device="cpu"):
    torch.manual_seed(seed)
    d_in = x.shape[1]
    n = x.shape[0]
    model = SphericalSAE(d_in=d_in, n_latents=n_latents, k=k,
                         latent_norm=mode, k_aux=k_aux).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_aux = k_aux > 0
    t0 = time.time()

    for step in range(steps):
        # linear lr decay to 0
        for g in opt.param_groups:
            g["lr"] = lr * (1.0 - step / steps)
        idx = torch.randint(0, n, (batch_size,), device=device)
        b = x[idx]
        x_hat, h, e_hat = model.forward_train(b, dead_after_steps if use_aux else None)
        losses = reconstruction_losses(b, x_hat)
        obj = losses["cos_loss"] if loss == "cos" else losses["mse"]
        aux_val = 0.0
        if e_hat is not None:
            resid = (b - x_hat).detach()
            aux_loss = (resid - e_hat).pow(2).sum(-1).mean() / (resid.pow(2).sum(-1).mean() + 1e-8)
            obj = obj + aux_alpha * aux_loss
            aux_val = float(aux_loss.detach())
        opt.zero_grad()
        obj.backward()
        opt.step()
        model.normalize_decoder()
        model.update_dead_stats(h)
        if step % log_every == 0 or step == steps - 1:
            n_dead = int(model.dead_mask(dead_after_steps).sum()) if use_aux else 0
            rate = (step + 1) / (time.time() - t0)
            print(f"  [{mode}] step {step:5d}/{steps} | cos_sim {losses['cos_sim']:.4f} "
                  f"| nmse {losses['nmse']:.4f} | aux {aux_val:.4f} | dead(now) {n_dead} "
                  f"| {rate:.1f} it/s", flush=True)

    metrics = evaluate(model, x)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    return model, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default=str(ROOT / "data" / "embeddings.npy"))
    ap.add_argument("--modes", nargs="+", default=["none", "l2"])
    ap.add_argument("--n-latents", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--k-aux", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux-alpha", type=float, default=1.0 / 32.0)
    ap.add_argument("--loss", type=str, default="cos", choices=["cos", "mse"])
    ap.add_argument("--dead-after-steps", type=int, default=None,
                    help="dead if silent this many steps (default: ~1 epoch)")
    ap.add_argument("--subset", type=int, default=None, help="use only N samples (debug)")
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = load_embeddings(args.embeddings).to(device)
    if args.subset:
        x = x[: args.subset]
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"device: {device} ({dev_name})")
    print(f"embeddings: {tuple(x.shape)} | unit-norm: {torch.allclose(x.norm(dim=-1), torch.ones(x.shape[0], device=device), atol=1e-4)}")

    dead_after = args.dead_after_steps or max(20, math.ceil(x.shape[0] / args.batch_size))
    cfg = dict(n_latents=args.n_latents, k=args.k, k_aux=args.k_aux, steps=args.steps,
               batch_size=args.batch_size, lr=args.lr, aux_alpha=args.aux_alpha,
               loss=args.loss, dead_after_steps=dead_after, seed=args.seed)
    print(f"config: {cfg}\ndead_after_steps = {dead_after} (~1 epoch)\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for mode in args.modes:
        print(f"=== training latent_norm = {mode} ===")
        model, metrics = train_one(x, mode=mode, log_every=args.log_every,
                                   device=device, **cfg)
        ckpt = CKPT_DIR / f"sae_{mode}_{args.tag}.pt"
        cpu_state = {kk: v.cpu() for kk, v in model.state_dict().items()}
        torch.save({"state_dict": cpu_state, "config": cfg, "mode": mode}, ckpt)
        metrics["checkpoint"] = str(ckpt.relative_to(ROOT))
        all_metrics[mode] = metrics
        print(f"  -> {mode}: cos_sim {metrics['cos_sim']:.4f} | nmse {metrics['nmse']:.4f} "
              f"| dead {metrics['dead_latents']}/{args.n_latents} "
              f"({metrics['dead_frac']*100:.1f}%) | {metrics['train_seconds']}s\n")

    out = RESULTS_DIR / f"metrics_{args.tag}.json"
    out.write_text(json.dumps({"config": cfg, "results": all_metrics}, indent=2))
    print("=" * 70)
    print(f"{'mode':>8} | {'cos_sim':>8} | {'nmse':>8} | {'dead':>10} | {'mean_act':>8}")
    print("-" * 70)
    for mode, m in all_metrics.items():
        print(f"{mode:>8} | {m['cos_sim']:>8.4f} | {m['nmse']:>8.4f} | "
              f"{m['dead_latents']:>4}/{args.n_latents:<5} | {m['mean_active']:>8.1f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
