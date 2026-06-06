"""Quantify feature monosemanticity, without an LLM.

A feature is monosemantic if the abstracts that activate it form a *tight*
cluster in embedding space. We score each feature by the mean pairwise cosine
similarity of its top-N activating abstracts (embeddings are unit-norm, so this
is just the mean off-diagonal Gram entry). We compare modes against each other
and against a random-set baseline, and also compare density-matched (since
niche features are naturally tighter).

    python scripts/monosemanticity.py --modes none l2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from interpret import feature_top_docs, load_model
from spherical_sae.data import load_embeddings

RESULTS_DIR = ROOT / "results"


def coherence_of_sets(emb: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine within each set of N document indices.

    emb: [n_docs, d] unit-norm. idx: [F, N] doc indices. -> [F] coherence.
    """
    E = emb[idx]                                   # [F, N, d]
    G = torch.bmm(E, E.transpose(1, 2))            # [F, N, N] cosine (unit-norm)
    N = E.shape[1]
    off = (G.sum(dim=(1, 2)) - N) / (N * (N - 1))  # remove diagonal (==1)
    return off


def random_baseline(emb: torch.Tensor, n_sets: int, n: int, seed: int) -> float:
    g = torch.Generator(device=emb.device).manual_seed(seed)
    idx = torch.randint(0, emb.shape[0], (n_sets, n), generator=g, device=emb.device)
    return float(coherence_of_sets(emb, idx).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["none", "l2"])
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = load_embeddings(ROOT / "data" / "embeddings.npy").to(device)
    base = random_baseline(emb, n_sets=4096, n=args.top_n, seed=args.seed)
    print(f"random-set baseline coherence (N={args.top_n}): {base:.4f}\n")

    results = {"random_baseline": round(base, 4), "top_n": args.top_n, "modes": {}}
    per_mode = {}
    for mode in args.modes:
        model = load_model(mode, args.tag, emb.shape[1], device)
        _, top_i, density, fired, _ = feature_top_docs(model, emb, top_n=args.top_n)
        alive = fired > 0
        idx = top_i.to(device).clamp(min=0)         # [F, N]
        coh = coherence_of_sets(emb, idx).cpu()      # [F]
        coh_a, dens_a = coh[alive], density[alive]
        per_mode[mode] = (coh_a, dens_a)
        results["modes"][mode] = {
            "n_alive": int(alive.sum()),
            "median_coherence": round(float(coh_a.median()), 4),
            "mean_coherence": round(float(coh_a.mean()), 4),
            "frac_above_baseline": round(float((coh_a > base).float().mean()), 4),
            "frac_tight_0.5": round(float((coh_a > 0.5).float().mean()), 4),
        }
        m = results["modes"][mode]
        print(f"[{mode}] alive {m['n_alive']} | median coh {m['median_coherence']:.4f} "
              f"| mean {m['mean_coherence']:.4f} | >baseline {m['frac_above_baseline']*100:.1f}% "
              f"| >0.5 {m['frac_tight_0.5']*100:.1f}%")

    # density-matched comparison (only meaningful for two modes)
    if len(args.modes) == 2:
        a, b = args.modes
        ca, da = per_mode[a]; cb, db = per_mode[b]
        edges = torch.quantile(torch.cat([da, db]),
                               torch.linspace(0, 1, 7))
        print(f"\ndensity-matched median coherence ({a} vs {b}):")
        print(f"  {'density bin':>22} | {a:>8} | {b:>8} | {'Δ':>7}")
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            ma = (da >= lo) & (da < hi); mb = (db >= lo) & (db < hi)
            if ma.sum() < 5 or mb.sum() < 5:
                continue
            va, vb = float(ca[ma].median()), float(cb[mb].median())
            label = f"{lo*100:.2f}-{hi*100:.2f}%"
            print(f"  {label:>22} | {va:>8.4f} | {vb:>8.4f} | {vb-va:>+7.4f}")
            bins.append({"bin": label, a: round(va, 4), b: round(vb, 4), "delta": round(vb - va, 4)})
        results["density_matched"] = bins

    out = RESULTS_DIR / f"monosemanticity_{args.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
