"""Experiment D (real data) -- retrieval rank-fidelity on the actual arXiv embeddings.

Label-free test of the project's core claim: does the spherical reconstruction
PRESERVE the nearest-neighbour ranking that real cosine search relies on? For a
set of queries we take the true top-K neighbours on the raw embeddings, then check
how many survive (and in what order) under each mode's reconstruction.

Requires the real dataset (run `python scripts/build_dataset.py` first, ~670 MB).

    python scripts/proof/retrieval_real.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _common import MODES, train
from spherical_sae.data import load_embeddings

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def rank_fidelity(raw_q, raw_c, rep_q, rep_c, k=10, ktau=100):
    """How well rep preserves raw's nearest-neighbour ranking."""
    rqn, rcn = F.normalize(raw_q, dim=-1), F.normalize(raw_c, dim=-1)
    pqn, pcn = F.normalize(rep_q, dim=-1), F.normalize(rep_c, dim=-1)
    raw_sims = rqn @ rcn.t()
    rep_sims = pqn @ pcn.t()
    raw_top = raw_sims.topk(k, dim=-1).indices              # (nq, k) gold neighbours
    rep_top = rep_sims.topk(k, dim=-1).indices
    # Recall@k of the gold top-k
    recall = 0.0
    nq = raw_q.shape[0]
    for i in range(nq):
        recall += len(set(raw_top[i].tolist()) & set(rep_top[i].tolist())) / k
    recall /= nq
    # Spearman-ish: correlation of rep-similarity ordering vs raw over the gold top-ktau
    gold = raw_sims.topk(ktau, dim=-1).indices
    corr = 0.0
    for i in range(nq):
        g = gold[i]
        a = raw_sims[i, g]
        b = rep_sims[i, g]
        ar = a.argsort().argsort().float()
        br = b.argsort().argsort().float()
        am, bm = ar - ar.mean(), br - br.mean()
        corr += float((am * bm).sum() / (am.norm() * bm.norm() + 1e-9))
    corr /= nq
    return {"recall@%d_of_raw" % k: recall, "rank_corr@%d" % ktau: corr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default=str(ROOT / "data" / "embeddings.npy"))
    ap.add_argument("--subsample", type=int, default=20000)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--n-latents", type=int, default=4096)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.embeddings)
    if not path.exists():
        print(f"!! Embeddings not found at {path}\n"
              f"   Build them first:  python scripts/build_dataset.py   (~670 MB)\n"
              f"   Then re-run this script.")
        return

    X = load_embeddings(str(path))
    g = torch.Generator().manual_seed(args.seed)
    if X.shape[0] > args.subsample:
        keep = torch.randperm(X.shape[0], generator=g)[:args.subsample]
        X = X[keep]
    print(f"Embeddings: {tuple(X.shape)}")
    perm = torch.randperm(X.shape[0], generator=g)
    qi, ci = perm[:args.n_queries], perm[args.n_queries:]
    Xq, Xc = X[qi], X[ci]

    t0 = time.time()
    results = {}
    for mode in MODES:
        model, _ = train(X[ci], n_latents=args.n_latents, k=args.k, latent_norm=mode,
                         loss="cos", steps=args.steps, seed=args.seed, verbose=False)
        with torch.no_grad():
            rq, _ = model(Xq)
            rc, _ = model(Xc)
        results[f"recon_{mode}"] = rank_fidelity(Xq, Xc, rq, rc)
        print(f"  mode={mode}: {results[f'recon_{mode}']}", flush=True)

    print("\n" + "=" * 64)
    print(f"{'mode':>10} | {'Recall@10 of raw top-10':>26} | {'rank corr':>10}")
    print("-" * 64)
    for mode in MODES:
        r = results[f"recon_{mode}"]
        print(f"{mode:>10} | {r['recall@10_of_raw']:>26.4f} | {r['rank_corr@100']:>10.4f}")
    print("=" * 64)
    print(f"(elapsed {time.time()-t0:.1f}s)")

    dest = ROOT / "results" / "proof" / "retrieval_real.json"
    dest.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
