"""Experiment D -- Retrieval on ground truth (roadmap step #3, synthetic version).

The project's whole premise is that the geometry helps cosine RETRIEVAL, yet the
repo never measures it. Here we can: each synthetic doc has a known class (its
dominant true atom), so "relevant" = same class. We compare cosine retrieval over
several representations:
  - raw            : the original unit-norm embeddings (reference ceiling)
  - recon_<mode>   : the SAE reconstruction x_hat for each mode
  - code_<mode>    : the sparse latent code h for each mode
and report Recall@10 / nDCG@10 / Precision@10. This answers: does spherical
reconstruction PRESERVE retrieval, and is the sparse code itself usable for search?

A ready-to-run real-data counterpart (arXiv categories) is in retrieval_real.py.

    python scripts/proof/run_retrieval.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import MODES, make_synthetic, train


@torch.no_grad()
def retrieval_metrics(Q, Cset, qlab, clab, k=10):
    """Cosine retrieval of Q against corpus Cset; relevance = same label."""
    Qn = F.normalize(Q, dim=-1)
    Cn = F.normalize(Cset, dim=-1)
    sims = Qn @ Cn.t()                                   # (nq, nc)
    topv, topi = sims.topk(k, dim=-1)
    rel = (clab[topi] == qlab[:, None]).float()          # (nq, k)
    # total relevant per query in the corpus
    tot = (clab[None, :] == qlab[:, None]).float().sum(-1).clamp(min=1)
    precision = rel.mean().item()
    recall = (rel.sum(-1) / tot).mean().item()
    # nDCG@k
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float))
    dcg = (rel * discounts).sum(-1)
    ideal_n = torch.minimum(tot, torch.tensor(float(k)))
    idcg = torch.stack([discounts[:int(t)].sum() for t in ideal_n])
    ndcg = (dcg / idcg.clamp(min=1e-9)).mean().item()
    return {"recall@10": recall, "ndcg@10": ndcg, "precision@10": precision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-latents", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-features", type=int, default=128)
    ap.add_argument("--active", type=int, default=6)
    ap.add_argument("--d-in", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=9000)
    ap.add_argument("--n-queries", type=int, default=1500)
    args = ap.parse_args()

    print(f"== Retrieval (synthetic, relevance = same dominant atom) ==\n"
          f"   {args.n_features} classes, {args.n_samples} docs, "
          f"{args.n_queries} held-out queries\n")
    t0 = time.time()
    X, D, lab = make_synthetic(args.n_samples, args.d_in, args.n_features,
                               args.active, seed=args.seed)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(X.shape[0], generator=g)
    qi, ci = perm[:args.n_queries], perm[args.n_queries:]
    Xq, Xc = X[qi], X[ci]
    lq, lc = lab[qi], lab[ci]

    reps = {"raw": (Xq, Xc)}
    models = {}
    for mode in MODES:
        model, _ = train(X[ci], n_latents=args.n_latents, k=args.k, latent_norm=mode,
                         loss="cos", steps=args.steps, seed=args.seed, verbose=False)
        models[mode] = model
        with torch.no_grad():
            rq, _ = model(Xq)
            rc, _ = model(Xc)
            hq = model.encode(Xq)
            hc = model.encode(Xc)
        reps[f"recon_{mode}"] = (rq, rc)
        reps[f"code_{mode}"] = (hq, hc)
        print(f"  trained mode={mode}", flush=True)

    results = {}
    for name, (Q, Cset) in reps.items():
        results[name] = retrieval_metrics(Q, Cset, lq, lc)

    print("\n" + "=" * 64)
    print(f"{'representation':>16} | {'Recall@10':>10} | {'nDCG@10':>9} | {'Prec@10':>9}")
    print("-" * 64)
    order = ["raw"] + [f"recon_{m}" for m in MODES] + [f"code_{m}" for m in MODES]
    for name in order:
        r = results[name]
        print(f"{name:>16} | {r['recall@10']:>10.4f} | {r['ndcg@10']:>9.4f} "
              f"| {r['precision@10']:>9.4f}")
    print("=" * 64)
    print(f"(elapsed {time.time()-t0:.1f}s)")

    out = {"config": vars(args), "results": results}
    dest = Path(__file__).resolve().parents[2] / "results" / "proof" / "retrieval.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
