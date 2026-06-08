"""Experiment E -- feature STEERING on the real arXiv embeddings.

The practical payoff of interpretable concepts, and the strongest CAUSAL test that
the concepts are real: turn a concept's "knob" up inside a query's sparse code and
show the retrieved results move toward that concept.

  query (without concept c) --encode--> code h --set h[c] high--> decode --> steered query
  retrieve top-k, then measure how many results actually express concept c.

Compared against the SAME query reconstructed WITHOUT the boost (the control), across
modes (standard `none` vs spherical `l2`) and several knob strengths.

Concept membership of a doc = the SAE fires concept c on it (model-grounded), plus a
keyword check on titles (human-readable). No sentence encoder required.

Requires data/embeddings.npy + data/papers.parquet (build_dataset.py).

    python scripts/proof/steering_real.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from _common import MODES  # noqa
from spherical_sae.data import load_embeddings
from spherical_sae.model import SphericalSAE

ROOT = Path(__file__).resolve().parents[2]

STOP = set("""the a an of to in for on with and or by from using used use based via toward towards
into over under between within across this that these those we our their its is are be can may also
new novel approach approaches method methods methodology model models learning deep machine via
network networks neural data analysis algorithm algorithms problem problems framework system systems
application applications study studies paper results result performance training generalization
efficient effective robust large scale general use case via through both more most than less when
what which while where about high low non via use""".split())


def pick_device(arg):
    if arg != "auto":
        return arg
    return "mps" if torch.backends.mps.is_available() else "cpu"


def train_one(Xtr, n_latents, k, mode, steps, device, lr=1e-3, batch=512, seed=0):
    torch.manual_seed(seed)
    model = SphericalSAE(Xtr.shape[1], n_latents, k, latent_norm=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    for step in range(steps):
        b = Xtr[torch.randint(0, n, (batch,), device=device)]
        x_hat, _ = model(b)
        obj = (1.0 - F.cosine_similarity(b, x_hat, dim=-1)).mean()
        opt.zero_grad(); obj.backward(); opt.step()
        model.normalize_decoder()
        if device == "mps" and step % 500 == 499:
            torch.mps.empty_cache()
    return model


@torch.no_grad()
def encode_all(model, X, device, batch=8192):
    out = []
    for i in range(0, X.shape[0], batch):
        out.append(model.encode(X[i:i + batch].to(device)).cpu())
    return torch.cat(out, 0)  # (N, n_latents) on cpu


def select_concepts(h, Xc, n_pick, dens_lo, dens_hi, top_n=20):
    """Pick clean concepts: alive, density in a band, highest top-doc coherence."""
    N, nl = h.shape
    density = (h > 0).float().mean(0)
    topv, topi = h.t().topk(top_n, dim=1)               # (nl, top_n)
    E = F.normalize(Xc[topi], dim=-1)                   # (nl, top_n, d)
    G = torch.bmm(E, E.transpose(1, 2))                 # (nl, top_n, top_n)
    coh = (G.sum((1, 2)) - top_n) / (top_n * (top_n - 1))
    mask = (density >= dens_lo) & (density <= dens_hi)
    cand = torch.where(mask)[0]
    order = cand[coh[cand].argsort(descending=True)]
    return order[:n_pick].tolist(), topi, density, coh


def concept_keywords(topi_row, titles_low, max_kw=3, min_titles=3):
    cnt = Counter()
    for di in topi_row.tolist():
        toks = {t for t in re.findall(r"[a-z]{4,}", titles_low[di]) if t not in STOP}
        cnt.update(toks)
    return [w for w, c in cnt.most_common(25) if c >= min_titles][:max_kw]


@torch.no_grad()
def retrieve(qvecs, corpus_n, k, self_idx):
    qn = F.normalize(qvecs, dim=-1)
    sims = qn @ corpus_n.t()                            # (B, N)
    sims[torch.arange(qvecs.shape[0], device=sims.device), self_idx] = -1e9
    return sims.topk(k, dim=1).indices                 # (B, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default=str(ROOT / "data" / "embeddings.npy"))
    ap.add_argument("--papers", type=str, default=str(ROOT / "data" / "papers.parquet"))
    ap.add_argument("--modes", type=str, nargs="+", default=["none", "l2"])
    ap.add_argument("--subsample", type=int, default=40000)
    ap.add_argument("--n-latents", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-concepts", type=int, default=15)
    ap.add_argument("--queries-per-concept", type=int, default=40)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--strengths", type=float, nargs="+", default=[1.0, 3.0, 6.0])
    ap.add_argument("--dens-lo", type=float, default=0.004)
    ap.add_argument("--dens-hi", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if not Path(args.embeddings).exists():
        print(f"!! {args.embeddings} not found. Run scripts/build_dataset.py first.")
        return
    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)

    X = load_embeddings(args.embeddings)
    titles_all = pd.read_parquet(args.papers)["title"].astype(str).tolist()
    g = torch.Generator().manual_seed(args.seed)
    if X.shape[0] > args.subsample:
        keep = torch.randperm(X.shape[0], generator=g)[:args.subsample]
        X = X[keep]
        titles = [titles_all[i] for i in keep.tolist()]
    else:
        titles = titles_all[:X.shape[0]]
    titles_low = [t.lower() for t in titles]
    N = X.shape[0]
    Xc = X.clone()                                    # cpu copy for coherence
    corpus_n = F.normalize(X, dim=-1).to(device)
    query_pool = torch.from_numpy(rng.choice(N, size=min(3000, N), replace=False))
    print(f"== Steering (real) ==  device={device}  N={N}  modes={args.modes}  "
          f"concepts={args.n_concepts}  strengths={args.strengths}\n", flush=True)

    report = {"config": vars(args), "modes": {}}
    t0 = time.time()
    for mode in args.modes:
        model = train_one(X.to(device), args.n_latents, args.k, mode, args.steps, device, seed=args.seed)
        h = encode_all(model, X, device)               # (N, nl) cpu
        active = h > 0
        picked, topi, density, coh = select_concepts(h, Xc, args.n_concepts, args.dens_lo, args.dens_hi)

        ctrl_a, examples = [], []
        steer_a = {s: [] for s in args.strengths}       # concept-active precision
        steer_kw = {s: [] for s in args.strengths}      # keyword precision
        ctrl_kw = []
        wins = 0; total = 0

        for c in picked:
            kws = concept_keywords(topi[c], titles_low)
            kwmask = torch.tensor([any(w in titles_low[d] for w in kws) for d in range(N)]) \
                if kws else torch.zeros(N, dtype=torch.bool)
            # queries that do NOT already have concept c
            qc = [int(q) for q in query_pool.tolist() if not bool(active[q, c])]
            if len(qc) < 5:
                continue
            qsel = torch.tensor(rng.choice(qc, size=min(args.queries_per_concept, len(qc)), replace=False))
            hq = h[qsel].clone()
            maxrow = hq.max(dim=1).values.clamp(min=1e-6)         # (B,)
            self_idx = qsel.to(device)

            x_ctrl = model.decode(hq.to(device))
            ci = retrieve(x_ctrl, corpus_n, args.topk, self_idx).cpu()
            pc = active[:, c][ci].float().mean().item()
            ctrl_a.append(pc)
            ctrl_kw.append(kwmask[ci].float().mean().item())

            best_lift = None
            for s in args.strengths:
                hs = hq.clone()
                hs[:, c] = maxrow * s
                x_st = model.decode(hs.to(device))
                si = retrieve(x_st, corpus_n, args.topk, self_idx).cpu()
                pa = active[:, c][si].float().mean().item()
                steer_a[s].append(pa)
                steer_kw[s].append(kwmask[si].float().mean().item())
                if s == args.strengths[-1]:
                    best_lift = pa - pc
                    # per-query win rate at max strength
                    per_q_ctrl = active[:, c][ci].float().mean(1)
                    per_q_st = active[:, c][si].float().mean(1)
                    wins += int((per_q_st > per_q_ctrl).sum().item())
                    total += per_q_st.shape[0]
            if len(examples) < 6 and kws:
                # one concrete example: a query and its top-3 steered results
                ex_q = int(qsel[0])
                hs = hq[:1].clone(); hs[:, c] = maxrow[:1] * args.strengths[-1]
                ex_top = retrieve(model.decode(hs.to(device)), corpus_n, 3,
                                  qsel[:1].to(device)).cpu()[0].tolist()
                examples.append({
                    "concept": " / ".join(kws),
                    "density": round(float(density[c]) * 100, 2),
                    "query_title": titles[ex_q][:90],
                    "steered_top3": [titles[i][:80] for i in ex_top],
                    "lift": round(best_lift, 3),
                })

        def m(x): return round(float(np.mean(x)), 4) if x else None
        report["modes"][mode] = {
            "n_concepts_used": len(ctrl_a),
            "control_concept_precision": m(ctrl_a),
            "steered_concept_precision": {str(s): m(steer_a[s]) for s in args.strengths},
            "control_keyword_precision": m(ctrl_kw),
            "steered_keyword_precision": {str(s): m(steer_kw[s]) for s in args.strengths},
            "lift_at_max": round(m(steer_a[args.strengths[-1]]) - m(ctrl_a), 4),
            "win_rate_at_max": round(wins / max(total, 1), 3),
            "examples": examples,
        }
        print(f"[{mode}] control {m(ctrl_a):.3f} -> steered "
              + " ".join(f"x{s}:{m(steer_a[s]):.3f}" for s in args.strengths)
              + f"  | win-rate {wins/max(total,1):.2f}  ({len(ctrl_a)} concetti, {time.time()-t0:.0f}s)",
              flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    # report
    print("\n" + "=" * 84)
    print(f"{'mode':>6} | {'controllo':>10} | "
          + " | ".join(f"manopola x{s}" for s in args.strengths) + f" | {'win-rate':>9}")
    print("-" * 84)
    for mode in args.modes:
        r = report["modes"][mode]
        cells = " | ".join(f"{r['steered_concept_precision'][str(s)]:>11.3f}" for s in args.strengths)
        print(f"{mode:>6} | {r['control_concept_precision']:>10.3f} | {cells} | "
              f"{r['win_rate_at_max']:>9.2f}")
    print("=" * 84)
    print("(precisione = quota dei top-10 risultati che esprimono davvero il concetto pilotato)")

    print("\nEsempi concreti (modalita l2):")
    for ex in report["modes"].get("l2", report["modes"][args.modes[-1]])["examples"][:4]:
        print(f"  • concetto [{ex['concept']}]  (lift +{ex['lift']})")
        print(f"      query:   {ex['query_title']}")
        for t in ex["steered_top3"]:
            print(f"      -> {t}")

    dest = ROOT / "results" / "proof" / "steering_real.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {dest}")


if __name__ == "__main__":
    main()
