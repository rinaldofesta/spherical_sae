"""M0 -- dose-response PILOT for calibrated/proportional control (PR #1).

The single check that decides whether the "proportional knob" bet is alive BEFORE
building the full calibration harness. For each mode {none, l1, softmax} we train one
SAE, pick a few clean concepts, and sweep the COMMANDED share p over a fine grid; for
each p we rewrite the query code with the mode's exact-share operator (_calib), decode
WITHOUT re-normalisation (decode_from_shares), retrieve top-K and measure the REALISED
share three ways:

  kw   -- fraction of top-K titles matching the concept keywords (model-independent)
  sae  -- fraction of top-K on which the SAE fires concept c   (model-grounded)
  cos  -- mean cosine of top-K to the concept centroid          (CONTINUOUS; the one
          that distinguishes "linear-recalibrable" from "binary switch / saturating")

Output: realised-vs-commanded curves per mode -> results/proof/pilot_calibration.json
plus an ASCII table. The measured SHAPE fixes the p-domain and the GO thresholds; we do
NOT hard-code MACE<=0.10 on {0.1..0.9} (already falsified by the earlier pilot).

    python scripts/proof/pilot_calibration.py
    python scripts/proof/pilot_calibration.py --modes none l1 softmax --n-concepts 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
PROOF = Path(__file__).resolve().parent
for p in (str(ROOT), str(PROOF)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _calib  # noqa: E402
from spherical_sae.data import load_embeddings  # noqa: E402
from steering_real import (  # noqa: E402
    concept_keywords, encode_all, pick_device, retrieve, select_concepts, train_one,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default=str(ROOT / "data" / "embeddings.npy"))
    ap.add_argument("--papers", type=str, default=str(ROOT / "data" / "papers.parquet"))
    ap.add_argument("--modes", type=str, nargs="+", default=["none", "l1", "softmax"])
    ap.add_argument("--subsample", type=int, default=40000)
    ap.add_argument("--n-latents", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-concepts", type=int, default=5)
    ap.add_argument("--queries-per-concept", type=int, default=40)
    ap.add_argument("--topk", type=int, default=30)
    # fine grid in [0.05, 0.6]; the measured curve fixes the useful domain.
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60])
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
    Xc = X.clone()
    Xc_n = F.normalize(Xc, dim=-1)                       # cpu, for centroid cosine
    corpus_n = F.normalize(X, dim=-1).to(device)
    query_pool = torch.from_numpy(rng.choice(N, size=min(3000, N), replace=False))
    print(f"== Dose-response PILOT ==  device={device}  N={N}  modes={args.modes}  "
          f"K={args.topk}  grid={args.grid}\n", flush=True)

    report = {"config": vars(args), "modes": {}}
    t0 = time.time()
    for mode in args.modes:
        model = train_one(X.to(device), args.n_latents, args.k, mode, args.steps,
                          device, seed=args.seed)
        h = encode_all(model, X, device)                 # (N, nl) cpu
        active = h > 0
        picked, topi, density, coh = select_concepts(
            h, Xc, args.n_concepts, args.dens_lo, args.dens_hi)

        # accumulate realised share across concepts, per grid point
        kw = {p: [] for p in args.grid}
        sae = {p: [] for p in args.grid}
        cos = {p: [] for p in args.grid}
        ctrl = {"kw": [], "sae": [], "cos": []}
        concepts_used = []

        for c in picked:
            kws = concept_keywords(topi[c], titles_low)
            kwmask = torch.tensor([any(w in titles_low[d] for w in kws) for d in range(N)]) \
                if kws else torch.zeros(N, dtype=torch.bool)
            centroid = F.normalize(Xc_n[topi[c]].mean(0), dim=-1)        # (d,)
            doc_cos = Xc_n @ centroid                                    # (N,) continuous
            # queries that do NOT already have concept c
            qc = [int(q) for q in query_pool.tolist() if not bool(active[q, c])]
            if len(qc) < 5:
                continue
            qsel = torch.tensor(rng.choice(qc, size=min(args.queries_per_concept, len(qc)),
                                           replace=False))
            hq = h[qsel].clone()
            self_idx = qsel.to(device)
            concepts_used.append(" / ".join(kws) if kws else f"#{c}")

            # control: knob OFF (decode the original code as the model would)
            ci = retrieve(model.decode_from_shares(hq.to(device)), corpus_n,
                          args.topk, self_idx).cpu()
            ctrl["kw"].append(kwmask[ci].float().mean().item())
            ctrl["sae"].append(active[:, c][ci].float().mean().item())
            ctrl["cos"].append(doc_cos[ci].mean().item())

            for p in args.grid:
                hs = _calib.set_share(hq, c, p, mode)
                idx = retrieve(model.decode_from_shares(hs.to(device)), corpus_n,
                               args.topk, self_idx).cpu()
                kw[p].append(kwmask[idx].float().mean().item())
                sae[p].append(active[:, c][idx].float().mean().item())
                cos[p].append(doc_cos[idx].mean().item())

        def m(xs):
            return round(float(np.mean(xs)), 4) if xs else None

        report["modes"][mode] = {
            "n_concepts_used": len(concepts_used),
            "concepts": concepts_used,
            "control": {kk: m(vv) for kk, vv in ctrl.items()},
            "realised_kw": {str(p): m(kw[p]) for p in args.grid},
            "realised_sae": {str(p): m(sae[p]) for p in args.grid},
            "realised_cos": {str(p): m(cos[p]) for p in args.grid},
        }
        print(f"[{mode}] {len(concepts_used)} concepts, {time.time()-t0:.0f}s", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    # ----- ASCII dose-response tables (commanded p -> realised share) -----
    grid = args.grid
    for signal, key in (("keyword share", "realised_kw"),
                        ("SAE-fires share", "realised_sae"),
                        ("centroid-cosine (continuous)", "realised_cos")):
        print("\n" + "=" * (14 + 7 * len(grid)))
        print(f"REALISED {signal}  (rows=mode, cols=commanded p)")
        print(f"{'mode':>8} | {'ctrl':>6} | " + " ".join(f"{p:>5.2f}" for p in grid))
        print("-" * (14 + 7 * len(grid)))
        cks = {"realised_kw": "kw", "realised_sae": "sae", "realised_cos": "cos"}[key]
        for mode in args.modes:
            r = report["modes"][mode]
            ctrlv = r["control"][cks]
            cells = " ".join(
                f"{(r[key][str(p)] if r[key][str(p)] is not None else float('nan')):>5.2f}"
                for p in grid)
            print(f"{mode:>8} | {ctrlv:>6.2f} | {cells}")
    print("=" * (14 + 7 * len(grid)))
    print("\nLettura: se la riga sale ~linearmente con p -> quadrante (controllo proporzionale).")
    print("Se salta da ~ctrl a ~saturazione e resta piatta -> interruttore binario (null onesto).")

    dest = ROOT / "results" / "proof" / "pilot_calibration.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {dest}")


if __name__ == "__main__":
    main()
