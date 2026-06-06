"""Feature interpretability for a trained (Spherical) SAE.

For each dictionary feature, find the abstracts that activate it most strongly
(its "meaning"), report firing density, and decompose a few example documents
as a *mixture of concepts*. No LLM needed for this first pass -- we eyeball the
top-activating titles to judge monosemanticity.

    python scripts/interpret.py --mode l2
    python scripts/interpret.py --mode none
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spherical_sae.data import load_embeddings
from spherical_sae.model import SphericalSAE

CKPT_DIR = ROOT / "data" / "checkpoints"
RESULTS_DIR = ROOT / "results"


def load_model(mode: str, tag: str, d_in: int, device: str) -> SphericalSAE:
    ckpt = torch.load(CKPT_DIR / f"sae_{mode}_{tag}.pt", map_location=device, weights_only=False)
    c = ckpt["config"]
    model = SphericalSAE(d_in=d_in, n_latents=c["n_latents"], k=c["k"],
                         latent_norm=ckpt["mode"], k_aux=c.get("k_aux", 0)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def feature_top_docs(model, x, top_n=8, batch=4096):
    """Per feature: the top-N documents by activation, plus firing density."""
    n, device = x.shape[0], x.device
    nl = model.n_latents
    top_v = torch.full((nl, top_n), -1.0, device=device)
    top_i = torch.full((nl, top_n), -1, dtype=torch.long, device=device)
    fired = torch.zeros(nl, device=device)
    act_sum = torch.zeros(nl, device=device)
    for o in range(0, n, batch):
        xb = x[o:o + batch]
        h = model.encode(xb)                      # [b, nl] sparse (post top-k)
        fired += (h > 0).sum(0).float()
        act_sum += h.sum(0)
        hb = h.t()                                # [nl, b]
        idx = torch.arange(o, o + xb.shape[0], device=device).expand(nl, -1)
        v = torch.cat([top_v, hb], dim=1)
        i = torch.cat([top_i, idx], dim=1)
        v, sel = v.topk(top_n, dim=1)
        top_v, top_i = v, torch.gather(i, 1, sel)
    density = fired / n
    mean_act = act_sum / fired.clamp(min=1)
    return top_v.cpu(), top_i.cpu(), density.cpu(), fired.cpu(), mean_act.cpu()


def short(title: str, width: int = 90) -> str:
    title = " ".join(str(title).split())
    return title[:width] + ("…" if len(title) > width else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="l2", choices=["none", "l2", "l1", "softmax"])
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--n-features-show", type=int, default=20)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = load_embeddings(ROOT / "data" / "embeddings.npy").to(device)
    papers = pd.read_parquet(ROOT / "data" / "papers.parquet")
    titles = papers["title"].tolist()
    model = load_model(args.mode, args.tag, x.shape[1], device)
    print(f"model: latent_norm={args.mode} | n_latents={model.n_latents} | k={model.k} "
          f"| docs={x.shape[0]} | device={device}")

    top_v, top_i, density, fired, mean_act = feature_top_docs(model, x, top_n=args.top_n)
    dead = int((fired == 0).sum())
    alive = torch.where(fired > 0)[0]
    print(f"features: {model.n_latents} | dead {dead} | alive {len(alive)} "
          f"| median density {density[alive].median():.4f} "
          f"| median firings {fired[alive].median():.0f}\n")

    # choose a spread of ALIVE features to display: by descending density, then
    # evenly spaced through the density rank so we see broad + niche concepts.
    order = alive[torch.argsort(density[alive], descending=True)]
    picks = order[torch.linspace(0, len(order) - 1, args.n_features_show).long()]

    lines = [f"# Feature interpretability — latent_norm = {args.mode}\n",
             f"{model.n_latents} features · {x.shape[0]} abstracts · "
             f"k={model.k} · dead={dead}\n",
             "Each feature is shown by the abstract titles that activate it most "
             "strongly. A coherent list ⇒ a monosemantic, interpretable feature.\n"]
    for f in picks.tolist():
        hdr = (f"\n## feature {f}  ·  density {density[f]*100:.2f}%  "
               f"({int(fired[f])} abstracts)  ·  mean act {mean_act[f]:.3f}")
        print(hdr)
        lines.append(hdr)
        for rank in range(args.top_n):
            di = int(top_i[f, rank])
            if di < 0:
                break
            row = f"  [{top_v[f, rank]:.3f}] {short(titles[di])}"
            print(row)
            lines.append(row)

    # ---- document-as-mixture-of-concepts demo --------------------------
    rng = np.random.default_rng(args.seed)
    demo_docs = [0, int(rng.integers(x.shape[0])), int(rng.integers(x.shape[0]))]
    # label each feature by its #1 abstract title
    feat_label = {f: short(titles[int(top_i[f, 0])], 60) for f in range(model.n_latents)}
    lines.append("\n\n# Documents as a mixture of concepts\n")
    print("\n\n=== documents as a mixture of concepts ===")
    with torch.no_grad():
        for d in demo_docs:
            h = model.encode(x[d:d + 1])[0]
            act = torch.where(h > 0)[0]
            vals = h[act]
            prop = (vals / vals.sum()).cpu()              # L1 proportion (readable)
            srt = torch.argsort(prop, descending=True)
            block = [f"\n### doc {d}: {short(titles[d], 80)}",
                     f"active concepts: {len(act)}"]
            for j in srt[:8].tolist():
                fi = int(act[j])
                block.append(f"  {prop[j]*100:5.1f}%  · feat {fi:>4} · {feat_label[fi]}")
            for b in block:
                print(b)
            lines.extend(block)

    out = RESULTS_DIR / f"interpret_{args.mode}_{args.tag}.md"
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
