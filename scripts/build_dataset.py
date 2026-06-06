"""Build the combined ML-ArXiv dataset: title + abstract + embedding.

Sources (HuggingFace, by CShorten):
  - CShorten/ML-ArXiv-Papers           -> title, abstract        (117,592 rows)
  - CShorten/ArXiv-ML-Abstract-Embeddings -> 384-dim embeddings  (aligned by row order)

The two are aligned positionally (the embeddings were generated from the
abstracts in the same order). This script verifies the alignment, then writes:

  data/embeddings.npy   float32 [N, 384]   -- ready for the SAE loader
  data/papers.parquet   columns: title, abstract  -- the text side
  data/meta.json        provenance + shapes

Usage:
  python scripts/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"

PAPERS_REPO = "CShorten/ML-ArXiv-Papers"
PAPERS_FILE = "ML-Arxiv-Papers.csv"
EMB_REPO = "CShorten/ArXiv-ML-Abstract-Embeddings"
EMB_FILE = "Abstract-Embeds.csv"


def download() -> tuple[Path, Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading papers CSV (~147 MB)...")
    papers = hf_hub_download(PAPERS_REPO, PAPERS_FILE, repo_type="dataset", local_dir=RAW)
    print("Downloading embeddings CSV (~525 MB)...")
    emb = hf_hub_download(EMB_REPO, EMB_FILE, repo_type="dataset", local_dir=RAW)
    return Path(papers), Path(emb)


def build(papers_csv: Path, emb_csv: Path) -> None:
    print("Loading papers...")
    papers = pd.read_csv(papers_csv)
    # Drop the unnamed index columns, keep text.
    papers = papers[["title", "abstract"]].reset_index(drop=True)

    print("Loading embeddings (this is the big one)...")
    emb = pd.read_csv(emb_csv)
    emb_cols = [c for c in emb.columns if c.startswith("x")]
    print(f"  embedding dim: {len(emb_cols)}")

    if len(papers) != len(emb):
        raise SystemExit(
            f"Row count mismatch: papers={len(papers)} vs embeddings={len(emb)}; "
            "alignment by row order is not safe."
        )
    print(f"  aligned rows: {len(papers)}")

    X = emb[emb_cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(X, axis=1)
    print(f"  embedding norms: mean={norms.mean():.3f} min={norms.min():.3f} "
          f"max={norms.max():.3f} (unit-norm? {np.allclose(norms, 1.0, atol=1e-2)})")

    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / "embeddings.npy", X)
    papers.to_parquet(DATA / "papers.parquet", index=False)
    meta = {
        "n_rows": int(len(papers)),
        "embedding_dim": int(len(emb_cols)),
        "papers_repo": PAPERS_REPO,
        "embeddings_repo": EMB_REPO,
        "embeddings_unit_norm": bool(np.allclose(norms, 1.0, atol=1e-2)),
        "files": {"embeddings": "embeddings.npy", "papers": "papers.parquet"},
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote:\n  {DATA/'embeddings.npy'}  {X.shape} {X.dtype}\n"
          f"  {DATA/'papers.parquet'}  ({len(papers)} rows)\n  {DATA/'meta.json'}")


if __name__ == "__main__":
    p, e = download()
    build(p, e)
