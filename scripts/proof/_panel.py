"""Frozen NEUTRAL concept panel for the M2 calibration harness (PR #2).

Kills the per-model `select_concepts` confound: a concept is a title keyword
plus its document set -- defined entirely OUTSIDE any SAE -- frozen to JSON
once (before seeing any calibration result) and identical for every arm and
seed. Each trained model then gets the panel ASSIGNED: per concept, the latent
whose firing pattern best F1-matches the concept's documents.

Pure helpers here (tested in test_panel.py); corpus I/O stays in the harness.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch

TOKEN_RE = re.compile(r"[a-z]{4,}")


def token_doc_freq(titles_low: list[str], stop: set[str]) -> Counter:
    """Document frequency of each title token (>=4 chars, not a stopword)."""
    df: Counter = Counter()
    for t in titles_low:
        toks = {tok for tok in TOKEN_RE.findall(t) if tok not in stop}
        df.update(toks)
    return df


def band_tokens(df: dict, n_docs: int, lo: float, hi: float) -> list[str]:
    """Tokens whose doc-frequency fraction lies in [lo, hi], densest first."""
    kept = [(c, t) for t, c in df.items() if lo <= c / n_docs <= hi]
    return [t for c, t in sorted(kept, reverse=True)]


def dedup_by_jaccard(tokens: list[str], docsets: dict, max_jaccard: float = 0.5) -> list[str]:
    """Greedily keep tokens whose doc-set overlaps every earlier pick < max_jaccard."""
    kept: list[str] = []
    for t in tokens:
        s = docsets[t]
        ok = True
        for k in kept:
            o = docsets[k]
            inter = len(s & o)
            if inter and inter / len(s | o) > max_jaccard:
                ok = False
                break
        if ok:
            kept.append(t)
    return kept


def split_panel(tokens: list[str], freqs: dict, n_holdout: int, seed: int = 0):
    """Density-stratified concept-level split -> (calibration, holdout).

    Sort by frequency, cut into n_holdout contiguous strata, draw ONE holdout
    concept per stratum -- so the held-out set spans rare to dense and the map
    must transfer across the whole density range. Deterministic in `seed`.
    """
    rng = np.random.default_rng(seed)
    ordered = sorted(tokens, key=lambda t: freqs[t])
    strata = np.array_split(np.arange(len(ordered)), n_holdout)
    hold = sorted(ordered[int(rng.choice(s))] for s in strata)
    calib = sorted(t for t in ordered if t not in set(hold))
    return calib, hold


def f1_scores(active: torch.Tensor, kwmask: torch.Tensor) -> torch.Tensor:
    """Per-latent F1 between firing pattern and the concept's documents.

    active: (N, n_latents) bool -- latent fires on doc; kwmask: (N,) bool.
    """
    a = active.float()
    kw = kwmask.float()
    tp = kw @ a                              # (n_latents,)
    fired = a.sum(0)
    pos = kw.sum()
    precision = tp / fired.clamp(min=1e-9)
    recall = tp / max(float(pos), 1e-9)
    return 2 * precision * recall / (precision + recall).clamp(min=1e-9)


def match_latent(active: torch.Tensor, kwmask: torch.Tensor, floor: float = 0.15):
    """Best-F1 latent for ONE concept (no injectivity -- see assign_latents).

    Returns (latent_idx, f1) or (None, best_f1) if best F1 < floor.
    """
    f1 = f1_scores(active, kwmask)
    best = int(torch.argmax(f1))
    best_f1 = float(f1[best])
    if best_f1 < floor:
        return None, best_f1
    return best, best_f1


def assign_latents(active: torch.Tensor, masks: dict, floor: float = 0.15) -> dict:
    """ONE-TO-ONE concept -> latent assignment (Hungarian, maximising total F1).

    Independent per-concept argmax can hand the same latent to two correlated
    panel concepts, which silently collapses set_mix's targets dict and makes
    "held-out" knobs reuse a calibration axis. Returns {token: (latent, f1)},
    dropping tokens whose ASSIGNED F1 falls below the floor.
    """
    from scipy.optimize import linear_sum_assignment

    tokens = list(masks)
    F1 = torch.stack([f1_scores(active, masks[t]) for t in tokens])  # (T, L)
    ri, ci = linear_sum_assignment((-F1).numpy())
    out = {}
    for r, c in zip(ri, ci):
        f1 = float(F1[r, c])
        if f1 >= floor:
            out[tokens[r]] = (int(c), f1)
    return out


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard overlap of two boolean doc masks (0 when both empty)."""
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return int((a & b).sum()) / union


__all__ = ["token_doc_freq", "band_tokens", "dedup_by_jaccard", "split_panel",
           "f1_scores", "match_latent", "assign_latents", "jaccard"]
