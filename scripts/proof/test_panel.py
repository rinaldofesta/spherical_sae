"""Unit tests for the frozen NEUTRAL concept panel (PR #2 / M2).

Run:  .venv/bin/python -m pytest scripts/proof/test_panel.py -q

The panel kills the per-model `select_concepts` confound: concepts are defined
MODEL-FREE (a title keyword + its document set), frozen once, identical for every
arm. These tests pin:
  - token_doc_freq counts documents (not occurrences), drops short/stop tokens;
  - band_tokens keeps only tokens whose doc-frequency is inside [lo, hi];
  - dedup_by_jaccard greedily drops tokens whose doc-sets overlap an earlier pick;
  - split_panel is deterministic and density-stratified (calib and holdout both
    span the frequency range, no concept in both);
  - match_latent returns the latent whose firing pattern best F1-matches the
    concept's documents, or None below the floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROOF = Path(__file__).resolve().parent
if str(PROOF) not in sys.path:
    sys.path.insert(0, str(PROOF))

import _panel  # noqa: E402


# ----------------------------------------------------------- token_doc_freq
def test_token_doc_freq_counts_documents_not_occurrences():
    titles = ["graph graph graph networks", "graph attention", "speech models"]
    df = _panel.token_doc_freq(titles, stop=set())
    assert df["graph"] == 2          # 2 docs, despite 4 occurrences
    assert df["speech"] == 1


def test_token_doc_freq_drops_short_and_stop_tokens():
    titles = ["the gan of gans", "a gan for all"]
    df = _panel.token_doc_freq(titles, stop={"gans"})
    assert "the" not in df           # < 4 chars
    assert "gans" not in df          # stopword
    assert "gan" not in df           # < 4 chars


# --------------------------------------------------------------- band_tokens
def test_band_tokens_keeps_only_in_band():
    df = {"common": 50, "mid": 10, "rare": 1}
    kept = _panel.band_tokens(df, n_docs=100, lo=0.05, hi=0.2)
    assert kept == ["mid"]


# ---------------------------------------------------------- dedup_by_jaccard
def test_dedup_by_jaccard_drops_overlapping_docsets():
    docsets = {
        "federated": {1, 2, 3, 4},
        "federating": {1, 2, 3, 5},   # jaccard 3/5 = 0.6 with federated
        "speech": {10, 11},
    }
    kept = _panel.dedup_by_jaccard(["federated", "federating", "speech"],
                                   docsets, max_jaccard=0.5)
    assert kept == ["federated", "speech"]


# --------------------------------------------------------------- split_panel
def test_split_panel_is_deterministic_and_disjoint():
    tokens = [f"t{i}" for i in range(15)]
    freqs = {t: i + 1 for i, t in enumerate(tokens)}
    a = _panel.split_panel(tokens, freqs, n_holdout=5, seed=0)
    b = _panel.split_panel(tokens, freqs, n_holdout=5, seed=0)
    assert a == b
    calib, hold = a
    assert len(hold) == 5 and len(calib) == 10
    assert not set(calib) & set(hold)


def test_split_panel_stratifies_by_frequency():
    """Holdout must span the density range, not be the 5 rarest/densest."""
    tokens = [f"t{i}" for i in range(15)]
    freqs = {t: i + 1 for i, t in enumerate(tokens)}
    _, hold = _panel.split_panel(tokens, freqs, n_holdout=5, seed=0)
    ranks = sorted(freqs[t] for t in hold)
    assert ranks[0] <= 5 and ranks[-1] >= 11   # at least one rare, one dense


# -------------------------------------------------------------- match_latent
def test_match_latent_finds_best_f1_latent():
    # 6 docs, 3 latents; concept docs = {0,1,2}
    active = torch.tensor([
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=torch.bool)
    kwmask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
    idx, f1 = _panel.match_latent(active, kwmask, floor=0.1)
    assert idx == 0                  # latent 0: precision 1.0, recall 1.0
    assert abs(f1 - 1.0) < 1e-6


def test_match_latent_returns_none_below_floor():
    active = torch.zeros(6, 3, dtype=torch.bool)
    active[5, 2] = True
    kwmask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
    idx, f1 = _panel.match_latent(active, kwmask, floor=0.5)
    assert idx is None


# ------------------------------------------------------------ assign_latents
def _collision_setup():
    """Latent 0 is INDIVIDUALLY the best match for BOTH concepts (a: f1=0.75,
    b: f1=0.571 > its runner-up latent 1 at 0.5) -- a genuine collision."""
    active = torch.tensor([
        # lat0 lat1 lat2
        [1, 1, 0],   # docs 0-2: concept a
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],   # docs 3-4: concept b
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=torch.bool)
    masks = {"a": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
             "b": torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)}
    return active, masks


def test_collision_setup_really_collides_under_independent_argmax():
    active, masks = _collision_setup()
    ia, _ = _panel.match_latent(active, masks["a"], floor=0.1)
    ib, _ = _panel.match_latent(active, masks["b"], floor=0.1)
    assert ia == ib == 0          # the bug assign_latents must fix


def test_assign_latents_is_injective_and_resolves_collisions():
    active, masks = _collision_setup()
    out = _panel.assign_latents(active, masks, floor=0.1)
    assert set(out) == {"a", "b"}
    latents = [out[t][0] for t in out]
    assert len(set(latents)) == len(latents)        # injective
    # global assignment: 'a' keeps latent 0 (0.75), 'b' moves to latent 1 (0.5)
    assert out["a"][0] == 0
    assert out["b"][0] == 1


def test_assign_latents_drops_below_floor_after_assignment():
    active, masks = _collision_setup()
    out = _panel.assign_latents(active, masks, floor=0.6)
    assert "a" in out                                # f1 = 0.75 survives
    assert "b" not in out                            # assigned f1 = 0.5 < 0.6


# ------------------------------------------------------------------- jaccard
def test_jaccard_of_masks():
    a = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    b = torch.tensor([0, 1, 1, 1], dtype=torch.bool)
    assert abs(_panel.jaccard(a, b) - 0.5) < 1e-9    # 2 / 4
    assert _panel.jaccard(a, torch.zeros(4, dtype=torch.bool)) == 0.0
