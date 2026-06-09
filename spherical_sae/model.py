"""Sparse autoencoders for dense text embeddings, with a *spherical* variant.

Standard top-k SAE (O'Neill et al., 2024, arXiv:2408.00657):

    h  = TopK(W_e (x - b_pre) + b_e)      # sparse latent code
    x_hat = W_d h + b_d                    # linear reconstruction

Spherical variant -- the idea explored here -- normalises the latent code
*before* the decoder:

    h_tilde = normalize(h)                 # L2 -> sphere, L1/softmax -> simplex
    x_hat   = W_d h_tilde + b_d

This discards the absolute activation magnitude and keeps only the
*distributional* information: which features fire and in what proportion.
With unit-norm input embeddings (e.g. OpenAI text-embedding-3) and unit-norm
decoder columns, normalising h also pushes ||x_hat|| ~ 1, so the
reconstruction stays close to the unit sphere "for free".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def top_k_mask(pre_acts: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest activations per row, zero out the rest."""
    if k >= pre_acts.shape[-1]:
        return pre_acts
    topk = torch.topk(pre_acts, k, dim=-1)
    out = torch.zeros_like(pre_acts)
    out.scatter_(-1, topk.indices, topk.values)
    return out


def normalize_latent(h: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    """Project the (sparse) latent code onto a normalised manifold.

    mode:
      "none"    -> identity (standard SAE)
      "l2"      -> unit L2 norm, point on the sphere S^{k-1} (directional)
      "l1"      -> sums to 1 over active feats, point on the simplex (true distribution)
      "softmax" -> softmax over active feats (zeros stay zero), a smooth distribution
    """
    if mode == "none":
        return h
    if mode == "l2":
        return h / (h.norm(dim=-1, keepdim=True) + eps)
    if mode == "l1":
        return h / (h.abs().sum(dim=-1, keepdim=True) + eps)
    if mode == "softmax":
        # softmax only over the active (non-zero) support; inactive stay 0.
        mask = h != 0
        neg_inf = torch.finfo(h.dtype).min
        logits = torch.where(mask, h, torch.full_like(h, neg_inf))
        probs = torch.softmax(logits, dim=-1)
        return probs * mask  # guard against all-zero rows
    raise ValueError(f"unknown latent normalization mode: {mode!r}")


class SphericalSAE(nn.Module):
    """Top-k sparse autoencoder with an optional spherical (normalised) latent.

    Args:
        d_in:        input/embedding dimension.
        n_latents:   number of dictionary features (n >> d_in).
        k:           number of active latents (top-k sparsity).
        latent_norm: one of {"none", "l2", "l1", "softmax"} (see normalize_latent).
        use_pre_bias: subtract a learned bias from the input before encoding
                      (standard SAE trick; recenters the data).
        tie_decoder_unit_norm: keep decoder columns at unit L2 norm.
    """

    def __init__(
        self,
        d_in: int,
        n_latents: int,
        k: int,
        latent_norm: str = "l2",
        k_aux: int = 0,
        use_pre_bias: bool = True,
        tie_decoder_unit_norm: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_latents = n_latents
        self.k = k
        self.k_aux = k_aux
        self.latent_norm = latent_norm
        self.tie_decoder_unit_norm = tie_decoder_unit_norm

        self.W_enc = nn.Parameter(torch.empty(n_latents, d_in))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.W_dec = nn.Parameter(torch.empty(d_in, n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.b_pre = nn.Parameter(torch.zeros(d_in)) if use_pre_bias else None

        # number of optimizer steps since each latent last fired (for AuxK).
        self.register_buffer("steps_since_fired", torch.zeros(n_latents))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialise decoder as a random unit-norm dictionary; tie encoder to it.
        nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True) + 1e-8)
            self.W_enc.copy_(self.W_dec.t())

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        if self.tie_decoder_unit_norm:
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True) + 1e-8)

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        """x -> dense post-ReLU activations (before top-k selection)."""
        if self.b_pre is not None:
            x = x - self.b_pre
        return F.relu(x @ self.W_enc.t() + self.b_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x -> sparse latent code h (pre-normalisation)."""
        return top_k_mask(self.pre_activations(x), self.k)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """sparse latent code h -> reconstruction (normalises h first)."""
        h_tilde = normalize_latent(h, self.latent_norm)
        return h_tilde @ self.W_dec.t() + self.b_dec

    def decode_from_shares(self, s: torch.Tensor) -> torch.Tensor:
        """Decode a code that is ALREADY on its target manifold (no re-normalisation).

        For calibrated intervention we build the latent point explicitly (e.g. exact
        simplex shares via _calib.set_share) and must decode it verbatim -- applying
        ``normalize_latent`` again would, for softmax, re-distort the commanded shares.
        This is the plain affine decoder ``s @ W_dec.T + b_dec``.
        """
        return s @ self.W_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    # ------------------------------------------------------------------ AuxK
    @torch.no_grad()
    def update_dead_stats(self, h: torch.Tensor) -> None:
        """Increment the silence counter; reset latents that fired this batch."""
        fired = (h != 0).any(dim=0)
        self.steps_since_fired += 1
        self.steps_since_fired[fired] = 0

    def dead_mask(self, dead_after_steps: int) -> torch.Tensor:
        return self.steps_since_fired > dead_after_steps

    def aux_decode(self, pre: torch.Tensor, dead_mask: torch.Tensor) -> torch.Tensor | None:
        """Reconstruct the residual with the top-k_aux *dead* latents (ghost grads).

        Returns a linear decode (no latent normalisation, no decoder bias) of the
        residual using only dead latents -- this is what gives them gradient and
        revives them. None if there is nothing to do.
        """
        if self.k_aux <= 0 or not bool(dead_mask.any()):
            return None
        pre_dead = pre * dead_mask.to(pre.dtype)          # keep only dead columns
        k_aux = min(self.k_aux, int(dead_mask.sum()))
        z_aux = top_k_mask(pre_dead, k_aux)
        return z_aux @ self.W_dec.t()

    def forward_train(self, x: torch.Tensor, dead_after_steps: int | None = None):
        """Forward returning everything the training loop needs for AuxK."""
        pre = self.pre_activations(x)
        h = top_k_mask(pre, self.k)
        x_hat = self.decode(h)
        e_hat = None
        if dead_after_steps is not None:
            e_hat = self.aux_decode(pre, self.dead_mask(dead_after_steps))
        return x_hat, h, e_hat
