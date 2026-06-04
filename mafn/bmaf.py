"""Bidirectional Modality-Adaptive Fusion (BMAF).

Fuses the radar and IMU temporal embeddings:

* ``BiDirectionalCrossAttn``: two cross-attention modules; radar attends to IMU
  and IMU attends to radar, each with an Add&Norm. Safe when a whole modality is
  absent for a sample (the key sequence is fully masked).
* ``MeanMaxPool``         : temporal mean + max pooling, concatenated -> (B, 2d).
* ``MaskedFeatureMix``    : replace an absent modality's pooled descriptor with a
  learned ``[MASK]`` vector before gating.
* ``ResidualAdaptiveGate``: presence-aware gate that starts as the presence-mean
  and learns to deviate only when it lowers the loss; an absent slot receives
  zero weight.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanMaxPool(nn.Module):
    """Temporal mean + max pooling concatenated along the channel axis.

    Input  : (B, C, T)
    Output : (B, 2*C)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        mx = x.max(dim=-1).values
        return torch.cat([mean, mx], dim=-1)


class BiDirectionalCrossAttn(nn.Module):
    """Two-stream bidirectional cross-attention with Add&Norm.

    Each stream queries the other; both come out enhanced. Inputs / outputs are
    in ``(B, T, d)`` layout for ``nn.MultiheadAttention(batch_first=True)``.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_a_q = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
        self.attn_b_q = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

    @staticmethod
    def _safe_attn(attn: nn.MultiheadAttention, q: torch.Tensor, k: torch.Tensor,
                   v: torch.Tensor, kpm: torch.Tensor | None) -> torch.Tensor:
        """Cross-attention that stays finite when an entire key sequence is masked.

        When the cross modality is absent for a sample, every key is masked and
        the softmax would degenerate to NaN, corrupting both the output and the
        gradient. Such rows are temporarily unmasked so the kernel sees valid
        input, then their output is zeroed — the stream keeps only its own
        features (``LN(a + 0)``) and the absent slot is dropped later by the gate.
        """
        if kpm is None:
            out, _ = attn(q, k, v, need_weights=False)
            return out
        all_masked = kpm.all(dim=1)
        safe_kpm = kpm
        if all_masked.any():
            safe_kpm = kpm.clone()
            safe_kpm[all_masked] = False
        out, _ = attn(q, k, v, key_padding_mask=safe_kpm, need_weights=False)
        if all_masked.any():
            out = out.masked_fill(all_masked.view(-1, 1, 1), 0.0)
        return out

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                a_kpm: torch.Tensor | None = None,
                b_kpm: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """a, b: (B, T, d). Optional key-padding masks (True = ignore)."""
        a_enh = self.norm_a(a + self._safe_attn(self.attn_a_q, a, b, b, b_kpm))
        b_enh = self.norm_b(b + self._safe_attn(self.attn_b_q, b, a, a, a_kpm))
        return a_enh, b_enh


class MaskedFeatureMix(nn.Module):
    """Replace an absent slot's feature with a learned [MASK] vector before fusion."""

    def __init__(self, d_model: int, n_slots: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(n_slots, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, feats: torch.Tensor, present_mask: torch.Tensor) -> torch.Tensor:
        """feats: (B, N_slots, d_model). present_mask: (B, N_slots) of 0/1."""
        B = feats.shape[0]
        tok = self.mask_token.unsqueeze(0).expand(B, -1, -1)
        m = present_mask.unsqueeze(-1)
        return feats * m + tok * (1 - m)


class ResidualAdaptiveGate(nn.Module):
    """Presence-aware gate that starts as the presence-mean and learns to deviate.

    A learnable scalar ``scale`` (initialised to 0) multiplies each per-slot score
    before the masked softmax, so at initialisation the weights are uniform over
    the present slots — exactly the presence-mean — and training can only move
    away from it when that lowers the loss. An absent slot (``-inf``) still
    receives zero weight.
    """

    def __init__(self, d_model: int, n_slots: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_model
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, feats: torch.Tensor, present_mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(feats).squeeze(-1) * self.scale
        scores = scores.masked_fill(present_mask < 0.5, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        return (feats * weights.unsqueeze(-1)).sum(dim=1)
