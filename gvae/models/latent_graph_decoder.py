# gvae/models/latent_graph_decoder.py
# Honest graph_hat from Z only: fixed slot count, cross-attn on spatial tokens (no h, no GT p).

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from gvae.models.decoder import sample_volume
from gvae.models.splatting import make_voxel_centers


class LatentGraphDecoder(nn.Module):
    """
    Decode supernode (p, r, s) from latent volume Z alone.

    Position: soft-argmax over voxel centres (slot-specific spatial attention).
    Semantics / size: Z sampled at predicted p (and optional slot trunk).
    """

    def __init__(self, d: int, *, max_slots: int | None = None, num_heads: int | None = None):
        super().__init__()
        self.d = d
        self.max_slots = max_slots if max_slots is not None else config.LATENT_GRAPH_MAX_SLOTS
        heads = num_heads if num_heads is not None else config.LATENT_GRAPH_ATTN_HEADS
        while d % heads != 0 and heads > 1:
            heads -= 1
        self.num_heads = heads

        self.z_proj = nn.Conv3d(d, d, kernel_size=1)
        self.slot_index_embed = nn.Embedding(self.max_slots, d)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
        )
        self.mlp_s = nn.Linear(d, config.NUM_CLASSES)
        self.mlp_r = nn.Linear(d, 3)
        self.softplus = nn.Softplus()
        self._pos_scale = math.sqrt(d)

    def _spatial_positions(
        self,
        slots: torch.Tensor,
        tokens: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable expected position in [-1, 1]³ per slot (N, 3)."""
        logits = (slots @ tokens.T) / self._pos_scale
        weights = F.softmax(logits, dim=-1)
        return (weights @ centers).clamp(-1.0, 1.0)

    def forward(self, Z: torch.Tensor, n_nodes: int) -> dict[str, torch.Tensor]:
        """
        Z: (C, H, W, D) with C == d.
        n_nodes: supernode count for this level (no GT coordinates used).
        """
        if n_nodes <= 0:
            empty = Z.new_zeros(0, 3)
            empty_s = Z.new_zeros(0, config.NUM_CLASSES)
            return {'s': empty_s, 'p': empty, 'r': empty}

        if n_nodes > self.max_slots:
            raise ValueError(
                f"n_nodes={n_nodes} exceeds LATENT_GRAPH_MAX_SLOTS={self.max_slots}",
            )

        feat = self.z_proj(Z.unsqueeze(0))
        _, _, H, W, D = feat.shape
        grid = (H, W, D)
        tokens = feat.flatten(2).transpose(1, 2)
        centers = make_voxel_centers(grid, Z.device)

        slot_ids = torch.arange(n_nodes, device=Z.device, dtype=torch.long)
        queries = self.slot_index_embed(slot_ids).unsqueeze(0)
        slots, _ = self.cross_attn(queries, tokens, tokens)
        slots = self.norm(slots).squeeze(0)

        p_hat = self._spatial_positions(slots, tokens.squeeze(0), centers)

        z_samp = sample_volume(Z, p_hat.unsqueeze(1)).reshape(n_nodes, -1)
        feat_at = self.readout(z_samp)
        if config.LATENT_GRAPH_REFINE_FROM_Z_SAMPLE:
            trunk = self.readout(slots)
            feat_at = feat_at + trunk

        return {
            's': torch.softmax(self.mlp_s(feat_at), dim=1),
            'p': p_hat,
            'r': self.softplus(self.mlp_r(feat_at)),
        }
