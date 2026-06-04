# gvae/models/latent_graph_decoder.py
# Honest graph_hat from Z only: fixed slot count, cross-attn on spatial tokens (no h, no GT p).

from __future__ import annotations

import torch
import torch.nn as nn

import config
from gvae.models.decoder import bound_position, sample_volume


class LatentGraphDecoder(nn.Module):
    """
    Decode supernode (p, r, s) from latent volume Z alone.

    Uses N learned slot queries (N = number of supernodes on the encoded graph).
    Cardinality comes from the graph topology at encode time, not from GT positions.
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
        # One query per supernode index in graph tensor order (not GT position).
        self.slot_index_embed = nn.Embedding(self.max_slots, d)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.readout = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
        )
        self.mlp_s = nn.Linear(d, config.NUM_CLASSES)
        self.mlp_p = nn.Linear(d, 3)
        self.mlp_r = nn.Linear(d, 3)
        self.softplus = nn.Softplus()

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
        tokens = feat.flatten(2).transpose(1, 2)
        slot_ids = torch.arange(n_nodes, device=Z.device, dtype=torch.long)
        queries = self.slot_index_embed(slot_ids).unsqueeze(0)
        slots, _ = self.cross_attn(queries, tokens, tokens)
        slots = self.norm(slots)
        slots = self.readout(slots.squeeze(0))

        if config.LATENT_GRAPH_REFINE_FROM_Z_SAMPLE:
            p_coarse = bound_position(self.mlp_p(slots))
            z_samp = sample_volume(Z, p_coarse.unsqueeze(1)).reshape(n_nodes, -1)
            slots = slots + self.readout(z_samp)

        return {
            's': torch.softmax(self.mlp_s(slots), dim=1),
            'p': bound_position(self.mlp_p(slots)),
            'r': self.softplus(self.mlp_r(slots)),
        }
