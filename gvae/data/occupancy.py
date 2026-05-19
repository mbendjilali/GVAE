# gvae/data/occupancy.py
# Occupancy readout head + query sampling from voxelised point-cloud GT

import torch
import torch.nn as nn

import config
from gvae.data.voxelize import sample_occupancy_queries

__all__ = ['OccupancyReadout', 'sample_occupancy_queries']


def fourier_encode(q, num_freqs=6):
    freqs = 2.0 ** torch.arange(num_freqs, device=q.device)
    x = q[:, :, None] * freqs[None, None, :] * torch.pi
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=2)
    return enc.reshape(q.shape[0], -1)


class OccupancyReadout(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        fourier_dim = 3 * 2 * 6

        self.input_proj = nn.Linear(fourier_dim, d)
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, q, Z):
        q_feat = self.input_proj(fourier_encode(q))

        H, W, D, d = Z.shape
        assert d == self.d, f"Z has {d} channels but readout expects {self.d}"
        Z_flat = Z.reshape(-1, d)

        Q = self.W_Q(q_feat)
        K = self.W_K(Z_flat)
        V = self.W_V(Z_flat)

        scale = self.d ** 0.5
        attn_scores = torch.softmax(Q @ K.T / scale, dim=1)
        z_q = attn_scores @ V

        return self.mlp(z_q).squeeze(1)
