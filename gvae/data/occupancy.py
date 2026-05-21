# gvae/data/occupancy.py
# Occupancy readout head + query sampling from voxelised point-cloud GT

import torch
import torch.nn as nn

import config
from gvae.data.voxelize import sample_occupancy_queries

__all__ = ['OccupancyReadout', 'sample_occupancy_queries', 'occ_query_count']


def fourier_encode(q, num_freqs=6):
    freqs = 2.0 ** torch.arange(num_freqs, device=q.device)
    x = q[:, :, None] * freqs[None, None, :] * torch.pi
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=2)
    return enc.reshape(q.shape[0], -1)


def occ_query_count(occ_grid: torch.Tensor) -> int:
    """Fewer queries on large grids to limit attention memory."""
    voxels = occ_grid.numel()
    if voxels >= 64 ** 3:
        return min(config.OCC_QUERY_POINTS, 512)
    if voxels >= 32 ** 3:
        return min(config.OCC_QUERY_POINTS, 1024)
    return config.OCC_QUERY_POINTS


def _chunked_cross_attention(Q, K, V, scale: float, key_chunk: int) -> torch.Tensor:
    """Softmax attention over keys in chunks (memory-safe for large voxel grids)."""
    n_keys = K.shape[0]
    if n_keys <= key_chunk:
        attn = torch.softmax(Q @ K.T / scale, dim=1)
        return attn @ V

    out = torch.zeros(Q.shape[0], V.shape[1], device=Q.device, dtype=Q.dtype)
    max_scores = torch.full((Q.shape[0], 1), -float('inf'), device=Q.device, dtype=Q.dtype)
    denom = torch.zeros(Q.shape[0], 1, device=Q.device, dtype=Q.dtype)

    for start in range(0, n_keys, key_chunk):
        Kc = K[start:start + key_chunk]
        Vc = V[start:start + key_chunk]
        scores = Q @ Kc.T / scale
        chunk_max = scores.max(dim=1, keepdim=True).values
        new_max = torch.maximum(max_scores, chunk_max)
        exp_scores = torch.exp(scores - new_max)
        out = out * torch.exp(max_scores - new_max) + exp_scores @ Vc
        denom = denom * torch.exp(max_scores - new_max) + exp_scores.sum(dim=1, keepdim=True)
        max_scores = new_max

    return out / denom.clamp(min=1e-8)


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

        C, H, W, D = Z.shape
        assert C == self.d, f"Z has {C} channels but readout expects {self.d}"
        Z_flat = Z.reshape(C, -1).T

        K = self.W_K(Z_flat)
        V = self.W_V(Z_flat)
        scale = self.d ** 0.5

        q_chunk = config.OCC_READOUT_QUERY_CHUNK
        key_chunk = config.OCC_READOUT_KEY_CHUNK
        outputs = []
        for start in range(0, q_feat.shape[0], q_chunk):
            Q = self.W_Q(q_feat[start:start + q_chunk])
            z_q = _chunked_cross_attention(Q, K, V, scale, key_chunk)
            outputs.append(self.mlp(z_q).squeeze(1))

        return torch.cat(outputs, dim=0)
