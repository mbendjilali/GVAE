# gvae/models/splatting.py
# Truncated anisotropic Gaussian splatting with scatter_add.
# Full dense path when N×V is small; chunked dense for large scenes.

import torch
import torch.nn as nn
from torch_scatter import scatter_add

import config


def make_voxel_centers(grid, device):
    H, W, D = grid
    x_centers = torch.linspace(-1, 1, H, device=device)
    y_centers = torch.linspace(-1, 1, W, device=device)
    z_centers = torch.linspace(-1, 1, D, device=device)
    gx, gy, gz = torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    return torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)


def _splat_dense(h, p, r, grid, sigma, eps, vox_centers=None):
    """N × V masked scatter (fast when N×V fits in memory)."""
    device = h.device
    N, d = h.shape
    H, W, D = grid
    if vox_centers is None:
        vox_centers = make_voxel_centers(grid, device)
    diff = vox_centers[None] - p[:, None, :]
    within_trunc = (diff.abs() <= (sigma * r)[:, None, :]).all(dim=2)
    node_idx, voxel_idx = within_trunc.nonzero(as_tuple=True)

    valid_diff = diff[node_idx, voxel_idx]
    valid_r = r[node_idx]
    diff_normalised = valid_diff / (valid_r + 1e-6)
    w = torch.exp(-0.5 * (diff_normalised ** 2).sum(dim=1))

    V = vox_centers.shape[0]
    weighted_features = w.unsqueeze(1) * h[node_idx]
    voxel_features = scatter_add(weighted_features, voxel_idx, dim=0, dim_size=V)
    weight_sum = scatter_add(w, voxel_idx, dim=0, dim_size=V)
    voxel_features = voxel_features / (weight_sum.unsqueeze(1) + eps)
    return voxel_features.T.reshape(d, H, W, D)


def _splat_dense_chunked(h, p, r, grid, sigma, eps, node_chunk: int):
    """Process nodes in chunks to avoid materialising full N×V mask."""
    device = h.device
    N, d = h.shape
    H, W, D = grid
    V = H * W * D

    if N == 0:
        return h.new_zeros(h.shape[1], H, W, D)

    vox_centers = make_voxel_centers(grid, device)
    voxel_features = torch.zeros(V, d, device=device, dtype=h.dtype)
    weight_sum = torch.zeros(V, device=device, dtype=h.dtype)

    for start in range(0, N, node_chunk):
        end = min(start + node_chunk, N)
        h_c = h[start:end]
        p_c = p[start:end]
        r_c = r[start:end]

        diff = vox_centers[None] - p_c[:, None, :]
        within_trunc = (diff.abs() <= (sigma * r_c)[:, None, :]).all(dim=2)
        node_idx, voxel_idx = within_trunc.nonzero(as_tuple=True)

        if node_idx.numel() == 0:
            continue

        valid_diff = diff[node_idx, voxel_idx]
        valid_r = r_c[node_idx]
        diff_normalised = valid_diff / (valid_r + 1e-6)
        w = torch.exp(-0.5 * (diff_normalised ** 2).sum(dim=1))

        weighted_features = w.unsqueeze(1) * h_c[node_idx]
        voxel_features = scatter_add(weighted_features, voxel_idx, dim=0, out=voxel_features)
        weight_sum = scatter_add(w, voxel_idx, dim=0, out=weight_sum)

    voxel_features = voxel_features / (weight_sum.unsqueeze(1) + eps)
    return voxel_features.T.reshape(d, H, W, D)


class GaussianSplatting(nn.Module):
    def __init__(self, grid, feature_dim: int):
        super().__init__()
        self.grid = grid
        self.feature_dim = feature_dim
        self.sigma = config.SPLAT_TRUNCATION_SIGMA

    def forward(self, h, p, r):
        voxels = self.grid[0] * self.grid[1] * self.grid[2]
        n_nodes = p.shape[0]
        pairs = n_nodes * voxels

        if pairs <= config.SPLAT_DENSE_MAX_PAIRS:
            return _splat_dense(h, p, r, self.grid, self.sigma, config.SPLAT_EPS)
        return _splat_dense_chunked(
            h, p, r, self.grid, self.sigma, config.SPLAT_EPS,
            node_chunk=config.SPLAT_NODE_CHUNK,
        )
