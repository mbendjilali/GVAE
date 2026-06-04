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


def _voxel_spacing(grid) -> torch.Tensor:
    """Per-axis center spacing in normalised [-1, 1] space."""
    sizes = grid
    return torch.tensor(
        [2.0 / max(g - 1, 1) for g in sizes],
        dtype=torch.float32,
    )


def _truncation_box(sigma: float, r: torch.Tensor, grid, voxel_cap_radius: float | None):
    """Per-node, per-axis truncation half-width: min(σ·r, cap·spacing), with voxel floor."""
    spacing = _voxel_spacing(grid).to(device=r.device, dtype=r.dtype)
    trunc = sigma * r
    if voxel_cap_radius is not None:
        cap = voxel_cap_radius * spacing.unsqueeze(0)
        trunc = torch.minimum(trunc, cap)
    floor = config.SPLAT_MIN_TRUNC_VOXEL_FRAC * spacing.unsqueeze(0)
    return torch.maximum(trunc, floor)


def _splat_dense(h, p, r, grid, sigma, eps, voxel_cap_radius=None, vox_centers=None):
    """N × V masked scatter (fast when N×V fits in memory)."""
    device = h.device
    N, d = h.shape
    H, W, D = grid
    if vox_centers is None:
        vox_centers = make_voxel_centers(grid, device)
    diff = vox_centers[None] - p[:, None, :]
    trunc = _truncation_box(sigma, r, grid, voxel_cap_radius)
    within_trunc = (diff.abs() <= trunc[:, None, :]).all(dim=2)
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


def _splat_dense_chunked(h, p, r, grid, sigma, eps, node_chunk: int, voxel_cap_radius=None):
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
        trunc = _truncation_box(sigma, r_c, grid, voxel_cap_radius)
        within_trunc = (diff.abs() <= trunc[:, None, :]).all(dim=2)
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


def center_splat_grid(features: torch.Tensor) -> torch.Tensor:
    """Subtract spatial mean per channel so U-Net does not amplify a DC blob."""
    if not config.SPLAT_SUBTRACT_SPATIAL_MEAN:
        return features
    return features - features.mean(dim=(1, 2, 3), keepdim=True)


class GaussianSplatting(nn.Module):
    def __init__(
        self,
        grid,
        feature_dim: int,
        sigma: float | None = None,
        voxel_cap_radius: float | None = None,
    ):
        super().__init__()
        self.grid = grid
        self.feature_dim = feature_dim
        self.sigma = sigma if sigma is not None else config.SPLAT_TRUNCATION_SIGMA
        self.voxel_cap_radius = voxel_cap_radius

    def forward(self, h, p, r):
        voxels = self.grid[0] * self.grid[1] * self.grid[2]
        n_nodes = p.shape[0]
        pairs = n_nodes * voxels

        if pairs <= config.SPLAT_DENSE_MAX_PAIRS:
            return _splat_dense(
                h, p, r, self.grid, self.sigma, config.SPLAT_EPS,
                voxel_cap_radius=self.voxel_cap_radius,
            )
        return _splat_dense_chunked(
            h, p, r, self.grid, self.sigma, config.SPLAT_EPS,
            node_chunk=config.SPLAT_NODE_CHUNK,
            voxel_cap_radius=self.voxel_cap_radius,
        )
