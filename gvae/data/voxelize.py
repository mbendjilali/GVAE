# gvae/data/voxelize.py
# Point-cloud voxelisation and query sampling for occupancy supervision

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

import config


def scene_normalization(positions: Tensor) -> tuple[Tensor, float]:
    """
    Same rule as SceneGraph: centre on mean instance position, scale by max |coordinate|.
    positions: (N, 3) in metres
    """
    centroid = positions.mean(dim=0)
    centred = positions - centroid
    scale = centred.abs().max().clamp(min=1e-6).item()
    return centroid, scale


def normalize_positions(positions: Tensor, centroid: Tensor, scale: float) -> Tensor:
    return (positions - centroid) / scale


def _points_to_indices(points: Tensor, grid: tuple[int, int, int]) -> tuple[Tensor, Tensor, Tensor]:
    """Map normalised points in [-1, 1]³ to voxel indices (matches splatting linspace grid)."""
    H, W, D = grid
    p = points.clamp(-1.0, 1.0)

    def axis_idx(coord: Tensor, n: int) -> Tensor:
        if n == 1:
            return torch.zeros(coord.shape[0], dtype=torch.long, device=coord.device)
        idx = ((coord + 1.0) * 0.5 * (n - 1)).round().long()
        return idx.clamp(0, n - 1)

    return axis_idx(p[:, 0], H), axis_idx(p[:, 1], W), axis_idx(p[:, 2], D)


def voxelize_points(points: Tensor, grid: tuple[int, int, int]) -> Tensor:
    """
    Mark voxels True if at least one point maps to that cell.

    points: (P, 3) in normalised [-1, 1]³
    Returns: (H, W, D) bool tensor
    """
    H, W, D = grid
    occ = torch.zeros(H, W, D, dtype=torch.bool, device=points.device)
    if points.numel() == 0:
        return occ

    i, j, k = _points_to_indices(points, grid)
    occ[i, j, k] = True
    return occ


def voxelize_points_np(points: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """NumPy variant used by utils/build_scene_graph.py."""
    H, W, D = grid
    occ = np.zeros((H, W, D), dtype=bool)
    if points.size == 0:
        return occ

    p = np.clip(points, -1.0, 1.0)

    def axis_idx(coord: np.ndarray, n: int) -> np.ndarray:
        if n == 1:
            return np.zeros(coord.shape[0], dtype=np.int64)
        idx = np.rint((coord + 1.0) * 0.5 * (n - 1)).astype(np.int64)
        return np.clip(idx, 0, n - 1)

    i = axis_idx(p[:, 0], H)
    j = axis_idx(p[:, 1], W)
    k = axis_idx(p[:, 2], D)
    occ[i, j, k] = True
    return occ


def voxel_centers(grid: tuple[int, int, int], device: torch.device) -> Tensor:
    """Voxel centre coordinates (H*W*D, 3), same layout as splatting.make_voxel_centers."""
    H, W, D = grid
    xs = torch.linspace(-1, 1, H, device=device)
    ys = torch.linspace(-1, 1, W, device=device)
    zs = torch.linspace(-1, 1, D, device=device)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)


def sample_occupancy_queries(
    occ: Tensor,
    n_queries: int = config.OCC_QUERY_POINTS,
    pos_ratio: float = config.OCC_POS_RATIO,
) -> tuple[Tensor, Tensor]:
    """
    Sample query locations and binary occupancy labels from a voxel grid.

    pos_ratio of queries are drawn from occupied voxels; the rest from empty voxels
    (or uniform [-1,1]³ if the scene has no empty voxels in the grid).
    """
    device = occ.device
    H, W, D = occ.shape
    centres = voxel_centers((H, W, D), device)

    flat_occ = occ.flatten()
    occupied_idx = flat_occ.nonzero(as_tuple=True)[0]
    empty_idx = (~flat_occ).nonzero(as_tuple=True)[0]

    n_pos = int(n_queries * pos_ratio)
    n_neg = n_queries - n_pos

    q_parts: list[Tensor] = []
    label_parts: list[Tensor] = []

    if n_pos > 0:
        if occupied_idx.numel() > 0:
            pick = occupied_idx[torch.randint(0, occupied_idx.numel(), (n_pos,), device=device)]
            q_parts.append(centres[pick])
            label_parts.append(torch.ones(n_pos, device=device))
        else:
            q_parts.append(torch.rand(n_pos, 3, device=device) * 2 - 1)
            label_parts.append(torch.zeros(n_pos, device=device))

    if n_neg > 0:
        if empty_idx.numel() > 0:
            pick = empty_idx[torch.randint(0, empty_idx.numel(), (n_neg,), device=device)]
            q_parts.append(centres[pick])
            label_parts.append(torch.zeros(n_neg, device=device))
        else:
            q_unif = torch.rand(n_neg, 3, device=device) * 2 - 1
            ii, jj, kk = _points_to_indices(q_unif, (H, W, D))
            q_parts.append(q_unif)
            label_parts.append(occ[ii, jj, kk].float())

    if not q_parts:
        return torch.zeros(1, 3, device=device), torch.zeros(1, device=device)

    return torch.cat(q_parts, dim=0), torch.cat(label_parts, dim=0)


def occ_cache_paths(json_path: str) -> tuple[str, str, str]:
    base = json_path[: -len(".json")] if json_path.endswith(".json") else json_path
    return (
        base + config.OCC_CACHE_SUFFIX_FINE,
        base + config.OCC_CACHE_SUFFIX_MID,
        base + config.OCC_CACHE_SUFFIX_COARSE,
    )
