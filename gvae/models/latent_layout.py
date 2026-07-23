# gvae/models/latent_layout.py
# Shared localized ||Z|| readout (peak loss, metrics, LatentGraphDecoder).

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import config
from gvae.models.splatting import make_voxel_centers


def _peak_k(grid_shape: tuple[int, int, int], n_nodes: int) -> int:
    return min(config.LATENT_PEAK_NEIGHBORS, math.prod(grid_shape), max(n_nodes, 1))


def positions_in_windows(
    centers: torch.Tensor,
    mag: torch.Tensor,
    window_idx: torch.Tensor,
    *,
    extra_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft-argmax positions from ||Z|| inside per-node voxel windows (N, k)."""
    temp = max(config.LATENT_PEAK_TEMP, 1e-4)
    logits = mag[window_idx] / temp
    if extra_logits is not None:
        logits = logits + extra_logits
    weights = F.softmax(logits, dim=-1)
    return (weights.unsqueeze(-1) * centers[window_idx]).sum(dim=1).clamp(-1.0, 1.0)


def peak_windows_around_gt(centers: torch.Tensor, p_gt: torch.Tensor, k: int) -> torch.Tensor:
    """(N, k) voxel indices: k nearest centres to each supernode GT position."""
    dist2 = torch.cdist(centers, p_gt, p=2).pow(2)
    _, topi = dist2.topk(k, dim=0, largest=False)
    return topi.T.contiguous()


def peak_windows_from_queries(
    queries: torch.Tensor,
    tokens: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, k) indices + query logits in window for decode without GT."""
    scale = math.sqrt(queries.shape[-1]) * max(config.LATENT_GRAPH_SPATIAL_TEMP, 1e-4)
    sim = queries @ tokens.T
    local_sim, topi = sim.topk(k, dim=-1, largest=True)
    q_logits = local_sim / scale if config.LATENT_DECODE_QUERY_WEIGHT > 0 else None
    return topi, q_logits


def layout_magnitude(z: torch.Tensor, layout: torch.Tensor | None) -> torch.Tensor:
    """Spatial activation map for peaks / decode (layout head, else ||Z||)."""
    if layout is not None:
        if layout.shape[0] == 1:
            return layout.squeeze(0).flatten()
        return layout.norm(dim=0).flatten()
    return z.norm(dim=0).flatten()


def slot_window_centroids(
    queries: torch.Tensor,
    tokens: torch.Tensor,
    centers: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Per-slot soft centroid of the query k-NN window (no GT)."""
    topi, q_logits = peak_windows_from_queries(queries, tokens, k)
    if q_logits is None:
        weights = torch.ones_like(topi, dtype=torch.float32) / topi.shape[1]
    else:
        weights = F.softmax(q_logits.float(), dim=-1)
    return (weights.unsqueeze(-1) * centers[topi]).sum(dim=1)


def nms_peaks_assigned_to_slots(
    mag: torch.Tensor,
    centers: torch.Tensor,
    slot_centroids: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """NMS peaks, then Hungarian-assign to slots via query centroids (honest, no GT)."""
    from scipy.optimize import linear_sum_assignment

    if n_nodes <= 0:
        return centers.new_zeros(0, 3)
    peaks = nms_peak_positions(mag, centers, n_nodes)
    n_peaks = peaks.shape[0]
    n_slots = slot_centroids.shape[0]
    if n_peaks == 0 or n_slots == 0:
        return slot_centroids.new_zeros(n_nodes, 3)
    cost = torch.cdist(peaks, slot_centroids, p=2)
    n = max(n_peaks, n_slots)
    pad_cost = cost.new_full((n, n), 1e3)
    pad_cost[:n_peaks, :n_slots] = cost
    row, col = linear_sum_assignment(pad_cost.detach().cpu().numpy())
    p_hat = slot_centroids.new_zeros(n_slots, 3)
    for r, c in zip(row, col):
        if r < n_peaks and c < n_slots:
            p_hat[c] = peaks[r]
    return p_hat.clamp(-1.0, 1.0)


def nms_peak_positions(
    mag: torch.Tensor,
    centers: torch.Tensor,
    n_nodes: int,
    *,
    suppress_sigma: float | None = None,
) -> torch.Tensor:
    """Greedy NMS on layout activation → up to n_nodes positions (honest, no GT)."""
    if n_nodes <= 0:
        return centers.new_zeros(0, 3)
    sigma = suppress_sigma if suppress_sigma is not None else config.LATENT_NMS_SUPPRESS_SIGMA
    sup2 = sigma * sigma
    # float32 loop: AMP may pass float16 mag; -1e9 overflows half
    work = mag.float().clone()
    suppress_val = -1e9
    picks: list[int] = []
    for _ in range(min(n_nodes, mag.numel())):
        v = int(work.argmax().item())
        if work[v].item() < suppress_val * 0.5:
            break
        picks.append(v)
        d2 = ((centers - centers[v]) ** 2).sum(-1)
        work[d2 < sup2] = suppress_val
    if not picks:
        return centers.new_zeros(0, 3)
    idx = torch.tensor(picks, device=mag.device, dtype=torch.long)
    return centers[idx].clamp(-1.0, 1.0)


def index_aligned_peak_positions(
    z: torch.Tensor,
    p_gt: torch.Tensor,
    layout: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per supernode i: soft-argmax on layout inside k-NN(p_gt[i]) — slot order aligned."""
    _, H, W, D = z.shape
    centers = make_voxel_centers((H, W, D), z.device)
    mag = layout_magnitude(z, layout)
    k = _peak_k((H, W, D), p_gt.shape[0])
    windows = peak_windows_around_gt(centers, p_gt, k)
    return positions_in_windows(centers, mag, windows)


def peak_positions_at_gt(
    z: torch.Tensor,
    p_gt: torch.Tensor,
    layout: torch.Tensor | None = None,
) -> torch.Tensor:
    """Alias: index-aligned oracle layout readout (one row per supernode)."""
    return index_aligned_peak_positions(z, p_gt, layout=layout)
