# gvae/data/occupancy.py
# Voxel-grid occupancy loss (OccGridHead supervision)

import torch
import torch.nn.functional as F

import config

__all__ = [
    'occ_grid_pos_weight',
    'loss_occ_grid',
]


def occ_grid_pos_weight(occ_gt: torch.Tensor) -> float:
    """Positive-class weight for sparse voxel BCE (neg count / pos count)."""
    if config.OCC_GRID_POS_WEIGHT is not None:
        return float(config.OCC_GRID_POS_WEIGHT)
    n_pos = occ_gt.sum().float().clamp(min=1.0)
    n_neg = (occ_gt.numel() - n_pos).clamp(min=1.0)
    ratio = (n_neg / n_pos).item()
    cap = config.OCC_GRID_POS_WEIGHT_CAP
    if cap is not None and cap > 0:
        ratio = min(ratio, cap)
    return ratio


def loss_occ_grid(logits: torch.Tensor, occ_gt: torch.Tensor) -> torch.Tensor:
    """BCE on full voxel grid aligned with Z."""
    target = occ_gt.to(dtype=logits.dtype)
    pos_weight = logits.new_tensor([occ_grid_pos_weight(occ_gt)])
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight,
    )
