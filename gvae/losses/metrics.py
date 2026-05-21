# gvae/losses/metrics.py
# Primary validation metrics (PR1): inst layout, occ, Z-only pos, soft mIoU

from __future__ import annotations

import torch
import torch.nn.functional as F

import config
from gvae.data.voxelize import voxel_centers
from gvae.losses.gvae_loss import soft_semantic_loss


def soft_miou(pred_probs: torch.Tensor, true_soft: torch.Tensor) -> float:
    """Soft mIoU: mean per-class IoU on probability mass (for soft supernode labels)."""
    if true_soft.numel() == 0:
        return float("nan")
    true = true_soft / true_soft.sum(dim=1, keepdim=True).clamp(min=config.SOFT_MIOU_EPS)
    inter = (pred_probs * true).sum(dim=0)
    union = (pred_probs + true - pred_probs * true).sum(dim=0)
    iou = inter / union.clamp(min=config.SOFT_MIOU_EPS)
    present = true.sum(dim=0) > config.SOFT_MIOU_EPS
    if not present.any():
        return float("nan")
    return iou[present].mean().item()


def mean_position_error(pred_positions: torch.Tensor, true_positions: torch.Tensor) -> float:
    if pred_positions.numel() == 0:
        return float("nan")
    return torch.norm(pred_positions - true_positions, dim=1).mean().item()


def _occupancy_iou_precision(
    occ_readout,
    z: torch.Tensor,
    occ_gt: torch.Tensor,
) -> tuple[float, float]:
    if occ_gt.numel() == 0:
        return float("nan"), float("nan")

    centres = voxel_centers(tuple(occ_gt.shape), z.device)
    logits = occ_readout(centres, z)
    probs = torch.sigmoid(logits)
    pred = probs >= config.METRICS_OCC_THRESHOLD
    flat_gt = occ_gt.flatten()

    tp = (pred & flat_gt).sum().float()
    fp = (pred & ~flat_gt).sum().float()
    fn = (~pred & flat_gt).sum().float()
    union = tp + fp + fn
    iou = (tp / union.clamp(min=1)).item()
    precision = (tp / (tp + fp).clamp(min=1)).item()
    return iou, precision


def _instance_pos_err_mid(outputs, graph) -> float:
    idx = graph.coarsen_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0 or outputs["S1"].numel() == 0:
        return float("nan")
    assign_mid = outputs["S1"].argmax(dim=1)
    pred_p = outputs["recon_mid"]["p"][assign_mid]
    return mean_position_error(pred_p, graph.p[idx])


def compute_metrics(outputs, graph, stage: int, step: int = 0) -> dict[str, float]:
    """Primary monitoring metrics (console + TensorBoard)."""
    if stage < 2:
        return {}

    with torch.no_grad():
        metrics: dict[str, float] = {}

        if outputs["p_lm1"].numel() > 0:
            metrics["pos_err_mid"] = mean_position_error(
                outputs["recon_mid"]["p"], outputs["p_lm1"],
            )
            metrics["soft_miou_mid"] = soft_miou(
                outputs["recon_mid"]["s"], outputs["s_lm1"],
            )

        iou, prec = _occupancy_iou_precision(
            outputs["occ_readout_mid"], outputs["z_mid"], graph.occ_mid,
        )
        metrics["occ_iou_mid"] = iou
        metrics["occ_precision_mid"] = prec
        metrics["inst_pos_err_mid"] = _instance_pos_err_mid(outputs, graph)

        if config.LOG_FULL_METRICS:
            metrics.update(_full_metrics(outputs, graph, stage, step))

    return metrics


def _full_metrics(outputs, graph, stage: int, step: int) -> dict[str, float]:
    """Extended debug metrics (TensorBoard only when LOG_FULL_METRICS=True)."""
    from gvae.losses.gvae_loss import KL_loss, kl_weight

    metrics: dict[str, float] = {"lambda_kl_metric": kl_weight(step)}

    if outputs["p_lm1"].numel() > 0:
        metrics["recon_sem_mid"] = soft_semantic_loss(
            outputs["recon_mid"]["s"], outputs["s_lm1"],
        ).item()

    coarse_prefix = "coarse" if stage >= 3 else "coarse_preview"
    if (stage >= 3 or config.LOG_COARSE_PREVIEW_AT_STAGE2) and outputs["p_1"].numel() > 0:
        metrics[f"pos_err_{coarse_prefix}"] = mean_position_error(
            outputs["recon_coarse"]["p"], outputs["p_1"],
        )
        metrics[f"soft_miou_{coarse_prefix}"] = soft_miou(
            outputs["recon_coarse"]["s"], outputs["s_1"],
        )
        iou_c, prec_c = _occupancy_iou_precision(
            outputs["occ_readout_coarse"], outputs["z_coarse"], graph.occ_coarse,
        )
        metrics[f"occ_iou_{coarse_prefix}"] = iou_c
        metrics[f"occ_precision_{coarse_prefix}"] = prec_c

    if outputs["mu_mid"].numel() > 0:
        metrics["kl_mid"] = KL_loss(outputs["mu_mid"], outputs["logvar_mid"]).item()

    return metrics
