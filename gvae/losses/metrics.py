# gvae/losses/metrics.py
# Primary validation metrics: fine instance layout, mid supernode, reconstruction

from __future__ import annotations

import torch

import config
from gvae.losses.gvae_loss import soft_cross_entropy_loss


def soft_miou(pred_probs: torch.Tensor, true_soft: torch.Tensor) -> float:
    """Per-supernode soft IoU: mean_n sum_c min(p,t) / sum_c max(p,t).

    Matches GT soft labels on each supernode independently; oracle = 1.0 when pred == true.
    """
    if true_soft.numel() == 0:
        return float("nan")
    true = true_soft / true_soft.sum(dim=1, keepdim=True).clamp(min=config.SOFT_MIOU_EPS)
    pred = pred_probs.clamp(min=0.0)
    pred = pred / pred.sum(dim=1, keepdim=True).clamp(min=config.SOFT_MIOU_EPS)
    inter = torch.minimum(pred, true).sum(dim=1)
    union = torch.maximum(pred, true).sum(dim=1)
    return (inter / union.clamp(min=config.SOFT_MIOU_EPS)).mean().item()


def hard_miou(pred_probs: torch.Tensor, true_onehot: torch.Tensor) -> float:
    """Hard mIoU on argmax class (instance one-hot labels)."""
    if true_onehot.numel() == 0:
        return float("nan")
    pred_cls = pred_probs.argmax(dim=1)
    true_cls = true_onehot.argmax(dim=1)
    ious = []
    for c in range(true_onehot.shape[1]):
        true_c = true_cls == c
        if not true_c.any():
            continue
        pred_c = pred_cls == c
        inter = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        ious.append((inter / union.clamp(min=1)).item())
    return sum(ious) / len(ious) if ious else float("nan")


def mean_position_error(pred_positions: torch.Tensor, true_positions: torch.Tensor) -> float:
    if pred_positions.numel() == 0:
        return float("nan")
    return torch.norm(pred_positions - true_positions, dim=1).mean().item()


def mean_footprint_error(pred_r: torch.Tensor, true_r: torch.Tensor) -> float:
    """Mean L1 error on footprint semi-axes (x, y, z)."""
    if pred_r.numel() == 0:
        return float("nan")
    return (pred_r - true_r).abs().mean().item()


def _instance_pos_err_mid(outputs, graph) -> float:
    idx = graph.coarsen_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0 or outputs["S0"].numel() == 0 or outputs["S1"].numel() == 0:
        return float("nan")
    assign_fine = outputs["S0"].argmax(dim=1)
    assign_mid = outputs["S1"].argmax(dim=1)[assign_fine]
    pred_p = outputs["recon_mid"]["p"][assign_mid]
    return mean_position_error(pred_p, graph.p[idx])


def compute_metrics(outputs, graph, step: int = 0) -> dict[str, float]:
    """Primary monitoring metrics (console + TensorBoard)."""
    with torch.no_grad():
        metrics: dict[str, float] = {}
        if outputs.get("latent_graph_vae"):
            metrics["latent_graph_vae"] = 1.0

        recon_fine = outputs.get("recon_fine")
        if recon_fine is not None and outputs["p_fine"].numel() > 0:
            metrics["pos_err_fine"] = mean_position_error(
                recon_fine["p"], outputs["p_fine"],
            )
            metrics["size_err_fine"] = mean_footprint_error(
                recon_fine["r"], outputs["r_fine"],
            )
            metrics["soft_miou_fine"] = soft_miou(recon_fine["s"], outputs["s_fine"])

        recon_fine_z = outputs.get("recon_fine_zonly")
        if recon_fine_z is not None and outputs["p_fine"].numel() > 0:
            metrics["pos_err_zonly_fine"] = mean_position_error(
                recon_fine_z["p"], outputs["p_fine"],
            )
            metrics["size_err_zonly_fine"] = mean_footprint_error(
                recon_fine_z["r"], outputs["r_fine"],
            )
            metrics["soft_miou_zonly_fine"] = soft_miou(
                recon_fine_z["s"], outputs["s_fine"],
            )

        recon_fine_z_h = outputs.get("recon_fine_zonly_hanchor")
        if recon_fine_z_h is not None and outputs["p_fine"].numel() > 0:
            metrics["pos_err_zonly_hanchor_fine"] = mean_position_error(
                recon_fine_z_h["p"], outputs["p_fine"],
            )

        p_anchor_f = outputs.get("p_anchor_fine")
        if p_anchor_f is not None and p_anchor_f.numel() > 0 and outputs["p_fine"].numel() > 0:
            metrics["anchor_err_fine"] = mean_position_error(p_anchor_f, outputs["p_fine"])
        r_anchor_f = outputs.get("r_anchor_fine")
        if r_anchor_f is not None and r_anchor_f.numel() > 0 and outputs["r_fine"].numel() > 0:
            metrics["anchor_size_err_fine"] = mean_footprint_error(
                r_anchor_f, outputs["r_fine"],
            )

        if outputs.get("recon_mid") is not None and outputs["p_lm1"].numel() > 0:
            metrics["pos_err_mid"] = mean_position_error(
                outputs["recon_mid"]["p"], outputs["p_lm1"],
            )
            metrics["size_err_mid"] = mean_footprint_error(
                outputs["recon_mid"]["r"], outputs["r_lm1"],
            )
            metrics["soft_miou_mid"] = soft_miou(
                outputs["recon_mid"]["s"], outputs["s_lm1"],
            )

        recon_mid_z = outputs.get("recon_mid_zonly")
        if recon_mid_z is not None and outputs["p_lm1"].numel() > 0:
            metrics["pos_err_zonly_mid"] = mean_position_error(
                recon_mid_z["p"], outputs["p_lm1"],
            )

        recon_mid_z_h = outputs.get("recon_mid_zonly_hanchor")
        if recon_mid_z_h is not None and outputs["p_lm1"].numel() > 0:
            metrics["pos_err_zonly_hanchor_mid"] = mean_position_error(
                recon_mid_z_h["p"], outputs["p_lm1"],
            )

        p_anchor_m = outputs.get("p_anchor_mid")
        if p_anchor_m is not None and p_anchor_m.numel() > 0 and outputs["p_lm1"].numel() > 0:
            metrics["anchor_err_mid"] = mean_position_error(p_anchor_m, outputs["p_lm1"])
        r_anchor_m = outputs.get("r_anchor_mid")
        if r_anchor_m is not None and r_anchor_m.numel() > 0 and outputs["r_lm1"].numel() > 0:
            metrics["anchor_size_err_mid"] = mean_footprint_error(
                r_anchor_m, outputs["r_lm1"],
            )

        if outputs.get("recon_mid") is not None:
            metrics["inst_pos_err_mid"] = _instance_pos_err_mid(outputs, graph)

        if config.LOG_FULL_METRICS:
            metrics.update(_full_metrics(outputs, graph, step))

    return metrics


def _full_metrics(outputs, graph, step: int) -> dict[str, float]:
    """Extended debug metrics (TensorBoard only when LOG_FULL_METRICS=True)."""
    from gvae.losses.gvae_loss import KL_loss, kl_weight

    metrics: dict[str, float] = {"lambda_kl_metric": kl_weight(step)}

    recon_fine = outputs.get("recon_fine")
    if recon_fine is not None and outputs["p_fine"].numel() > 0:
        metrics["miou_fine"] = hard_miou(recon_fine["s"], outputs["s_fine"])
        metrics["recon_sem_fine"] = soft_cross_entropy_loss(
            recon_fine["s"], outputs["s_fine"],
        ).item()

    if outputs.get("recon_mid") is not None and outputs["p_lm1"].numel() > 0:
        metrics["recon_sem_mid"] = soft_cross_entropy_loss(
            outputs["recon_mid"]["s"], outputs["s_lm1"],
        ).item()

    if outputs.get("recon_coarse") is not None:
        if outputs["p_1"].numel() > 0:
            metrics["pos_err_coarse"] = mean_position_error(
                outputs["recon_coarse"]["p"], outputs["p_1"],
            )
            metrics["soft_miou_coarse"] = soft_miou(
                outputs["recon_coarse"]["s"], outputs["s_1"],
            )

    if outputs["mu_fine"].numel() > 0:
        metrics["kl_fine"] = KL_loss(outputs["mu_fine"], outputs["logvar_fine"]).item()
    if outputs["mu_mid"].numel() > 0:
        metrics["kl_mid"] = KL_loss(outputs["mu_mid"], outputs["logvar_mid"]).item()

    return metrics
