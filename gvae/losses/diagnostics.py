# gvae/losses/diagnostics.py
# Per-scene loss breakdown for NaN / edge-connectivity debugging

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import config
from gvae.losses.gvae_loss import (
    KL_loss,
    anchor_footprint_loss,
    anchor_loss,
    kl_weight,
    norm_contrastive_loss,
    reconstruction_loss,
    _lambda_anchor,
    _lambda_anchor_r,
)
from gvae.data.occupancy import loss_occ_grid


@dataclass
class LossBreakdown:
    path: str = ""
    n_instance: int = 0
    n_fine: int = 0
    n_mid: int = 0
    n_coarse: int = 0
    e_instance: int = 0
    e_fine: int = 0
    e_mid: int = 0
    e_coarse: int = 0
    terms: dict[str, float] = field(default_factory=dict)
    nan_terms: list[str] = field(default_factory=list)
    total: float = float("nan")

    @property
    def has_nan(self) -> bool:
        return bool(self.nan_terms) or not math.isfinite(self.total)


def _store_term(breakdown: LossBreakdown, name: str, tensor: torch.Tensor) -> None:
    val = tensor.detach().float().item()
    breakdown.terms[name] = val
    if not math.isfinite(val):
        breakdown.nan_terms.append(name)


def _maybe_occ_grid(bd: LossBreakdown, name: str, head, z, occ_gt, lambda_grid: float) -> float:
    if lambda_grid <= 0 or head is None:
        return 0.0
    loss = loss_occ_grid(head(z), occ_gt)
    _store_term(bd, name, loss)
    return lambda_grid * bd.terms[name]


def _branch_terms(
    bd: LossBreakdown,
    *,
    name: str,
    recon,
    recon_zonly,
    recon_hzonly,
    p_true,
    r_true,
    s_true,
    p_anchor,
    r_anchor,
    mu,
    logvar,
    occ_grid_head,
    z,
    occ_grid,
    lam_kl: float,
) -> float:
    if p_true.numel() == 0:
        return 0.0

    total = 0.0
    prefix = name

    w_size_z = config.LAMBDA_SIZE_ZONLY
    w_sem_z = config.LAMBDA_SEM_ZONLY

    if recon is not None and config.LAMBDA_RECON_H > 0:
        _store_term(
            bd, f"recon_{prefix}",
            reconstruction_loss(recon, p_true, r_true, s_true),
        )
        total += config.LAMBDA_RECON_H * bd.terms[f"recon_{prefix}"]

    if (
        recon_zonly is not None
        and config.USE_Z_ONLY_DECODER
        and config.LAMBDA_RECON_ZONLY > 0
    ):
        _store_term(
            bd, f"recon_zonly_{prefix}",
            reconstruction_loss(
                recon_zonly, p_true, r_true, s_true,
                size_weight=w_size_z, sem_weight=w_sem_z,
            ),
        )
        total += config.LAMBDA_RECON_ZONLY * bd.terms[f"recon_zonly_{prefix}"]

    if (
        recon_hzonly is not None
        and config.USE_Z_ONLY_DECODER
        and config.LAMBDA_RECON_HZONLY > 0
    ):
        _store_term(
            bd, f"recon_hzonly_{prefix}",
            reconstruction_loss(
                recon_hzonly, p_true, r_true, s_true,
                size_weight=w_size_z, sem_weight=w_sem_z,
            ),
        )
        total += config.LAMBDA_RECON_HZONLY * bd.terms[f"recon_hzonly_{prefix}"]

    lambda_anchor = _lambda_anchor(prefix)
    if lambda_anchor > 0 and p_anchor is not None and p_anchor.numel() > 0:
        _store_term(bd, f"anchor_{prefix}", anchor_loss(p_anchor, p_true))
        total += lambda_anchor * bd.terms[f"anchor_{prefix}"]

    lambda_anchor_r = _lambda_anchor_r(prefix)
    if lambda_anchor_r > 0 and r_anchor is not None and r_anchor.numel() > 0:
        _store_term(bd, f"anchor_r_{prefix}", anchor_footprint_loss(r_anchor, r_true))
        total += lambda_anchor_r * bd.terms[f"anchor_r_{prefix}"]

    _store_term(bd, f"KL_{prefix}", KL_loss(mu, logvar))
    total += lam_kl * bd.terms[f"KL_{prefix}"]

    lambda_grid = {
        'fine': config.LAMBDA_OCC_GRID_FINE,
        'mid': config.LAMBDA_OCC_GRID_MID,
        'coarse': config.LAMBDA_OCC_GRID_COARSE,
    }[prefix]
    total += _maybe_occ_grid(
        bd, f"occ_{prefix}", occ_grid_head, z, occ_grid, lambda_grid,
    )

    lambda_norm = {
        'fine': config.LAMBDA_NORM_CONTRAST_FINE,
        'mid': config.LAMBDA_NORM_CONTRAST_MID,
        'coarse': config.LAMBDA_NORM_CONTRAST_COARSE,
    }[prefix]
    if lambda_norm > 0 and occ_grid.numel() > 0:
        _store_term(
            bd, f"norm_contrast_{prefix}",
            norm_contrastive_loss(z, p_true, occ_grid),
        )
        total += lambda_norm * bd.terms[f"norm_contrast_{prefix}"]

    return total


def loss_breakdown(
    outputs,
    graph,
    step: int,
    *,
    path: str = "",
) -> LossBreakdown:
    """Decompose the training loss without side effects."""
    bd = LossBreakdown(path=path)
    p, edge_index = graph.p, graph.edge_index
    bd.n_instance = int(p.shape[0])
    bd.e_instance = int(edge_index.shape[1])
    bd.n_fine = int(outputs["p_fine"].shape[0])
    bd.e_fine = int(outputs["edge_index_fine"].shape[1])
    bd.n_mid = int(outputs["p_lm1"].shape[0])
    bd.e_mid = int(outputs["edge_index_lm1"].shape[1])
    bd.n_coarse = int(outputs["p_1"].shape[0])
    bd.e_coarse = int(outputs["edge_index_1"].shape[1])

    lam_kl = kl_weight(step)
    bd.terms["lambda_kl"] = lam_kl

    total = 0.0
    total += _branch_terms(
        bd,
        name='fine',
        recon=outputs.get("recon_fine"),
        recon_zonly=outputs.get("recon_fine_zonly"),
        recon_hzonly=outputs.get("recon_fine_zonly_hanchor"),
        p_true=outputs["p_fine"],
        r_true=outputs["r_fine"],
        s_true=outputs["s_fine"],
        p_anchor=outputs.get("p_anchor_fine"),
        r_anchor=outputs.get("r_anchor_fine"),
        mu=outputs["mu_fine"],
        logvar=outputs["logvar_fine"],
        occ_grid_head=outputs["occ_grid_head_fine"],
        z=outputs["z_fine"],
        occ_grid=graph.occ_fine,
        lam_kl=lam_kl,
    )
    total += _branch_terms(
        bd,
        name='mid',
        recon=outputs.get("recon_mid"),
        recon_zonly=outputs.get("recon_mid_zonly"),
        recon_hzonly=outputs.get("recon_mid_zonly_hanchor"),
        p_true=outputs["p_lm1"],
        r_true=outputs["r_lm1"],
        s_true=outputs["s_lm1"],
        p_anchor=outputs.get("p_anchor_mid"),
        r_anchor=outputs.get("r_anchor_mid"),
        mu=outputs["mu_mid"],
        logvar=outputs["logvar_mid"],
        occ_grid_head=outputs["occ_grid_head_mid"],
        z=outputs["z_mid"],
        occ_grid=graph.occ_mid,
        lam_kl=lam_kl,
    )
    total += _branch_terms(
        bd,
        name='coarse',
        recon=outputs.get("recon_coarse"),
        recon_zonly=outputs.get("recon_coarse_zonly"),
        recon_hzonly=outputs.get("recon_coarse_zonly_hanchor"),
        p_true=outputs["p_1"],
        r_true=outputs["r_1"],
        s_true=outputs["s_1"],
        p_anchor=outputs.get("p_anchor_coarse"),
        r_anchor=outputs.get("r_anchor_coarse"),
        mu=outputs["mu_coarse"],
        logvar=outputs["logvar_coarse"],
        occ_grid_head=outputs["occ_grid_head_coarse"],
        z=outputs["z_coarse"],
        occ_grid=graph.occ_coarse,
        lam_kl=lam_kl,
    )

    bd.total = total
    if not math.isfinite(bd.total):
        if "total" not in bd.nan_terms:
            bd.nan_terms.append("total")
    return bd
