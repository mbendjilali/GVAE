# gvae/losses/diagnostics.py
# Per-scene loss breakdown for NaN / edge-connectivity debugging

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import config
from gvae.losses.gvae_loss import (
    KL_loss,
    kl_weight,
    loss_occupancy,
    reconstruction_loss,
)


@dataclass
class LossBreakdown:
    path: str = ""
    n_instance: int = 0
    n_mid: int = 0
    n_coarse: int = 0
    e_instance: int = 0
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


def loss_breakdown(
    outputs,
    graph,
    step: int,
    stage: int,
    *,
    path: str = "",
    include_coarse_in_total: bool | None = None,
) -> LossBreakdown:
    """Decompose the training loss without side effects."""
    if include_coarse_in_total is None:
        include_coarse_in_total = stage >= 1

    bd = LossBreakdown(path=path)
    p, edge_index = graph.p, graph.edge_index
    bd.n_instance = int(p.shape[0])
    bd.e_instance = int(edge_index.shape[1])
    bd.n_mid = int(outputs["p_lm1"].shape[0])
    bd.e_mid = int(outputs["edge_index_lm1"].shape[1])
    bd.n_coarse = int(outputs["p_1"].shape[0])
    bd.e_coarse = int(outputs["edge_index_1"].shape[1])

    lam_kl = kl_weight(step)
    bd.terms["lambda_kl"] = lam_kl

    L_recon = 0.0
    L_kl = 0.0
    L_occ = 0.0

    if outputs.get("recon_fine") is not None and outputs["p_inst"].numel() > 0:
        _store_term(
            bd,
            "recon_fine",
            reconstruction_loss(
                outputs["recon_fine"],
                outputs["p_inst"],
                outputs["r_inst"],
                outputs["s_inst"],
                outputs["edge_index_inst"],
                edge_margin=config.EDGE_PROXIMITY,
            ),
        )
        _store_term(bd, "KL_fine", KL_loss(outputs["mu_fine"], outputs["logvar_fine"]))
        _store_term(
            bd,
            "occ_fine",
            loss_occupancy(outputs["occ_readout_fine"], outputs["z_fine"], graph.occ_fine),
        )
        L_recon += bd.terms["recon_fine"]
        L_kl += bd.terms["KL_fine"]
        L_occ += bd.terms["occ_fine"]

    if outputs.get("recon_mid") is not None and outputs["p_lm1"].numel() > 0:
        _store_term(
            bd,
            "recon_mid",
            reconstruction_loss(
                outputs["recon_mid"],
                outputs["p_lm1"],
                outputs["r_lm1"],
                outputs["s_lm1"],
                outputs["edge_index_lm1"],
                edge_margin=config.BALL_QUERY_RADIUS_LEVELS[0],
            ),
        )
        _store_term(bd, "KL_mid", KL_loss(outputs["mu_mid"], outputs["logvar_mid"]))
        _store_term(
            bd,
            "occ_mid",
            loss_occupancy(outputs["occ_readout_mid"], outputs["z_mid"], graph.occ_mid),
        )
        L_recon += bd.terms["recon_mid"]
        L_kl += bd.terms["KL_mid"]
        L_occ += bd.terms["occ_mid"]

    if include_coarse_in_total and outputs.get("recon_coarse") is not None and outputs["p_1"].numel() > 0:
        _store_term(
            bd,
            "recon_coarse",
            reconstruction_loss(
                outputs["recon_coarse"],
                outputs["p_1"],
                outputs["r_1"],
                outputs["s_1"],
                outputs["edge_index_1"],
                edge_margin=config.BALL_QUERY_RADIUS_LEVELS[1],
            ),
        )
        _store_term(bd, "KL_coarse", KL_loss(outputs["mu_coarse"], outputs["logvar_coarse"]))
        _store_term(
            bd,
            "occ_coarse",
            loss_occupancy(outputs["occ_readout_coarse"], outputs["z_coarse"], graph.occ_coarse),
        )
        L_recon += bd.terms["recon_coarse"]
        L_kl += bd.terms["KL_coarse"]
        L_occ += bd.terms["occ_coarse"]

    bd.total = L_recon + lam_kl * L_kl + config.LAMBDA_OCC * L_occ
    if not math.isfinite(bd.total):
        if "total" not in bd.nan_terms:
            bd.nan_terms.append("total")
    return bd
