#!/usr/bin/env python3
"""Oracle sanity checks for validation metrics (run: python utils/metrics_sanity.py)."""

from __future__ import annotations

import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import torch
import torch.nn as nn

import config
from gvae.losses.gvae_loss import KL_loss, kl_weight, soft_semantic_loss
from gvae.losses.metrics import (
    _instance_pos_err_mid,
    _occupancy_grid_iou,
    compute_metrics,
    hard_miou,
    mean_position_error,
    soft_miou,
)


class _FixedHead(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._logits


class _FakeGraph:
    def __init__(self, occ_fine, occ_mid, occ_coarse, p_inst):
        self.occ_fine = occ_fine
        self.occ_mid = occ_mid
        self.occ_coarse = occ_coarse
        self.coarsen_mask = torch.ones(p_inst.shape[0], dtype=torch.bool)
        self.p = p_inst


def run_oracle_checks() -> list[str]:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            failures.append(f"{name}: {detail}")

    true_soft = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.5, 0.5]])
    check("soft_miou oracle", abs(soft_miou(true_soft, true_soft) - 1.0) < 1e-5)
    check("soft_miou empty", math.isnan(soft_miou(true_soft[:0], true_soft[:0])))

    s_true = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    check("hard_miou oracle", abs(hard_miou(s_true, s_true) - 1.0) < 1e-5)

    p = torch.randn(5, 3)
    check("pos_err oracle", mean_position_error(p, p) < 1e-6)

    occ = torch.zeros(4, 4, 2, dtype=torch.bool)
    occ[0, 0, 0] = True
    occ[1, 1, 0] = True
    logits = torch.where(occ, torch.tensor(30.0), torch.tensor(-30.0))
    head = _FixedHead(logits)
    z = torch.randn(8, 4, 4, 2)
    iou, prec, rec, pred_rate, gt_rate = _occupancy_grid_iou(head, z, occ)
    check("occ_iou oracle", abs(iou - 1.0) < 1e-5, f"got {iou}")
    check("occ_precision oracle", abs(prec - 1.0) < 1e-5, f"got {prec}")
    check("occ_recall oracle", abs(rec - 1.0) < 1e-5, f"got {rec}")
    check("occ_pred_rate oracle", abs(pred_rate - gt_rate) < 1e-5, f"got {pred_rate}")

    occ_sparse = torch.zeros(3, 3, 2, dtype=torch.bool)
    occ_sparse[0, 0, 0] = True
    head_empty = _FixedHead(torch.full((3, 3, 2), -30.0))
    iou0, _, _, _, _ = _occupancy_grid_iou(head_empty, z[:8, :3, :3, :2], occ_sparse)
    check("occ_iou zero pred", abs(iou0) < 1e-5, f"got {iou0}")

    outputs = {
        "S0": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "S1": torch.tensor([[1.0, 0.0]]),
        "recon_mid": {"p": torch.tensor([[0.1, 0.2, 0.3]])},
    }
    graph = _FakeGraph(
        occ, occ, occ,
        torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
    )
    check(
        "inst_pos_err_mid oracle",
        _instance_pos_err_mid(outputs, graph) < 1e-6,
    )

    check("recon_sem oracle", soft_semantic_loss(s_true, s_true).item() < 1e-6)
    check("kl oracle", abs(KL_loss(torch.zeros(50), torch.zeros(50)).item()) < 1e-6)

    old_steps = config.KL_TOTAL_STEPS
    config.KL_TOTAL_STEPS = 400
    check("kl_weight step0", kl_weight(0) == 0.0)
    check("kl_weight ramp", kl_weight(25) > 0)
    config.KL_TOTAL_STEPS = old_steps

    # Occ metrics omitted when lambda=0; present when lambda>0
    old_l = config.LAMBDA_OCC_GRID_FINE
    config.LAMBDA_OCC_GRID_FINE = 0.0
    fake_out = {
        "recon_fine": {"p": p[:2], "s": s_true, "r": p[:2]},
        "p_fine": p[:2],
        "s_fine": s_true,
        "r_fine": p[:2],
        "z_fine": z,
        "occ_grid_head_fine": head,
        "recon_mid": None,
        "p_lm1": torch.zeros(0, 3),
        "mu_fine": torch.zeros(1),
        "logvar_fine": torch.zeros(1),
        "mu_mid": torch.zeros(1),
        "logvar_mid": torch.zeros(1),
    }
    config.LOG_FULL_METRICS = False
    m_off = compute_metrics(fake_out, graph, step=0)
    check("occ_iou gated off", "occ_iou_fine" not in m_off)
    config.LAMBDA_OCC_GRID_FINE = old_l

    return failures


def main() -> int:
    failures = run_oracle_checks()
    if failures:
        print("FAILED checks:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All metric oracle checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
