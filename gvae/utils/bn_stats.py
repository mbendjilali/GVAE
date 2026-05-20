# gvae/utils/bn_stats.py
# BatchNorm running-stat management for per-scene (batch_size=1) 3D U-Net training

from __future__ import annotations

import torch
import torch.nn as nn

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def iter_batch_norm_modules(module: nn.Module):
    for m in module.modules():
        if isinstance(m, _BN_TYPES):
            yield m


def reset_bn_running_stats(module: nn.Module) -> None:
    """Clear running_mean / running_var (call when a U-Net branch is first trained)."""
    for bn in iter_batch_norm_modules(module):
        bn.reset_running_stats()


def set_bn_momentum(module: nn.Module, momentum: float | None) -> None:
    """
    momentum=None → cumulative moving average (recommended for batch_size=1).
    See PyTorch BatchNorm docs: each forward refines the global mean/var estimate.
    """
    for bn in iter_batch_norm_modules(module):
        bn.momentum = momentum


def set_unet_bn_train(encoder, stage: int) -> None:
    """
    Only U-Nets that are being trained should update BN running stats.
    Inactive branches stay in eval() so their BN stats stay frozen.
    """
    if stage >= 2:
        encoder.unet_mid.train()
    else:
        encoder.unet_mid.eval()

    if stage >= 3:
        encoder.unet_coarse.train()
    else:
        encoder.unet_coarse.eval()


@torch.no_grad()
def calibrate_batch_norm(model, loader, device, stage: int) -> int:
    """
    Refresh BatchNorm running statistics on the training set before validation.
    Runs full-graph forwards in train() mode (BN layers only on active U-Nets update).

    Returns the number of graphs processed.
    """
    was_training = model.training
    model.train()
    set_unet_bn_train(model.encoder, stage)

    n = 0
    for batch in loader:
        for graph in batch:
            model(graph.to(device))
            n += 1

    model.train(was_training)
    set_unet_bn_train(model.encoder, stage)
    return n


def init_unet_batch_norm(encoder) -> None:
    """Configure both 3D U-Nets for stable running-stat accumulation."""
    for unet in (encoder.unet_mid, encoder.unet_coarse):
        set_bn_momentum(unet, None)
