# gvae/training/freeze.py — selective requires_grad for finetune runs

from __future__ import annotations

import torch.nn as nn

# Prefixes matched against model.named_parameters() keys.
TRAIN_MODULE_PREFIXES: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "zonly": ("zonly_decoder_",),
    "decoders": ("decoder_", "zonly_decoder_"),
    "latent_decoders": ("latent_decoder_",),
}


def apply_trainable_modules(
    model: nn.Module,
    *,
    train_modules: str = "all",
    freeze_encoder: bool = False,
) -> tuple[int, int, int]:
    """
    Set requires_grad on model parameters.

    Returns (n_trainable_params, n_frozen_params, n_trainable_tensors).
    """
    if train_modules not in TRAIN_MODULE_PREFIXES:
        raise ValueError(
            f"unknown train_modules={train_modules!r}; "
            f"choices: {', '.join(TRAIN_MODULE_PREFIXES)}",
        )
    prefixes = TRAIN_MODULE_PREFIXES[train_modules]
    n_train = n_frozen = n_tensors = 0
    for name, param in model.named_parameters():
        trainable = prefixes is None or any(name.startswith(p) for p in prefixes)
        if freeze_encoder and name.startswith("encoder."):
            trainable = False
        param.requires_grad = trainable
        n = param.numel()
        if trainable:
            n_train += n
            n_tensors += 1
        else:
            n_frozen += n
    return n_train, n_frozen, n_tensors


def reinit_zonly_decoders(model: nn.Module) -> list[str]:
    """Re-randomize Z-only readout heads (after loading a frozen-encoder checkpoint)."""
    touched: list[str] = []
    for attr in ("zonly_decoder_fine", "zonly_decoder_mid", "zonly_decoder_coarse"):
        dec = getattr(model, attr, None)
        if dec is None:
            continue
        dec.apply(_reset_parameters)
        touched.append(attr)
    return touched


def _reset_parameters(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
