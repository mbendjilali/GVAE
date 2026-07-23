# gvae/checkpoint_compat.py
# Load older checkpoints into models with anchor MLP / Z-only layout changes.

from __future__ import annotations

import torch

import config


def _inject_z_pred_trunk(state: dict) -> dict:
    """Add identity-initialized z_pred_trunk if checkpoint predates readout MLP."""
    if not config.USE_Z_PRED_READOUT_MLP:
        return state
    out = dict(state)
    for level in ("fine", "mid", "coarse"):
        prefix = f"decoder_{level}.z_pred_trunk"
        if any(k.startswith(f"{prefix}.") for k in state):
            continue
        w_key = f"decoder_{level}.mlp_s.weight"
        if w_key not in state:
            continue
        d = int(state[w_key].shape[1])
        for idx in (0, 2):
            out[f"{prefix}.{idx}.weight"] = torch.eye(d)
            out[f"{prefix}.{idx}.bias"] = torch.zeros(d)
    return out


def _drop_occ_grid_heads(state: dict) -> dict:
    """Remove OccGridHead weights from checkpoints predating its removal."""
    return {
        k: v for k, v in state.items()
        if not k.startswith(
            ("occ_grid_head_fine.", "occ_grid_head_mid.", "occ_grid_head_coarse."),
        )
    }


def infer_latent_levels_from_state_dict(state: dict) -> tuple[int, int, int]:
    """Read Z channel widths from U-Net final_conv (out = 2×d, in = d)."""
    def _z_dim(level: str) -> int:
        key = f"encoder.unet_{level}.final_conv.weight"
        if key not in state:
            raise KeyError(f"Cannot infer latent dim: missing {key}")
        return int(state[key].shape[1])

    return _z_dim("fine"), _z_dim("mid"), _z_dim("coarse")


def configure_dims_from_checkpoint(state: dict) -> tuple[int, int, int]:
    """Set config.D_LATENT_LEVELS to match checkpoint before constructing GVAE."""
    levels = infer_latent_levels_from_state_dict(state)
    config.apply_latent_levels(*levels)
    return levels


def migrate_state_dict(state: dict) -> dict:
    """Map legacy Linear anchor heads into 2-layer MLP final layer (index 2)."""
    out = _drop_occ_grid_heads(state)
    for key in list(state.keys()):
        for stem in ("mlp_p_anchor", "mlp_r_anchor"):
            old_w = f".{stem}.weight"
            if key.endswith(old_w) and f".{stem}.0." not in key:
                prefix = key[: -len(".weight")]
                out[f"{prefix}.2.weight"] = state[key]
                bias_key = f"{prefix}.bias"
                if bias_key in state:
                    out[f"{prefix}.2.bias"] = state[bias_key]
                del out[key]
                if bias_key in out:
                    del out[bias_key]
    return _inject_z_pred_trunk(out)


def prepare_checkpoint(
    state: dict,
    *,
    latent_levels: tuple[int, int, int] | None = None,
) -> dict:
    """Set config.D_LATENT_LEVELS (infer or override), then migrate legacy keys."""
    if latent_levels is not None:
        config.apply_latent_levels(*latent_levels)
    else:
        configure_dims_from_checkpoint(state)
    return migrate_state_dict(state)
