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


def migrate_state_dict(state: dict) -> dict:
    """Map legacy Linear anchor heads into 2-layer MLP final layer (index 2)."""
    out = dict(state)
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
