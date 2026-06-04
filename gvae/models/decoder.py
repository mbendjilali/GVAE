# gvae/models/decoder.py
# h+Z deformable readout (SceneGraphDecoder) and Z-only point readout (ZOnlyDecoder).
# GT p,r in SceneGraphDecoder only when DECODER_GT_ANCHOR_MIX > 0 (probe ablations).

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def bound_position(logits: torch.Tensor) -> torch.Tensor:
    """Map position logits to normalized [-1, 1]³ (clamp avoids tanh edge saturation)."""
    if config.POSITION_BOUND == "tanh":
        return torch.tanh(logits)
    return logits.clamp(-1.0, 1.0)


def position_from_logits(
    logits: torch.Tensor,
    p_query: torch.Tensor,
) -> torch.Tensor:
    """Absolute position or residual refinement relative to query (anchor / GT slot)."""
    if config.POSITION_RESIDUAL:
        return (p_query + logits).clamp(-1.0, 1.0)
    return bound_position(logits)


def _anchor_mlp(d: int, out_dim: int) -> nn.Module:
    """Anchor predictor: Linear (legacy) or 2-layer MLP when USE_ANCHOR_MLP."""
    if not config.USE_ANCHOR_MLP:
        return nn.Linear(d, out_dim)
    hidden = d if config.ANCHOR_MLP_HIDDEN <= 0 else config.ANCHOR_MLP_HIDDEN
    return nn.Sequential(
        nn.Linear(d, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


def deformable_predicts_position() -> bool:
    """Whether SceneGraphDecoder outputs p (vs anchor-only when Z-only patches position)."""
    return (not config.USE_Z_ONLY_DECODER) or (not config.Z_ONLY_PATCH_DEFORMABLE_POSITION)


def _z_pred_trunk(d: int) -> nn.Module:
    """Shared 2-layer MLP on cross-attn z_pred before mlp_s / mlp_r / mlp_p."""
    if not config.USE_Z_PRED_READOUT_MLP:
        return nn.Identity()
    hidden = d if config.Z_PRED_READOUT_MLP_HIDDEN <= 0 else config.Z_PRED_READOUT_MLP_HIDDEN
    return nn.Sequential(
        nn.Linear(d, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, d),
        nn.ReLU(inplace=True),
    )


def make_ref_grid(p, r):
    offsets_1d = torch.linspace(-1, 1, 3, device=p.device)
    gx, gy, gz = torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij')
    grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    ref_pts = p[:, None, :] + r[:, None, :] * grid[None, :, :]
    return ref_pts


def sample_volume(Z, ref_pts):
    """Z: (C, H, W, D). grid_sample layout matches legacy (C, D, W, H) ordering."""
    C, H, W, D = Z.shape
    Z_in = Z.permute(0, 3, 2, 1).unsqueeze(0)
    grid = ref_pts.unsqueeze(0).unsqueeze(3)
    out = F.grid_sample(Z_in, grid, mode='bilinear',
                        align_corners=True, padding_mode='border')
    return out.squeeze(0).squeeze(-1).permute(1, 2, 0)


class SceneGraphDecoder(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        P = config.NUM_REF_POINTS

        self.mlp_p_anchor = _anchor_mlp(d, 3)
        self.mlp_r_anchor = _anchor_mlp(d, 3)
        self.mlp_offset = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, P * 3),
        )
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        self.z_pred_trunk = _z_pred_trunk(d)
        self.mlp_s = nn.Linear(d, config.NUM_CLASSES)
        self.mlp_r = nn.Linear(d, 3)
        if deformable_predicts_position():
            self.mlp_p = nn.Linear(d, 3)
        self.softplus = nn.Softplus()

    def _reference_geometry(self, h, p_gt=None, r_gt=None):
        p_anchor = bound_position(self.mlp_p_anchor(h))
        r_anchor = self.softplus(self.mlp_r_anchor(h))
        mix = config.DECODER_GT_ANCHOR_MIX
        if mix > 0.0 and p_gt is not None and r_gt is not None:
            p_ref = (1.0 - mix) * p_anchor + mix * p_gt
            r_ref = (1.0 - mix) * r_anchor + mix * r_gt
            return p_ref, r_ref
        return p_anchor, r_anchor

    def forward(self, h, Z, p_gt=None, r_gt=None):
        p_ref, r_ref = self._reference_geometry(h, p_gt=p_gt, r_gt=r_gt)
        ref_pts = make_ref_grid(p_ref, r_ref)
        delta = self.mlp_offset(h).reshape(-1, config.NUM_REF_POINTS, 3)
        ref_pts = (ref_pts + delta).clamp(-1, 1)

        Z_sampled = sample_volume(Z, ref_pts)

        Q = self.W_Q(h).unsqueeze(1)
        K = self.W_K(Z_sampled)
        V = self.W_V(Z_sampled)

        scale = self.d ** 0.5
        attn = torch.softmax((Q @ K.transpose(1, 2)) / scale, dim=2)
        z_pred = (attn @ V).squeeze(1)
        feat = self.z_pred_trunk(z_pred)

        if deformable_predicts_position():
            p_hat = position_from_logits(self.mlp_p(feat), p_ref)
        else:
            # GVAE.forward patches p from ZOnlyDecoder @ p_anchor when Z_ONLY_PATCH_DEFORMABLE_POSITION.
            p_hat = p_ref

        return {
            's': torch.softmax(self.mlp_s(feat), dim=1),
            'p': p_hat,
            'r': self.softplus(self.mlp_r(feat)),
        }

    def predict_anchors(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h-predicted reference box (used for anchor loss and hz readout)."""
        return bound_position(self.mlp_p_anchor(h)), self.softplus(self.mlp_r_anchor(h))


def zonly_query_points(p_gt: torch.Tensor, *, training: bool) -> torch.Tensor:
    """Sample locations for Z-only readout; jitter during training for localisation pressure."""
    if not training or config.Z_ONLY_QUERY_JITTER <= 0:
        return p_gt
    noise = torch.randn_like(p_gt) * config.Z_ONLY_QUERY_JITTER
    return (p_gt + noise).clamp(-1.0, 1.0)


class ZOnlyDecoder(nn.Module):
    """Read s, p, r from bilinear Z samples at query points — no h or cross-attention."""

    def __init__(self, d: int):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.mlp_s = nn.Linear(d, config.NUM_CLASSES)
        self.mlp_p = nn.Linear(d, 3)
        self.mlp_r = nn.Linear(d, 3)
        self.softplus = nn.Softplus()

    def forward(self, Z: torch.Tensor, p_query: torch.Tensor) -> dict[str, torch.Tensor]:
        z = sample_volume(Z, p_query.unsqueeze(1)).reshape(p_query.shape[0], -1)
        z = self.readout(z)
        return {
            's': torch.softmax(self.mlp_s(z), dim=1),
            'p': position_from_logits(self.mlp_p(z), p_query),
            'r': self.softplus(self.mlp_r(z)),
        }

    def forward_at_gt(
        self,
        Z: torch.Tensor,
        p_gt: torch.Tensor,
        *,
        training: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if training is None:
            training = self.training
        return self.forward(Z, zonly_query_points(p_gt, training=training))
