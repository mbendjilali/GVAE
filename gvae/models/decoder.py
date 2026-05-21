# gvae/models/decoder.py
# Deformable cross-attention readout + MLP heads (s, p, r)
# Reference queries are anchored on h-predicted (p, r), not GT geometry.

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def make_ref_grid(p, r):
    offsets_1d = torch.linspace(-1, 1, 3, device=p.device)
    gx, gy, gz = torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij')
    grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    ref_pts = p[:, None, :] + r[:, None, :] * grid[None, :, :]
    return ref_pts


def sample_volume(Z, ref_pts):
    H, W, D, d = Z.shape
    Z_in = Z.permute(3, 2, 1, 0).unsqueeze(0)
    grid = ref_pts.unsqueeze(0).unsqueeze(3)
    out = F.grid_sample(Z_in, grid, mode='bilinear',
                        align_corners=True, padding_mode='border')
    return out.squeeze(0).squeeze(-1).permute(1, 2, 0)


class SceneGraphDecoder(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        P = config.NUM_REF_POINTS

        self.mlp_p_anchor = nn.Linear(d, 3)
        self.mlp_r_anchor = nn.Linear(d, 3)
        self.mlp_offset = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, P * 3),
        )
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        self.mlp_s = nn.Linear(d, config.NUM_CLASSES)
        self.mlp_p = nn.Linear(d, 3)
        self.mlp_r = nn.Linear(d, 3)
        self.softplus = nn.Softplus()

    def _reference_geometry(self, h, p_gt=None, r_gt=None):
        p_anchor = torch.tanh(self.mlp_p_anchor(h))
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

        return {
            's': torch.softmax(self.mlp_s(z_pred), dim=1),
            'p': torch.tanh(self.mlp_p(z_pred)),
            'r': self.softplus(self.mlp_r(z_pred)),
        }
