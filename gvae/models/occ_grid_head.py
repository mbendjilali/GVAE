# gvae/models/occ_grid_head.py
# Voxel-aligned occupancy logits from latent Z (1×1 Conv3d on (C,H,W,D))

import torch
import torch.nn as nn

import config


class OccGridHead(nn.Module):
    """Predict per-voxel occupancy logits directly from Z."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.conv = nn.Conv3d(d, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (C, H, W, D) → logits (H, W, D)."""
        assert z.shape[0] == self.d, f"Z has {z.shape[0]} channels but head expects {self.d}"
        x = z.unsqueeze(0)
        if config.UNET_CHANNELS_LAST:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return self.conv(x).squeeze(0).squeeze(0)
