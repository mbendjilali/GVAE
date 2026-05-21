# gvae/models/unet3d.py
# Configurable-depth 3D U-Net with skip connections for spatial densification
# Latent layout: (C, H, W, D) — Conv3d-native, no permute at boundaries.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import config


def _num_groups(channels: int) -> int:
    g = config.UNET_NUM_GROUPS
    while g > 1 and channels % g != 0:
        g //= 2
    return max(1, g)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, d: int, depth: int):
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        ch = d
        for _ in range(depth):
            self.encoders.append(ConvBlock(ch, ch * 2))
            ch = ch * 2

        self.bottleneck = ConvBlock(ch, ch)

        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.decoders.append(ConvBlock(ch * 2, ch // 2))
            ch = ch // 2

        self.final_conv = nn.Conv3d(ch, d * 2, kernel_size=1)

    def _use_checkpoint(self, x) -> bool:
        if not config.UNET_USE_CHECKPOINT:
            return False
        return x.shape[-1] >= config.UNET_CHECKPOINT_MIN_SIDE

    def _run_block(self, block, x):
        if self._use_checkpoint(x) and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def _to_conv_input(self, x: torch.Tensor) -> torch.Tensor:
        """(C, H, W, D) → (1, C, H, W, D), optionally channels_last_3d."""
        x = x.unsqueeze(0)
        if config.UNET_CHANNELS_LAST:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return x

    def forward(self, x):
        # x: (C, H, W, D)
        x = self._to_conv_input(x)

        skips = []
        for encoder in self.encoders:
            x = self._run_block(encoder, x)
            skips.append(x)
            x = nn.functional.max_pool3d(x, kernel_size=2)

        x = self._run_block(self.bottleneck, x)

        for decoder in self.decoders:
            x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = self._run_block(decoder, x)

        out = self.final_conv(x)
        mu, logvar = out.chunk(2, dim=1)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # (1, C, H, W, D) → (C, H, W, D)
        return z.squeeze(0), mu.squeeze(0), logvar.squeeze(0)
