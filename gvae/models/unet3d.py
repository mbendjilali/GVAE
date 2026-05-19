# gvae/models/unet3d.py
# Configurable-depth 3D U-Net with skip connections for spatial densification
# TODO: implement (todo id: unet3d)

import torch
import torch.nn as nn

import config

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    
class UNet3D(nn.Module):
    def __init__(self, d: int, depth: int):
        super().__init__()
        self.depth = depth # number of encoder/decoder levels (2 for coarse, 3 for mid)
        
        # Encoder stages
        self.encoders = nn.ModuleList()
        ch = d
        for i in range(depth):
            self.encoders.append(ConvBlock(ch, ch*2))
            ch = ch * 2
        
        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch)

        # Decoder stages
        self.decoders = nn.ModuleList()
        for i in range(depth):
            self.decoders.append(ConvBlock(ch * 2, ch//2)) # input channels are doubled due to skip connections
            ch = ch // 2

        
        # variational bottleneck (final output)
        # ch is back to d after the decoder stages
        self.final_conv = nn.Conv3d(ch, d * 2, kernel_size=1) # output mean and logvar for each feature dimension

    def forward(self, x):
        # x: (H, W, D, d) input features on the voxel grid (after splatting)
        # we need to permute it to (1, d, H, W, D) for Conv3d
        x = x.permute(3, 0, 1, 2).unsqueeze(0) # (1, d, H, W, D)

        # Encoder path
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x) # save for skip connection
            x = nn.functional.max_pool3d(x, kernel_size=2) # downsample by 2 in each spatial dimension H, W, D
        
        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        # upsample + concat skip + convblock
        for decoder in self.decoders:
            x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False) # upsample by 2
            skip = skips.pop() # get corresponding skip connection
            x = torch.cat([x, skip], dim=1) # concatenate along channel dimension
            x = decoder(x)
        
        # Final conv to get mean and logvar
        out = self.final_conv(x) # (1, d*2, H, W, D)
        mu, logvar = out.chunk(2, dim=1) # split into mean and logvar each of shape (1, d, H, W, D)

        std = torch.exp(0.5 * logvar) # standard deviation
        eps = torch.randn_like(std) # random noise
        z = mu + std * eps # reparameterisation trick to sample from N(mu, std^2)

        # permute back to (H, W, D, d)
        z = z.squeeze(0).permute(1, 2, 3, 0) # (H, W, D, d)
        mu = mu.squeeze(0).permute(1, 2, 3, 0) # (H, W, D, d)
        logvar = logvar.squeeze(0).permute(1, 2, 3, 0) # (H, W, D, d)

        return z, mu, logvar
    


    
