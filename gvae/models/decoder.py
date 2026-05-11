# gvae/models/decoder.py
# Deformable cross-attention readout + MLP heads (s, p, r) + edge losses
# TODO: implement (todo id: decoder)

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# function to build the reference points
def make_ref_grid(p, r):
    # create 3 values along each axis: -1, 0, +1 in local object space
    offsets_1d = torch.linspace(-1, 1, 3, device=p.device) # (3,)
    # build the 3d grid of offsets: shape (27, 3)
    gx, gy, gz = torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij') 
    grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1) # (27, 3)
    # scale offsets by the node's size (half-extents) and shift to the node's center position
    ref_pts = p[:, None, :] + r[:, None, :] * grid[None, :, :] # (N, 27, 3)
    return ref_pts

# Interpolation of Z at the reference points
# Z (H, W, D) : 3D feature grid after splatting, 
# # ref_pts : reference points (N, 27, 3)
# returns (N, 27, d) sampled features
def sample_volume(Z, ref_pts):
    """
    Z (H,W,D,d)          ref_pts (N,27,3)
     ↓ permute              ↓ unsqueeze
(1, d, D, W, H)      (1, N, 27, 3)
          ↓                ↓
          └── F.grid_sample ──┘
                   ↓
           (1, d, N, 27)
                   ↓ squeeze + permute
               (N, 27, d)
    """
    H, W, D, d = Z.shape
    N = ref_pts.shape[0]
    # Z is (H, W, D, d), we need to permute it to (d, D, W, H) for grid_sample
    Z_in = Z.permute(3, 2, 1, 0).unsqueeze(0) # (1, d, D, W, H) 
    grid = ref_pts.unsqueeze(0) # (1, N, 27, 3)
    out = F.grid_sample(Z_in, grid, mode='bilinear', 
                        align_corners=True, padding_mode='border') # (1, d, N, 27)
    return out.squeeze(0).permute(1, 2, 0) # (N, 27, d)

class SceneGraphDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = config.D.MODEL
        P = 27 # number of reference points

        # MLP for offsets to deform the reference points
        self.mlp_offset = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, P * 3) # output offset for each reference point
        )

        # G, K, V projections for cross-attention
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)

        # MLP heads for reconstruciton of s, p, r
        self.mlp_s = nn.Linear(d, config.NUM_CLASSES) # output class probabilities
        self.mlp_p = nn.Linear(d, 3) # output position (x, y, z)
        self.mlp_r = nn.Linear(d, 3) # output size (apply Softplus)
        self.softplus = nn.Softplus() # to ensure positive outputs for r

    def forward(self, h, p, r, Z):
        """
        h: (N, d) node features from the encoder
        p: (N, 3) node positions from the encoder
        r: (N, 3) node sizes from the encoder
        Z: (H, W, D, d) voxel features after splatting

        Returns:
        s: (N, NUM_CLASSES) predicted class probabilities
        p: (N, 3) predicted positions
        r: (N, 3) predicted sizes
        """
        
        # 1. Build reference points + apply learned offsets to deform them
        ref_pts = make_ref_grid(p, r) # (N, 27, 3)
        delta = self.mlp_offset(h).reshape(-1, 27, 3) # (N, 27, 3) learned offsets to deform the reference points)
        ref_pts = ref_pts + delta # (N, 27, 3) deformed reference points
        ref_pts = ref_pts.clamp(-1, 1) # ensure the deformed reference points are still within the latent space bounds of [-1, 1]

        # 2. Sample features from Z at the deformed reference points using trilinear interpolation
        Z_sampled = sample_volume(Z, ref_pts) # (N, 27, d) 

        # 3. Cross-attention: use h as query, Z_sampled as key and value
        Q = self.W_Q(h).unsqueeze(1) # (N, 1, d) one query per node
        K = self.W_K(Z_sampled) # (N, 27, d)
        V = self.W_V(Z_sampled) # (N, 27, d)

        scale = config.D_MODEL ** 0.5
        attn = torch.softmax((Q @ K.transpose(1, 2)) / scale, dim=2) # (N, 1, 27) attention weights
        z_pred = (attn @ V).squeeze(1) # (N, d) one feature vector for each node

        # 4. MLP heads to predict s, p, r
        s_pred = torch.softmax(self.mlp_s(z_pred), dim=1) # (N, NUM_CLASSES) class probabilities
        p_pred = self.mlp_p(z_pred) # (N, 3) predicted positions
        r_pred = self.softplus(self.mlp_r(z_pred)) # (N, 3) predicted sizes, apply Softplus to ensure positivity

        return {
            's': s_pred,
            'p': p_pred,
            'r': r_pred
        }