# gvae/data/occupancy.py
# 3D scene voxelisation for occupancy ground truth


import torch
import torch.nn as nn
import torch.nn.functional as F
import config

def compute_occupancy_gt(p, r, n_queries = config.OCC_QUERY_POINTS):
    # sample query points
    n_pos = n_queries // 2
    N = p.shape[0]

    # pick a random object index for each positive sample
    obj_idx = torch.randint(0, N, (n_pos,), device=p.device) # (n_pos,)
    # random offsets in [-1, 1]^3 scaled by the object's half-extents
    offsets = (torch.rand(n_pos, 3, device=p.device) * 2 - 1) # (n_pos, 3) in [-1, 1]
    q_pos = p[obj_idx] + r[obj_idx] * offsets # (n_pos, 3) inside the bounding box
    q_pos = q_pos.clamp(-1, 1) # ensure within scene bounds

    # sample points uniformly
    n_neg = n_queries - n_pos
    q_neg = torch.rand(n_neg, 3, device=p.device) * 2 - 1 # (n_neg, 3) in [-1, 1]^3

    # compute true labels for all query points
    # A point is occupied (label=1) if it is inside any object's bounding box
    q_all = torch.cat([q_pos, q_neg], dim=0) # (n_queries, 3)
    diff = (q_all[:, None] - p[None, :, :]).abs() # (n_queries, N, 3)
    inside = (diff <= r[None, :, :]).all(dim=2) # (n_queries, N) True if inside object i
    labels = inside.any(dim=1).float()

    return q_all, labels # (n_queries, 3), (n_queries,)

# Fourier feature encoding 
def fourier_encode(q, num_freqs = 6):
    # q: (n_queries, 3)
    # num_freqs: number of frequency bands
    freqs = 2.0 ** torch.arange(num_freqs, device=q.device) # (num_freqs,) = [1, 2, 4, 8, 16, 32]
    # q[:, :, None] → (n, 3, 1), freqs → (num_freqs,)
    x = q[:, :, None] * freqs[None, None, :] * torch.pi # (n, 3, num_freqs)
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=2) # (n, 3, 2*num_freqs)
    return enc.reshape(q.shape[0], -1) # (n, 3*2*num_freqs)

class OccupancyReadout(nn.Module):
    def __init__(self):
        super().__init__()
        d = config.D_MODEL
        fourier_dim = 3 * 2 * 6 # 3 coords, 2 (sin+cos), 6 frequencies = 36

        #project Fourier features up to d
        self.input_proj = nn.Linear(fourier_dim, d)

        # cross-attention projections (query point → reads from Z)
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)

        # # final MLP: d → 1 (occupancy probability)
        self.mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1)
        )

    def forward(self, q, Z):
        # encode query positions with Fourier features
        q_enc = fourier_encode(q) # (n_queries, fourier_dim)
        q_feat = self.input_proj(q_enc) # (n_queries, d)

        # flatten Zto a set of key-value pairs for attention
        H, W, D, d = Z.shape
        Z_flat = Z.reshape(-1, d)

        # cross-attention: each query point attends to all voxels
        Q = self.W_Q(q_feat) # (n_queries, d)
        K = self.W_K(Z_flat)
        V = self.W_V(Z_flat)

        scale = d ** 0.5
        attn_scores = torch.softmax(Q @ K.T / scale, dim=1) 
        z_q = attn_scores @ V

        # predict occupancy logit — sigmoid is applied inside the loss for numerical stability
        logit = self.mlp(z_q).squeeze(1)
        return logit  # raw logit, NOT sigmoid
    

