# gvae/models/coarsening.py
# FPS + ball-query (hard) coarsening + soft-S MLP + supernode attribute derivation

import torch
import torch.nn as nn
from torch_cluster import fps
from torch_geometric.nn import radius_graph

import config


def _empty_coarsening(device, feature_dim: int):
    z3 = torch.zeros(0, 3, device=device)
    zc = torch.zeros(0, config.NUM_CLASSES, device=device)
    zh = torch.zeros(0, feature_dim, device=device)
    return {
        'p': z3,
        'r': z3,
        's': zc,
        'edge_index': torch.zeros(2, 0, dtype=torch.long, device=device),
        'S': torch.zeros(0, 0, device=device),
        'hard_assign': torch.zeros(0, dtype=torch.long, device=device),
        'h_pooled': zh,
    }


class FPSCoarsening(nn.Module):
    def __init__(self, coarsening_level: int = 0):
        super().__init__()
        self.ratio = config.REDUCTION_RATIO
        self.ball_query_radius = config.BALL_QUERY_RADIUS_LEVELS[coarsening_level]

        self.mlp_S = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, p, r, s, h, coarsen_mask: torch.Tensor | None = None):
        """
        p, r, s, h: (N, …) all graph nodes at this level.

        coarsen_mask: (N,) bool — if given, only True rows are FPS seeds / pool members
        (B policy: background stays at instance G_L, objects form supernodes).
        """
        device = p.device
        N = p.shape[0]
        if coarsen_mask is None:
            coarsen_mask = torch.ones(N, dtype=torch.bool, device=device)
        else:
            coarsen_mask = coarsen_mask.to(device)

        idx = coarsen_mask.nonzero(as_tuple=True)[0]
        n_c = idx.numel()
        if n_c == 0:
            return _empty_coarsening(device, h.shape[1])

        p_c = p[idx]
        r_c = r[idx]
        s_c = s[idx]
        h_c = h[idx]

        batch = torch.zeros(n_c, dtype=torch.long, device=device)
        seed_local = fps(p_c, batch=batch, ratio=self.ratio)
        seed_indices = idx[seed_local]

        p_seeds = p[seed_indices]
        dist = torch.cdist(p_c, p_seeds)
        hard_assign = dist.argmin(dim=1)

        M = seed_indices.shape[0]
        r_seeds = r[seed_indices]
        dist_normalised = dist / (r_seeds.mean(dim=1).unsqueeze(0) + 1e-6)

        scores = self.mlp_S(dist_normalised.reshape(-1, 1)).reshape(n_c, M)
        S = torch.softmax(scores, dim=1)
        # Stop recon/KL/occ gradients through soft-S; pool loss still trains mlp_S.
        S_feat = S.detach()

        S_col_sum = S_feat.sum(dim=0)
        p_super = (S_feat.T @ p_c) / (S_col_sum.unsqueeze(1) + 1e-6)

        r_super = torch.zeros(M, 3, device=device)
        for j in range(M):
            members = hard_assign == j
            if members.sum() > 0:
                p_members = p_c[members]
                r_super[j] = (p_members.max(dim=0).values - p_members.min(dim=0).values) / 2

        s_super = (S_feat.T @ s_c) / (S_col_sum.unsqueeze(1) + 1e-6)

        edge_index_super = radius_graph(
            p_super,
            r=self.ball_query_radius,
            loop=False,
            max_num_neighbors=config.MAX_NUM_NEIGHBORS,
        )

        return {
            'p': p_super,
            'r': r_super,
            's': s_super,
            'edge_index': edge_index_super,
            'S': S,
            'hard_assign': hard_assign,
            'h_pooled': S_feat.T @ h_c / (S_col_sum.unsqueeze(1) + 1e-6),
        }
