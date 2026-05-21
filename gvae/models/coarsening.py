# gvae/models/coarsening.py
# FPS coarsening + configurable assignment (hard Voronoi | soft distance softmax)

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


def _supernode_radii(p_c: torch.Tensor, hard_assign: torch.Tensor, M: int) -> torch.Tensor:
    r_super = torch.zeros(M, 3, device=p_c.device)
    for j in range(M):
        members = hard_assign == j
        if members.any():
            p_members = p_c[members]
            r_super[j] = (p_members.max(dim=0).values - p_members.min(dim=0).values) / 2
    return r_super


def _build_assignment(
    dist: torch.Tensor,
    hard_assign: torch.Tensor,
    log_temperature: nn.Parameter | None,
) -> torch.Tensor:
    mode = config.COARSEN_ASSIGNMENT
    if mode == "hard":
        n_c, M = dist.shape
        S = torch.zeros(n_c, M, device=dist.device)
        S[torch.arange(n_c, device=dist.device), hard_assign] = 1.0
        return S
    if mode == "soft":
        if log_temperature is None:
            raise RuntimeError("soft coarsening requires log_temperature parameter")
        temperature = log_temperature.exp().clamp(min=1e-2, max=10.0)
        return torch.softmax(-dist / temperature, dim=1)
    raise ValueError(f"unknown COARSEN_ASSIGNMENT: {mode!r}")


class FPSCoarsening(nn.Module):
    """FPS seeds + hard or soft member assignment → supernode graph."""

    def __init__(self, coarsening_level: int = 0):
        super().__init__()
        self.coarsening_level = coarsening_level
        self.ratio = config.REDUCTION_RATIO_LEVELS[coarsening_level]
        self.ball_query_radius = config.BALL_QUERY_RADIUS_LEVELS[coarsening_level]
        self.log_temperature: nn.Parameter | None = None
        if config.COARSEN_ASSIGNMENT == "soft":
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(float(config.SOFTMAX_TEMPERATURE)))
            )

    def forward(self, p, r, s, h, coarsen_mask: torch.Tensor | None = None):
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

        S = _build_assignment(dist, hard_assign, self.log_temperature)
        S_feat = S.detach() if config.COARSEN_DETACH_FEATURES and config.COARSEN_ASSIGNMENT == "soft" else S
        col_sum = S_feat.sum(dim=0).clamp(min=1e-6)

        p_super = (S_feat.T @ p_c) / col_sum.unsqueeze(1)
        s_super = (S_feat.T @ s_c) / col_sum.unsqueeze(1)
        h_pooled = (S_feat.T @ h_c) / col_sum.unsqueeze(1)
        r_super = _supernode_radii(p_c, hard_assign, M)

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
            'h_pooled': h_pooled,
        }
