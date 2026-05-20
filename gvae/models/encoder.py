# gvae/models/encoder.py
# Per-level encoder: R-GAT + PointROPE → Splat → UNet3D → variational bottleneck


import torch
import torch.nn as nn

import config
from gvae.models.gps import GPSLayer
from gvae.models.coarsening import FPSCoarsening
from gvae.models.splatting import GaussianSplatting
from gvae.models.unet3d import UNet3D

# Raw node input: semantic + log(r) + p
_NODE_INPUT_DIM = config.NUM_CLASSES + 3 + 3


class SceneGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d_L, d_L1, d_1 = config.D_MODEL_LEVELS
        h_L, h_L1, h_1 = config.D_NUM_HEADS

        # Level L — instance graph
        self.input_proj_L = nn.Linear(_NODE_INPUT_DIM, d_L)
        self.gps_L = nn.ModuleList([
            GPSLayer(d_L, num_heads=h_L, use_global_attention=False),
            GPSLayer(d_L, num_heads=h_L, use_global_attention=False),
        ])
        self.coarsen_1 = FPSCoarsening(coarsening_level=0)

        # Level L-1 — region graph → mid latent
        self.input_proj_L1 = nn.Linear(d_L + config.NUM_CLASSES + 3, d_L1)
        self.gps_L1 = nn.ModuleList([
            GPSLayer(d_L1, num_heads=h_L1, use_global_attention=False),
            GPSLayer(d_L1, num_heads=h_L1, use_global_attention=False),
        ])
        self.coarsen_2 = FPSCoarsening(coarsening_level=1)

        # Level 1 — scene graph → coarse latent
        self.input_proj_1 = nn.Linear(d_L1 + config.NUM_CLASSES + 3, d_1)
        self.gps_1 = nn.ModuleList([
            GPSLayer(d_1, num_heads=h_1, use_global_attention=True),
            GPSLayer(d_1, num_heads=h_1, use_global_attention=True),
        ])

        self.splat_mid = GaussianSplatting(config.GRID_MID, feature_dim=d_L1)
        self.splat_coarse = GaussianSplatting(config.GRID_COARSE, feature_dim=d_1)
        self.unet_mid = UNet3D(d_L1, depth=3)
        self.unet_coarse = UNet3D(d_1, depth=2)

    def forward(self, graph):
        p, r, s, edge_index = graph.p, graph.r, graph.s, graph.edge_index
        edge_attr = torch.norm(
            p[edge_index[0]] - p[edge_index[1]], dim=1, keepdim=True
        )

        x_L = torch.cat([s, torch.log(r + 1e-6), p], dim=1)
        h = self.input_proj_L(x_L)
        for gps in self.gps_L:
            h = gps(h, edge_index, edge_attr, p)
        h_L = h

        coarsen_mask = (
            graph.coarsen_mask
            if config.COARSEN_EXCLUDE_NON_INSTANTIABLE
            else None
        )
        c1 = self.coarsen_1(p, r, s, h_L, coarsen_mask=coarsen_mask)
        p1, r1, s1, edge_index_1 = c1['p'], c1['r'], c1['s'], c1['edge_index']
        edge_attr_1 = torch.norm(
            p1[edge_index_1[0]] - p1[edge_index_1[1]], dim=1, keepdim=True
        )

        x_Lm1 = torch.cat([c1['h_pooled'], s1, torch.log(r1 + 1e-6)], dim=1)
        h = self.input_proj_L1(x_Lm1)
        for gps in self.gps_L1:
            h = gps(h, edge_index_1, edge_attr_1, p1)
        h_Lm1 = h

        F_mid = self.splat_mid(h_Lm1, p1, r1)
        z_mid, mu_mid, logvar_mid = self.unet_mid(F_mid)

        c2 = self.coarsen_2(p1, r1, s1, h_Lm1)
        p2, r2, s2, edge_index_2 = c2['p'], c2['r'], c2['s'], c2['edge_index']
        edge_attr_2 = torch.norm(
            p2[edge_index_2[0]] - p2[edge_index_2[1]], dim=1, keepdim=True
        )
        x_1 = torch.cat([c2['h_pooled'], s2, torch.log(r2 + 1e-6)], dim=1)
        h = self.input_proj_1(x_1)
        for gps in self.gps_1:
            h = gps(h, edge_index_2, edge_attr_2, p2)
        h_1 = h

        F_coarse = self.splat_coarse(h_1, p2, r2)
        z_coarse, mu_coarse, logvar_coarse = self.unet_coarse(F_coarse)

        return {
            'z_mid': z_mid,
            'mu_mid': mu_mid,
            'logvar_mid': logvar_mid,
            'z_coarse': z_coarse,
            'mu_coarse': mu_coarse,
            'logvar_coarse': logvar_coarse,
            "h_lm1": h_Lm1,
            "p_lm1": p1,
            "r_lm1": r1,
            "s_lm1": s1,
            "h_1": h_1,
            "p_1": p2,
            "r_1": r2,
            "s_1": s2,
            'S1': c1['S'],
            'S2': c2['S'],
            'edge_index_lm1': edge_index_1,
            'edge_index_1': edge_index_2,
        }
