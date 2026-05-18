# gvae/models/encoder.py
# Per-level encoder: R-GAT + PointROPE → Splat → UNet3D → variational bottleneck


import torch
import torch.nn as nn

import config
from gvae.models.gps import GPSLayer
from gvae.models.coarsening import FPSCoarsening
from gvae.models.splatting import GaussianSplatting
from gvae.models.unet3d import UNet3D

class SceneGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = config.D_MODEL

        # ------------------------------------------------------------
        self.input_proj_L = nn.Linear( # Input projection for level L
            config.NUM_CLASSES + 3 + 3, # semantic one-hot + position + radius = 21
            d
        )
        self.gps_L = nn.ModuleList([ # GPS layer for level L
            GPSLayer(d, num_heads = 8, use_global_attention = False),
            GPSLayer(d, num_heads = 8, use_global_attention = False),
        ])
        self.coarsen_1 = FPSCoarsening() # First coarsening (L → L-1)
        # ------------------------------------------------------------
        self.input_proj_L1 = nn.Linear(d + config.NUM_CLASSES + 3, d) # Input projection for level L-1 (after coarsening)
        self.gps_L1 = nn.ModuleList([ # GPS layer for level L-1 
            GPSLayer(d, num_heads = 8, use_global_attention = False),
            GPSLayer(d, num_heads = 8, use_global_attention = False),
        ])
        self.coarsen_2 = FPSCoarsening() # Second coarsening (L-1 → 1)
        # ------------------------------------------------------------
        self.input_proj_1 = nn.Linear(d + config.NUM_CLASSES + 3, d) # Input projection for level 1 (same shape as L-1)
        self.gps_1 = nn.ModuleList([ # GPS layer for level 1
            GPSLayer(d, num_heads = 8, use_global_attention = True), # global attention at the coarsest level
            GPSLayer(d, num_heads = 8, use_global_attention = True),
        ])
        # ------------------------------------------------------------
        # Gaussian splatting layer to convert node features to voxel grid features
        self.splat_mid = GaussianSplatting(config.GRID_MID) # for mid-level features
        self.splat_coarse = GaussianSplatting(config.GRID_COARSE) # for coarse-level features

        # 3D U-Net for spatial densification on the voxel grid
        self.unet_mid = UNet3D(d, depth=3) # deeper U-Net for mid-level features: sparser splat needs more inpainting
        self.unet_coarse = UNet3D(d, depth=2) # shallower U-Net for coarse-level features: already denser splat

    def forward(self, graph):
        """
        graph: input scene graph with node attributes:
            p: (N, 3) node positions
            r: (N, 3) node radii
            s: (N, config.NUM_CLASSES) node semantic one-hot vectors
            edge_index: (2, E) edge indices

        Returns:
        z_coarse: (config.D_MODEL * 2,) latent code for the coarsest level (mean and logvar concatenated)
        z_mid: (config.D_MODEL * 2,) latent code for the mid level (mean and logvar concatenated)
        """
        # Build edge attributes
        p, r, s, edge_index = graph.p, graph.r, graph.s, graph.edge_index
        edge_attr = torch.norm(
            p[edge_index[0]] - p[edge_index[1]], dim=1, keepdim=True) # (E, 1) edge attributes = distances between connected nodes
        
        # Level L: full graph, project raw feature (semantic + log(size) + position)
        x_L = torch.cat([s, torch.log(r + 1e-6), p], dim=1) # (N, NUM_CLASSES + 3 + 3) concatenate semantic, size, and position
        h = self.input_proj_L(x_L) # (N, d) initial node features for level L

        for gps in self.gps_L:
            h = gps(h, edge_index, edge_attr, p) # (N, d) updated node features after GPS layers at level L
        h_L = h # save level L features for splatting later

        # Coarsening L → L-1
        c1 = self.coarsen_1(p, r, s, h_L) # (M1, 3), (M1, 3), (M1, NUM_CLASSES), (M1, d) coarsened positions, radii, semantics, and features
        # c1 is a dict with: p, r, s edge_index, S, h_pooled

        # Level L-1: project coarsened features, concatenate with coarsened semantic and geometric info
        p1, r1, s1, edge_index_1 = c1['p'], c1['r'], c1['s'], c1['edge_index']
        edge_attr_1 = torch.norm(
            p1[edge_index_1[0]] - p1[edge_index_1[1]], dim=1, keepdim=True) # (E1, 1) edge attributes for coarsened graph
        
        x_Lm1 = torch.cat([c1['h_pooled'], s1, torch.log(r1 + 1e-6)], dim=1) # (M1, d + NUM_CLASSES + 3) concatenate coarsened features with semantic and geometric info
        h = self.input_proj_L1(x_Lm1) # (M1, d) initial node features for level L-1

        for gps in self.gps_L1:
            h = gps(h, edge_index_1, edge_attr_1, p1) # (M1, d) updated node features after GPS layers at level L-1
        h_Lm1 = h # save level L-1 features for splatting later

        # Splat level L-1 features to voxel grid
        F_mid = self.splat_mid(h_Lm1, p1, r1) # (H_mid*W_mid*D_mid, d) splatted features on mid-level voxel grid
        z_mid, mu_mid, logvar_mid, = self.unet_mid(F_mid) 

        # Coarsening L-1 → 1
        c2 = self.coarsen_2(p1, r1, s1, h_Lm1) # (M2, 3), (M2, 3),
        # Level 1: project coarsened features, concatenate with coarsened semantic and geometric info
        p2, r2, s2, edge_index_2 = c2['p'], c2['r'], c2['s'], c2['edge_index']
        edge_attr_2 = torch.norm(
            p2[edge_index_2[0]] - p2[edge_index_2[1]], dim=1, keepdim=True) # (E2, 1) edge attributes for coarsened graph
        x_1 = torch.cat([c2['h_pooled'], s2, torch.log(r2 + 1e-6)], dim=1) # (M2, d + NUM_CLASSES + 3) concatenate coarsened features with semantic and geometric info
        h = self.input_proj_1(x_1) # (M2, d)
        for gps in self.gps_1:
            h = gps(h, edge_index_2, edge_attr_2, p2) # (M2, d) updated node features after GPS layers at level 1
        h_1 = h # save level 1 features for splatting later

        # Splat level 1 features to voxel grid
        F_coarse = self.splat_coarse(h_1, p2, r2) # (H_coarse*W_coarse*D_coarse, d) splatted features
        z_coarse, mu_coarse, logvar_coarse = self.unet_coarse(F_coarse)

        return {
            'z_mid': z_mid, 
            'mu_mid': mu_mid, 
            'logvar_mid': logvar_mid,
            'z_coarse': z_coarse,
            'mu_coarse': mu_coarse, 
            'logvar_coarse': logvar_coarse,
            # Return intermediate node data needed for the decoder
            # mid lvel nodes
            "h_lm1": h_Lm1, 
            "p_lm1": p1, 
            "r_lm1": r1, 
            "s_lm1": s1,
            # coarse level nodes
            "h_1": h_1,
            "p_1": p2,
            "r_1": r2,
            "s_1": s2,
            # soft assignment matrices for pool losses
            'S1': c1['S'],
            'S2': c2['S'],
            # coarsened edge indices (needed for reconstruction loss)
            'edge_index_lm1': edge_index_1,
            'edge_index_1':   edge_index_2,
        }