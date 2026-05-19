# gvae/models/splatting.py
# Truncated anisotropic Gaussian splatting with scatter_add; O(Σ|N_i|) complexity
# TODO: implement (todo id: splatting)

import torch
import torch.nn as nn
from torch_scatter import scatter_add
import config

# Return the 3D position and center of each voxel in the grid
def make_voxel_centers(grid, device):
    # Create a grid of voxel centers
    H, W, D = grid 

    # create a 1D list of center positions along each axis
    x_centers = torch.linspace(-1, 1, H, device=device)
    y_centers = torch.linspace(-1, 1, W, device=device)
    z_centers = torch.linspace(-1, 1, D, device=device)

    # Build the grid 
    gx, gy, gz = torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij') # generate 3D grid of shape (H, W, D)

    # flatten and stack them as column
    centers = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1) # (H*W*D, 3)

    return centers

class GaussianSplatting(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid  # (H, W, D) tuple defining the voxel grid dimensions
        self.sigma = config.SPLAT_TRUNCATION_SIGMA

    def forward(self, h, p, r):
        """
        h: (N, d) node features
        p: (N, 3) node positions
        r: (N, 3) node radii

        Returns:
        voxel_features: (H*W*D, d) features for each voxel after splatting
        """
        device = h.device
        N, d = h.shape
        H, W, D = self.grid

        # 1. Get voxel centers
        vox_centers = make_voxel_centers(self.grid, device)  # (H*W*D, 3)

        # 2. Find which voxels are within truncation distance of each node
        # we need to know for each object i and each voxel v, whether the voxel is within +- 2r_i of the object's center p_i along each axis.
        # difference vector between each node and each voxel center
        diff = vox_centers[None] - p[:, None, :] 
        # boolean mask : True if voxel is within truncation distance of node i along all axes
        within_trunc = (diff.abs() <= (self.sigma * r)[:, None, :]).all(dim=2) # (N, H*W*D) .all(dim=2) checks if all 3 axes satisfy the condition
        
        # 3. Extract valid (object, voxel) pairs
        node_idx, voxel_idx = within_trunc.nonzero(as_tuple=True)

        # 4. compute Gaussian weights for valid pairs
        # select the valid pairs
        valid_diff = diff[node_idx, voxel_idx]  # (num_valid, 3) difference vectors for valid pairs
        valid_r = r[node_idx]  # (num_valid, 3) radius for valid nodes

        # normalise the offset by the object size to get a scale-invariant distance
        diff_normalised = valid_diff / (valid_r + 1e-6)  # (num_valid, 3)
        w = torch.exp(-0.5 * (diff_normalised ** 2).sum(dim=1))  # (num_valid,) one weights per valid pair

        # 5. Scatter onto the voxel grid
        # now we accumulate : each valid pair contributes h[node_idx] * w to voxel_idx
        V = vox_centers.shape[0]

        # compute weighted features for valid pairs
        weighted_features = w.unsqueeze(1) * h[node_idx]  # (num_valid, d)
        # sum weighted features for each voxel using scatter_add
        voxel_features = scatter_add(weighted_features, voxel_idx, dim=0, dim_size=V)  # (H*W*D, d)
        # sum weights into each voxel for normalisation
        weight_sum = scatter_add(w, voxel_idx, dim=0, dim_size=V)
        # normalise weighted average
        voxel_features = voxel_features / (weight_sum.unsqueeze(1) + config.SPLAT_EPS)

        # 6. Reshape the 3D grid and return
        H, W, D = self.grid
        return voxel_features.reshape(H, W, D, config.D_MODEL)