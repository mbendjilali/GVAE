"""
scene_graph.py
i
This is the version of SceneGraph generated strictly from the plan file
(scene_graph_vae_4b057e4d.plan.md). 

Differences from your current scene_graph.py:
  - Node attributes follow the plan exactly: (p, r, s, label)
  - Only proximity edges (computed from positions)
  - Per-scene normalisation to [-1,1]³ (same as your version)
  - Voxelised occupancy ground truth at both resolutions (same as your version)
  - to_pyg() method to convert to PyG Data object for batching
  - Loading from JSON (same as your version)
"""

import json
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


class SceneGraph:
    """
    SceneGraph as described in the plan file.

    From the plan:
      Node attributes: (p, r, s, label)
        p    — 3D position (centroid)
        r    — radius / footprint (half-extents)
        s    — semantic one-hot vector
        label — integer class label

      Edges:
        proximity edges — physical proximity (computed from positions)

      Per-scene normalisation to [-1, 1]³
      Voxelised occupancy ground truth at both resolutions
    """

    @classmethod
    def from_json(cls, json_path: str) -> "SceneGraph":
        """
        Load a scene from a JSON file.

        Expected JSON format:
        {
          "instances": [
            {"id": 0, "position": [x,y,z], "radius": [rx,ry,rz], "label": "ground"},
            {"id": 1, "position": [x,y,z], "radius": [rx,ry,rz], "label": "car"},
            ...
          ]
        }

        Proximity edges are computed automatically from positions.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        instances = data["instances"]

        positions = [inst["position"] for inst in instances]
        radii     = [inst["radius"]   for inst in instances]
        labels    = [inst["label"]    for inst in instances]

        p = torch.tensor(positions, dtype=torch.float32)   # (N, 3)
        r = torch.tensor(radii,     dtype=torch.float32)   # (N, 3)

        return cls(p, r, labels)

    def __init__(
        self,
        position: Tensor,   # (N, 3) raw positions in metres
        radius: Tensor,     # (N, 3) half-extents in metres
        label: list,        # list of N strings
    ):
        N = position.shape[0]
        self.num_nodes = N

        #  Per-scene normalisation to [-1, 1]³ 
        # subtract scene centroid, then scale so the furthest point is at distance 1.
        # We scale r by the SAME factor as p so sizes stay consistent.
        centroid_scene = position.mean(dim=0)             # (3,) average position
        p_centred = position - centroid_scene             # centre at origin
        scale = p_centred.abs().max().clamp(min=1e-6)

        self.p: Tensor = p_centred / scale          # (N, 3) — plan attribute p
        self.r: Tensor = radius / scale             # (N, 3) — plan attribute r
        self.centroid_scene = centroid_scene
        self.scale = scale.item()

        #  Label integer 
        label_idx = torch.tensor(
            [config.SEMANTIC_CLASSES.index(lbl) for lbl in label],
            dtype=torch.long,
        )
        self.label: Tensor = label_idx               # (N,) — plan attribute label
        #  Semantic one-hot vector 
        self.s: Tensor = torch.zeros(N, config.NUM_CLASSES)
        self.s.scatter_(1, label_idx.unsqueeze(1), 1.0)
        # (N, NUM_CLASSES) — plan attribute s

        #  Proximity edges 
        if N > 1:
            self.edge_index: Tensor = radius_graph(
                self.p,
                r=config.EDGE_PROXIMITY,
                loop=False,
                max_num_neighbors=32,
            )                                      # (2, E)
        else:
            self.edge_index = torch.zeros(2, 0, dtype=torch.long)

    #  Voxel occupancy ground truth 
    def voxel_occupancy(self, grid: tuple) -> Tensor:
        """
        From the plan: "voxelised occupancy ground truth"

        Returns a binary (H, W, D) tensor.
        True = at least one object occupies this voxel.

        Call with:
          occ_coarse = scene.voxel_occupancy(config.GRID_COARSE)  # (8,  8,  4)
          occ_mid    = scene.voxel_occupancy(config.GRID_MID)     # (16, 16, 8)

        
        For example, if we call 
            occ_coarse = scene.voxel_occupancy(config.GRID_COARSE)  # (8,  8,  4)
        we will get a (8, 8, 4) grid where each voxel is True if at least one object occupies that voxel, and False otherwise.
        """
        H, W, D = grid

        # create voxel coordinate grid in normalised [-1, 1]³ space
        xs = torch.linspace(-1, 1, H)
        ys = torch.linspace(-1, 1, W)
        zs = torch.linspace(-1, 1, D)

        # build the grid
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
        centres = torch.stack([gx, gy, gz], dim=-1)   # (H, W, D, 3)

        # create an empty occupancy grid (full of False)
        occ = torch.zeros(H, W, D, dtype=torch.bool)

        for i in range(self.num_nodes):
            lower_corner = self.p[i] - self.r[i] # p center position, r radius (half-extent size)
            upper_corner = self.p[i] + self.r[i]
            # Check which voxels are inside the bounding box defined by (p, r)
            # inside is a 3D array of True/False where True means the voxel centre is inside the box of node i
            inside = (
                (centres[..., 0] >= lower_corner[0]) & (centres[..., 0] <= upper_corner[0]) &
                (centres[..., 1] >= lower_corner[1]) & (centres[..., 1] <= upper_corner[1]) &
                (centres[..., 2] >= lower_corner[2]) & (centres[..., 2] <= upper_corner[2])
            )
            occ |= inside

        return occ # 3D array (H, W, D) of True/False values

    #  Convert to PyTorch Geometric Data object 
    def to_pyg(self) -> Data:
        """
        Convert to a PyG Data object for use with PyG's DataLoader (batching).
        """
        return Data(
            p=self.p,                    # (N, 3)
            r=self.r,                    # (N, 3)
            s=self.s,                    # (N, NUM_CLASSES)
            label=self.label,            # (N,)
            edge_index=self.edge_index,  # (2, E)
            num_nodes=self.num_nodes,
        )

    def __repr__(self) -> str:
        return (
            f"SceneGraph(N={self.num_nodes}, "
            f"E={self.edge_index.shape[1]}, "
            f"classes={config.NUM_CLASSES})"
        )
