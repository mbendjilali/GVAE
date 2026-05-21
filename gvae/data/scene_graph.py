"""
scene_graph.py

Scene graph loader with point-cloud voxel occupancy ground truth at mid and coarse
resolutions (precomputed by utils/build_scene_graph.py).
"""

import json
import os

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

import config
from gvae.data.graph_masks import coarsen_mask_from_labels
from gvae.data.voxelize import (
    occ_cache_paths,
    scene_normalization,
    normalize_positions,
)


class SceneGraph:
    """
    Node attributes: (p, r, s, label)
    Edges: proximity (radius graph on p)
    occ_mid / occ_coarse: bool voxel grids from LiDAR (not bounding-box proxies)
    """

    @classmethod
    def from_json(cls, json_path: str) -> "SceneGraph":
        with open(json_path, "r") as f:
            data = json.load(f)

        instances = data["instances"]
        if config.REMOVE_NON_INSTANTIABLE:
            instances = [
                inst for inst in instances
                if inst["label"] not in config.NON_INSTANTIABLE_CLASSES
            ]

        positions = [inst["position"] for inst in instances]
        radii = [inst["radius"] for inst in instances]
        labels = [inst["label"] for inst in instances]
        coarsen_mask = coarsen_mask_from_labels(labels)
        if config.REMOVE_NON_INSTANTIABLE:
            coarsen_mask = torch.ones(len(labels), dtype=torch.bool)

        p = torch.tensor(positions, dtype=torch.float32)
        r = torch.tensor(radii, dtype=torch.float32)

        occ_fine, occ_mid, occ_coarse = cls._load_occ_caches(json_path)

        # Prefer normalization stored at build time (matches LiDAR voxelisation)
        if "normalization" in data:
            centroid = torch.tensor(data["normalization"]["centroid"], dtype=torch.float32)
            scale = float(data["normalization"]["scale"])
        else:
            centroid, scale = scene_normalization(p)

        return cls(
            p, r, labels,
            centroid=centroid,
            scale=scale,
            occ_fine=occ_fine,
            occ_mid=occ_mid,
            occ_coarse=occ_coarse,
            coarsen_mask=coarsen_mask,
            source_path=json_path,
        )

    @staticmethod
    def _load_occ_caches(json_path: str) -> tuple[Tensor, Tensor, Tensor]:
        fine_path, mid_path, coarse_path = occ_cache_paths(json_path)
        paths = (fine_path, mid_path, coarse_path)
        expected = (config.GRID_FINE, config.GRID_MID, config.GRID_COARSE)
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            msg = (
                "Point-cloud occupancy cache missing. Run:\n"
                f"  python utils/build_scene_graph.py <data_root>\n"
                f"Missing: {missing[0]}"
            )
            if config.OCC_REQUIRE_CACHE:
                raise FileNotFoundError(msg)
            import warnings
            warnings.warn(msg + " — using empty occupancy grids.")
            return (
                torch.zeros(*config.GRID_FINE, dtype=torch.bool),
                torch.zeros(*config.GRID_MID, dtype=torch.bool),
                torch.zeros(*config.GRID_COARSE, dtype=torch.bool),
            )

        grids = []
        for path, shape in zip(paths, expected):
            arr = np.load(path)
            if tuple(arr.shape) != shape:
                if config.OCC_REQUIRE_CACHE:
                    raise ValueError(
                        f"Occupancy cache {path} has shape {arr.shape}, "
                        f"expected {shape}. Rebuild caches with utils/build_scene_graph.py"
                    )
                import warnings
                warnings.warn(
                    f"Occupancy cache {path} shape {arr.shape} != {shape}; using empty grid.",
                    stacklevel=2,
                )
                grids.append(torch.zeros(*shape, dtype=torch.bool))
            else:
                grids.append(torch.from_numpy(arr))
        return grids[0], grids[1], grids[2]

    def __init__(
        self,
        position: Tensor,
        radius: Tensor,
        label: list,
        centroid: Tensor | None = None,
        scale: float | None = None,
        occ_fine: Tensor | None = None,
        occ_mid: Tensor | None = None,
        occ_coarse: Tensor | None = None,
        coarsen_mask: Tensor | None = None,
        source_path: str = "",
    ):
        N = position.shape[0]
        self.num_nodes = N

        if centroid is None or scale is None:
            centroid, scale = scene_normalization(position)
        else:
            scale = max(float(scale), 1e-6)

        self.centroid_scene = centroid
        self.scale = scale
        self.p = normalize_positions(position, centroid, scale)
        self.r = radius / scale

        label_idx = torch.tensor(
            [config.SEMANTIC_CLASSES.index(lbl) for lbl in label],
            dtype=torch.long,
        )
        self.label = label_idx
        self.s = torch.zeros(N, config.NUM_CLASSES)
        self.s.scatter_(1, label_idx.unsqueeze(1), 1.0)

        if N > 1:
            self.edge_index = radius_graph(
                self.p,
                r=config.EDGE_PROXIMITY,
                loop=False,
                max_num_neighbors=32,
            )
        else:
            self.edge_index = torch.zeros(2, 0, dtype=torch.long)

        Hf, Wf, Df = config.GRID_FINE
        Hm, Wm, Dm = config.GRID_MID
        Hc, Wc, Dc = config.GRID_COARSE
        self.occ_fine = occ_fine if occ_fine is not None else torch.zeros(Hf, Wf, Df, dtype=torch.bool)
        self.occ_mid = occ_mid if occ_mid is not None else torch.zeros(Hm, Wm, Dm, dtype=torch.bool)
        self.occ_coarse = occ_coarse if occ_coarse is not None else torch.zeros(Hc, Wc, Dc, dtype=torch.bool)

        if coarsen_mask is None:
            coarsen_mask = torch.ones(N, dtype=torch.bool)
        self.coarsen_mask = coarsen_mask
        self.source_path = source_path

    @property
    def num_coarsenable(self) -> int:
        return int(self.coarsen_mask.sum().item())

    def to(self, device):
        self.p = self.p.to(device)
        self.r = self.r.to(device)
        self.s = self.s.to(device)
        self.label = self.label.to(device)
        self.edge_index = self.edge_index.to(device)
        self.occ_fine = self.occ_fine.to(device)
        self.occ_mid = self.occ_mid.to(device)
        self.occ_coarse = self.occ_coarse.to(device)
        self.coarsen_mask = self.coarsen_mask.to(device)
        return self

    def on_device(self, device, non_blocking: bool = False):
        """Copy tensor fields to device without mutating a cached CPU instance."""
        import copy
        g = copy.copy(self)
        kw = {"non_blocking": non_blocking}
        g.p = self.p.to(device, **kw)
        g.r = self.r.to(device, **kw)
        g.s = self.s.to(device, **kw)
        g.label = self.label.to(device, **kw)
        g.edge_index = self.edge_index.to(device, **kw)
        g.occ_fine = self.occ_fine.to(device, **kw)
        g.occ_mid = self.occ_mid.to(device, **kw)
        g.occ_coarse = self.occ_coarse.to(device, **kw)
        g.coarsen_mask = self.coarsen_mask.to(device, **kw)
        return g

    def to_pyg(self) -> Data:
        return Data(
            p=self.p,
            r=self.r,
            s=self.s,
            label=self.label,
            edge_index=self.edge_index,
            occ_fine=self.occ_fine,
            occ_mid=self.occ_mid,
            occ_coarse=self.occ_coarse,
            num_nodes=self.num_nodes,
        )

    def __repr__(self) -> str:
        occ_fine = int(self.occ_fine.sum().item())
        occ_mid = int(self.occ_mid.sum().item())
        occ_coarse = int(self.occ_coarse.sum().item())
        return (
            f"SceneGraph(N={self.num_nodes}, N_coarsen={self.num_coarsenable}, "
            f"E={self.edge_index.shape[1]}, occ_fine={occ_fine}, "
            f"occ_mid={occ_mid}, occ_coarse={occ_coarse})"
        )
