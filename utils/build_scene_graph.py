"""
build_scene_graph.py

Converts every .laz tile in dales2/train and dales2/test into:
  - a JSON scene graph (SceneGraph.from_json)
  - point-cloud occupancy caches at mid and coarse resolutions

Output (per tile, same directory):
    <stem>.json
    <stem>_occ_mid.npy      bool (16, 16, 8)
    <stem>_occ_coarse.npy   bool (8, 8, 4)

Run from the repo root:
    python utils/build_scene_graph.py <data_root>
"""

import os
import json
import sys
import numpy as np
import laspy

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)

import config
from gvae.data.voxelize import voxelize_points_np

INSTANCE_FIELD = "instance"
LABEL_FIELD = "classification"

LABEL_MAP = {
    0: 'ground',
    1: 'vegetation',
    2: 'car',
    3: 'powerline',
    4: 'fence',
    5: 'tree',
    6: 'pickup',
    7: 'van_truck',
    8: 'heavy_duty',
    9: 'utility_pole',
    10: 'light_pole',
    11: 'traffic_pole',
    12: 'habitat',
    13: 'complex',
    14: 'annex',
}


def _filter_instances(instances):
    if not config.REMOVE_NON_INSTANTIABLE:
        return instances
    return [
        inst for inst in instances
        if inst["label"] not in config.NON_INSTANTIABLE_CLASSES
    ]


def laz_to_scene(laz_path):
    """
    Returns:
        graph_dict: JSON-serialisable scene graph + normalization
        points_norm: (P, 3) all LiDAR points in [-1, 1]³ (same frame as SceneGraph)
    """
    laz = laspy.read(laz_path)
    xyz = np.stack([np.array(laz.x), np.array(laz.y), np.array(laz.z)], axis=1).astype(np.float64)
    instance_ids = np.array(laz[INSTANCE_FIELD], dtype=np.int64)
    label_ids = np.array(laz[LABEL_FIELD], dtype=np.int64)

    instances = []
    for inst in np.unique(instance_ids):
        mask = instance_ids == inst
        points = xyz[mask]
        labels = label_ids[mask]

        centroid = points.mean(axis=0)
        half_extent = (points.max(axis=0) - points.min(axis=0)) / 2
        half_extent = np.maximum(half_extent, 0.01)

        instances.append({
            "id": int(inst),
            "position": centroid.tolist(),
            "radius": half_extent.tolist(),
            "label": LABEL_MAP[int(labels[0])],
        })

    instances = _filter_instances(instances)
    if len(instances) == 0:
        return None, None

    positions = np.array([inst["position"] for inst in instances], dtype=np.float64)
    centroid = positions.mean(axis=0)
    centred = positions - centroid
    scale = max(float(np.abs(centred).max()), 1e-6)

    points_norm = (xyz - centroid) / scale
    if points_norm.shape[0] > config.OCC_MAX_POINTS:
        idx = np.random.choice(points_norm.shape[0], config.OCC_MAX_POINTS, replace=False)
        points_norm = points_norm[idx]

    graph_dict = {
        "normalization": {
            "centroid": centroid.tolist(),
            "scale": scale,
        },
        "instances": instances,
    }
    return graph_dict, points_norm


def write_occupancy_caches(stem_path, points_norm):
    """Write *_occ_mid.npy and *_occ_coarse.npy next to <stem_path>.json."""
    occ_mid = voxelize_points_np(points_norm, config.GRID_MID)
    occ_coarse = voxelize_points_np(points_norm, config.GRID_COARSE)
    np.save(stem_path + config.OCC_CACHE_SUFFIX_MID, occ_mid)
    np.save(stem_path + config.OCC_CACHE_SUFFIX_COARSE, occ_coarse)


def process_split(split, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    laz_files = sorted(f for f in os.listdir(in_dir) if f.endswith('.laz'))
    print(f"\n[{split}]  {len(laz_files)} tiles found")

    for fname in laz_files:
        stem = os.path.splitext(fname)[0]
        in_path = os.path.join(in_dir, fname)
        json_path = os.path.join(out_dir, stem + '.json')
        occ_mid_path = os.path.join(out_dir, stem + config.OCC_CACHE_SUFFIX_MID)
        occ_coarse_path = os.path.join(out_dir, stem + config.OCC_CACHE_SUFFIX_COARSE)

        if (
            os.path.exists(json_path)
            and os.path.exists(occ_mid_path)
            and os.path.exists(occ_coarse_path)
        ):
            print(f"  skip  {fname}")
            continue

        try:
            graph_dict, points_norm = laz_to_scene(in_path)
            if graph_dict is None:
                print(f"  warn  {fname}  → 0 instances after filtering")
                continue

            with open(json_path, 'w') as f:
                json.dump(graph_dict, f, indent=2)

            write_occupancy_caches(os.path.join(out_dir, stem), points_norm)
            n = len(graph_dict["instances"])
            print(f"  ok    {fname}  → {n} nodes + occ caches")
        except Exception as e:
            print(f"  ERROR {fname}: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python utils/build_scene_graph.py <data_root>")
        sys.exit(1)
    data_root = sys.argv[1]
    in_dir = os.path.join(data_root, 'dales2')
    out_dir = os.path.join(data_root, 'graphs')
    for split in ('train', 'test'):
        split_dir = os.path.join(in_dir, split)
        if os.path.isdir(split_dir):
            process_split(split, split_dir, out_dir)
        else:
            print(f"[{split}]  not found, skipping")


if __name__ == '__main__':
    main()
