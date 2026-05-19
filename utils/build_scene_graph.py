"""
build_scene_graph.py

Converts every .laz tile in dales2/train and dales2/test into a JSON
scene graph that SceneGraph.from_json() can read.

Output structure:
    data/scenes/train/<tile>.json
    data/scenes/test/<tile>.json

Run from the repo root:
    python utils/build_scene_graph.py
"""

import os
import json
import numpy as np
import laspy
import sys


INSTANCE_FIELD = "instance"  
LABEL_FIELD    = "classification"  

# mapping
LABEL_MAP = {
    0:  'ground',
    1:  'vegetation',
    2:  'car',
    3:  'powerline',
    4:  'fence',
    5:  'tree',
    6:  'pickup',
    7:  'van_truck',
    8:  'heavy_duty',
    9:  'utility_pole',
    10: 'light_pole',
    11: 'traffic_pole',
    12: 'habitat',
    13: 'complex',
    14: 'annex',
}


def laz_to_graph(laz_path):
    # takes one .laz and returns a dict with the scene graph
    laz = laspy.read(laz_path)
    xyz = np.stack([np.array(laz.x), 
                    np.array(laz.y), 
                    np.array(laz.z)], 
                    axis=1).astype(np.float64) 
    instance = np.array(laz[INSTANCE_FIELD], dtype=np.int64)
    label = np.array(laz[LABEL_FIELD], dtype=np.int64)

    # loop over unique instances
    instances = []
    for inst in np.unique(instance):
        # extract the points 
        mask = instance == inst
        points = xyz[mask]
        labels = label[mask]

        centroid = points.mean(axis=0)

        half_extent = (points.max(axis=0) - points.min(axis=0)) / 2
        half_extent = np.maximum(half_extent, 0.01) # clamp at 0.01 to avoid zero-size node

        instances.append({
            "id": int(inst),
            "position": centroid.tolist(),
            "radius": half_extent.tolist(),
            "label": LABEL_MAP[labels[0]] # assume all points in instance have same label
        })

    return {"instances": instances}


def process_split(split, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    laz_files = sorted(f for f in os.listdir(in_dir) if f.endswith('.laz'))
    print(f"\n[{split}]  {len(laz_files)} tiles found")

    for fname in laz_files:
        stem     = os.path.splitext(fname)[0]
        in_path  = os.path.join(in_dir,  fname)
        out_path = os.path.join(out_dir, stem + '.json')

        if os.path.exists(out_path):
            print(f"  skip  {fname}")
            continue

        try:
            graph = laz_to_graph(in_path)
            n = len(graph["instances"])
            if n == 0:
                print(f"  warn  {fname}  → 0 instances")
                continue
            with open(out_path, 'w') as f:
                json.dump(graph, f, indent=2)
            print(f"  ok    {fname}  → {n} nodes")
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