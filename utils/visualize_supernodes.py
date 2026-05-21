"""
visualize_supernodes.py

After stage 1 training, run the model on a single JSON scene graph and produce:
  1. A new JSON  (<name>_supernodes.json) — same as input with two extra fields per instance:
        "supernode_mid"    : which M1 supernode (1st coarsening) this instance belongs to
        "supernode_coarse" : which M2 supernode (2nd coarsening) this instance belongs to
  2. A new LAS   (<name>_supernodes.las)  — the original .laz tile with two extra point attributes:
        supernode_mid    : per-point, inherited from the instance it belongs to
        supernode_coarse : per-point, inherited from the instance it belongs to

Usage (from repo root):
    python utils/visualize_supernodes.py data/graphs/train/tile_001.json

The checkpoint is set directly in this file (see CHECKPOINT below).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import laspy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

from gvae.models.gvae import GVAE
from gvae.data.scene_graph import SceneGraph

# ── Edit these before running ─────────────────────────────────────────────────
CHECKPOINT = "checkpoint/20260520_110645/stage1_best.pth"  
DATA_ROOT  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
# ─────────────────────────────────────────────────────────────────────────────

INSTANCE_FIELD = "instance"   # name of the instance-ID field in the .laz files


def run(json_path: str, ckpt_path: str):

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device('cpu')
    model = GVAE().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── Load scene graph and run forward pass ─────────────────────────────────
    graph = SceneGraph.from_json(json_path).to(device)
    N = graph.num_nodes

    with torch.no_grad():
        outputs = model(graph)

    S1 = outputs['S1']   # (N_coarsen, M1) instantiable nodes → mid supernodes
    S2 = outputs['S2']   # (M1, M2)
    M1, M2 = S1.shape[1], S2.shape[1]

    assign_mid = [-1] * N
    assign_coarse = [-1] * N
    coarsen_idx = graph.coarsen_mask.nonzero(as_tuple=True)[0].tolist()
    if S1.numel() > 0:
        mid_local = S1.argmax(dim=1).tolist()
        coarse_from_mid = S2.argmax(dim=1).tolist()
        for k, node_i in enumerate(coarsen_idx):
            m = mid_local[k]
            assign_mid[node_i] = m
            assign_coarse[node_i] = coarse_from_mid[m]

    print(f"  {N} nodes ({graph.num_coarsenable} coarsenable)  →  {M1} mid  →  {M2} coarse")

    # ── 1. Annotate and save JSON ─────────────────────────────────────────────
    with open(json_path) as f:
        data = json.load(f)

    if config.REMOVE_NON_INSTANTIABLE:
        instances_json = [
            inst for inst in data['instances']
            if inst['label'] not in config.NON_INSTANTIABLE_CLASSES
        ]
    else:
        instances_json = data['instances']

    for i, instance in enumerate(instances_json):
        instance['supernode_mid'] = assign_mid[i]
        instance['supernode_coarse'] = assign_coarse[i]

    stem  = os.path.splitext(os.path.basename(json_path))[0]
    parts = json_path.replace('\\', '/').split('/')
    split = 'train' if 'train' in parts else 'test'

    out_dir = os.path.join(DATA_ROOT, 'with_supernodes', split)
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, stem + '_supernodes.json')
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON  → {out_json}")

    # ── 2. Build instance_id → supernode mapping ──────────────────────────────
    # Each instance in the JSON has an "id" field = original LAZ instance ID
    mid_map    = {inst['id']: inst['supernode_mid']    for inst in data['instances']}
    coarse_map = {inst['id']: inst['supernode_coarse'] for inst in data['instances']}

    # ── 3. Find the matching .laz tile ────────────────────────────────────────
    # JSON lives at  data/graphs/<split>/tile_001.json
    # LAZ lives at   data/dales2/<split>/tile_001.laz
    laz_path = os.path.join(DATA_ROOT, 'dales2', split, stem + '.laz')

    if not os.path.exists(laz_path):
        print(f"  ⚠  LAZ file not found: {laz_path} — skipping LAS export")
        return

    # ── 4. Read LAZ and annotate every point ──────────────────────────────────
    las = laspy.read(laz_path)
    instance_ids = np.array(las[INSTANCE_FIELD], dtype=np.int64)

    # map each point's instance ID to its supernode label (-1 if not found)
    mid_arr    = np.array([mid_map.get(int(i), -1)    for i in instance_ids], dtype=np.int32)
    coarse_arr = np.array([coarse_map.get(int(i), -1) for i in instance_ids], dtype=np.int32)

    # ── 5. Write new LAS with extra dimensions ────────────────────────────────
    # Save dimension names now — before creating new_header mutates the shared point_format object
    original_dims = list(las.point_format.dimension_names)

    new_header = laspy.LasHeader(point_format=las.point_format, version=las.header.version)
    new_header.offsets = las.header.offsets
    new_header.scales  = las.header.scales
    new_header.add_extra_dim(laspy.ExtraBytesParams(name='supernode_mid',    type=np.int32))
    new_header.add_extra_dim(laspy.ExtraBytesParams(name='supernode_coarse', type=np.int32))

    new_las = laspy.LasData(header=new_header)

    # copy all existing point dimensions (using the saved list, not the mutated one)
    for dim_name in original_dims:
        new_las[dim_name] = las[dim_name]

    new_las['supernode_mid']    = mid_arr
    new_las['supernode_coarse'] = coarse_arr

    out_las = os.path.join(out_dir, stem + '_supernodes.las')
    new_las.write(out_las)
    print(f"Saved LAS   → {out_las}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate a tile with supernode assignments.')
    parser.add_argument('json',   help='Input scene graph JSON (e.g. data/graphs/train/tile_001.json)')
    parser.add_argument('--ckpt', default=CHECKPOINT,
                        help='Override the CHECKPOINT path set at the top of the script')
    args = parser.parse_args()

    run(args.json, args.ckpt)
