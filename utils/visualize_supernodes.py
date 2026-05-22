"""
visualize_supernodes.py

Run the model on a single JSON scene graph and produce two files:

  1. <name>_supernodes.json   — input JSON with three extra fields per instance:
        "supernode_fine"   : fine-level index (0-based position in the instance list)
        "supernode_mid"    : M1 supernode index (1st coarsening), -1 if non-coarsenable
        "supernode_coarse" : M2 supernode index (2nd coarsening), -1 if non-coarsenable

  2. <name>.las              — original .laz tile with three new per-point attributes
        supernode_fine    : inherited from the point's instance index
        supernode_mid     : inherited from the point's instance
        supernode_coarse  : inherited from the point's instance

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
CHECKPOINT = "checkpoint/20260522_111554/best.pth"  
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

    # ── Coarsening assignment matrices ────────────────────────────────────────
    S0 = outputs['S0']   # (n_c,   M_fine) coarsenable instances → fine supernodes
    S1 = outputs['S1']   # (M_fine, M_mid)  fine supernodes        → mid supernodes
    S2 = outputs['S2']   # (M_mid,  M_crs)  mid supernodes         → coarse supernodes
    M_fine   = S0.shape[1] if S0.numel() > 0 else 0
    M_mid    = S1.shape[1] if S1.numel() > 0 else 0
    M_coarse = S2.shape[1] if S2.numel() > 0 else 0

    # Per-instance hard assignments (-1 for non-coarsenable instances)
    assign_fine   = [-1] * N
    assign_mid    = [-1] * N
    assign_coarse = [-1] * N
    coarsen_idx   = graph.coarsen_mask.nonzero(as_tuple=True)[0].tolist()
    if S0.numel() > 0:
        fine_local      = S0.argmax(dim=1).tolist()
        mid_from_fine   = S1.argmax(dim=1).tolist() if S1.numel() > 0 else []
        coarse_from_mid = S2.argmax(dim=1).tolist() if S2.numel() > 0 else []
        for k, node_i in enumerate(coarsen_idx):
            f = fine_local[k]
            assign_fine[node_i] = f
            if mid_from_fine:
                m = mid_from_fine[f]
                assign_mid[node_i] = m
                if coarse_from_mid:
                    assign_coarse[node_i] = coarse_from_mid[m]

    print(f"  {N} nodes ({graph.num_coarsenable} coarsenable)  →  {M_fine} fine  →  {M_mid} mid  →  {M_coarse} coarse")
    fine_vals   = [v for v in assign_fine   if v != -1]
    mid_vals    = [v for v in assign_mid    if v != -1]
    coarse_vals = [v for v in assign_coarse if v != -1]
    print(f"  fine   unique values: {sorted(set(fine_vals))}   ({len(set(fine_vals))} groups)")
    print(f"  mid    unique values: {sorted(set(mid_vals))}   ({len(set(mid_vals))} groups)")
    print(f"  coarse unique values: {sorted(set(coarse_vals))}   ({len(set(coarse_vals))} groups)")

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
        instance['supernode_fine']   = assign_fine[i]
        instance['supernode_mid']    = assign_mid[i]
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

    # ── 2. Build instance_id → supernode index mappings ───────────────────────
    fine_map   = {inst['id']: inst['supernode_fine']   for inst in instances_json}
    mid_map    = {inst['id']: inst['supernode_mid']    for inst in instances_json}
    coarse_map = {inst['id']: inst['supernode_coarse'] for inst in instances_json}

    # ── 3. Find the matching .laz tile ────────────────────────────────────────
    laz_path = os.path.join(DATA_ROOT, 'dales2', split, stem + '.laz')

    if not os.path.exists(laz_path):
        print(f"  ⚠  LAZ file not found: {laz_path} — skipping LAS export")
        return

    # ── 4. Read LAZ and annotate every point ──────────────────────────────────
    las = laspy.read(laz_path)
    instance_ids = np.array(las[INSTANCE_FIELD], dtype=np.int64)

    fine_arr   = np.array([fine_map.get(int(i),   -1) for i in instance_ids], dtype=np.int32)
    mid_arr    = np.array([mid_map.get(int(i),    -1) for i in instance_ids], dtype=np.int32)
    coarse_arr = np.array([coarse_map.get(int(i), -1) for i in instance_ids], dtype=np.int32)

    # ── 5. Write annotated point-cloud LAS ────────────────────────────────────
    original_dims = list(las.point_format.dimension_names)
    new_header = laspy.LasHeader(point_format=las.point_format, version=las.header.version)
    new_header.offsets = las.header.offsets
    new_header.scales  = las.header.scales
    new_header.add_extra_dim(laspy.ExtraBytesParams(name='supernode_fine',   type=np.int32))
    new_header.add_extra_dim(laspy.ExtraBytesParams(name='supernode_mid',    type=np.int32))
    new_header.add_extra_dim(laspy.ExtraBytesParams(name='supernode_coarse', type=np.int32))

    new_las = laspy.LasData(header=new_header)
    for dim_name in original_dims:
        new_las[dim_name] = las[dim_name]
    new_las['supernode_fine']   = fine_arr
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
