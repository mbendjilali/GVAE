#!/usr/bin/env python3
"""
Scan train/test graphs and report loss breakdowns, edge counts, and NaN terms.

Usage (repo root):
  python utils/diagnose_nan_losses.py [--split train] [--stage 2] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

import torch

import config
from gvae.data.scene_graph import SceneGraph
from gvae.losses.diagnostics import loss_breakdown
from gvae.models.gvae import GVAE
from train import SceneGraphDataset, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--stage", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--limit", type=int, default=0, help="0 = all scenes")
    args = parser.parse_args()

    device = get_device()
    data_dir = os.path.join(config.GRAPH_DATA_DIR, args.split)
    ds = SceneGraphDataset(data_dir)
    files = ds.files[: args.limit] if args.limit else ds.files

    model = GVAE().to(device)
    model.eval()

    zero_coarse = zero_mid = nan_scenes = 0
    nan_term_counts: dict[str, int] = {}

    print(f"split={args.split}  stage={args.stage}  scenes={len(files)}  device={device}")
    print(f"BALL_QUERY_RADIUS_LEVELS={config.BALL_QUERY_RADIUS_LEVELS}")
    print(f"REMOVE_NON_INSTANTIABLE={config.REMOVE_NON_INSTANTIABLE}")
    print(f"COARSEN_EXCLUDE_NON_INSTANTIABLE={config.COARSEN_EXCLUDE_NON_INSTANTIABLE}")
    print(f"OCC_FILTER_NON_INSTANTIABLE={config.OCC_FILTER_NON_INSTANTIABLE}")
    print("-" * 72)

    for path in files:
        graph = SceneGraph.from_json(path).to(device)
        with torch.no_grad():
            outputs = model(graph)
        bd = loss_breakdown(
            outputs, graph, step=0, stage=args.stage, path=path,
            include_coarse_in_total=(args.stage >= 3),
        )

        if bd.e_mid == 0:
            zero_mid += 1
        if bd.e_coarse == 0:
            zero_coarse += 1
        if bd.has_nan:
            nan_scenes += 1
            for t in bd.nan_terms:
                nan_term_counts[t] = nan_term_counts.get(t, 0) + 1
            print(f"NAN  {os.path.basename(path)}  E=({bd.e_instance},{bd.e_mid},{bd.e_coarse})  "
                  f"N=({bd.n_instance},{bd.n_mid},{bd.n_coarse})  terms={bd.nan_terms}")

    print("-" * 72)
    print(f"graphs with E_coarse=0: {zero_coarse}/{len(files)}")
    print(f"graphs with E_mid=0:     {zero_mid}/{len(files)}")
    print(f"graphs with any NaN:   {nan_scenes}/{len(files)}")
    if nan_term_counts:
        print("NaN term counts:")
        for k, v in sorted(nan_term_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
