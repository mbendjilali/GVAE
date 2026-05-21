#!/usr/bin/env python3
"""
Profile forward + backward on the heaviest scene (most nodes).

Usage (repo root):
  python utils/profile_scene.py
  python utils/profile_scene.py --split train --top-k 10
  python utils/profile_scene.py --path data/graphs/train/largest.json
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

import torch

import config
from gvae.losses.gvae_loss import compute_branch_losses
from gvae.models.gvae import GVAE
from train import SceneGraphDataset, get_device, TRAIN_STAGE


def _pick_heaviest(dataset: SceneGraphDataset, top_k: int) -> list[tuple[int, str, int]]:
    ranked = sorted(
        ((g.num_nodes, path, i) for i, (g, path) in enumerate(zip(dataset.graphs, dataset.files))),
        reverse=True,
    )
    return [(nodes, path, idx) for nodes, path, idx in ranked[:top_k]]


def _train_step(model, graph, device, use_amp: bool, sequential: bool):
    model.train()
    graph = graph.on_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    with torch.amp.autocast('cuda', enabled=use_amp):
        outputs = model(graph, stage=TRAIN_STAGE)
        branches, _ = compute_branch_losses(outputs, graph, step=0, stage=TRAIN_STAGE)

    if not branches:
        raise RuntimeError("No loss branches for this scene.")

    if sequential and len(branches) > 1 and not use_amp:
        for i, (_, loss, _) in enumerate(branches):
            scaler.scale(loss).backward(retain_graph=(i < len(branches) - 1))
    else:
        total = sum(b[1] for b in branches)
        scaler.scale(total).backward()

    scaler.unscale_(optimizer)
    if config.GRAD_CLIP_NORM > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.GRAD_CLIP_NORM,
        )
    scaler.step(optimizer)
    scaler.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--path", type=str, default="", help="Profile this JSON instead of heaviest")
    parser.add_argument("--top-k", type=int, default=5, help="List top-K scenes by node count")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=3, help="Profiler steps after warmup")
    args = parser.parse_args()

    device = get_device()
    use_amp = config.USE_AMP and device.type == "cuda"
    data_dir = os.path.join(config.GRAPH_DATA_DIR, args.split)
    dataset = SceneGraphDataset(data_dir)

    if args.path:
        from gvae.data.scene_graph import SceneGraph
        graph = SceneGraph.from_json(args.path)
        graph.source_path = args.path
        label = os.path.basename(args.path)
        nodes = graph.num_nodes
    else:
        heaviest = _pick_heaviest(dataset, max(args.top_k, 1))
        nodes, path, _ = heaviest[0]
        graph = dataset.graphs[dataset.files.index(path)]
        label = os.path.basename(path)
        print(f"Top scenes by node count ({args.split}):")
        for n, p, _ in heaviest:
            print(f"  {n:5d}  {os.path.basename(p)}")

    print(f"\nProfiling: {label}  N={nodes}  device={device}  amp={use_amp}  "
          f"sequential_backward={config.SEQUENTIAL_BACKWARD}")

    model = GVAE().to(device)

    for _ in range(args.warmup):
        _train_step(model, graph, device, use_amp, config.SEQUENTIAL_BACKWARD)
        if device.type == "cuda":
            torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.active):
            _train_step(model, graph, device, use_amp, config.SEQUENTIAL_BACKWARD)
            if device.type == "cuda":
                torch.cuda.synchronize()

    print(f"\nTop ops by {sort_key}:")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"\nPeak CUDA memory this run: {peak_mb:.0f} MiB")


if __name__ == "__main__":
    main()
