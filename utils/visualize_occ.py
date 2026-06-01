#!/usr/bin/env python3
"""
Occupancy visual sanity check: predicted vs LiDAR GT grids on selected scenes.

Usage (repo root):
  python utils/visualize_occ.py --checkpoint checkpoint/norm_depth3_150/best.pth
  python utils/visualize_occ.py --checkpoint checkpoint/norm_depth3_150/best.pth \\
      --scenes 5080_54400 5140_54390 -o checkpoint/norm_depth3_150/occ_viz
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

import numpy as np
import torch

import config
from gvae.losses.metrics import _occupancy_grid_iou
from gvae.models.gvae import GVAE
from train import SceneGraphDataset, get_device

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _load_model(checkpoint: str, device: torch.device) -> GVAE:
    model = GVAE().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _max_projection(grid: np.ndarray) -> np.ndarray:
    """Collapse Z axis for a quick 2D view (H, W)."""
    return grid.max(axis=2)


def _scene_stem(graph) -> str:
    path = getattr(graph, "source_path", "")
    return os.path.splitext(os.path.basename(path))[0]


def _pick_scenes(dataset: SceneGraphDataset, names: list[str] | None, n_default: int):
    if names:
        wanted = set(names)
        picked = [g for g in dataset.graphs if _scene_stem(g) in wanted]
        missing = wanted - {_scene_stem(g) for g in picked}
        if missing:
            print(f"Warning: scenes not found in split: {', '.join(sorted(missing))}")
        return picked
    return dataset.graphs[:n_default]


@torch.no_grad()
def visualize_scene(
    model: GVAE,
    graph,
    device: torch.device,
    out_dir: str,
) -> dict[str, float]:
    graph = graph.on_device(device)
    outputs = model(graph)
    stats: dict[str, float] = {"scene": _scene_stem(graph)}

    level_specs = (
        ("fine", "z_fine", "occ_fine", "occ_grid_head_fine", config.LAMBDA_OCC_GRID_FINE),
        ("mid", "z_mid", "occ_mid", "occ_grid_head_mid", config.LAMBDA_OCC_GRID_MID),
    )

    arrays: dict[str, np.ndarray] = {}
    if _HAS_MPL:
        fig, axes = plt.subplots(len(level_specs), 3, figsize=(9, 3 * len(level_specs)))
        if len(level_specs) == 1:
            axes = np.expand_dims(axes, axis=0)

    for row, (level, z_key, occ_key, head_key, lam) in enumerate(level_specs):
        occ_gt = getattr(graph, occ_key)
        if _HAS_MPL:
            ax_gt, ax_pred, ax_ov = axes[row]

        if lam <= 0 or occ_gt.numel() == 0:
            if _HAS_MPL:
                for ax in (ax_gt, ax_pred, ax_ov):
                    ax.axis("off")
            continue

        head = outputs[head_key]
        z = outputs[z_key]
        iou, prec, rec, pred_rate, gt_rate = _occupancy_grid_iou(head, z, occ_gt)
        stats[f"occ_iou_{level}"] = iou
        stats[f"occ_pred_rate_{level}"] = pred_rate
        stats[f"occ_gt_rate_{level}"] = gt_rate
        stats[f"occ_recall_{level}"] = rec
        stats[f"occ_precision_{level}"] = prec

        logits = head(z)
        pred = (torch.sigmoid(logits) >= config.METRICS_OCC_THRESHOLD).cpu().numpy()
        gt = occ_gt.cpu().numpy()
        arrays[f"{level}_gt"] = gt
        arrays[f"{level}_pred"] = pred

        if not _HAS_MPL:
            continue

        gt_2d = _max_projection(gt.astype(np.float32))
        pred_2d = _max_projection(pred.astype(np.float32))
        overlay = np.zeros((*gt_2d.shape, 3), dtype=np.float32)
        overlay[..., 0] = gt_2d
        overlay[..., 1] = pred_2d
        overlay[..., 2] = gt_2d * pred_2d

        ax_gt.imshow(gt_2d, origin="lower", cmap="Greys", vmin=0, vmax=1)
        ax_gt.set_title(f"{level} GT (rate={gt_rate:.1%})")
        ax_pred.imshow(pred_2d, origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax_pred.set_title(f"{level} pred (rate={pred_rate:.1%})")
        ax_ov.imshow(overlay, origin="lower")
        ax_ov.set_title(f"{level} R=GT G=pred B=both IoU={iou:.0%}")

        for ax in (ax_gt, ax_pred, ax_ov):
            ax.set_xticks([])
            ax.set_yticks([])

    stem = _scene_stem(graph)
    if _HAS_MPL:
        fig.suptitle(stem)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{stem}_occ.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        stats["image"] = out_path
    else:
        npz_path = os.path.join(out_dir, f"{stem}_occ.npz")
        np.savez(npz_path, **arrays)
        stats["image"] = npz_path
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize occupancy pred vs GT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--split", choices=("train", "test"), default="test",
        help="Which graph split to sample scenes from",
    )
    parser.add_argument(
        "--scenes", nargs="*", default=None,
        help="Tile stems (e.g. 5080_54400). Default: first --count scenes.",
    )
    parser.add_argument("--count", type=int, default=3, help="Default number of scenes")
    parser.add_argument(
        "-o", "--output", type=str, default="",
        help="Output directory (default: beside checkpoint)",
    )
    args = parser.parse_args()

    device = get_device()
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir = args.output or os.path.join(ckpt_dir, "occ_viz")
    os.makedirs(out_dir, exist_ok=True)

    dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, args.split))
    scenes = _pick_scenes(dataset, args.scenes, args.count)
    if not scenes:
        print("No scenes to visualize.")
        return 1

    model = _load_model(args.checkpoint, device)
    if not _HAS_MPL:
        print("Note: matplotlib not installed — saving .npz grids instead of PNG.")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Scenes ({len(scenes)}): {', '.join(_scene_stem(g) for g in scenes)}")
    print(f"Output:     {out_dir}\n")

    for graph in scenes:
        stats = visualize_scene(model, graph, device, out_dir)
        print(
            f"{stats['scene']}: "
            f"fine IoU={stats.get('occ_iou_fine', float('nan')):.0%} "
            f"pred={stats.get('occ_pred_rate_fine', float('nan')):.0%} "
            f"gt={stats.get('occ_gt_rate_fine', float('nan')):.0%} "
            f"rec={stats.get('occ_recall_fine', float('nan')):.0%} "
            f"→ {stats['image']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
