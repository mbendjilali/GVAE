#!/usr/bin/env python3
"""
Per-scene loss breakdown for NaN / val-instability debugging.

Usage (repo root):
  python utils/diagnose_nan_losses.py --split test --stage 2
  python utils/diagnose_nan_losses.py --split test --stage 2 \\
      --checkpoint checkpoint/<run>/best.pth --mode eval
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

import torch

import config
from gvae.data.scene_graph import SceneGraph
from gvae.losses.diagnostics import loss_breakdown
from gvae.losses.gvae_loss import compute_loss
from gvae.models.gvae import GVAE
from train import SceneGraphDataset, get_device


def _load_checkpoint(model: GVAE, path: str, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {path}")


def _scene_stats(outputs) -> dict[str, float]:
    p = outputs["recon_mid"]["p"]
    z = outputs["z_mid"]
    return {
        "max_abs_p": float(p.abs().max().item()),
        "max_abs_z": float(z.abs().max().item()),
    }


def _flag_reason(bd, stats: dict[str, float], loss_total: float) -> list[str]:
    reasons = []
    if bd.has_nan or not math.isfinite(loss_total):
        reasons.append("nan")
    if loss_total > 1e3:
        reasons.append(f"total>{loss_total:.2e}")
    if stats["max_abs_p"] > 2.0:
        reasons.append(f"|p|={stats['max_abs_p']:.2e}")
    if stats["max_abs_z"] > 1e3:
        reasons.append(f"|z|={stats['max_abs_z']:.2e}")
    kl = bd.terms.get("KL_mid", bd.terms.get("KL", 0))
    if kl > 100:
        reasons.append(f"KL={kl:.1f}")
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--stage", type=int, default=1, choices=(1,), help="kept for CLI compat; always full forward")
    parser.add_argument("--limit", type=int, default=0, help="0 = all scenes")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--mode",
        choices=("eval", "train"),
        default="eval",
        help="eval matches validate(); train matches training forward",
    )
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    device = get_device()
    data_dir = os.path.join(config.GRAPH_DATA_DIR, args.split)
    ds = SceneGraphDataset(data_dir)
    files = ds.files[: args.limit] if args.limit else ds.files

    model = GVAE().to(device)
    if args.checkpoint:
        _load_checkpoint(model, args.checkpoint, device)
    else:
        print("NOTE: no --checkpoint → random weights.\n")

    if args.mode == "eval":
        model.eval()
    else:
        model.train()

    print(f"split={args.split}  stage={args.stage}  mode={args.mode}  scenes={len(files)}  device={device}")
    print(f"U-Net: GroupNorm (num_groups={config.UNET_NUM_GROUPS})")
    print("-" * 88)

    zero_coarse = zero_mid = flagged = 0
    rows: list[tuple[float, str, dict]] = []

    for path in files:
        graph = SceneGraph.from_json(path).to(device)
        with torch.no_grad():
            outputs = model(graph)
        bd = loss_breakdown(
            outputs, graph, step=0, stage=args.stage, path=path,
            include_coarse_in_total=(args.stage >= 1),
        )
        total_tensor, components = compute_loss(outputs, graph, step=0, stage=args.stage)
        loss_total = total_tensor.item()
        stats = _scene_stats(outputs) if args.stage >= 1 else {"max_abs_p": 0.0, "max_abs_z": 0.0}
        rows.append((loss_total if math.isfinite(loss_total) else float("inf"), path, {
            "components": {k: (v.item() if hasattr(v, "item") else v) for k, v in components.items()},
            "stats": stats,
            "bd": bd,
        }))

        if bd.e_mid == 0:
            zero_mid += 1
        if bd.e_coarse == 0:
            zero_coarse += 1
        reasons = _flag_reason(bd, stats, loss_total)
        if reasons:
            flagged += 1
            c = rows[-1][2]["components"]
            print(
                f"FLAG {os.path.basename(path)}  {', '.join(reasons)}\n"
                f"     total={loss_total}  recon={c.get('recon')}  KL={c.get('KL')}  occ={c.get('occ')}\n"
                f"     max|p|={stats['max_abs_p']:.4g}  max|z|={stats['max_abs_z']:.4g}"
            )

    rows.sort(key=lambda x: x[0], reverse=True)
    if args.show_all or (not flagged and rows):
        print("-" * 88)
        title = "All scenes (worst first):" if args.show_all else "Top 5 by total loss:"
        print(title)
        for total, path, payload in (rows if args.show_all else rows[:5]):
            c = payload["components"]
            s = payload["stats"]
            print(
                f"  {os.path.basename(path):40s}  total={total:12.4g}  "
                f"recon={c.get('recon', 0):8.4g}  KL={c.get('KL', 0):8.4g}  "
                f"max|p|={s['max_abs_p']:.4g}"
            )

    print("-" * 88)
    n_seen = len(rows)
    print(f"graphs scanned:         {n_seen}/{len(files)}")
    print(f"graphs with E_coarse=0: {zero_coarse}/{n_seen}")
    print(f"graphs with E_mid=0:    {zero_mid}/{n_seen}")
    print(f"graphs flagged:         {flagged}/{n_seen}")


if __name__ == "__main__":
    main()
