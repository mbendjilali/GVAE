#!/usr/bin/env python3
"""
Run latent probes on a trained checkpoint (anchor ablation, linear Z→p, signal vs background).

Usage (repo root):
  python utils/probe_latent.py --checkpoint checkpoint/20260521_163301/best.pth
  python utils/probe_latent.py --checkpoint checkpoint/20260521_163301/best.pth --linear-epochs 500
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

import torch

import config
from gvae.models.gvae import GVAE
from gvae.probes.latent import format_report, run_latent_probes, save_probe_artifacts
from train import SceneGraphDataset, get_device


def _load_model(checkpoint: str, device: torch.device) -> GVAE:
    model = GVAE().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Latent probes on a GVAE checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best.pth or last.pth",
    )
    parser.add_argument("--train-split", choices=("train", "test"), default="train")
    parser.add_argument("--val-split", choices=("train", "test"), default="test")
    parser.add_argument("--linear-epochs", type=int, default=300)
    parser.add_argument("--linear-lr", type=float, default=1e-2)
    parser.add_argument("--n-empty", type=int, default=256, help="Empty voxels sampled per scene")
    parser.add_argument(
        "--anchor-mixes", type=str, default="0,0.5,1",
        help="Comma-separated DECODER_GT_ANCHOR_MIX values",
    )
    parser.add_argument(
        "--probe-target", choices=("supernode", "instance", "both"), default="both",
        help="Sample Z at supernode GT, raw instances, or both (fine level)",
    )
    parser.add_argument("-o", "--output", type=str, default="", help="Optional text report path")
    args = parser.parse_args()

    device = get_device()
    mixes = tuple(float(x.strip()) for x in args.anchor_mixes.split(",") if x.strip())

    train_ds = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, args.train_split))
    val_ds = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, args.val_split))

    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Train:      {len(train_ds)} graphs ({args.train_split})")
    print(f"Val:        {len(val_ds)} graphs ({args.val_split})")
    print(f"Anchor mix: {mixes}")
    print(f"Probe target: {args.probe_target}")
    print()

    model = _load_model(args.checkpoint, device)
    report = run_latent_probes(
        model,
        train_ds.graphs,
        val_ds.graphs,
        device,
        anchor_mixes=mixes,
        linear_epochs=args.linear_epochs,
        linear_lr=args.linear_lr,
        n_empty_per_scene=args.n_empty,
        probe_target=args.probe_target,
    )
    text = format_report(report, probe_target=args.probe_target)
    print(text)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        json_path = os.path.join(
            out_dir,
            os.path.splitext(os.path.basename(args.output))[0] + ".json",
        )
        from gvae.probes.latent import report_to_dict
        import json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                report_to_dict(
                    report,
                    checkpoint=args.checkpoint,
                    probe_target=args.probe_target,
                ),
                f,
                indent=2,
            )
            f.write("\n")
        print(f"Report written to {args.output}")
        print(f"JSON written to {json_path}")


if __name__ == "__main__":
    main()
