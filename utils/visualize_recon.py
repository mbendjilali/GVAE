#!/usr/bin/env python3
"""
Graph reconstruction visual sanity check: GT vs h+Z vs Z-only vs h-anchors (BEV).

Usage (repo root):
  python utils/visualize_recon.py --checkpoint checkpoint/20260526_100316/best.pth
  python utils/visualize_recon.py --checkpoint checkpoint/<run>/best.pth \\
      --scenes 5080_54400 5140_54390 5175_54395 --levels fine mid
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.axes import Axes

import config
from gvae.checkpoint_compat import migrate_state_dict
from gvae.data.voxelize import _points_to_indices
from gvae.losses.metrics import compute_metrics
from gvae.models.decoder import sample_volume
from gvae.models.gvae import GVAE
from train import SceneGraphDataset, get_device

LEVEL_SPECS = {
    "fine": {
        "p_gt": "p_fine",
        "r_gt": "r_fine",
        "s_gt": "s_fine",
        "recon": "recon_fine",
        "recon_z": "recon_fine_zonly",
        "recon_z_h": "recon_fine_zonly_hanchor",
        "h_key": "h_fine",
        "decoder_attr": "decoder_fine",
        "z_key": "z_fine",
        "occ_key": "occ_fine",
    },
    "mid": {
        "p_gt": "p_lm1",
        "r_gt": "r_lm1",
        "s_gt": "s_lm1",
        "recon": "recon_mid",
        "recon_z": "recon_mid_zonly",
        "recon_z_h": "recon_mid_zonly_hanchor",
        "h_key": "h_lm1",
        "decoder_attr": "decoder_mid",
        "z_key": "z_mid",
        "occ_key": "occ_mid",
    },
}


_LEGACY_DECODER_MLP_P = (
    "decoder_fine.mlp_p.",
    "decoder_mid.mlp_p.",
    "decoder_coarse.mlp_p.",
)


def _load_model(
    checkpoint: str,
    device: torch.device,
    *,
    use_zonly_decoder: bool | None = None,
) -> GVAE:
    state = migrate_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
    )
    if use_zonly_decoder is None:
        config.LATENT_GRAPH_VAE_MODE = any(
            k.startswith("latent_decoder_fine.") for k in state
        )
        if not config.LATENT_GRAPH_VAE_MODE:
            config.USE_Z_ONLY_DECODER = any(
                k.startswith("zonly_decoder_fine.") for k in state
            )
    else:
        config.USE_Z_ONLY_DECODER = use_zonly_decoder
    model = GVAE().to(device)
    incompatible = model.load_state_dict(state, strict=False)
    legacy_prefixes = _LEGACY_DECODER_MLP_P
    if config.LATENT_GRAPH_VAE_MODE:
        legacy_prefixes = legacy_prefixes + (
            "decoder_fine.",
            "decoder_mid.",
            "decoder_coarse.",
            "zonly_decoder_fine.",
            "zonly_decoder_mid.",
            "zonly_decoder_coarse.",
        )
    else:
        legacy_prefixes = legacy_prefixes + (
            "occ_grid_head_fine.",
            "occ_grid_head_mid.",
            "occ_grid_head_coarse.",
        )
    unexpected = [
        k
        for k in incompatible.unexpected_keys
        if not any(k.startswith(p) for p in legacy_prefixes)
    ]
    if unexpected:
        preview = ", ".join(unexpected[:6])
        suffix = " ..." if len(unexpected) > 6 else ""
        raise RuntimeError(f"Checkpoint keys not in model: {preview}{suffix}")
    if incompatible.unexpected_keys:
        print(
            "Note: ignored legacy SceneGraphDecoder mlp_p weights "
            "(unused when Z-only readout is enabled)."
        )
    anchor_missing = [
        k for k in incompatible.missing_keys
        if "mlp_p_anchor.0." in k or "mlp_r_anchor.0." in k
    ]
    other_missing = [k for k in incompatible.missing_keys if k not in anchor_missing]
    if anchor_missing and not other_missing:
        print(
            "Note: anchor MLP first layer random (checkpoint had Linear anchor head)."
        )
    elif other_missing:
        preview = ", ".join(other_missing[:6])
        suffix = " ..." if len(other_missing) > 6 else ""
        raise RuntimeError(f"Model missing checkpoint keys: {preview}{suffix}")
    model.eval()
    return model


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


def _class_colors(s_onehot: torch.Tensor) -> np.ndarray:
    """Map one-hot / soft labels to RGBA via tab20."""
    if s_onehot.numel() == 0:
        return np.zeros((0, 4))
    cls = s_onehot.argmax(dim=1).cpu().numpy()
    cmap = plt.cm.tab20
    return cmap(cls % 20)


def _draw_footprints(
    ax: Axes,
    p: torch.Tensor,
    r: torch.Tensor,
    s: torch.Tensor,
    *,
    edgecolor: str | None = None,
    facecolor: str | None = None,
    alpha: float = 0.35,
    linewidth: float = 1.0,
    use_class_color: bool = False,
) -> None:
    if p.numel() == 0:
        return
    p_np = p.detach().cpu().numpy()
    r_np = r.detach().cpu().numpy()
    colors = _class_colors(s) if use_class_color else None
    for i in range(p_np.shape[0]):
        cx, cy = p_np[i, 0], p_np[i, 1]
        rx, ry = max(r_np[i, 0], 1e-4), max(r_np[i, 1], 1e-4)
        ec = edgecolor
        fc = facecolor
        if use_class_color and colors is not None:
            ec = colors[i]
            fc = (*colors[i][:3], alpha)
        rect = mpatches.Rectangle(
            (cx - rx, cy - ry), 2 * rx, 2 * ry,
            linewidth=linewidth,
            edgecolor=ec,
            facecolor=fc if fc is not None else "none",
            alpha=alpha if fc is None else 1.0,
        )
        ax.add_patch(rect)


def _draw_points(
    ax: Axes,
    p: torch.Tensor,
    *,
    color: str,
    marker: str = "o",
    size: float = 12,
    label: str | None = None,
) -> None:
    if p.numel() == 0:
        return
    xy = p.detach().cpu().numpy()[:, :2]
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=color, s=size, marker=marker,
        linewidths=0.8 if marker != "x" else 0,
        edgecolors="white" if marker != "x" else "none",
        label=label, zorder=5,
    )


def _setup_bev_ax(ax: Axes, title: str) -> None:
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2, linewidth=0.5)


def _h_anchors(model: GVAE, outputs: dict, level: str) -> torch.Tensor:
    spec = LEVEL_SPECS[level]
    h = outputs[spec["h_key"]]
    if h.numel() == 0:
        return h.new_zeros(0, 3)
    decoder = getattr(model, spec["decoder_attr"])
    p_anchor, _ = decoder.predict_anchors(h)
    return p_anchor


def _z_norm_bev(z: torch.Tensor) -> np.ndarray:
    """Max-project L2 norm of Z channels onto the H×W plane (BEV)."""
    if z.numel() == 0:
        return np.zeros((1, 1))
    norm = z.norm(dim=0).detach().cpu().numpy()
    return norm.max(axis=2)


def _z_norm_at_points(z: torch.Tensor, p: torch.Tensor) -> np.ndarray:
    """‖Z‖ at bilinear sample locations (N,)."""
    if p.numel() == 0:
        return np.zeros(0)
    feat = sample_volume(z, p.unsqueeze(1)).reshape(p.shape[0], -1)
    return feat.norm(dim=1).detach().cpu().numpy()


def _points_to_bev_pixels(p: torch.Tensor, grid_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    H, W = grid_hw
    device = p.device
    if p.numel() == 0:
        return np.array([]), np.array([])
    i, j, _ = _points_to_indices(p, (H, W, 1))
    return i.cpu().numpy(), j.cpu().numpy()


def _format_metrics(metrics: dict[str, float], level: str) -> list[str]:
    lines = []
    prefix = level
    keys = [
        (f"pos_err_{prefix}", "pos"),
        (f"size_err_{prefix}", "size"),
        (f"pos_err_zonly_{prefix}", "zpos"),
        (f"size_err_zonly_{prefix}", "zsize"),
        (f"pos_err_zonly_hanchor_{prefix}", "hzpos"),
        (f"anchor_err_{prefix}", "anc"),
        (f"anchor_size_err_{prefix}", "asz"),
        (f"soft_miou_{prefix}", "smiou"),
        (f"soft_miou_zonly_{prefix}", "zsmiou"),
    ]
    if level == "mid":
        keys.append(("inst_pos_err_mid", "inst"))
    parts = []
    for key, label in keys:
        if key in metrics:
            v = metrics[key]
            if "miou" in key:
                parts.append(f"{label}={v:.0%}")
            else:
                parts.append(f"{label}={v:.3f}")
    if parts:
        lines.append(f"  [{level}] " + "  ".join(parts))
    return lines


@torch.no_grad()
def visualize_scene(
    model: GVAE,
    graph,
    device: torch.device,
    out_dir: str,
    levels: tuple[str, ...],
) -> dict[str, str | float]:
    graph = graph.on_device(device)
    outputs = model(graph)
    metrics = compute_metrics(outputs, graph, step=0)
    stem = _scene_stem(graph)
    paths: dict[str, str | float] = {"scene": stem}

    idx = graph.coarsen_mask.nonzero(as_tuple=True)[0]
    p_inst = graph.p[idx] if idx.numel() else graph.p[:0]
    r_inst = graph.r[idx] if idx.numel() else graph.r[:0]
    s_inst = graph.s[idx] if idx.numel() else graph.s[:0]

    n_levels = len(levels)
    fig, axes = plt.subplots(n_levels, 4, figsize=(14, 3.5 * n_levels))
    if n_levels == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, level in enumerate(levels):
        spec = LEVEL_SPECS[level]
        p_gt = outputs[spec["p_gt"]]
        r_gt = outputs[spec["r_gt"]]
        s_gt = outputs[spec["s_gt"]]
        recon = outputs.get(spec["recon"])
        recon_z = outputs.get(spec["recon_z"])
        p_h = _h_anchors(model, outputs, level)

        ax_gt, ax_hz, ax_z, ax_all = axes[row]

        _setup_bev_ax(ax_gt, f"{level} — GT supernodes")
        if p_inst.numel():
            _draw_footprints(ax_gt, p_inst, r_inst, s_inst, edgecolor="#888888", alpha=0.15, linewidth=0.6)
        _draw_footprints(ax_gt, p_gt, r_gt, s_gt, use_class_color=True, alpha=0.45, linewidth=1.2)

        _setup_bev_ax(ax_hz, f"{level} — h+Z recon")
        if recon is not None:
            _draw_footprints(ax_hz, p_gt, r_gt, s_gt, edgecolor="#2ca02c", alpha=0.12, linewidth=0.8)
            _draw_footprints(ax_hz, recon["p"], recon["r"], recon["s"], edgecolor="#1f77b4", alpha=0.35, linewidth=1.2)

        _setup_bev_ax(ax_z, f"{level} — Z-only @ GT slot")
        if recon_z is not None:
            _draw_footprints(ax_z, p_gt, r_gt, s_gt, edgecolor="#2ca02c", alpha=0.12, linewidth=0.8)
            _draw_points(ax_z, recon_z["p"], color="#ff7f0e", size=14, label="Z-only p̂")

        _setup_bev_ax(ax_all, f"{level} — overlay")
        if p_inst.numel():
            _draw_footprints(ax_all, p_inst, r_inst, s_inst, edgecolor="#cccccc", alpha=0.1, linewidth=0.5)
        _draw_footprints(ax_all, p_gt, r_gt, s_gt, edgecolor="#2ca02c", facecolor="#2ca02c", alpha=0.12, linewidth=1.0)
        if recon is not None:
            _draw_footprints(ax_all, recon["p"], recon["r"], recon["s"], edgecolor="#1f77b4", alpha=0.25, linewidth=1.0)
        if recon_z is not None:
            _draw_points(ax_all, recon_z["p"], color="#ff7f0e", size=10)
        if p_h.numel():
            _draw_points(ax_all, p_h, color="#d62728", marker="x", size=28, label="h-anchor")

        if row == 0:
            handles = [
                mpatches.Patch(edgecolor="#2ca02c", facecolor="#2ca02c", alpha=0.3, label="GT"),
                mpatches.Patch(edgecolor="#1f77b4", facecolor="none", label="h+Z"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=6, label="Z-only"),
                plt.Line2D([0], [0], marker="x", color="#d62728", linestyle="None", markersize=6, label="h-anchor"),
            ]
            ax_all.legend(handles=handles, loc="upper right", fontsize=7)

    fig.suptitle(f"{stem} — graph reconstruction (BEV)", fontsize=12)
    fig.tight_layout()
    bev_path = os.path.join(out_dir, f"{stem}_recon_bev.png")
    fig.savefig(bev_path, dpi=130)
    plt.close(fig)
    paths["recon_bev"] = bev_path

    z_level = "fine" if "fine" in levels else levels[0]
    z_spec = LEVEL_SPECS[z_level]
    z = outputs[z_spec["z_key"]]
    z_bev = _z_norm_bev(z)
    H, W = z_bev.shape
    med = float(np.median(z_bev)) + 1e-6
    ratio_bev = np.clip(z_bev / med, 0.0, 3.0)
    p_gt = outputs[z_spec["p_gt"]]
    norms_gt = _z_norm_at_points(z, p_gt)
    norms_bg_median = float(np.median(z_bev))

    fig2, ax2 = plt.subplots(1, 3, figsize=(13, 4))

    vmin, vmax = np.percentile(z_bev, [10, 92])
    im0 = ax2[0].imshow(z_bev.T, origin="lower", cmap="magma", aspect="auto", vmin=vmin, vmax=vmax)
    ax2[0].set_title(f"{z_level} ‖Z‖ max-Z (p10–p92)")
    plt.colorbar(im0, ax=ax2[0], fraction=0.046)

    im1 = ax2[1].imshow(ratio_bev.T, origin="lower", cmap="magma", aspect="auto", vmin=0.8, vmax=1.8)
    ax2[1].set_title(f"{z_level} ‖Z‖ / median (1≈typical)")
    plt.colorbar(im1, ax=ax2[1], fraction=0.046)

    occ_gt = getattr(graph, z_spec["occ_key"]).cpu().numpy()
    occ_bev = occ_gt.max(axis=2).T.astype(np.float32)
    ax2[2].imshow(occ_bev, origin="lower", cmap="Greys", aspect="auto", vmin=0, vmax=1)
    ax2[2].set_title(f"{z_level} occ GT (LiDAR)")

    for ax, p_pts, marker, size, label in (
        (ax2[0], p_inst, "o", 10, "instances"),
        (ax2[0], p_gt, "o", 22, "supernodes"),
        (ax2[1], p_gt, "o", 22, "supernodes"),
    ):
        if p_pts.numel() == 0:
            continue
        ii, jj = _points_to_bev_pixels(p_pts, (H, W))
        if ax is ax2[0] and p_pts is p_gt and norms_gt.size:
            ax.scatter(
                ii, jj, s=size, c=norms_gt, cmap="viridis", vmin=norms_gt.min(), vmax=norms_gt.max(),
                edgecolors="white", linewidths=0.4, label=label, zorder=5,
            )
        else:
            ax.scatter(ii, jj, s=size, marker=marker, c="#00ffff" if p_pts is p_gt else "#2ca02c",
                       edgecolors="white", linewidths=0.3, label=label, zorder=5)

    ax2[0].legend(loc="upper right", fontsize=7)
    for ax in ax2:
        ax.set_xlabel("grid i")
        ax.set_ylabel("grid j")

    fig2.suptitle(
        f"{stem} — Z spatial structure (median ‖Z‖={norms_bg_median:.1f}; "
        f"at supernodes mean={norms_gt.mean():.1f})"
    )
    fig2.tight_layout()
    peaks_path = os.path.join(out_dir, f"{stem}_z_peaks.png")
    fig2.savefig(peaks_path, dpi=130)
    plt.close(fig2)
    paths["z_peaks"] = peaks_path

    metrics_path = os.path.join(out_dir, f"{stem}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"scene: {stem}\n")
        for level in levels:
            for line in _format_metrics(metrics, level):
                f.write(line + "\n")
        if "inst_pos_err_mid" in metrics and "mid" not in levels:
            for line in _format_metrics(metrics, "mid"):
                if "inst" in line:
                    f.write(line + "\n")
    paths["metrics"] = metrics_path

    for k, v in metrics.items():
        if isinstance(v, float):
            paths[k] = v

    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize graph reconstruction vs GT")
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
        "--levels", nargs="+", default=["fine", "mid"],
        choices=list(LEVEL_SPECS),
        help="Coarsening levels to plot",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="",
        help="Output directory (default: <ckpt_dir>/recon_viz)",
    )
    parser.add_argument(
        "--no-zonly-decoder",
        action="store_true",
        help="Load checkpoint trained without Z-only decoders (overrides auto-detect)",
    )
    args = parser.parse_args()

    device = get_device()
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir = args.output or os.path.join(ckpt_dir, "recon_viz")
    os.makedirs(out_dir, exist_ok=True)
    levels = tuple(args.levels)

    dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, args.split))
    scenes = _pick_scenes(dataset, args.scenes, args.count)
    if not scenes:
        print("No scenes to visualize.")
        return 1

    use_zonly = False if args.no_zonly_decoder else None
    model = _load_model(args.checkpoint, device, use_zonly_decoder=use_zonly)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Levels:     {', '.join(levels)}")
    print(f"Scenes ({len(scenes)}): {', '.join(_scene_stem(g) for g in scenes)}")
    print(f"Output:     {out_dir}\n")

    for graph in scenes:
        paths = visualize_scene(model, graph, device, out_dir, levels)
        print(f"{paths['scene']}:")
        print(f"  BEV   → {paths['recon_bev']}")
        print(f"  peaks → {paths['z_peaks']}")
        print(f"  stats → {paths['metrics']}")
        pos = paths.get("pos_err_fine")
        size = paths.get("size_err_fine")
        zpos = paths.get("pos_err_zonly_fine")
        hzpos = paths.get("pos_err_zonly_hanchor_fine")
        if pos is not None:
            parts = [f"pos={pos:.3f}"]
            if zpos is not None:
                parts.append(f"zpos={zpos:.3f}")
            if hzpos is not None:
                parts.append(f"hzpos={hzpos:.3f}")
            if size is not None:
                parts.append(f"size={size:.3f}")
            print("  fine " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
