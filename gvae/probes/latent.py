# gvae/probes/latent.py
# Offline probes: anchor ablation, linear Z→position/class, signal vs background

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

import config
from gvae.data.voxelize import sample_occupancy_queries
from gvae.losses.metrics import hard_miou, mean_position_error
from gvae.models.decoder import sample_volume
from gvae.models.gvae import GVAE


@dataclass
class LevelSpec:
    name: str
    z_key: str
    p_key: str
    r_key: str
    h_key: str
    s_key: str
    occ_key: str
    grid: tuple[int, int, int]
    d: int


LEVELS = (
    LevelSpec(
        "fine", "z_fine", "p_fine", "r_fine", "h_fine", "s_fine",
        "occ_fine", config.GRID_FINE, config.D_FINE_LATENT,
    ),
    LevelSpec(
        "mid", "z_mid", "p_lm1", "r_lm1", "h_lm1", "s_lm1",
        "occ_mid", config.GRID_MID, config.D_MID_LATENT,
    ),
    LevelSpec(
        "coarse", "z_coarse", "p_1", "r_1", "h_1", "s_1",
        "occ_coarse", config.GRID_COARSE, config.D_COARSE_LATENT,
    ),
)


def sample_z_at_points(Z: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinear sample latent Z at normalized positions. Returns (N, C)."""
    if points.numel() == 0:
        return points.new_zeros(0, Z.shape[0])
    ref_pts = points.unsqueeze(1)
    out = sample_volume(Z, ref_pts)
    return out.reshape(points.shape[0], -1)


@torch.no_grad()
def _forward_graph(model: GVAE, graph, device: torch.device) -> dict:
    model.eval()
    graph = graph.on_device(device)
    return model(graph)


def _decoder_for_level(model: GVAE, level: str):
    return {
        "fine": model.decoder_fine,
        "mid": model.decoder_mid,
        "coarse": model.decoder_coarse,
    }[level]


@torch.no_grad()
def anchor_ablation(
    model: GVAE,
    graphs: list,
    device: torch.device,
    mixes: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> dict[str, dict[str, float]]:
    """Decoder position / semantic error vs DECODER_GT_ANCHOR_MIX (no retraining)."""
    old_mix = config.DECODER_GT_ANCHOR_MIX
    results: dict[str, dict[str, float]] = {level: {} for level in ("fine", "mid", "coarse")}

    pos_pred: dict[str, dict[float, list[torch.Tensor]]] = {
        level: {m: [] for m in mixes} for level in ("fine", "mid", "coarse")
    }
    pos_true: dict[str, list[torch.Tensor]] = {level: [] for level in ("fine", "mid", "coarse")}
    sem_pred: dict[str, dict[float, list[torch.Tensor]]] = {
        level: {m: [] for m in mixes} for level in ("fine", "mid", "coarse")
    }
    sem_true: dict[str, list[torch.Tensor]] = {level: [] for level in ("fine", "mid", "coarse")}

    try:
        cached_outs = [_forward_graph(model, graph, device) for graph in graphs]
        for mix in mixes:
            config.DECODER_GT_ANCHOR_MIX = mix
            for out in cached_outs:
                for level, spec in zip(("fine", "mid", "coarse"), LEVELS):
                    h = out[spec.h_key]
                    if h.numel() == 0:
                        continue
                    dec = _decoder_for_level(model, level)
                    recon = dec(
                        h=h,
                        Z=out[spec.z_key],
                        p_gt=out[spec.p_key],
                        r_gt=out[spec.r_key],
                    )
                    pos_pred[level][mix].append(recon["p"])
                    sem_pred[level][mix].append(recon["s"])
                    if mix == mixes[0]:
                        pos_true[level].append(out[spec.p_key])
                        sem_true[level].append(out[spec.s_key])

        for level in ("fine", "mid", "coarse"):
            if not pos_true[level]:
                continue
            p_true = torch.cat(pos_true[level], dim=0)
            s_true = torch.cat(sem_true[level], dim=0)
            for mix in mixes:
                if not pos_pred[level][mix]:
                    continue
                p_pred = torch.cat(pos_pred[level][mix], dim=0)
                s_pred = torch.cat(sem_pred[level][mix], dim=0)
                key = f"mix_{mix:g}"
                results[level][f"{key}_pos_err"] = mean_position_error(p_pred, p_true)
                if level == "fine":
                    results[level][f"{key}_miou"] = hard_miou(s_pred, s_true)
    finally:
        config.DECODER_GT_ANCHOR_MIX = old_mix

    return results


@dataclass
class ProbeDataset:
    z: torch.Tensor
    positions: torch.Tensor
    labels: torch.Tensor


def _collect_probe_samples(
    model: GVAE,
    graphs: list,
    device: torch.device,
    spec: LevelSpec,
) -> ProbeDataset:
    z_parts: list[torch.Tensor] = []
    p_parts: list[torch.Tensor] = []
    s_parts: list[torch.Tensor] = []

    with torch.no_grad():
        for graph in graphs:
            out = _forward_graph(model, graph, device)
            p = out[spec.p_key]
            if p.numel() == 0:
                continue
            z_parts.append(sample_z_at_points(out[spec.z_key], p))
            p_parts.append(p)
            s_parts.append(out[spec.s_key])

    if not z_parts:
        return ProbeDataset(
            torch.zeros(0, spec.d),
            torch.zeros(0, 3),
            torch.zeros(0, config.NUM_CLASSES),
        )
    return ProbeDataset(
        torch.cat(z_parts, dim=0),
        torch.cat(p_parts, dim=0),
        torch.cat(s_parts, dim=0),
    )


def _train_linear_head(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    out_dim: int,
    *,
    epochs: int = 300,
    lr: float = 1e-2,
    task: str = "regression",
) -> tuple[float, float]:
    if X_train.numel() == 0 or X_val.numel() == 0:
        return float("nan"), float("nan")

    device = X_train.device
    head = nn.Linear(X_train.shape[1], out_dim).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    if task == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        head.train()
        opt.zero_grad(set_to_none=True)
        pred = head(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        pred_val = head(X_val)
        pred_tr = head(X_train)
        if task == "regression":
            val_metric = torch.norm(pred_val - Y_val, dim=1).mean().item()
            train_metric = torch.norm(pred_tr - Y_train, dim=1).mean().item()
            return val_metric, train_metric
        val_acc = (pred_val.argmax(1) == Y_val).float().mean().item()
        train_acc = (pred_tr.argmax(1) == Y_train).float().mean().item()
        return val_acc, train_acc


def linear_probe(
    model: GVAE,
    train_graphs: list,
    val_graphs: list,
    device: torch.device,
    *,
    epochs: int = 300,
    lr: float = 1e-2,
) -> dict[str, dict[str, float]]:
    """
    Fit linear maps Z(p) → position and Z(p) → class on train; report error on val.
    Also reports mean |Z| at GT vs random empty voxels (same graph, no learned head).
    """
    results: dict[str, dict[str, float]] = {}

    for spec in LEVELS:
        train = _collect_probe_samples(model, train_graphs, device, spec)
        val = _collect_probe_samples(model, val_graphs, device, spec)
        level: dict[str, float] = {"n_train": float(train.z.shape[0]), "n_val": float(val.z.shape[0])}

        if train.z.shape[0] >= 4 and val.z.shape[0] >= 1:
            val_pos, train_pos = _train_linear_head(
                train.z, train.positions, val.z, val.positions, 3,
                epochs=epochs, lr=lr, task="regression",
            )
            level["linear_pos_err_val"] = val_pos
            level["linear_pos_err_train"] = train_pos

            y_train_cls = train.labels.argmax(dim=1)
            y_val_cls = val.labels.argmax(dim=1)
            val_acc, train_acc = _train_linear_head(
                train.z, y_train_cls, val.z, y_val_cls, config.NUM_CLASSES,
                epochs=epochs, lr=lr, task="classification",
            )
            level["linear_cls_acc_val"] = val_acc
            level["linear_cls_acc_train"] = train_acc

        results[spec.name] = level

    return results


@torch.no_grad()
def signal_vs_background(
    model: GVAE,
    graphs: list,
    device: torch.device,
    *,
    n_empty_per_scene: int = 256,
) -> dict[str, dict[str, float]]:
    """Compare ‖Z(p)‖ at GT node positions vs random empty occ voxels."""
    results: dict[str, dict[str, float]] = {}

    for spec in LEVELS:
        gt_norms: list[torch.Tensor] = []
        empty_norms: list[torch.Tensor] = []
        gt_occ_scores: list[torch.Tensor] = []
        empty_occ_scores: list[torch.Tensor] = []

        for graph in graphs:
            out = _forward_graph(model, graph, device)
            p = out[spec.p_key]
            if p.numel() == 0:
                continue
            z_gt = sample_z_at_points(out[spec.z_key], p)
            gt_norms.append(z_gt.norm(dim=1))
            gt_occ_scores.append(z_gt.abs().mean(dim=1))

            occ_grid = getattr(graph, spec.occ_key).to(device)
            q, labels = sample_occupancy_queries(
                occ_grid,
                n_queries=n_empty_per_scene,
                pos_ratio=0.0,
            )
            empty_pts = q[labels < 0.5]
            if empty_pts.numel() == 0:
                continue
            z_empty = sample_z_at_points(out[spec.z_key], empty_pts)
            empty_norms.append(z_empty.norm(dim=1))
            empty_occ_scores.append(z_empty.abs().mean(dim=1))

        level: dict[str, float] = {}
        if gt_norms:
            gt = torch.cat(gt_norms)
            level["gt_norm_mean"] = gt.mean().item()
            level["gt_norm_std"] = gt.std(unbiased=False).item()
        if empty_norms:
            em = torch.cat(empty_norms)
            level["empty_norm_mean"] = em.mean().item()
            level["empty_norm_std"] = em.std(unbiased=False).item()
        if gt_norms and empty_norms:
            gt = torch.cat(gt_norms)
            em = torch.cat(empty_norms)
            level["norm_ratio_gt_over_empty"] = (gt.mean() / em.mean().clamp(min=1e-8)).item()
            pooled_std = torch.cat([gt, em]).std(unbiased=False).clamp(min=1e-8)
            level["norm_separation"] = ((gt.mean() - em.mean()) / pooled_std).item()

        if gt_occ_scores and empty_occ_scores:
            g = torch.cat(gt_occ_scores)
            e = torch.cat(empty_occ_scores)
            level["abs_mean_ratio_gt_over_empty"] = (g.mean() / e.mean().clamp(min=1e-8)).item()

        results[spec.name] = level

    return results


@dataclass
class LatentProbeReport:
    anchor: dict[str, dict[str, float]] = field(default_factory=dict)
    linear: dict[str, dict[str, float]] = field(default_factory=dict)
    signal: dict[str, dict[str, float]] = field(default_factory=dict)


def run_latent_probes(
    model: GVAE,
    train_graphs: list,
    val_graphs: list,
    device: torch.device,
    *,
    anchor_mixes: tuple[float, ...] = (0.0, 0.5, 1.0),
    linear_epochs: int = 300,
    linear_lr: float = 1e-2,
    n_empty_per_scene: int = 256,
) -> LatentProbeReport:
    eval_graphs = val_graphs if val_graphs else train_graphs
    return LatentProbeReport(
        anchor=anchor_ablation(model, eval_graphs, device, mixes=anchor_mixes),
        linear=linear_probe(
            model, train_graphs, eval_graphs, device,
            epochs=linear_epochs, lr=linear_lr,
        ),
        signal=signal_vs_background(
            model, eval_graphs, device, n_empty_per_scene=n_empty_per_scene,
        ),
    )


def format_report(report: LatentProbeReport) -> str:
    lines = ["=" * 72, "Latent probes", "=" * 72]

    lines.append("\n── A. Anchor ablation (decoder pos_err / miou vs GT anchor mix) ──")
    for level, metrics in report.anchor.items():
        if not metrics:
            lines.append(f"  [{level}]  (no nodes)")
            continue
        parts = [f"  [{level}]"]
        for k in sorted(metrics):
            v = metrics[k]
            if "miou" in k and (v != v):
                continue
            if "miou" in k:
                parts.append(f"{k}={v:.1%}")
            else:
                parts.append(f"{k}={v:.4f}")
        lines.append("  ".join(parts))

    lines.append("\n── B. Linear probe Z(p) → position / class ──")
    for level, metrics in report.linear.items():
        parts = [f"  [{level}]  n_train={int(metrics.get('n_train', 0))}  n_val={int(metrics.get('n_val', 0))}"]
        for k in ("linear_pos_err_val", "linear_pos_err_train", "linear_cls_acc_val", "linear_cls_acc_train"):
            if k in metrics:
                v = metrics[k]
                if "acc" in k:
                    parts.append(f"{k}={v:.1%}")
                else:
                    parts.append(f"{k}={v:.4f}")
        lines.append("  ".join(parts))

    lines.append("\n── C. Signal vs background (‖Z(p)‖ at GT vs empty occ voxels) ──")
    for level, metrics in report.signal.items():
        if not metrics:
            lines.append(f"  [{level}]  (no data)")
            continue
        parts = [f"  [{level}]"]
        for k in (
            "gt_norm_mean", "empty_norm_mean", "norm_ratio_gt_over_empty",
            "norm_separation", "abs_mean_ratio_gt_over_empty",
        ):
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.4f}")
        lines.append("  ".join(parts))

    lines.append("")
    return "\n".join(lines)
