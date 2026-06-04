# train.py — Single-stage training loop for the Scene Graph VAE
# All branches (fine + mid + coarse); configurable FPS coarsening; LR decay mid-run

import math
import os
import sys
import argparse
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from gvae.training.console import Style, Term, strip_ansi
from gvae.models.gvae import GVAE
from gvae.data.scene_graph import SceneGraph
from gvae.losses.gvae_loss import compute_branch_losses, compute_loss, decoder_gt_anchor_mix_for_epoch
from gvae.losses.metrics import compute_metrics
from gvae.probes.latent import run_latent_probes, save_probe_artifacts


class SceneGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.files: list[str] = []
        self.graphs: list[SceneGraph] = []
        self.skipped: list[str] = []
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.json'):
                continue
            path = os.path.join(data_dir, f)
            graph = SceneGraph.from_json(path)
            if graph.num_coarsenable == 0:
                self.skipped.append(f)
                continue
            graph.source_path = path
            self.files.append(path)
            self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def make_dataloader(dataset, shuffle: bool) -> DataLoader:
    pin = config.DATALOADER_PIN_MEMORY and torch.cuda.is_available()
    kwargs = {
        "batch_size": config.BATCH_SIZE,
        "shuffle": shuffle,
        "collate_fn": lambda x: x,
        "num_workers": config.DATALOADER_NUM_WORKERS,
        "pin_memory": pin,
    }
    if config.DATALOADER_NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def _use_amp(device: torch.device) -> bool:
    return config.USE_AMP and device.type == "cuda"


def _forward_loss(model, graph, step, device, use_amp: bool):
    graph = graph.on_device(device, non_blocking=True)
    with torch.amp.autocast('cuda', enabled=use_amp):
        outputs = model(graph)
        branches, lambda_kl = compute_branch_losses(outputs, graph, step)

    zero = graph.p.new_zeros(())
    L_recon = L_KL = L_pool = zero
    for _, _, parts in branches:
        L_recon = L_recon + parts.get('recon', zero)
        L_recon = L_recon + parts.get('recon_latent', zero)
        L_KL = L_KL + parts.get('KL', zero)
        if 'pool' in parts:
            L_pool = L_pool + parts['pool']
    components = {
        'recon': L_recon, 'KL': L_KL, 'lambda_kl': lambda_kl,
    }
    if config.USE_POOL_LOSS and config.COARSEN_ASSIGNMENT == "soft":
        components['pool'] = L_pool
    return graph, outputs, branches, components


def _backward_branches(
    branches,
    scaler,
    sequential: bool,
    use_amp: bool,
    loss_scale: float = 1.0,
) -> None:
    if not branches:
        return
    # AMP + retain_graph backward mixes fp16/fp32 autograd dtypes → inf / runtime error.
    use_sequential = sequential and not use_amp and len(branches) > 1
    if use_sequential:
        for i, (_, loss, _) in enumerate(branches):
            scaler.scale(loss * loss_scale).backward(retain_graph=(i < len(branches) - 1))
    else:
        total = sum(b[1] for b in branches) * loss_scale
        scaler.scale(total).backward()


def _strict_mean(values: list[float]) -> tuple[float, int]:
    """Mean over all values; any non-finite input makes the mean non-finite."""
    if not values:
        return float('nan'), 0
    n_bad = sum(1 for v in values if not math.isfinite(v))
    return sum(values) / len(values), n_bad


def _average_components(component_dicts: list[dict]) -> dict:
    if not component_dicts:
        return {}
    out = {}
    for key in component_dicts[0]:
        vals = [c[key] for c in component_dicts]
        out[key], _ = _strict_mean(vals)
    return out


def _graph_label(graph) -> str:
    path = getattr(graph, 'source_path', None)
    return os.path.basename(path) if path else repr(graph)


def _lr_for_epoch(epoch: int) -> float:
    """epoch is 0-indexed."""
    if epoch >= config.LR_DECAY_EPOCH:
        return config.LEARNING_RATE_LATE
    return config.LEARNING_RATE


def _set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = lr


def _tqdm_kwargs(desc: str, colour: str) -> dict:
    return {
        "desc": desc,
        "leave": False,
        "dynamic_ncols": True,
        "colour": colour,
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    }


def validate(model, loader, device, step=0, desc='val', use_amp: bool = False):
    model.eval()
    saved_mix = config.DECODER_GT_ANCHOR_MIX
    config.DECODER_GT_ANCHOR_MIX = 0.0

    per_graph_losses = []
    all_metrics = []
    all_components = []
    failed_paths: list[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, **_tqdm_kwargs(desc, "cyan")):
            for graph in batch:
                graph = graph.on_device(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(graph)
                    loss, components = compute_loss(
                        outputs, graph, step=step,
                    )
                val = loss.item()
                per_graph_losses.append(val)
                if not math.isfinite(val):
                    failed_paths.append(_graph_label(graph))
                all_components.append({
                    k: v.item() if hasattr(v, 'item') else v for k, v in components.items()
                })

                m = compute_metrics(outputs, graph, step=step)
                if m:
                    all_metrics.append(m)

    n_graphs = len(per_graph_losses)
    avg_loss, n_failed = _strict_mean(per_graph_losses)
    avg_components = _average_components(all_components)

    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            avg_metrics[key], _ = _strict_mean(vals)

    model.train()
    config.DECODER_GT_ANCHOR_MIX = saved_mix
    return avg_loss, avg_metrics, avg_components, n_graphs, n_failed, failed_paths


def _run_probes(
    model,
    train_graphs,
    val_graphs,
    device,
    ckpt_dir,
    term: Term,
    *,
    epoch: int | None = None,
    probe_target: str = "supernode",
    text_name: str = "probe_report.txt",
    json_name: str = "probe_summary.json",
) -> None:
    model.eval()
    report = run_latent_probes(
        model,
        train_graphs,
        val_graphs,
        device,
        probe_target=probe_target,
    )
    text_path, json_path = save_probe_artifacts(
        report,
        ckpt_dir,
        epoch=epoch,
        probe_target=probe_target,
        text_name=text_name,
        json_name=json_name,
    )
    model.train()
    ep_note = f" (ep {epoch})" if epoch is not None else ""
    term.ok(f"probes saved{ep_note} → {os.path.basename(text_path)}, {os.path.basename(json_path)}")


def train(
    model,
    loader,
    val_loader,
    device,
    ckpt_dir,
    writer,
    term: Term,
):
    use_amp = _use_amp(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    step_counter = 0
    epoch_counter = 0
    best_loss = float('inf')
    current_lr = config.LEARNING_RATE
    config.DECODER_GT_ANCHOR_MIX = decoder_gt_anchor_mix_for_epoch(0)

    for epoch in range(config.NUM_EPOCHS):
        lr = _lr_for_epoch(epoch)
        if lr != current_lr:
            current_lr = lr
            _set_optimizer_lr(optimizer, current_lr)
            tqdm.write(term.paint(
                f"  LR → {current_lr:.1e} (epoch {epoch + 1})",
                Style.YELLOW,
            ))

        mix = decoder_gt_anchor_mix_for_epoch(epoch)
        if mix != config.DECODER_GT_ANCHOR_MIX:
            config.DECODER_GT_ANCHOR_MIX = mix
            if config.ANCHOR_MIX_CURRICULUM:
                tqdm.write(term.paint(
                    f"  anchor mix → {mix:.2f} (epoch {epoch + 1})",
                    Style.YELLOW,
                ))

        per_graph_losses = []
        epoch_components = []
        running_loss = 0.0

        batch_bar = tqdm(
            loader,
            **_tqdm_kwargs(f"train {epoch + 1}/{config.NUM_EPOCHS}", "green"),
        )
        for batch in batch_bar:
            optimizer.zero_grad(set_to_none=True)
            batch_had_grad = False
            loss_scale = 1.0 / len(batch)
            for graph in batch:
                _, _, branches, components = _forward_loss(
                    model, graph, step_counter, device, use_amp,
                )
                val = sum(b[1].item() for b in branches) if branches else float('nan')
                if not math.isfinite(val):
                    raise RuntimeError(
                        f"Non-finite training loss on {_graph_label(graph)}: "
                        f"{ {k: (v.item() if hasattr(v, 'item') else v) for k, v in components.items()} }"
                    )
                _backward_branches(
                    branches, scaler, config.SEQUENTIAL_BACKWARD, use_amp, loss_scale,
                )
                batch_had_grad = True
                per_graph_losses.append(val)
                running_loss = sum(per_graph_losses) / len(per_graph_losses)
                epoch_components.append({
                    k: v.item() if hasattr(v, 'item') else v for k, v in components.items()
                })
                step_counter += 1

            if batch_had_grad:
                scaler.unscale_(optimizer)
                if config.GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        config.GRAD_CLIP_NORM,
                    )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    tqdm.write(term.paint(
                        "  ⚠ GradScaler reduced scale (skipped non-finite grads)",
                        Style.YELLOW,
                    ))

            batch_bar.set_postfix(
                loss=f"{per_graph_losses[-1]:.3f}" if per_graph_losses else "—",
                avg=f"{running_loss:.3f}" if per_graph_losses else "—",
                lr=f"{current_lr:.0e}",
                refresh=False,
            )

        n_train = len(per_graph_losses)
        avg_loss, n_train_failed = _strict_mean(per_graph_losses)
        avg_train_components = _average_components(epoch_components)

        avg_val_loss, avg_metrics, avg_val_components, n_val, n_val_failed, val_failed = validate(
            model, val_loader, device, step=step_counter,
            desc=f"val {epoch + 1}/{config.NUM_EPOCHS}",
            use_amp=use_amp,
        )

        is_best = math.isfinite(avg_val_loss) and avg_val_loss < best_loss
        term.epoch_header(
            epoch + 1, config.NUM_EPOCHS, current_lr,
            avg_loss if math.isfinite(avg_loss) else float('nan'),
            avg_val_loss if math.isfinite(avg_val_loss) else float('nan'),
            is_best=is_best,
        )
        term.metrics_line(avg_metrics)
        if n_train_failed or n_val_failed:
            term.warn(
                f"non-finite: train {n_train - n_train_failed}/{n_train}, "
                f"val {n_val - n_val_failed}/{n_val}"
            )
        if val_failed:
            term.warn(f"val scenes: {', '.join(val_failed[:5])}"
                      + (f" (+{len(val_failed) - 5})" if len(val_failed) > 5 else ""))

        writer.add_scalars('total', {'train': avg_loss, 'val': avg_val_loss}, step_counter)
        writer.add_scalar('lr', current_lr, epoch_counter)
        for k, v in avg_train_components.items():
            writer.add_scalar(f'train/{k}', v, step_counter)
        for k, v in avg_val_components.items():
            writer.add_scalar(f'val/{k}', v, step_counter)
        if avg_metrics:
            for k, v in avg_metrics.items():
                writer.add_scalar(f'metrics/{k}', v, step_counter)

        epoch_counter += 1
        writer.add_scalars('epoch/total', {'train': avg_loss, 'val': avg_val_loss}, epoch_counter)
        for k, v in avg_train_components.items():
            writer.add_scalar(f'epoch/train/{k}', v, epoch_counter)
        for k, v in avg_val_components.items():
            writer.add_scalar(f'epoch/val/{k}', v, epoch_counter)
        if avg_metrics:
            for k, v in avg_metrics.items():
                writer.add_scalar(f'epoch/metrics/{k}', v, epoch_counter)

        if is_best:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
            term.ok(f"saved best.pth ({best_loss:.4f})")
        elif not math.isfinite(avg_val_loss):
            term.warn("val loss non-finite; checkpoint not updated")


def get_device():
    if config.CUDA_DEVICE is None:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{config.CUDA_DEVICE}')
    print(f"CUDA_DEVICE={config.CUDA_DEVICE} requested but CUDA unavailable; using CPU.")
    return torch.device('cpu')


class _Tee:
    def __init__(self, *streams, strip_ansi_for: int | None = None):
        self.streams = streams
        self.strip_ansi_for = strip_ansi_for

    def write(self, data):
        for i, stream in enumerate(self.streams):
            payload = strip_ansi(data) if i == self.strip_ansi_for else data
            stream.write(payload)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def fileno(self):
        return self.streams[0].fileno()

    def isatty(self):
        return self.streams[0].isatty()


def _setup_run_logging(ckpt_dir: str):
    log_path = os.path.join(ckpt_dir, 'train.log')
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
    log_file.write(
        f"\n{'=' * 72}\n"
        f"Run started {datetime.now().isoformat(timespec='seconds')}\n"
        f"Command: {' '.join(sys.argv)}\n"
        f"CWD: {os.getcwd()}\n"
        f"{'=' * 72}\n"
    )
    log_file.flush()
    sys.stdout = _Tee(sys.__stdout__, log_file, strip_ansi_for=1)
    sys.stderr = _Tee(sys.__stderr__, log_file, strip_ansi_for=1)
    return log_file


def _teardown_run_logging(log_file):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if log_file and not log_file.closed:
        log_file.write(f"\nRun finished {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.close()


def _load_init_checkpoint(model, path: str, device, term: Term) -> None:
    """Load pretrained weights; allow architecture mismatches (e.g. phase-1 → phase-2)."""
    from gvae.checkpoint_compat import migrate_state_dict

    state = migrate_state_dict(
        torch.load(path, map_location=device, weights_only=True),
    )
    incompatible = model.load_state_dict(state, strict=False)
    n_unexp = len(incompatible.unexpected_keys)
    n_miss = len(incompatible.missing_keys)
    if n_unexp:
        term.dim(
            f"init from {path}: ignored {n_unexp} checkpoint key(s) "
            f"(not in this model)"
        )
    if n_miss:
        preview = ", ".join(incompatible.missing_keys[:6])
        suffix = " ..." if n_miss > 6 else ""
        term.warn(
            f"init: {n_miss} module(s) not in checkpoint — random init ({preview}{suffix})"
        )
    if not n_unexp and not n_miss:
        term.ok(f"init from {path} (strict match)")


def main(
    ckpt_dir,
    *,
    run_probes: bool,
    probe_target: str,
    init_checkpoint: str | None = None,
):
    term = Term()
    device = get_device()
    use_amp = _use_amp(device)

    train_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'train'))
    val_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'test'))
    config.KL_TOTAL_STEPS = len(train_dataset) * config.NUM_EPOCHS

    skipped = train_dataset.skipped + val_dataset.skipped
    banner_lines = [
        f"{term.paint('device', Style.DIM)}  {device}"
        + (f"  {term.paint('amp', Style.DIM)} on" if use_amp else ""),
        f"{term.paint('data', Style.DIM)}     "
        f"{len(train_dataset)} train · {len(val_dataset)} val graphs (cached)",
        f"{term.paint('train', Style.DIM)}    "
        f"{config.NUM_EPOCHS} ep · lr {config.LEARNING_RATE:.0e}→{config.LEARNING_RATE_LATE:.0e} "
        f"@ ep {config.LR_DECAY_EPOCH + 1} · batch {config.BATCH_SIZE}",
        f"{term.paint('grid', Style.DIM)}     "
        f"fine {config.GRID_FINE} · mid {config.GRID_MID} · coarse {config.GRID_COARSE} · "
        f"unet depth fine/mid/coarse = "
        f"{config.UNET_DEPTH_FINE}/{config.UNET_DEPTH_MID}/{config.UNET_DEPTH_COARSE}",
        f"{term.paint('coarsen', Style.DIM)}  "
        f"{config.COARSEN_ASSIGNMENT} · ratios {config.REDUCTION_RATIO_LEVELS}"
        + (f" · pool λ={config.LAMBDA_POOL}" if config.USE_POOL_LOSS else ""),
    ]
    if config.LATENT_GRAPH_VAE_MODE:
        banner_lines.append(
            f"{term.paint('latent-vae', Style.DIM)} "
            f"Z→graph_hat (slot cross-attn) λ recon={config.LAMBDA_RECON_LATENT} "
            f"max_slots={config.LATENT_GRAPH_MAX_SLOTS} "
            f"refine_z={config.LATENT_GRAPH_REFINE_FROM_Z_SAMPLE}"
        )
    if config.USE_Z_ONLY_DECODER:
        pos_mode = (
            "patch p←Z@anchor"
            if config.Z_ONLY_PATCH_DEFORMABLE_POSITION
            else "aux losses only (deformable p)"
        )
        banner_lines.append(
            f"{term.paint('zonly', Style.DIM)} "
            f"λ h/z/hz = {config.LAMBDA_RECON_H}/{config.LAMBDA_RECON_ZONLY}/"
            f"{config.LAMBDA_RECON_HZONLY} jitter={config.Z_ONLY_QUERY_JITTER} · {pos_mode}"
        )
    if (
        config.LAMBDA_ANCHOR_FINE > 0
        or config.LAMBDA_ANCHOR_MID > 0
        or config.LAMBDA_ANCHOR_COARSE > 0
        or config.LAMBDA_ANCHOR_R_FINE > 0
        or config.LAMBDA_ANCHOR_R_MID > 0
        or config.LAMBDA_ANCHOR_R_COARSE > 0
    ):
        banner_lines.append(
            f"{term.paint('anchor', Style.DIM)} "
            f"λ p fine/mid/coarse = "
            f"{config.LAMBDA_ANCHOR_FINE}/{config.LAMBDA_ANCHOR_MID}/"
            f"{config.LAMBDA_ANCHOR_COARSE} · "
            f"λ r = {config.LAMBDA_ANCHOR_R_FINE}/"
            f"{config.LAMBDA_ANCHOR_R_MID}/{config.LAMBDA_ANCHOR_R_COARSE}"
        )
    banner_lines.append(
        f"{term.paint('recon', Style.DIM)} "
        f"λ sem/pos/size = {config.LAMBDA_SEM}/"
        f"{config.LAMBDA_POS}/{config.LAMBDA_SIZE} "
        f"(zonly sem/size = {config.LAMBDA_SEM_ZONLY}/{config.LAMBDA_SIZE_ZONLY})"
    )
    if config.ANCHOR_MIX_CURRICULUM:
        banner_lines.append(
            f"{term.paint('anchor', Style.DIM)} "
            f"mix curriculum {config.ANCHOR_MIX_START:.1f}→{config.ANCHOR_MIX_END:.1f} "
            f"over {config.ANCHOR_MIX_ANNEAL_EPOCHS} ep"
        )
    if (
        config.LAMBDA_NORM_CONTRAST_FINE > 0
        or config.LAMBDA_NORM_CONTRAST_MID > 0
        or config.LAMBDA_NORM_CONTRAST_COARSE > 0
    ):
        banner_lines.append(
            f"{term.paint('norm', Style.DIM)} "
            f"contrast λ fine/mid/coarse = "
            f"{config.LAMBDA_NORM_CONTRAST_FINE}/"
            f"{config.LAMBDA_NORM_CONTRAST_MID}/"
            f"{config.LAMBDA_NORM_CONTRAST_COARSE} "
            f"margin={config.NORM_CONTRAST_MARGIN}"
        )
    if config.USE_ANCHOR_MLP:
        banner_lines.append(
            f"{term.paint('anchor', Style.DIM)} "
            f"anchor MLP (hidden={'d' if config.ANCHOR_MLP_HIDDEN <= 0 else config.ANCHOR_MLP_HIDDEN})"
        )
    banner_lines.append(
        f"{term.paint('decoder', Style.DIM)} "
        f"position bound={config.POSITION_BOUND} "
        f"residual={config.POSITION_RESIDUAL}"
    )
    if config.USE_Z_PRED_READOUT_MLP:
        hidden = "d" if config.Z_PRED_READOUT_MLP_HIDDEN <= 0 else config.Z_PRED_READOUT_MLP_HIDDEN
        banner_lines.append(
            f"{term.paint('decoder', Style.DIM)} "
            f"z_pred readout MLP (hidden={hidden})"
        )
    if config.SPLAT_SUBTRACT_SPATIAL_MEAN:
        banner_lines.append(
            f"{term.paint('splat', Style.DIM)} "
            f"subtract spatial mean before U-Net"
        )
    if (
        config.SPLAT_TRUNCATION_SIGMA_FINE != config.SPLAT_TRUNCATION_SIGMA
        or config.SPLAT_FINE_VOXEL_CAP
    ):
        cap = f", cap={config.SPLAT_FINE_VOXEL_RADIUS}vox" if config.SPLAT_FINE_VOXEL_CAP else ""
        banner_lines.append(
            f"{term.paint('splat', Style.DIM)} "
            f"σ fine={config.SPLAT_TRUNCATION_SIGMA_FINE} "
            f"(mid/coarse={config.SPLAT_TRUNCATION_SIGMA}){cap} "
            f"trunc_floor={config.SPLAT_MIN_TRUNC_VOXEL_FRAC}×spacing"
        )
    banner_lines.append(
        f"{term.paint('tb', Style.DIM)}       "
        f"tensorboard --logdir {os.path.join(ckpt_dir, 'tb_logs')}",
    )
    if skipped:
        banner_lines.append(
            term.paint(f"skipped {len(skipped)} graph(s) (0 coarsenable nodes)", Style.YELLOW)
        )
    term.banner("GVAE training", banner_lines)

    train_dataloader = make_dataloader(train_dataset, shuffle=True)
    val_dataloader = make_dataloader(val_dataset, shuffle=False)

    model = GVAE().to(device)
    if init_checkpoint:
        _load_init_checkpoint(model, init_checkpoint, device, term)
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'tb_logs'))

    train(model, train_dataloader, val_dataloader, device, ckpt_dir, writer, term)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))

    writer.close()
    term.ok(f"done — checkpoints in {ckpt_dir}/ (best.pth, last.pth)")

    best_path = os.path.join(ckpt_dir, "best.pth")
    if run_probes and os.path.isfile(best_path):
        from gvae.checkpoint_compat import migrate_state_dict
        model.load_state_dict(
            migrate_state_dict(
                torch.load(best_path, map_location=device, weights_only=True),
            ),
            strict=False,
        )
        _run_probes(
            model, train_dataset.graphs, val_dataset.graphs, device, ckpt_dir, term,
            probe_target=probe_target,
        )
    elif run_probes:
        term.warn("no best.pth saved; skipping probes")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train GVAE")
    parser.add_argument(
        "--ckpt-dir", type=str, default="",
        help="Checkpoint directory (default: checkpoint/<timestamp>)",
    )
    parser.add_argument(
        "--init-checkpoint", type=str, default=None,
        help="Load weights from .pth before training (e.g. phase-1 best.pth); "
        "optimizer/epoch start fresh",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override NUM_EPOCHS")
    parser.add_argument(
        "--splat-sigma-fine", type=float, default=None,
        help="Override SPLAT_TRUNCATION_SIGMA_FINE",
    )
    parser.add_argument(
        "--lambda-recon-zonly", type=float, default=None,
        help="Override LAMBDA_RECON_ZONLY",
    )
    parser.add_argument(
        "--lambda-recon-h", type=float, default=None,
        help="Override LAMBDA_RECON_H",
    )
    parser.add_argument(
        "--lambda-recon-hzonly", type=float, default=None,
        help="Override LAMBDA_RECON_HZONLY (Z-only at h anchors)",
    )
    parser.add_argument(
        "--lambda-anchor-fine", type=float, default=None,
        help="Override LAMBDA_ANCHOR_FINE",
    )
    parser.add_argument(
        "--lambda-anchor-mid", type=float, default=None,
        help="Override LAMBDA_ANCHOR_MID",
    )
    parser.add_argument(
        "--lambda-size", type=float, default=None,
        help="Override LAMBDA_SIZE (h+Z footprint loss in L_recon)",
    )
    parser.add_argument(
        "--lambda-size-zonly", type=float, default=None,
        help="Override LAMBDA_SIZE_ZONLY (Z-only / hz footprint weight)",
    )
    parser.add_argument(
        "--lambda-sem-zonly", type=float, default=None,
        help="Override LAMBDA_SEM_ZONLY (Z-only / hz semantic CE weight)",
    )
    parser.add_argument(
        "--lambda-norm-contrast-coarse", type=float, default=None,
        help="Override LAMBDA_NORM_CONTRAST_COARSE",
    )
    parser.add_argument(
        "--no-anchor-mlp", action="store_true",
        help="Use Linear anchor heads (USE_ANCHOR_MLP=False)",
    )
    parser.add_argument(
        "--no-zpred-readout-mlp", action="store_true",
        help="Linear s/r/p heads on z_pred (USE_Z_PRED_READOUT_MLP=False)",
    )
    parser.add_argument(
        "--position-bound", choices=("clamp", "tanh"), default=None,
        help="Position head: clamp (default) or tanh",
    )
    parser.add_argument(
        "--no-position-residual", action="store_true",
        help="Predict absolute p from Z (POSITION_RESIDUAL=False)",
    )
    parser.add_argument(
        "--no-splat-center", action="store_true",
        help="Disable SPLAT_SUBTRACT_SPATIAL_MEAN before U-Net",
    )
    parser.add_argument(
        "--splat-min-trunc-frac", type=float, default=None,
        help="Override SPLAT_MIN_TRUNC_VOXEL_FRAC",
    )
    parser.add_argument(
        "--lambda-anchor-r-fine", type=float, default=None,
        help="Override LAMBDA_ANCHOR_R_FINE",
    )
    parser.add_argument(
        "--lambda-anchor-r-mid", type=float, default=None,
        help="Override LAMBDA_ANCHOR_R_MID",
    )
    parser.add_argument(
        "--no-anchor-curriculum", action="store_true",
        help="Disable DECODER_GT_ANCHOR_MIX curriculum (fixed mix=0)",
    )
    parser.add_argument(
        "--anchor-mix-anneal-epochs", type=int, default=None,
        help="Override ANCHOR_MIX_ANNEAL_EPOCHS",
    )
    parser.add_argument(
        "--latent-graph-vae", action="store_true",
        help="Honest graph VAE: encode→Z→LatentGraphDecoder→graph_hat (no h at decode)",
    )
    parser.add_argument(
        "--lambda-recon-latent", type=float, default=None,
        help="Override LAMBDA_RECON_LATENT (Z→graph_hat reconstruction)",
    )
    parser.add_argument(
        "--no-zonly-decoder", action="store_true",
        help="Disable Z-only decode path (USE_Z_ONLY_DECODER=False)",
    )
    parser.add_argument(
        "--zonly-aux-loss-only", action="store_true",
        help="Z-only @ GT/anchor for training losses only; deformable mlp_p sets "
        "recon p (Z_ONLY_PATCH_DEFORMABLE_POSITION=False)",
    )
    parser.add_argument(
        "--zonly-jitter", type=float, default=None,
        help="Override Z_ONLY_QUERY_JITTER",
    )
    parser.add_argument(
        "--unet-depth-fine", type=int, default=None,
        help="Override UNET_DEPTH_FINE",
    )
    parser.add_argument(
        "--lambda-norm-contrast-fine", type=float, default=None,
        help="Override LAMBDA_NORM_CONTRAST_FINE",
    )
    parser.add_argument(
        "--lambda-norm-contrast-mid", type=float, default=None,
        help="Override LAMBDA_NORM_CONTRAST_MID",
    )
    parser.add_argument(
        "--no-probe", action="store_true",
        help="Skip latent probes after training (default: probe best.pth once at end)",
    )
    parser.add_argument(
        "--probe-target", choices=("supernode", "instance", "both"), default="both",
        help="Probe sampling: supernode GT, raw instances, or both",
    )
    return parser.parse_args()


def _apply_config_overrides(args) -> None:
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.splat_sigma_fine is not None:
        config.SPLAT_TRUNCATION_SIGMA_FINE = args.splat_sigma_fine
    if args.lambda_recon_zonly is not None:
        config.LAMBDA_RECON_ZONLY = args.lambda_recon_zonly
    if args.lambda_recon_h is not None:
        config.LAMBDA_RECON_H = args.lambda_recon_h
    if args.lambda_recon_hzonly is not None:
        config.LAMBDA_RECON_HZONLY = args.lambda_recon_hzonly
    if args.lambda_anchor_fine is not None:
        config.LAMBDA_ANCHOR_FINE = args.lambda_anchor_fine
    if args.lambda_anchor_mid is not None:
        config.LAMBDA_ANCHOR_MID = args.lambda_anchor_mid
    if args.lambda_size is not None:
        config.LAMBDA_SIZE = args.lambda_size
    if args.lambda_size_zonly is not None:
        config.LAMBDA_SIZE_ZONLY = args.lambda_size_zonly
    if args.lambda_sem_zonly is not None:
        config.LAMBDA_SEM_ZONLY = args.lambda_sem_zonly
    if args.lambda_norm_contrast_coarse is not None:
        config.LAMBDA_NORM_CONTRAST_COARSE = args.lambda_norm_contrast_coarse
    if args.no_anchor_mlp:
        config.USE_ANCHOR_MLP = False
    if args.no_zpred_readout_mlp:
        config.USE_Z_PRED_READOUT_MLP = False
    if args.position_bound is not None:
        config.POSITION_BOUND = args.position_bound
    if args.no_position_residual:
        config.POSITION_RESIDUAL = False
    if args.no_splat_center:
        config.SPLAT_SUBTRACT_SPATIAL_MEAN = False
    if args.splat_min_trunc_frac is not None:
        config.SPLAT_MIN_TRUNC_VOXEL_FRAC = args.splat_min_trunc_frac
    if args.lambda_anchor_r_fine is not None:
        config.LAMBDA_ANCHOR_R_FINE = args.lambda_anchor_r_fine
    if args.lambda_anchor_r_mid is not None:
        config.LAMBDA_ANCHOR_R_MID = args.lambda_anchor_r_mid
    if args.no_anchor_curriculum:
        config.ANCHOR_MIX_CURRICULUM = False
        config.DECODER_GT_ANCHOR_MIX = 0.0
    if args.anchor_mix_anneal_epochs is not None:
        config.ANCHOR_MIX_ANNEAL_EPOCHS = args.anchor_mix_anneal_epochs
    if args.latent_graph_vae:
        config.LATENT_GRAPH_VAE_MODE = True
        config.USE_Z_ONLY_DECODER = False
        config.LAMBDA_RECON_H = 0.0
        config.LAMBDA_RECON_ZONLY = 0.0
        config.LAMBDA_RECON_HZONLY = 0.0
        config.LAMBDA_ANCHOR_FINE = 0.0
        config.LAMBDA_ANCHOR_MID = 0.0
        config.LAMBDA_ANCHOR_COARSE = 0.0
        config.LAMBDA_ANCHOR_R_FINE = 0.0
        config.LAMBDA_ANCHOR_R_MID = 0.0
        config.LAMBDA_ANCHOR_R_COARSE = 0.0
        config.ANCHOR_MIX_CURRICULUM = False
        config.DECODER_GT_ANCHOR_MIX = 0.0
    if args.lambda_recon_latent is not None:
        config.LAMBDA_RECON_LATENT = args.lambda_recon_latent
    if args.no_zonly_decoder:
        config.USE_Z_ONLY_DECODER = False
    if args.zonly_aux_loss_only:
        if args.no_zonly_decoder:
            raise SystemExit("--zonly-aux-loss-only requires Z-only decoder (omit --no-zonly-decoder)")
        config.Z_ONLY_PATCH_DEFORMABLE_POSITION = False
    if args.zonly_jitter is not None:
        config.Z_ONLY_QUERY_JITTER = args.zonly_jitter
    if args.unet_depth_fine is not None:
        config.UNET_DEPTH_FINE = args.unet_depth_fine
    if args.lambda_norm_contrast_fine is not None:
        config.LAMBDA_NORM_CONTRAST_FINE = args.lambda_norm_contrast_fine
    if args.lambda_norm_contrast_mid is not None:
        config.LAMBDA_NORM_CONTRAST_MID = args.lambda_norm_contrast_mid


if __name__ == "__main__":
    args = _parse_args()
    _apply_config_overrides(args)

    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join("checkpoint", timestamp)
        os.makedirs(ckpt_dir, exist_ok=True)
    log_file = _setup_run_logging(ckpt_dir)
    try:
        term = Term()
        term.dim(f"checkpoint  {ckpt_dir}")
        term.dim(f"log         {os.path.join(ckpt_dir, 'train.log')}")
        main(
            ckpt_dir,
            run_probes=not args.no_probe,
            probe_target=args.probe_target,
            init_checkpoint=args.init_checkpoint,
        )
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted.")
        raise
    finally:
        _teardown_run_logging(log_file)
