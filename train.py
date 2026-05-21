# train.py — Single-stage training loop for the Scene Graph VAE
# All branches (fine + mid + coarse); hard FPS coarsening; LR decay mid-run

import math
import os
import sys
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from gvae.training.console import Style, Term, strip_ansi
from gvae.models.gvae import GVAE
from gvae.data.scene_graph import SceneGraph
from gvae.losses.gvae_loss import compute_branch_losses, compute_loss
from gvae.losses.metrics import compute_metrics

TRAIN_STAGE = 1  # full forward (all branches)


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
        outputs = model(graph, stage=TRAIN_STAGE)
        branches, lambda_kl = compute_branch_losses(outputs, graph, step, TRAIN_STAGE)

    zero = graph.p.new_zeros(())
    L_recon = L_KL = L_occ = zero
    for _, _, parts in branches:
        L_recon = L_recon + parts['recon']
        L_KL = L_KL + parts['KL']
        L_occ = L_occ + parts['occ']
    components = {'recon': L_recon, 'KL': L_KL, 'occ': L_occ, 'lambda_kl': lambda_kl}
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

    per_graph_losses = []
    all_metrics = []
    all_components = []
    failed_paths: list[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, **_tqdm_kwargs(desc, "cyan")):
            for graph in batch:
                graph = graph.on_device(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(graph, stage=TRAIN_STAGE)
                    loss, components = compute_loss(
                        outputs, graph, step=step, stage=TRAIN_STAGE,
                    )
                val = loss.item()
                per_graph_losses.append(val)
                if not math.isfinite(val):
                    failed_paths.append(_graph_label(graph))
                all_components.append({
                    k: v.item() if hasattr(v, 'item') else v for k, v in components.items()
                })

                m = compute_metrics(outputs, graph, TRAIN_STAGE, step=step)
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
    return avg_loss, avg_metrics, avg_components, n_graphs, n_failed, failed_paths


def train(model, loader, val_loader, device, ckpt_dir, writer, term: Term):
    use_amp = _use_amp(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    step_counter = 0
    epoch_counter = 0
    best_loss = float('inf')
    current_lr = config.LEARNING_RATE

    for epoch in range(config.NUM_EPOCHS):
        lr = _lr_for_epoch(epoch)
        if lr != current_lr:
            current_lr = lr
            _set_optimizer_lr(optimizer, current_lr)
            tqdm.write(term.paint(
                f"  LR → {current_lr:.1e} (epoch {epoch + 1})",
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


def main(ckpt_dir):
    term = Term()
    device = get_device()
    use_amp = _use_amp(device)

    train_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'train'))
    val_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'test'))

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
        f"fine {config.GRID_FINE} · mid {config.GRID_MID} · coarse {config.GRID_COARSE}",
        f"{term.paint('tb', Style.DIM)}       "
        f"tensorboard --logdir {os.path.join(ckpt_dir, 'tb_logs')}",
    ]
    if skipped:
        banner_lines.append(
            term.paint(f"skipped {len(skipped)} graph(s) (0 coarsenable nodes)", Style.YELLOW)
        )
    term.banner("GVAE training", banner_lines)

    train_dataloader = make_dataloader(train_dataset, shuffle=True)
    val_dataloader = make_dataloader(val_dataset, shuffle=False)

    model = GVAE().to(device)
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'tb_logs'))

    train(model, train_dataloader, val_dataloader, device, ckpt_dir, writer, term)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))

    writer.close()
    term.ok(f"done — checkpoints in {ckpt_dir}/ (best.pth, last.pth)")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("checkpoint", timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = _setup_run_logging(ckpt_dir)
    try:
        term = Term()
        term.dim(f"checkpoint  {ckpt_dir}")
        term.dim(f"log         {os.path.join(ckpt_dir, 'train.log')}")
        main(ckpt_dir)
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted.")
        raise
    finally:
        _teardown_run_logging(log_file)
