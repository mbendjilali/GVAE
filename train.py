# train.py — Stage-wise training loop for the Scene Graph VAE
# Stage 1: coarsening only
# Stage 2: + mid encoder branch
# Stage 3: + coarse encoder branch
# Stage 4: joint fine-tune (all modules, reduced LR)

import math
import os
import sys
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from gvae.models.gvae import GVAE
from gvae.data.scene_graph import SceneGraph
from gvae.losses.gvae_loss import compute_loss
from gvae.losses.metrics import compute_metrics


class SceneGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.files = []
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.json'):
                continue
            path = os.path.join(data_dir, f)
            graph = SceneGraph.from_json(path)
            if graph.num_coarsenable == 0:
                print(f"  dataset: skip {f} (0 coarsenable nodes)")
                continue
            self.files.append(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        graph = SceneGraph.from_json(path)
        graph.source_path = path
        return graph


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(modules):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True


def _format_scalar(v) -> str:
    if isinstance(v, torch.Tensor):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-4):
        return f"{v:.3e}"
    return f"{v:.4f}"


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


def _print_loss_block(title: str, components: dict, metrics: dict | None = None,
                        n_graphs: int = 0, n_failed: int = 0):
    print(title, end='')
    if n_graphs:
        print(f"  ({n_graphs - n_failed}/{n_graphs} graphs finite)", end='')
    print()
    ordered = (
        'recon', 'KL', 'pool', 'pool_cut', 'pool_ortho', 'pool_spatial', 'occ', 'lambda_kl',
        'recon_coarse_preview', 'KL_coarse_preview', 'occ_coarse_preview',
    )
    for key in ordered:
        if key in components:
            print(f"    {key:10s} {_format_scalar(components[key])}")
    extra = [k for k in components if k not in ordered]
    for key in extra:
        print(f"    {key:10s} {_format_scalar(components[key])}")
    if metrics:
        print("    metrics:")
        print(f"      mid    acc={metrics.get('acc_mid', 0):.1%}  "
              f"pos={_format_scalar(metrics.get('pos_err_mid', 0))}  "
              f"size={_format_scalar(metrics.get('size_err_mid', 0))}")
        for ck in ('coarse', 'coarse_preview'):
            if f'acc_{ck}' in metrics:
                print(f"      {ck:16s} acc={metrics.get(f'acc_{ck}', 0):.1%}  "
                      f"pos={_format_scalar(metrics.get(f'pos_err_{ck}', 0))}  "
                      f"size={_format_scalar(metrics.get(f'size_err_{ck}', 0))}")
                break


def set_stage(model, stage):
    freeze_all(model)
    enc = model.encoder

    if stage >= 4:
        unfreeze([model])
        return

    if stage >= 1:
        unfreeze([enc.coarsen_1, enc.coarsen_2])

    if stage >= 2:
        unfreeze([enc.input_proj_L, enc.gps_L, enc.splat_mid,
                  enc.input_proj_L1, enc.gps_L1, enc.unet_mid,
                  model.decoder_mid, model.occ_readout_mid])

    if stage >= 3:
        unfreeze([enc.input_proj_1, enc.gps_1,
                  enc.splat_coarse, enc.unet_coarse,
                  model.decoder_coarse, model.occ_readout_coarse])


def validate(model, loader, stage, device):
    model.eval()

    per_graph_losses = []
    all_metrics = []
    all_components = []
    failed_paths: list[str] = []

    with torch.no_grad():
        for batch in loader:
            for graph in batch:
                graph = graph.to(device)
                outputs = model(graph)
                loss, components = compute_loss(outputs, graph, step=0, stage=stage)
                val = loss.item()
                per_graph_losses.append(val)
                if not math.isfinite(val):
                    failed_paths.append(_graph_label(graph))
                all_components.append({
                    k: v.item() if hasattr(v, 'item') else v for k, v in components.items()
                })

                m = compute_metrics(outputs, stage)
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


def train_stage(model, loader, val_loader, num_epochs, stage, step_counter, epoch_counter, lr, device, ckpt_dir, writer):
    set_stage(model, stage)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    best_loss = float('inf')

    for epoch in range(num_epochs):
        per_graph_losses = []
        epoch_components = []

        for batch in loader:
            optimizer.zero_grad()
            batch_had_grad = False
            for graph in batch:
                graph = graph.to(device)
                outputs = model(graph)
                loss, components = compute_loss(outputs, graph, step_counter[0], stage)
                val = loss.item()
                if not math.isfinite(val):
                    raise RuntimeError(
                        f"Non-finite training loss on {_graph_label(graph)}: "
                        f"{ {k: (v.item() if hasattr(v, 'item') else v) for k, v in components.items()} }"
                    )
                loss.backward()
                batch_had_grad = True
                per_graph_losses.append(val)
                epoch_components.append({
                    k: v.item() if hasattr(v, 'item') else v for k, v in components.items()
                })
                step_counter[0] += 1

            if batch_had_grad:
                optimizer.step()

        n_train = len(per_graph_losses)
        avg_loss, n_train_failed = _strict_mean(per_graph_losses)
        avg_train_components = _average_components(epoch_components)

        avg_val_loss, avg_metrics, avg_val_components, n_val, n_val_failed, val_failed = validate(
            model, val_loader, stage, device,
        )

        print(f"\n── Stage {stage}  Epoch {epoch + 1}/{num_epochs} ──")
        print(f"  total  train={_format_scalar(avg_loss)}   val={_format_scalar(avg_val_loss)}")
        _print_loss_block("  train:", avg_train_components, n_graphs=n_train, n_failed=n_train_failed)
        _print_loss_block("  val:  ", avg_val_components, avg_metrics if avg_metrics else None,
                          n_graphs=n_val, n_failed=n_val_failed)
        if val_failed:
            print(f"  val non-finite scenes: {', '.join(val_failed)}")

        global_epoch = step_counter[0]
        writer.add_scalars(f'stage{stage}/total', {'train': avg_loss, 'val': avg_val_loss}, global_epoch)
        for k, v in avg_train_components.items():
            writer.add_scalar(f'stage{stage}/train/{k}', v, global_epoch)
        for k, v in avg_val_components.items():
            writer.add_scalar(f'stage{stage}/val/{k}', v, global_epoch)
        if avg_metrics:
            for k, v in avg_metrics.items():
                writer.add_scalar(f'stage{stage}/metrics/{k}', v, global_epoch)

        # TensorBoard logging — per epoch (x-axis = global epoch across all stages)
        epoch_counter[0] += 1
        writer.add_scalars(f'stage{stage}_epoch/total', {'train': avg_loss, 'val': avg_val_loss}, epoch_counter[0])
        for k, v in avg_train_components.items():
            writer.add_scalar(f'stage{stage}_epoch/train/{k}', v, epoch_counter[0])
        for k, v in avg_val_components.items():
            writer.add_scalar(f'stage{stage}_epoch/val/{k}', v, epoch_counter[0])
        if avg_metrics:
            for k, v in avg_metrics.items():
                writer.add_scalar(f'stage{stage}_epoch/metrics/{k}', v, epoch_counter[0])

        if math.isfinite(avg_val_loss) and avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"stage{stage}_best.pth"))
            print(f"  → New best validation loss {best_loss:.4f}, model saved.")
        elif not math.isfinite(avg_val_loss):
            print("  → Val loss non-finite; checkpoint not updated.")

    return step_counter


def get_device():
    if config.CUDA_DEVICE is None:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{config.CUDA_DEVICE}')
    print(f"CUDA_DEVICE={config.CUDA_DEVICE} requested but CUDA unavailable; using CPU.")
    return torch.device('cpu')


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
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
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_file


def _teardown_run_logging(log_file):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if log_file and not log_file.closed:
        log_file.write(f"\nRun finished {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.close()


def main(ckpt_dir):
    device = get_device()
    print(f"Using device: {device}")

    train_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'train'))
    val_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'test'))
    print(f"Dataset: {len(train_dataset)} train graphs, {len(val_dataset)} val graphs")
    print(f"U-Net normalization: GroupNorm (num_groups={config.UNET_NUM_GROUPS})")

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=lambda x: x,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: x,
    )

    model = GVAE().to(device)
    step_counter = [0]
    epoch_counter = [0]

    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'tb_logs'))
    print(f"TensorBoard logs: tensorboard --logdir {os.path.join(ckpt_dir, 'tb_logs')}")

    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE1,
                stage=1, step_counter=step_counter, epoch_counter=epoch_counter, lr=config.LEARNING_RATE,
                device=device, ckpt_dir=ckpt_dir, writer=writer)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "stage1.pth"))

    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE2,
                stage=2, step_counter=step_counter, epoch_counter=epoch_counter, lr=config.LEARNING_RATE,
                device=device, ckpt_dir=ckpt_dir, writer=writer)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "stage2.pth"))

    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE3,
                stage=3, step_counter=step_counter, epoch_counter=epoch_counter, lr=config.LEARNING_RATE,
                device=device, ckpt_dir=ckpt_dir, writer=writer)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "stage3.pth"))

    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE4,
                stage=4, step_counter=step_counter, epoch_counter=epoch_counter, lr=config.LEARNING_RATE_FINETUNE,
                device=device, ckpt_dir=ckpt_dir, writer=writer)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pth"))

    writer.close()
    print(f"Training complete. Final model saved to {ckpt_dir}/final.pth")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("checkpoint", timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = _setup_run_logging(ckpt_dir)
    try:
        print(f"Checkpoints will be saved to: {ckpt_dir}")
        print(f"Terminal log: {os.path.join(ckpt_dir, 'train.log')}")
        main(ckpt_dir)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise
    finally:
        _teardown_run_logging(log_file)
