# train.py — Stage-wise training loop for the Scene Graph VAE
# Stage 1: coarsening only
# Stage 2: + mid encoder branch
# Stage 3: + coarse encoder branch
# Stage 4: joint fine-tune (all modules, reduced LR)
# TODO: implement (todo id: train)

import math
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from gvae.models.gvae import GVAE
from gvae.data.scene_graph import SceneGraph
from gvae.losses.gvae_loss import compute_loss
from gvae.losses.metrics import compute_metrics
from gvae.utils.bn_stats import (
    calibrate_batch_norm,
    init_unet_batch_norm,
    reset_bn_running_stats,
    set_unet_bn_train,
)

class SceneGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        # find all .json files in the folder
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.json') 
        ]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # load the scene graph from the JSON file
        return SceneGraph.from_json(self.files[idx])


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(modules):
    # modules: a list of nn.Module to unfreeze
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True

def _format_scalar(v) -> str:
    """Human-readable loss/metric value (handles nan/inf and large magnitudes)."""
    if isinstance(v, torch.Tensor):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-4):
        return f"{v:.3e}"
    return f"{v:.4f}"


def _finite_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float('nan')


def _average_components(component_dicts: list[dict]) -> dict:
    if not component_dicts:
        return {}
    keys = component_dicts[0].keys()
    out = {}
    for key in keys:
        vals = [c[key] for c in component_dicts]
        finite = [v for v in vals if math.isfinite(v)]
        out[key] = sum(finite) / len(finite) if finite else float('nan')
    return out


def _print_loss_block(title: str, components: dict, metrics: dict | None = None):
    print(title)
    ordered = (
        'recon', 'KL', 'pool', 'occ', 'lambda_kl',
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
        coarse_keys = ('coarse', 'coarse_preview')
        for ck in coarse_keys:
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

def validate(model, loader, stage, step_counter, device):
    model.eval()

    per_graph_losses = []
    all_metrics = []
    all_components = []

    with torch.no_grad():
        for batch in loader:
            for graph in batch:
                graph = graph.to(device)
                if graph.num_coarsenable == 0:
                    step_counter[0] += 1
                    continue
                outputs = model(graph)
                loss, components = compute_loss(outputs, graph, step_counter[0], stage)
                per_graph_losses.append(loss.item())
                all_components.append({k: v.item() if hasattr(v, 'item') else v for k, v in components.items()})

                m = compute_metrics(outputs, stage)
                if m:
                    all_metrics.append(m)

                step_counter[0] += 1

    n_graphs = len(all_components)
    avg_loss = _finite_mean(per_graph_losses)
    avg_components = _average_components(all_components)

    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            finite = [v for v in vals if math.isfinite(v)]
            avg_metrics[key] = sum(finite) / len(finite) if finite else float('nan')
    
    model.train()  # switch back to training mode
    return avg_loss, avg_metrics, avg_components


def _reset_bn_for_stage(encoder, stage: int) -> None:
    """Fresh running stats when a U-Net branch is first optimised."""
    if stage == 2:
        reset_bn_running_stats(encoder.unet_mid)
    if stage == 3:
        reset_bn_running_stats(encoder.unet_coarse)


def train_stage(model, loader, val_loader, num_epochs, stage, step_counter, lr, device, ckpt_dir, writer):
    set_stage(model, stage)
    set_unet_bn_train(model.encoder, stage)
    _reset_bn_for_stage(model.encoder, stage)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr  # use the lr argument, not hardcoded config value
    )

    best_loss = float('inf')  # track best (lowest) average loss seen so far

    for epoch in range(num_epochs):
        per_graph_losses = []
        epoch_components = []
        skipped = 0
        for batch in loader: # batch is a list of graph objects due to collate_fn in dataloader
            # we need to loop over the list of graphs and process them one by one, since our model and loss are not designed for batching multiple graphs together
            optimizer.zero_grad()  # clear gradients before processing the batch
            batch_had_grad = False
            for graph in batch:
                graph = graph.to(device)
                if graph.num_coarsenable == 0:
                    skipped += 1
                    step_counter[0] += 1
                    continue
                set_unet_bn_train(model.encoder, stage)
                outputs = model(graph)
                loss, components = compute_loss(outputs, graph, step_counter[0], stage)  # compute loss for each graph
                if not math.isfinite(loss.item()):
                    skipped += 1
                    step_counter[0] += 1
                    continue
                loss.backward()  # accumulate gradients for each graph
                batch_had_grad = True
                per_graph_losses.append(loss.item())
                epoch_components.append({k: v.item() if hasattr(v, 'item') else v for k, v in components.items()})
                step_counter[0] += 1  # increment global step counter for each graph

            if batch_had_grad:
                optimizer.step()  # update weights once per batch, after all graphs

        n_train = len(epoch_components)
        avg_loss = _finite_mean(per_graph_losses)
        if skipped:
            print(f"  skipped {skipped} graph(s) (no instantiable nodes or non-finite loss)")
        avg_train_components = _average_components(epoch_components)

        if config.BN_CALIBRATE_BEFORE_VAL and stage >= 2:
            n_calib = calibrate_batch_norm(model, loader, device, stage)
            print(f"  BN calibration: {n_calib} training graphs")

        avg_val_loss, avg_metrics, avg_val_components = validate(model, val_loader, stage, step_counter, device)

        print(f"\n── Stage {stage}  Epoch {epoch + 1}/{num_epochs} ──")
        print(f"  total  train={_format_scalar(avg_loss)}   val={_format_scalar(avg_val_loss)}")
        _print_loss_block("  train:", avg_train_components)
        _print_loss_block("  val:  ", avg_val_components, avg_metrics if avg_metrics else None)

        # TensorBoard logging
        global_epoch = step_counter[0]  # use global step so all stages share one x-axis
        writer.add_scalars(f'stage{stage}/total', {'train': avg_loss, 'val': avg_val_loss}, global_epoch)
        for k, v in avg_train_components.items():
            writer.add_scalar(f'stage{stage}/train/{k}', v, global_epoch)
        for k, v in avg_val_components.items():
            writer.add_scalar(f'stage{stage}/val/{k}', v, global_epoch)
        if avg_metrics:
            for k, v in avg_metrics.items():
                writer.add_scalar(f'stage{stage}/metrics/{k}', v, global_epoch)

        # save best model for this stage based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"stage{stage}_best.pth"))
            print(f"  → New best validation loss {best_loss:.4f}, model saved.")
 
    return step_counter


def get_device():
    if config.CUDA_DEVICE is None:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{config.CUDA_DEVICE}')
    print(f"CUDA_DEVICE={config.CUDA_DEVICE} requested but CUDA unavailable; using CPU.")
    return torch.device('cpu')


def main(ckpt_dir):
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset and create dataloader
    train_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'train'))
    val_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'test'))

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)  # collate_fn to return list of graphs
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

    model = GVAE().to(device)
    init_unet_batch_norm(model.encoder)
    print(f"U-Net BatchNorm: momentum={config.BN_MOMENTUM!r}, calibrate_before_val={config.BN_CALIBRATE_BEFORE_VAL}")

    step_counter = [0]

    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'tb_logs'))
    print(f"TensorBoard logs: tensorboard --logdir {os.path.join(ckpt_dir, 'tb_logs')}")

    # Stage 1
    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE1,
                stage=1, step_counter=step_counter, lr=config.LEARNING_RATE, device=device, ckpt_dir=ckpt_dir, writer=writer)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "stage1.pth"))

    # Stage 2
    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE2,
                stage=2, step_counter=step_counter, lr=config.LEARNING_RATE, device=device, ckpt_dir=ckpt_dir, writer=writer)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "stage2.pth"))

    # Stage 3
    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE3,
                stage=3, step_counter=step_counter, lr=config.LEARNING_RATE, device=device, ckpt_dir=ckpt_dir, writer=writer)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "stage3.pth"))

    # Stage 4 — reduced LR for joint fine-tune
    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE4,
                stage=4, step_counter=step_counter, lr=config.LEARNING_RATE_FINETUNE, device=device, ckpt_dir=ckpt_dir, writer=writer)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pth"))

    writer.close()
    print(f"Training complete. Final model saved to {ckpt_dir}/final.pth")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20260513_142301
    ckpt_dir  = os.path.join("checkpoint", timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    main(ckpt_dir)