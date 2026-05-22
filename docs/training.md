# Training

## Environment

**Recommended:** conda env from `environment.yml` (CUDA 12.4 example):

```bash
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
export PIP_FIND_LINKS=https://data.pyg.org/whl/torch-2.6.0+cu124.html
mamba env create -f environment.yml
conda activate gvae
pip install tensorboard   # if not already present
```

Alternative: `pip install -r requirements.txt` with matching PyG wheel links (see comments in that file).

Set GPU in `config.py`: `CUDA_DEVICE = 1` (or `None` for CPU).

---

## Data layout

Training expects split directories:

```
data/graphs/
├── train/*.json
└── test/*.json
```

Each JSON needs matching occupancy sidecars (built by `utils/build_scene_graph.py`):

- `{scene}_occ_fine.npy`
- `{scene}_occ_mid.npy`
- `{scene}_occ_coarse.npy`

See [data.md](data.md) for the build pipeline.

---

## Run training

```bash
python train.py
```

Short ablation (example — grid occ on all levels, 25 epochs):

```bash
python train.py --epochs 25 --ckpt-dir checkpoint/my_ablation \\
  --lambda-occ-grid-fine 1 --lambda-occ-grid-mid 1 --lambda-occ-grid-coarse 1
```

- Checkpoints and logs: `checkpoint/{timestamp}/` (or `--ckpt-dir`)
- TensorBoard: `tensorboard --logdir checkpoint/<run>/tb_logs`

Single-stage training: all branches (fine + mid + coarse) run every epoch. LR decays at `LR_DECAY_EPOCH` (default epoch 41).

---

## Checkpoints

| File | Use |
|------|-----|
| `best.pth` | **Use this** — lowest val total loss |
| `last.pth` | Final epoch weights |

Loading for inference or probes:

```python
import torch
from gvae.models.gvae import GVAE

model = GVAE()
model.load_state_dict(torch.load("checkpoint/<run>/best.pth", map_location="cpu"))
model.eval()
```

Checkpoints are **not compatible** across major architecture changes (e.g. adding fine coarsening or `OccGridHead`).

---

## Interpreting metrics

Primary val metrics (console + TensorBoard):

| Metric | Good direction | Notes |
|--------|----------------|-------|
| `pos_err_fine` | ↓ | Fine supernode positions (after S0 coarsening) |
| `miou_fine` | ↑ | Hard mIoU on fine supernode labels |
| `occ_iou_fine` | ↑ | Query readout vs LiDAR cache |
| `inst_pos_err_mid` | ↓ | Instance → S0 → S1 → mid decode chain |
| `pos_err_mid` | ↓ | Z-only mid supernode positions |
| `occ_iou_mid` / `occ_precision_mid` | ↑ | Occupancy vs LiDAR cache |
| `soft_miou_mid` | ↑ (slow) | Diagnostic only on merged supernode labels |

When grid occ is enabled (`LAMBDA_OCC_GRID_* > 0`), also watch `occ_grid_iou_*`.

Run latent probes after training:

```bash
python utils/probe_latent.py --checkpoint checkpoint/<run>/best.pth -o checkpoint/<run>/probe_report.txt
```

---

## Key config knobs

| Knob | Default | Effect |
|------|---------|--------|
| `DECODER_GT_ANCHOR_MIX` | `0.0` | Z-only decoder anchors |
| `REDUCTION_RATIO_LEVELS` | `[0.2, 0.2, 0.2]` | FPS keep ratio per coarsening step |
| `LAMBDA_OCC_GRID_*` | `0.0` | Voxel-aligned occ BCE on Z (per level) |
| `SPLAT_TRUNCATION_SIGMA_FINE` | `1.0` | Sharper fine splat vs mid/coarse (2.0) |
| `GRAD_CLIP_NORM` | `1.0` | Gradient clipping (0 = off) |
| `NUM_EPOCHS` | `150` | Override via `--epochs` |
| `LOG_FULL_METRICS` | `True` | Extended TensorBoard metrics |

---

## Stability notes

- **GroupNorm** in U-Net (not BatchNorm) — required for batch=1.
- **`COARSEN_DETACH_FEATURES`** — detaches soft assignment on feature pooling in soft mode.
- Empty `edge_index` warnings on sparse graphs are benign.
- Training skips graphs with zero coarsenable nodes.

---

## Utilities

| Script | Purpose |
|--------|---------|
| `utils/build_scene_graph.py` | LAZ → JSON + occ caches |
| `utils/probe_latent.py` | Anchor / linear / signal probes on a checkpoint |
| `utils/visualize_supernodes.py` | Export S0/S1/S2 assignments to JSON + LAS |
| `utils/diagnose_nan_losses.py` | Per-graph loss breakdown |
| `utils/smoke_test.py` | Quick forward + backward pass |
| `utils/profile_scene.py` | Timing on heaviest scenes |

---

## Next work

See [TODO.md](../TODO.md) for Z-localization ablations and probe integration.
