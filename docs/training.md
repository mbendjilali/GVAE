# Training

How to set up the environment, run training, read the logs, and check whether `Z` is good enough for diffusion handoff.

---

## Environment

**Recommended:** conda environment from `environment.yml`:

```bash
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
export PIP_FIND_LINKS=https://data.pyg.org/whl/torch-2.6.0+cu124.html
mamba env create -f environment.yml
conda activate gvae
pip install tensorboard   # if not already installed
```

Alternative: `pip install -r requirements.txt` with matching PyTorch Geometric wheel links (see comments in that file).

**GPU:** set `CUDA_DEVICE` in `config.py` (e.g. `1`). Use `None` for CPU (slow).

---

## Data layout

Training reads split folders:

```
data/graphs/
├── train/*.json          # training scenes
└── test/*.json           # validation scenes
```

Each JSON needs three occupancy sidecars (built by `utils/build_scene_graph.py`):

```
{scene}_occ_fine.npy
{scene}_occ_mid.npy
{scene}_occ_coarse.npy
```

See [data.md](data.md) for the full build pipeline. Scenes with **zero coarsenable objects** are skipped automatically.

---

## Run training

```bash
python train.py
```

Outputs go to `checkpoint/{timestamp}/` unless you pass `--ckpt-dir`:

| File | Purpose |
|------|---------|
| `best.pth` | **Use this** — weights at lowest validation loss |
| `last.pth` | Weights after the final epoch |
| `train.log` | Full console log (ANSI stripped) |
| `tb_logs/` | TensorBoard scalars |

Watch training:

```bash
tensorboard --logdir checkpoint/<run>/tb_logs
```

**Schedule:** single-stage — fine, mid, and coarse branches all train every epoch. Learning rate drops at epoch `LR_DECAY_EPOCH + 1` (default: 41) from `3e-4` to `1e-4`. Default run length: `150` epochs.

### Example ablation (25 epochs, all occupancy levels)

```bash
python train.py --epochs 25 --ckpt-dir checkpoint/my_ablation \
  --lambda-occ-grid-fine 1 --lambda-occ-grid-mid 1 --lambda-occ-grid-coarse 1
```

---

## Command-line overrides

These override `config.py` without editing the file:

| Flag | Config variable | Purpose |
|------|-----------------|---------|
| `--ckpt-dir PATH` | — | Output directory |
| `--epochs N` | `NUM_EPOCHS` | Training length |
| `--lambda-occ-grid-fine F` | `LAMBDA_OCC_GRID_FINE` | Fine occupancy loss (0 = off) |
| `--lambda-occ-grid-mid M` | `LAMBDA_OCC_GRID_MID` | Mid occupancy loss |
| `--lambda-occ-grid-coarse C` | `LAMBDA_OCC_GRID_COARSE` | Coarse occupancy loss |
| `--splat-sigma-fine S` | `SPLAT_TRUNCATION_SIGMA_FINE` | Fine splat sharpness |
| `--lambda-recon-h H` | `LAMBDA_RECON_H` | h+Z decoder loss weight |
| `--lambda-recon-zonly Z` | `LAMBDA_RECON_ZONLY` | Z-only decoder loss weight |
| `--no-zonly-decoder` | `USE_Z_ONLY_DECODER=False` | Disable Z-only path |
| `--zonly-jitter J` | `Z_ONLY_QUERY_JITTER` | Train-time query noise |
| `--unet-depth-fine D` | `UNET_DEPTH_FINE` | Fine U-Net depth |
| `--lambda-norm-contrast-fine F` | `LAMBDA_NORM_CONTRAST_FINE` | Fine norm contrastive loss |
| `--lambda-norm-contrast-mid M` | `LAMBDA_NORM_CONTRAST_MID` | Mid norm contrastive loss |

Norm contrastive weights are **not** exposed separately on the CLI beyond the two flags above; margin and empty-point count stay in `config.py` (`NORM_CONTRAST_MARGIN`, `NORM_CONTRAST_EMPTY_POINTS`).

---

## Loading a checkpoint

```python
import torch
from gvae.models.gvae import GVAE

model = GVAE()
model.load_state_dict(torch.load("checkpoint/<run>/best.pth", map_location="cpu"))
model.eval()
```

**Compatibility:** checkpoints from older runs may fail to load after architecture changes (fine coarsening, `OccGridHead`, Z-only decoders, U-Net depth, …). Retrain or filter `state_dict` keys when migrating.

---

## Reading the training console

Each epoch prints train/val loss, then a **metrics** line. Example shape:

```
Epoch 42/150  lr=3.0e-04  train 7.21  val 7.85
  │ metrics  fine pos=0.18 smiou=84% zpos=0.08 occ=50%  ·  mid inst=0.17 zpos=0.08 smiou=… occ=64%
```

| Console label | TensorBoard key | Meaning |
|---------------|-----------------|---------|
| `pos=` | `pos_err_fine` | h+Z decoder — can the model **find** fine supernodes? |
| `zpos=` | `pos_err_zonly_fine` | Z-only decoder — given GT slot, can `Z` refine position? |
| `smiou=` | `soft_miou_fine` | Semantic reconstruction quality |
| `occ=` | `occ_iou_fine` | Voxel occupancy vs LiDAR |
| `inst=` | `inst_pos_err_mid` | Instance positions via coarsening chain |

**Do not confuse** `pos=` (localization) with `zpos=` (slot-conditioned readout). Both should go down, but they measure different things — see [architecture.md](architecture.md#two-decoders-why-two).

### Occupancy IoU plateau

Fine `occ_iou` often plateaus around ~50% even when training is healthy. The model may over-predict occupied voxels relative to sparse LiDAR GT; IoU is capped by pred/gt rate ratio. Mid occupancy usually looks better (~60%+). Use probes and visual inspection for occ sanity, not IoU alone.

---

## Latent probes

After training, run **offline probes** on `best.pth`:

```bash
python utils/probe_latent.py --checkpoint checkpoint/<run>/best.pth \
  -o checkpoint/<run>/probe_report.txt
```

Three probe families (report sections A / B / C):

| Probe | What it tests | What “good” looks like |
|-------|-------------|------------------------|
| **A. Anchor ablation** | h-decoder position vs GT anchor mix (0 → 1) | Low `pos_err` at mix=0 means `h` localizes without cheating |
| **B. Linear Z→p** | Linear map from `Z(p_gt)` to position | Low `linear_pos_err_val` means position is linearly readable in `Z` |
| **C. Signal vs background** | `‖Z(p_gt)‖` vs `‖Z(empty)‖` | `norm_ratio_gt_over_empty` > 1 — peaks at true object sites |

These are the **decision metrics** for DDM handoff quality. Training also applies a **norm contrastive loss** (Probe C alignment) on fine and mid when `LAMBDA_NORM_CONTRAST_* > 0`.

Probe options:

```bash
python utils/probe_latent.py --help
# --linear-epochs 500   more training for linear head
# --anchor-mixes 0,0.5,1   custom anchor ablation grid
```

---

## Key config knobs (in `config.py`)

| Knob | Default | Effect |
|------|---------|--------|
| `USE_Z_ONLY_DECODER` | `True` | Enable Z-only readout path |
| `LAMBDA_RECON_H` / `LAMBDA_RECON_ZONLY` | `1.0` / `1.2` | Balance h+Z vs Z-only reconstruction |
| `Z_ONLY_QUERY_JITTER` | `0.05` | Train-time noise on Z sample locations |
| `UNET_DEPTH_FINE` | `1` | Shallow fine U-Net (less spatial blur) |
| `LAMBDA_NORM_CONTRAST_FINE/MID` | `0.1` | Norm contrastive (Probe C) |
| `DECODER_GT_ANCHOR_MIX` | `0.0` | **h-decoder only** — blend GT into anchors (probes; keep 0 for training) |
| `REDUCTION_RATIO_LEVELS` | `[0.2, 0.2, 0.2]` | FPS keep ratio per coarsening step |
| `LAMBDA_OCC_GRID_*` | `1.0` | Occupancy BCE per level |
| `SPLAT_TRUNCATION_SIGMA_FINE` | `1.0` | Sharper fine splat (mid/coarse use 2.0) |
| `GRAD_CLIP_NORM` | `1.0` | Gradient clipping (`0` = off) |
| `LOG_FULL_METRICS` | `True` | Extra TensorBoard metrics |

---

## Stability notes

- **GroupNorm** in the U-Net — required because effective batch size is small.
- **`COARSEN_DETACH_FEATURES`** — in soft coarsening mode, detaches assignments when pooling features (pool loss still trains temperature).
- Empty edge warnings on sparse graphs are normal.
- **AMP** (`USE_AMP=True`) speeds training on CUDA; sequential backward per branch is disabled under AMP.

---

## Utility scripts

| Script | Purpose |
|--------|---------|
| `utils/build_scene_graph.py` | LAZ → JSON + occ caches |
| `utils/probe_latent.py` | Anchor / linear / signal probes |
| `utils/visualize_supernodes.py` | Export coarsening assignments to JSON + LAS |
| `utils/diagnose_nan_losses.py` | Per-graph loss breakdown |
| `utils/metrics_sanity.py` | Oracle checks on metric definitions |
| `utils/smoke_test.py` | Quick forward + backward |
| `utils/profile_scene.py` | Timing on heavy scenes |

---

## Next work

See [TODO.md](../TODO.md) for ablation matrix, probe automation, and dataset tasks.
