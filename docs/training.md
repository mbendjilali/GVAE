# Training

How to set up the environment, run training, read the logs, and check whether `Z` is good enough for diffusion handoff.

## Latent graph VAE (honest Z → graph_hat)

```bash
python train.py --ckpt-dir checkpoint/latent_graph_vae \
  --latent-graph-vae \
  --lambda-recon-latent 1.0
```

Decode uses **only** `Z` and supernode **count** (no `h`, no GT `p` at decode). Val metrics `pos` / `size` / `smiou` are `graph_hat` vs GT.

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

### Localization recipe (Jun 2026 best)

Use a **fresh** `--ckpt-dir` per run. See [localization-progress.md](localization-progress.md) for the three-step arc and metrics.

```bash
python train.py --ckpt-dir checkpoint/anchor_v2 \
  --lambda-recon-h 1.5 --lambda-recon-zonly 1.0 --lambda-recon-hzonly 0.5 \
  --lambda-anchor-fine 2.5 --lambda-anchor-mid 1.0
```

At end of training, probes run automatically on `best.pth` → `probe_report.txt`, `probe_summary.json`. Pass `--no-probe` to skip; `--probe-target {supernode,instance,both}` (default `both`).

### Example ablation (25 epochs, norm contrast off)

```bash
python train.py --epochs 25 --ckpt-dir checkpoint/my_ablation \
  --lambda-norm-contrast-fine 0 --lambda-norm-contrast-mid 0
```

Use a fresh `--ckpt-dir` per ablation; reusing a directory overwrites the
previous run.

---

## Command-line overrides

These override `config.py` without editing the file:

| Flag | Config variable | Purpose |
|------|-----------------|---------|
| `--ckpt-dir PATH` | — | Output directory |
| `--epochs N` | `NUM_EPOCHS` | Training length |
| `--latent-graph-vae` | `LATENT_GRAPH_VAE_MODE` | Honest Z→graph_hat decode (no h, no occ head) |
| `--lambda-recon-latent` | `LAMBDA_RECON_LATENT` | Weight on latent-graph reconstruction |
| `--splat-sigma-fine S` | `SPLAT_TRUNCATION_SIGMA_FINE` | Fine splat sharpness |
| `--lambda-recon-h` / `--lambda-recon-zonly` / `--lambda-recon-hzonly` | recon weights |
| `--lambda-anchor-fine` / `--lambda-anchor-mid` | anchor supervision |
| `--no-anchor-curriculum` | disable GT anchor mix schedule |
| `--anchor-mix-anneal-epochs N` | curriculum length |
| `--no-zonly-decoder` | `USE_Z_ONLY_DECODER=False` | Disable Z-only path (restores deformable `mlp_p`) |
| `--zonly-jitter J` | `Z_ONLY_QUERY_JITTER` | Train-time query noise |
| `--unet-depth-fine D` | `UNET_DEPTH_FINE` | Fine U-Net depth |
| `--lambda-norm-contrast-fine F` | `LAMBDA_NORM_CONTRAST_FINE` | Fine norm contrastive loss |
| `--lambda-norm-contrast-mid M` | `LAMBDA_NORM_CONTRAST_MID` | Mid norm contrastive loss |
| `--no-probe` | — | Skip end-of-run latent probes (default: run once on `best.pth`) |
| `--probe-target` | `both` | Probe sampling: `supernode`, `instance`, or `both` |

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

**Compatibility:** `migrate_state_dict` drops legacy `occ_grid_head_*` keys. Retrain after major architecture changes (Z-only decoders, latent-graph VAE, U-Net depth, …).

---

## Reading the training console

Each epoch prints train/val loss, then a **metrics** line. Example shape:

```
Epoch 132/150  lr=1.0e-04  train 11.99  val 11.24  ★ best
  │ metrics  fine pos=0.102 smiou=76% zpos=0.082 zsmiou=42% anc=0.158 hzpos=0.102  ·  mid inst=0.127 zpos=0.076 anc=0.133 hzpos=0.100 smiou=75%
```

| Console label | TensorBoard key | Meaning |
|---------------|-----------------|---------|
| `pos=` | `pos_err_fine` | Final fine position (Z@anchor by default) — **localization** |
| `hzpos=` | `pos_err_zonly_hanchor_fine` | Z-only at h anchor; equals `pos=` when Z-only decoder is enabled |
| `zpos=` | `pos_err_zonly_fine` | Z-only at **GT slot** (oracle layout) |
| `anc=` | `anchor_err_fine` | Anchor vs GT before Z refinement — main bottleneck |
| `smiou=` / `zsmiou=` | `soft_miou_fine` / `soft_miou_zonly_fine` | Semantics: deformable h+Z vs Z-only @ GT |
| `inst=` | `inst_pos_err_mid` | Instance positions via coarsening chain |

With default config, **`pos` = `hzpos`**; compare both to **`anc`** to see how much Z refines a misplaced anchor. **`zpos`** is the oracle ceiling when layout slots are known. See [architecture.md](architecture.md#two-decoders-why-two) and [localization-progress.md](localization-progress.md).

## Latent probes

**Default:** `train.py` runs probes once on `best.pth` at end of training → `probe_report.txt`, `probe_summary.json` in the checkpoint dir. Re-run manually:

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
| `USE_Z_ONLY_DECODER` | `True` | Z-only readout; h+Z `pos` comes from ZOnlyDecoder at `p_anchor` |
| `LAMBDA_RECON_H` / `LAMBDA_RECON_ZONLY` / `LAMBDA_RECON_HZONLY` | `1.5` / `1.0` / `0.8` | h+Z recon / Z-only @ GT / Z-only @ h anchors |
| `LAMBDA_ANCHOR_FINE` / `MID` | `1.0` / `0.5` | MSE on `bound(mlp_p_anchor(h))` vs GT (`POSITION_BOUND=clamp` by default) |
| `ANCHOR_MIX_CURRICULUM` | `True` | Train h-decoder with GT anchor mix 1→0 over 40 ep |
| `Z_ONLY_QUERY_JITTER` | `0.05` | Train-time noise on Z sample locations |
| `UNET_DEPTH_FINE` | `3` | Fine U-Net depth (depth=1 regresses fine `zpos`; ablation confirmed) |
| `LAMBDA_NORM_CONTRAST_FINE/MID` | `0.1` | Norm contrastive (Probe C) |
| `DECODER_GT_ANCHOR_MIX` | `0.0` | **h-decoder only** — blend GT into anchors (probes; keep 0 for training) |
| `REDUCTION_RATIO_LEVELS` | `[0.2, 0.2, 0.2]` | FPS keep ratio per coarsening step |
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
| `utils/visualize_recon.py` | GT vs h+Z vs Z-only reconstruction (BEV) |
| `utils/visualize_supernodes.py` | Export coarsening assignments to JSON + LAS |
| `utils/diagnose_nan_losses.py` | Per-graph loss breakdown |
| `utils/metrics_sanity.py` | Oracle checks on metric definitions |
| `utils/smoke_test.py` | Quick forward + backward |
| `utils/profile_scene.py` | Timing on heavy scenes |

---

## Next work

- **Localization:** anchor MLP head — see [localization-progress.md](localization-progress.md)
- **Backlog / ablations:** [TODO.md](../TODO.md)
