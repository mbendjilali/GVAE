# Scene Graph VAE (GVAE)

A neural encoder that turns **3D outdoor scene graphs** (cars, poles, trees, …) into **three dense 3D latent volumes** — `Z_fine`, `Z_mid`, and `Z_coarse`. A downstream **diffusion model (DDM)** can then sample or edit layout using those volumes.

Think of it like compressing a city block into three maps at different zoom levels: fine detail, neighborhood scale, and whole-scene envelope.

---

## What you need to know first

| Term | Meaning |
|------|---------|
| **Scene graph** | A list of objects (nodes) with position, size, and class, plus edges between nearby objects |
| **Supernode** | A group of objects merged by coarsening — one representative point per group |
| **Latent volume `Z`** | A 3D grid of learned feature vectors (like a 3D image the model can read and write) |
| **Splatting** | Spreading each node's features onto nearby voxels in that grid |
| **Decoder** | A small network that reads `Z` and predicts object attributes (class, position, size) |
| **DDM handoff** | The diffusion model uses `Z` (and known layout slots), not the full graph encoder |

---

## Documentation

Read in this order if you are new:

| # | Doc | What it covers |
|---|-----|----------------|
| 1 | [docs/data.md](docs/data.md) | Input files: JSON graphs + occupancy caches |
| 2 | [docs/architecture.md](docs/architecture.md) | How the model works (encoder, decoders, losses) |
| 3 | [docs/training.md](docs/training.md) | How to train, read metrics, run probes |
| 4 | [docs/localization-progress.md](docs/localization-progress.md) | Jun 2026 `pos` / anchor experiment recap |
| 5 | [TODO.md](TODO.md) | Backlog and ablation recipes |

Older drafts in [docs/archive/](docs/archive/) are **not** up to date.

---

## Quick start

```bash
# 1. Create the conda environment (CUDA 12.4 example — see docs/training.md)
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
export PIP_FIND_LINKS=https://data.pyg.org/whl/torch-2.6.0+cu124.html
mamba env create -f environment.yml && conda activate gvae

# 2. Build scene graphs + occupancy caches from LiDAR
python utils/build_scene_graph.py <data_root>

# 3. Train (writes checkpoint/<timestamp>/)
python train.py

# 4. (Optional) Re-run probes or visualize recon
python utils/probe_latent.py --checkpoint checkpoint/<run>/best.pth \
  -o checkpoint/<run>/probe_report.txt
python utils/visualize_recon.py --checkpoint checkpoint/<run>/best.pth --graph data/graphs/test/5080_54400.json
```

Most hyperparameters live in `config.py`. Common overrides on the command line:

```bash
python train.py --epochs 25 --ckpt-dir checkpoint/my_run \
  --lambda-occ-grid-fine 1 --lambda-occ-grid-mid 1 --lambda-occ-grid-coarse 1
```

Full flag list: [docs/training.md](docs/training.md#command-line-overrides).

Set `CUDA_DEVICE` in `config.py` to pick a GPU (`None` = CPU).

---

## Model outputs

Three **independent** voxel grids (not nested sub-volumes):

| Latent | Grid (H×W×D) | Channels | Role |
|--------|----------------|----------|------|
| `Z_fine` | 64×64×8 | 72 | Finest layout (fine supernodes) |
| `Z_mid` | 32×32×8 | 144 | Regional layout |
| `Z_coarse` | 16×16×4 | 288 | Scene-scale envelope |

Each scene also has LiDAR occupancy sidecars aligned to those grids:

- `{scene}_occ_fine.npy`, `_occ_mid.npy`, `_occ_coarse.npy`

Use **`best.pth`** (lowest validation loss), not `last.pth`. Old checkpoints may not load after architecture changes (U-Net depth, Z-only decoders, etc.).

---

## Repository layout

```
gvae/           models, losses, data loaders, probes
train.py        training loop
config.py       hyperparameters (single source of truth)
utils/          build graphs, probes, diagnostics, visualization
data/graphs/    train/ and test/ scene JSON + occ sidecars
checkpoint/     run outputs (best.pth, train.log, probe_report.txt, …)
docs/           documentation
```
