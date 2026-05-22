# Scene Graph VAE (GVAE)

Encodes 3D outdoor scene graphs into three KL-regularised spatial latent volumes (`Z_fine`, `Z_mid`, `Z_coarse`) for hierarchical diffusion conditioning.

Each 500 m tile is processed through instance R-GAT, three FPS coarsening steps (instances → fine → mid → coarse supernodes), Gaussian splatting, and 3D U-Net encoders. Training is **single-stage** (all branches active every epoch).

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/architecture.md](docs/architecture.md) | Encoder, coarsening, splatting, decoder, losses |
| [docs/training.md](docs/training.md) | Setup, training, checkpoints, metrics, probes |
| [docs/data.md](docs/data.md) | Scene graphs, LiDAR, occupancy caches |
| [TODO.md](TODO.md) | Active backlog |

Superseded drafts are in [docs/archive/](docs/archive/) — do not use them as source of truth.

---

## Quick start

```bash
# 1. Environment (see docs/training.md for CUDA wheel links)
mamba env create -f environment.yml && conda activate gvae

# 2. Data: LAZ → JSON + occ caches (fine, mid, coarse), then train/test split
python utils/build_scene_graph.py <data_root>

# 3. Train
python train.py
# → checkpoint/{timestamp}/best.pth, train.log

# 4. (Optional) Latent probes on a checkpoint
python utils/probe_latent.py --checkpoint checkpoint/<run>/best.pth \\
  -o checkpoint/<run>/probe_report.txt
```

Configure GPU and hyperparameters in `config.py`. Override ablation settings via `train.py` flags (`--epochs`, `--lambda-occ-grid-*`, `--splat-sigma-fine`, `--ckpt-dir`) — see [docs/training.md](docs/training.md).

---

## Repository layout

```
gvae/           models, losses, data loaders, probes
train.py        single-stage training loop (LR decay mid-run)
config.py       hyperparameters
utils/          graph building, probes, diagnostics, visualization
data/graphs/    train/ and test/ scene JSON + occ sidecars
checkpoint/     training outputs (best.pth, probe_report.txt, …)
docs/           current documentation
TODO.md         backlog
```

---

## Outputs

| Latent | Grid (H×W×D) | Channels | Role |
|--------|----------------|----------|------|
| `Z_fine` | 64×64×8 | 72 | Finest layout level (fine supernodes after S0) |
| `Z_mid` | 32×32×8 | 144 | Regional layout |
| `Z_coarse` | 16×16×4 | 288 | Scene envelope |

Occupancy sidecars per scene: `{stem}_occ_fine.npy`, `_occ_mid.npy`, `_occ_coarse.npy`.

Use **`best.pth`** (lowest val loss), not `last.pth`. Checkpoints are not interchangeable across major architecture changes. Details in [docs/training.md](docs/training.md).
