# Scene Graph VAE (GVAE)

Encodes 3D outdoor scene graphs into KL-regularised spatial latent volumes (`Z_coarse`, `Z_mid`) for hierarchical diffusion conditioning.

**Status:** PR2 / Layer A implemented (`Z_fine`, 64³/32³/16³ grids). Rebuild occ caches before training.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/architecture.md](docs/architecture.md) | Encoder, coarsening, decoder, losses |
| [docs/training.md](docs/training.md) | Setup, training, checkpoints, metrics |
| [docs/data.md](docs/data.md) | Scene graphs, LiDAR, occupancy caches |
| [TODO.md](TODO.md) | Active backlog |

Older prose (`GVAE description.md`, `scene_graph_vae_4b057e4d.plan.md`) is archived under [docs/archive/](docs/archive/) and **must not** be used as source of truth.

---

## Quick start

```bash
# 1. Environment (see docs/training.md for CUDA wheel links)
mamba env create -f environment.yml && conda activate gvae

# 2. Data: LAZ → JSON + occ caches, then split train/test
python utils/build_scene_graph.py  # see script for args

# 3. Train
python train.py
# → checkpoint/{timestamp}/best.pth, train.log
```

Configure GPU and hyperparameters in `config.py`.

---

## Repository layout

```
gvae/           model, losses, data loaders
train.py        single-stage training loop (LR decay)
config.py       hyperparameters
utils/          graph building, diagnostics, visualization
data/graphs/    train/ and test/ scene JSON + occ sidecars
checkpoint/     training outputs
docs/           current documentation
TODO.md         backlog
```

---

## Outputs

| Latent | Grid | Role |
|--------|------|------|
| `Z_fine` | 64×64×64 | Diffusion level 3 / instance layout |
| `Z_coarse` | 16×16×16 | Diffusion level 1 conditioning |
| `Z_mid` | 32×32×32 | Diffusion level 2 conditioning |

Use `best.pth`, not `last.pth`. Details in [docs/training.md](docs/training.md).
