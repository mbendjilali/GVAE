# Scene Graph VAE (GVAE)

Encodes 3D outdoor scene graphs into KL-regularised spatial latent volumes (`Z_coarse`, `Z_mid`) for hierarchical diffusion conditioning.

**Status:** PR1 complete (Z-only decoder, soft semantic loss, stable training). PR2 / Layer A (`Z_fine`) in progress — see [TODO.md](TODO.md).

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
# → checkpoint/{timestamp}/stage*_best.pth, train.log
```

Configure GPU and hyperparameters in `config.py`.

---

## Repository layout

```
gvae/           model, losses, data loaders
train.py        four-stage training loop
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
| `Z_coarse` | 8×8×4 | Diffusion level 1 conditioning |
| `Z_mid` | 16×16×8 | Diffusion level 2 conditioning |
| `Z_fine` | *planned* | Instance-level layout (PR2) |

Use `stage{N}_best.pth`, not `final.pth`. Details in [docs/training.md](docs/training.md).
