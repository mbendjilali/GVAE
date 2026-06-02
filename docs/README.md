# GVAE documentation

This folder explains the Scene Graph VAE from scratch. You do **not** need prior VAE or diffusion experience — start with the reading order below.

---

## Reading order (recommended)

1. **[data.md](data.md)** — JSON graphs, LiDAR occupancy caches.
2. **[architecture.md](architecture.md)** —  Graphs → splat → U-Net → `Z` → decoders → losses.
3. **[training.md](training.md)** — How to run training, interpret the console, and run probes.
4. **[localization-progress.md](localization-progress.md)** — Jun 2026 anchor / `pos` experiment recap (half page).

Then skim **[../TODO.md](../TODO.md)** for remaining backlog.

---

## Document map

| File | Contents |
|------|----------|
| [data.md](data.md) | LAZ → JSON pipeline, class list, occupancy grids, train/val split |
| [architecture.md](architecture.md) | Encoder chain, two decoders, loss terms, metrics |
| [training.md](training.md) |Environment, CLI flags, checkpoints, probes |
| [localization-progress.md](localization-progress.md) |Three-run arc, current numbers, next experiment |

---
