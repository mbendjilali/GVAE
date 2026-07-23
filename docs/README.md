# GVAE documentation

This folder explains the Scene Graph VAE from scratch. You do **not** need prior VAE or diffusion experience — start with the reading order below.

---

## Reading order (recommended)

1. **[data.md](data.md)** — JSON graphs, LiDAR occupancy caches.
2. **[architecture.md](architecture.md)** —  Graphs → splat → U-Net → `Z` → decoders → losses.
3. **[training.md](training.md)** — How to run training, interpret the console, and run probes.
4. **[localization-progress.md](localization-progress.md)** — Jun 2026 anchor / `pos` experiment recap (half page).
5. **[probe-results.md](probe-results.md)** — probes A / B / C for every checkpoint that still exists, and what is only transcribed.
6. **[development-retrospective.md](development-retrospective.md)** — May 2026 record: how the objectives moved, the Phase E ablations, and what did not work.

Then skim **[../TODO.md](../TODO.md)** for remaining backlog.

---

## Document map

| File | Contents |
|------|----------|
| [data.md](data.md) | LAZ → JSON pipeline, class list, occupancy grids, train/val split |
| [architecture.md](architecture.md) | Encoder chain, two decoders, loss terms, metrics |
| [training.md](training.md) |Environment, CLI flags, checkpoints, probes |
| [localization-progress.md](localization-progress.md) |Three-run arc, current numbers, next experiment |
| [probe-results.md](probe-results.md) | Probe A / B / C tables, reproducibility tolerance, which runs are lost |
| [development-retrospective.md](development-retrospective.md) | May 2026 record: objectives, Phase E ablations, obstacles, failed experiments |

---
