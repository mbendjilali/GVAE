# GVAE documentation

This folder explains the Scene Graph VAE from scratch. You do **not** need prior VAE or diffusion experience — start with the reading order below.

---

## Reading order (recommended)

1. **[data.md](data.md)** — What files the model eats (JSON graphs, LiDAR occupancy caches).
2. **[architecture.md](architecture.md)** — What happens inside the network: graphs → splat → U-Net → `Z` → decoders → losses.
3. **[training.md](training.md)** — How to run training, interpret the console, and run probes.

Then skim **[../TODO.md](../TODO.md)** for active experiments and ablation commands.

---

## Document map

| File | Audience | Contents |
|------|----------|----------|
| [data.md](data.md) | Data engineers | LAZ → JSON pipeline, class list, occupancy grids |
| [architecture.md](architecture.md) | Model readers | Encoder chain, two decoders, loss terms, metrics |
| [training.md](training.md) | Practitioners | Environment, CLI flags, checkpoints, probes |

---

## Archive

Superseded drafts are in [archive/](archive/). **Do not use them** — they describe older designs (single mid/coarse grid, query-based occupancy, etc.).

| File | Replaced by |
|------|-------------|
| `archive/GVAE-description.md` | `architecture.md` |
| `archive/scene_graph_vae_implementation_plan.md` | `architecture.md` + `../TODO.md` |
