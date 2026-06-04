# GVAE — backlog

Living task list for the DALES scene-graph VAE. Docs: [docs/README.md](docs/README.md) · metrics: [docs/training.md](docs/training.md) · localization: [docs/localization-progress.md](docs/localization-progress.md).

## Reference checkpoints (Jun 2026)

| Run | Role | Fine `pos` | Fine `size` | Notes |
|-----|------|------------|-------------|--------|
| `checkpoint/anchor_localize/` | Localization SOTA | ~0.10 | — | `smiou` ~76%; no footprint loss |
| `checkpoint/r_focus_v2/` | Footprint + balanced loss | ~0.11 | ~0.006 | `smiou` ~64%; val loss ~15 |

**Decision metrics:** `pos`, `anc`, `zpos`, `size`, `asz`, `soft_miou_fine`, `occ_iou_fine`, probes (`norm_ratio_gt_over_empty`, `linear_pos_err_val`, anchor mix @0).

---

## Implemented (Jun 2026 — this branch)

- [x] Footprint loss: log Smooth-L1 + `LAMBDA_SIZE` / `LAMBDA_SIZE_ZONLY` + anchor-r
- [x] Loss scale fix (no Z-only footprint blow-up; `OCC_GRID_POS_WEIGHT_CAP`)
- [x] Metrics: `size_err_*`, `anchor_size_err_*`, console `size` / `asz`
- [x] **Anchor MLP** (`USE_ANCHOR_MLP`) — 2-layer p/r heads vs Linear
- [x] **Fine splat trunc floor** (`SPLAT_MIN_TRUNC_VOXEL_FRAC`) — non-empty `F_fine`
- [x] **`LAMBDA_SEM_ZONLY`** — stronger Z-only semantics
- [x] **Coarse norm contrast** (`LAMBDA_NORM_CONTRAST_COARSE`)
- [x] Checkpoint compat: Linear anchor → MLP last layer (`gvae/checkpoint_compat.py`)
- [x] Position head **`clamp`** instead of `tanh` (`POSITION_BOUND`) — full [-1, 1] range
- [x] **`POSITION_RESIDUAL`** — `p = p_query + Δp` for Z-only readout (fixes center pull)
- [x] **`SPLAT_SUBTRACT_SPATIAL_MEAN`** + **`UNET_ALIGN_CORNERS`** — reduce Z center blob
- [x] `z_peaks` viz: ratio-to-median panel + ‖Z‖ colored at supernodes

---

## Next training run (recommended)

**Requires retrain** after residual position + splat centering (`localize_v5`).

```bash
python train.py --ckpt-dir checkpoint/localize_v5 \
  --lambda-recon-h 1.5 --lambda-recon-zonly 1.0 --lambda-recon-hzonly 0.5 \
  --lambda-anchor-fine 2.5 --lambda-anchor-mid 1.0
```

Targets: **`anc` ≤ 0.10**, **`pos` ≤ 0.08**, edge GT sites reachable (|p̂|>0.9), **`zsmiou` ≥ 50%**.

After training: probes + `visualize_recon.py` on 3 val tiles.

---

## Active

| ID | Task | Priority |
|----|------|----------|
| L1 | Train `localize_v5` (residual p + splat centering) | **Now** |
| L2 | Compare `localize_v3` vs `anchor_localize` / `r_focus_v2` on same BEV scenes | High |
| L3 | Integrate probes `--probe-every N` in training loop | Medium |
| L4 | Phase 2B soft density target on fine splat | Medium |
| L5 | Rebuild occ caches if `GRID_*` changes (G3) | When needed |

---

## Diagnostics & tooling

| ID | Task |
|----|------|
| D1 | TensorBoard loss components (recon / anchor / occ per level) |
| D2 | Log `‖F_fine‖` fraction non-zero per scene (splat health) |
| D3 | Document fixed 11-scene val split in `docs/data.md` |

---

## Deferred

- Multi-tile world frame / stitching
- Road-edge reconstruction loss
- Cross-level Z consistency
- Full XCube sparse cascade (L4+)
- Ellipsoidal splat (replace axis box trunc)

---

## Quick commands

```bash
# Train (defaults include anchor MLP + splat floor)
python train.py --ckpt-dir checkpoint/<run> \
  --lambda-anchor-fine 2.5 --lambda-anchor-mid 1.0

# Probes
python utils/probe_latent.py --checkpoint checkpoint/<run>/best.pth

# Recon BEV
python utils/visualize_recon.py --checkpoint checkpoint/<run>/best.pth \
  --scenes 5080_54400 5140_54390 5175_54395 --levels fine mid \
  -o checkpoint/<run>/recon_viz
```
