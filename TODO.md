# GVAE — TODO

Documentation: [docs/](docs/) · Architecture: [docs/architecture.md](docs/architecture.md)

**Reference checkpoint (combined h+Z + Z-only):** `checkpoint/20260525_145547/best.pth` (ep 141, val ~7.85).

**Decision metrics for DDM handoff:** run `utils/probe_latent.py` after each ablation — `norm_ratio_gt_over_empty`, `linear_pos_err_val`, anchor mix pos errors, `occ_iou_fine`, `soft_miou_fine`, plus console `zpos` / `pos` from training.

---

## Done recently

- [x] Fine / mid / coarse latent grids + occ caches
- [x] Single-stage training (all branches)
- [x] Grid-aligned occ supervision (`OccGridHead`)
- [x] Sharper fine splat (σ + voxel cap)
- [x] Fine coarsening layer (S0: instances → fine supernodes)
- [x] Latent probe tooling (`utils/probe_latent.py`)
- [x] Z-only decoder path + dual recon losses (`LAMBDA_RECON_H`, `LAMBDA_RECON_ZONLY`)
- [x] Shallow fine U-Net (`UNET_DEPTH_FINE = 1`)
- [x] Norm contrastive loss on fine/mid Z (Probe C alignment)
- [x] Loss/metrics audit (soft mIoU, grid-only occ, oracle sanity checks)

---

## Active — Z localization

| ID | Task | Notes |
|----|------|-------|
| Z1 | Phase 2B — soft density target on fine | Run 4 in ablation matrix |
| Z2 | Integrate probes into training loop | End-of-run or `--probe-every N` |
| Z3 | Instance-level probe mode | Probes currently use fine supernode GT after S0 |
| Z4 | Re-run ablations post fine-coarsening | Old Run 1–3 numbers not comparable |
| Z5 | Z-only readout at h-predicted anchors | Stricter metric than GT-slot `zpos` |

---

## Metrics & diagnostics

| ID | Task |
|----|------|
| A1 | Full-metrics eval on best checkpoint with `LOG_FULL_METRICS=True` |
| A2 | ~~Fix `soft_miou` global-pool bug~~ → audit all losses & metrics (oracle checks) |
| A3 | Occ visual sanity — pred vs cache on 2–3 scenes |
| A4 | Save probe summary JSON next to `best.pth` on checkpoint save |
| A5 | Console `zsmiou` when both decoders active; log occ pred_rate/recall |

---

## Config / data

| ID | Task |
|----|------|
| G1 | Document fixed val split (11 test tiles) |
| G2 | Dataset expansion beyond 29 train scenes |
| G3 | Rebuild occ caches if grid shapes change |

---

## Deferred

- Multi-tile shared world frame for stitching
- Road-edge reconstruction loss
- Cross-level Z consistency loss
- Full XCube sparse cascade (L4+)

---

## Ablation quick reference

```bash
# Grid occ all levels, 25 ep
python train.py --epochs 25 --ckpt-dir checkpoint/ablation \
  --lambda-occ-grid-fine 1 --lambda-occ-grid-mid 1 --lambda-occ-grid-coarse 1

# Disable Z-only path (h-decoder only)
python train.py --no-zonly-decoder --lambda-recon-h 1 --lambda-recon-zonly 0

# Probes
python utils/probe_latent.py --checkpoint checkpoint/ablation/best.pth \
  -o checkpoint/ablation/probe_report.txt
```
