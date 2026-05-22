# GVAE — TODO

Documentation: [docs/](docs/) · Architecture: [docs/architecture.md](docs/architecture.md)

**Reference checkpoint:** `checkpoint/20260521_163301/best.pth` (150 ep, val best 1.06).

**Decision metrics for Z_fine pinning:** run `utils/probe_latent.py` after each ablation — `norm_ratio_gt_over_empty`, `linear_pos_err_val`, anchor mix pos errors, `occ_iou_fine`, `miou_fine`.

---

## Done recently

- [x] Fine / mid / coarse latent grids + occ caches
- [x] Single-stage training (all branches)
- [x] Grid-aligned occ supervision (`OccGridHead`)
- [x] Sharper fine splat (σ + voxel cap)
- [x] Fine coarsening layer (S0: instances → fine supernodes)
- [x] Latent probe tooling (`utils/probe_latent.py`)

---

## Active — Z localization

| ID | Task | Notes |
|----|------|-------|
| Z1 | Phase 2B — soft density target on fine | Run 4 in ablation matrix |
| Z2 | Integrate probes into training loop | End-of-run or `--probe-every N` |
| Z3 | Instance-level probe mode | Probes currently use fine supernode GT after S0 |
| Z4 | Re-run ablations post fine-coarsening | Old Run 1–3 numbers not comparable |

---

## Metrics & diagnostics

| ID | Task |
|----|------|
| A1 | Full-metrics eval on best checkpoint with `LOG_FULL_METRICS=True` |
| A2 | Decompose `soft_miou` ceiling (supernode label ambiguity) |
| A3 | Occ visual sanity — pred vs cache on 2–3 scenes |
| A4 | Save probe summary JSON next to `best.pth` on checkpoint save |

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
- Cross-level Z consistency loss (F2 in old backlog)
- Full XCube sparse cascade (L4+)

---

## Ablation quick reference

```bash
# Grid occ all levels, 25 ep
python train.py --epochs 25 --ckpt-dir checkpoint/ablation \\
  --lambda-occ-grid-fine 1 --lambda-occ-grid-mid 1 --lambda-occ-grid-coarse 1

# Probes
python utils/probe_latent.py --checkpoint checkpoint/ablation/best.pth \\
  -o checkpoint/ablation/probe_report.txt
```
