> **Superseded (Jun 2026).** See [../localization-progress.md](../localization-progress.md) for current status and next experiment. Kept for historical probe baselines and Phase 1–4 planning context.

# GVAE — next steps (agent handoff)

**Purpose:** Continue work in a fresh conversation. This file captures diagnosis, probe baselines, and a concrete implementation plan. **Not committed** (`docs/agent/` is gitignored).

**Repo:** `/home/moussabendjilali/libs/GVAE`  
**Branch at handoff:** `z-only-decoder-new-layer` (merged configurable coarsening; PR #3 → main pending review)  
**Reference checkpoint:** `checkpoint/20260521_163301/best.pth` (150-epoch run, val best **1.0646** @ ep 148)

---

## 1. Product goal (unchanged)

Encode each **500 m × 500 m** DALES-2 tile into three latents for **hierarchical DDM conditioning**:

| Latent | Grid | ~cell size (500 m tile) | Expectation |
|--------|------|-------------------------|-------------|
| `Z_coarse` | `(16, 16, 4)` | ~31 m | Regional layout / envelope — **not** instance pinning |
| `Z_mid` | `(32, 32, 8)` | ~16 m | Group / supernode layout |
| `Z_fine` | `(64, 64, 8)` | ~8 m | **Must pin individual instances** at finest level |

Ultimate aim: control **general layout** of very large scenes via DDM cascade; per-tile training on full 500 m tiles is in scope. **Multi-tile cities** later need a **shared world frame** (fixed tile norm or metadata alongside `Z`) when stitching adjacent tiles — not blocking single-tile work.

---

## 2. What works today

### Training run `20260521_163301` (150 ep, hard FPS, single stage)

- Val loss **7.09 → 1.06** (best ep 148); worth training past 70 ep.
- **Fine semantics:** `miou_fine` ~**96%** (decoder + `h`).
- **Mid layout:** `inst_pos_err_mid` ~**0.11** norm (~30 m); `soft_miou_mid` ~29%.
- **Coarse occ (TensorBoard):** `occ_iou_coarse` ~**75%** @ ep 90 then stalls; `occ_precision_coarse` ~78%; `pos_err_coarse` ~**0.14** norm (~39 m for scene supernodes).
- **Coarse semantics weak:** `soft_miou_coarse` 0.06 → 0.12 (slow, no plateau).

### Architecture already in place

- Three-level encoder/decoder, `(C,H,W,D)` latents, hard FPS coarsening (configurable soft mode exists but default **hard**).
- Occ sidecars **already** at `GRID_FINE/MID/COARSE` — same shapes as `Z_*` (no extra downsample step needed).
- **Probes implemented:** `utils/probe_latent.py`, `gvae/probes/latent.py`.
- Colorful training terminal, AMP, cached graphs, `best.pth` / `last.pth`.

---

## 3. Core diagnosis (probe-driven)

**The GVAE is a strong graph→semantics path; `Z` is not yet a spatial instance map.**

### Probe baselines (`best.pth`, val split)

Report: `checkpoint/20260521_163301/probe_report.txt`

**A. Anchor ablation (decoder reads `Z` with different anchor mix)**

| Level | mix=0 (Z-only anchor) | mix=1 (GT anchor) |
|-------|------------------------|-------------------|
| fine pos_err | **0.24**, miou **96.5%** | 0.73, miou 3% |
| mid pos_err | **0.12** | 0.66 |
| coarse pos_err | **0.13** | 0.87 |

→ Reading `Z` at **true** instance centres **hurts**. Learned anchors compensate because **signal in `Z` is offset from GT positions**.

**B. Linear probe `Z(p_gt) → position` (val)**

| Level | linear_pos_err_val | linear_cls_acc_val |
|-------|--------------------|--------------------|
| fine | **0.68** | 61.5% |
| mid | 0.61 | 65.5% |
| coarse | 0.56 | 72.3% |

→ Raw `Z` at GT does not encode position; decoder miou 96% comes mostly from **`h` + deformable sampling**, not `Z(p)`.

**C. Signal vs background (`‖Z(p)‖` at GT vs empty occ voxels)**

| Level | norm_ratio gt/empty | target |
|-------|---------------------|--------|
| fine | **0.45** | **> 1.0** |
| mid | 0.46 | > 1.0 |
| coarse | 0.50 | > 1.0 |

→ **Empty voxels have stronger `Z` norm than object centres** — splat + U-Net spreads energy; not peaked at instances.

### Fine metrics that plateau (training log)

- `occ_iou_fine` ~**48%** by epoch ~4, flat for 146 epochs.
- `pos_err_fine` ~**0.24** norm (~**67 m** at typical scale) — **does not meet** instance-pinning goal (~sub-voxel / <1 fine cell ≈ norm **< 0.03–0.1**).

### Root cause (one line)

**Splatting writes blurred node-centric features; recon/KL train decoder + `h`; occ uses detached cross-attention readout — nothing forces voxel `(i,j,k)` of `Z` to mean “instance here”.**

---

## 4. Redefine success metrics for `Z_fine` (and per level)

### Primary gates (run after every ablation)

```bash
python utils/probe_latent.py --checkpoint <run>/best.pth -o <run>/probe_report.txt
```

| Metric | Source | Baseline (fine) | Target (fine pinning) |
|--------|--------|-----------------|------------------------|
| `norm_ratio_gt_over_empty` | Probe C | 0.45 | **> 1.0** (ideally > 1.2) |
| `linear_pos_err_val` | Probe B | 0.68 | **< 0.10** (stretch: < 0.05) |
| `mix_1_pos_err` vs `mix_0_pos_err` | Probe A | 0.73 vs 0.24 | **mix_1 ≤ mix_0** |
| `mix_0_pos_err` | Probe A | 0.24 | **< 0.10** |
| `occ_iou_fine` (full grid) | val metrics | ~48% | **> 65%** (interim); align with DDM needs |
| `miou_fine` | val | ~96% | do not collapse (< 90%) |

### Secondary (TensorBoard / console)

- **Coarse:** keep `occ_iou_coarse` ≥ 75%; improve `soft_miou_coarse` above 0.12.
- **Mid:** `inst_pos_err_mid`, `soft_miou_mid` — maintain or improve while changing fine path.

### Logging changes to implement

- Log probe summary **automatically** at end of training (or every N epochs on val).
- Optionally log probe metrics to TensorBoard under `probe/fine/...`.
- Treat probe metrics as **release criteria for `Z_fine`**, not only `val loss`.

---

## 5. Implementation plan (ordered)

### Phase 0 — Baseline preservation

- [ ] Tag current `best.pth` run config in a small `configs/baseline_20260521.yaml` or comment block (optional).
- [ ] Ensure probe script runs in `gvae` conda env:  
  `/home/moussabendjilali/miniforge3/envs/gvae/bin/python utils/probe_latent.py ...`

### Phase 1 — Grid-aligned occ supervision (fits latest plan; occ caches already correct shape)

**Goal:** Supervise each `Z_level` on matching `occ_*` grid with BCE/Dice **on voxels**, not only `OccupancyReadout` + 2048 queries.

**Current:** `gvae/data/occupancy.py` → `OccupancyReadout` cross-attention; `loss_occupancy` samples queries.

**Proposed:**

1. Add `OccGridHead`: `1×1 Conv3d(C, 1, 1)` on `Z` (or split: keep readout for legacy, add grid head).
2. Add `loss_occ_grid(logits, occ_gt)` — BCE with `pos_weight` for sparsity, or Dice.
3. Per-level weights in `config.py`, e.g.  
   `LAMBDA_OCC_GRID_FINE`, `LAMBDA_OCC_GRID_MID`, `LAMBDA_OCC_GRID_COARSE`.
4. Wire in `compute_branch_losses` / `gvae_loss.py` using `graph.occ_fine/mid/coarse` (already loaded in `SceneGraph`).

**Note:** This is **primary for coarse/mid**; for **fine** it is **necessary but not sufficient** for instance pinning (occ = thin LiDAR shell, not centroids).

**Files:** `config.py`, `gvae/losses/gvae_loss.py`, new `gvae/models/occ_grid_head.py` (or extend `occupancy.py`), `train.py` (log components).

### Phase 2 — Fix how `Z_fine` is written (instance localization)

Pick one or combine (ablation matrix below):

**2A. Sharper fine splat**

- Reduce `SPLAT_TRUNCATION_SIGMA` for fine only, or separate `SPLAT_SIGMA_FINE`.
- Optional: cap Gaussian support to ~1 voxel at fine grid.

**Files:** `config.py`, `gvae/models/splatting.py`, `encoder.py` (fine splat call).

**2B. Soft density target (alternative/complement to hard occ)**

- Build target density from instance `(p, r)` at `(64,64,8)` — aligned with splat semantics.
- L1 or BCE against density head on `Z_fine`.

**Files:** `gvae/data/voxelize.py` (new helper), loss + head.

**2C. Centre / offset field (fine only, if 2A+2B insufficient)

- Auxiliary target: per-voxel vector to nearest instance centre, or soft instance id channel.
- Strongest signal for **pinning** but more engineering.

**Do not prioritize first:** bigger decoder (`gvae/models/decoder.py`) — probes show bottleneck is **write path**, not readout capacity.

### Phase 3 — Small ablation matrix (few epochs each, ~20–30 ep enough for probe direction)

Run same seed/data subset if needed for speed (optional: `--max-graphs 10` flag on train — **not implemented yet**).

| Run | Changes | Hypothesis |
|-----|---------|------------|
| **0** | Baseline (current) | Reproduce probe baselines |
| **1** | Grid occ BCE on coarse+mid only | Coarse occ IoU ↑; probes coarse unchanged or slightly better |
| **2** | Grid occ BCE on all three levels | Probe C ratio ↑; fine occ IoU ↑; linear pos err may still be high |
| **3** | (2) + sharper fine splat (2A) | norm_ratio **> 1**; mix_1 pos_err ↓ |
| **4** | (3) + soft density on fine (2B) | linear_pos_err ↓ toward 0.1 |

After each run: `probe_latent.py` + compare table to Section 4.

### Phase 4 — Integrate probes into training loop

- [ ] Call `run_latent_probes()` on val set at end of `train.py` (or `--probe-every 10`).
- [ ] Print compact probe line in epoch summary for fine (ratio, linear pos err, mix0/mix1).
- [ ] Save `probe_report.txt` next to `train.log`.

---

## 6. Key files map

| Area | Path |
|------|------|
| Config | `config.py` |
| Encoder / splat | `gvae/models/encoder.py`, `splatting.py`, `unet3d.py` |
| Decoder | `gvae/models/decoder.py` |
| Occ readout (current) | `gvae/data/occupancy.py` |
| Losses | `gvae/losses/gvae_loss.py`, `metrics.py` |
| Probes | `gvae/probes/latent.py`, `utils/probe_latent.py` |
| Occ caches build | `utils/build_scene_graph.py` |
| Scene load | `gvae/data/scene_graph.py` |
| Train | `train.py` |
| Docs (committed) | `docs/architecture.md`, `docs/training.md` |

---

## 7. Config snapshot (reference run)

```python
COARSEN_ASSIGNMENT = "hard"
GRID_FINE = (64, 64, 8)
GRID_MID = (32, 32, 8)
GRID_COARSE = (16, 16, 4)
NUM_EPOCHS = 150
LR 3e-4 → 1e-4 @ epoch 41
DECODER_GT_ANCHOR_MIX = 0.0   # Z-only anchors
USE_POOL_LOSS = False
LOG_FULL_METRICS = True
```

Occ sidecars: `{stem}_occ_fine.npy`, `_occ_mid.npy`, `_occ_coarse.npy` next to JSON.

---

## 8. Explicit non-goals for next sprint

- Multi-tile global coordinate system (document only; implement later).
- Soft coarsening / pool loss ablation (unless fine work is blocked).
- Long 150-epoch runs before probe gates move — use **short ablations** first.
- Committing `docs/agent/` (gitignored handoff only).

---

## 9. Suggested first message in new conversation

> Read `docs/agent/next_step.md`. Implement Phase 1 (grid-aligned occ supervision on Z_coarse/mid/fine) and Phase 3 run **1** and **2** with short training; re-run probes and compare to baselines in Section 4. Then proceed to Phase 2A if fine `norm_ratio` still < 1.

---

## 10. Open questions for user (if not decided)

1. Keep `OccupancyReadout` alongside grid head, or replace entirely?
2. Dice vs BCE for sparse occ; `pos_weight` for occupied voxels?
3. Short ablation epoch count (20 vs 40) and whether to add `--max-graphs` for speed?
4. Accept interim fine goal: **occ IoU + probe ratio** before demanding **pos_err < 0.05**?

---

*Generated for agent handoff — May 2026.*
