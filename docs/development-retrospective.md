# GVAE Development Retrospective

**Period:** May 2026 (~3 weeks)  
**Status:** Historical record, frozen at 2026-05-25. Not updated for the June work.  
**Production checkpoint:** `checkpoint/norm_depth3_150/best.pth` (ep 145)  
**Probes:** `checkpoint/norm_depth3_150/probe_report.txt`

Sources: git history, current docs (`docs/architecture.md`, `docs/training.md`, `docs/data.md`), archive drafts, training logs, and ablation runs from May 21–25.

> **Scope note (Jun 2026).** This document describes the state of the project at the
> end of May 2026 and is preserved as the record of that period — §2 Phase E and §4.3
> in particular. It is **not** a description of the current tip. Since it was written:
> the latent-graph VAE mode was added (`4a02377`), `OccGridHead` and all voxel
> occupancy prediction were removed from training and metrics (`a215503`), and the
> position readout moved to a spatial soft-argmax over `Z` voxels (`2d2fecd`).
> §3 and the occupancy figures in §5 therefore describe an architecture that no longer
> ships. For the current implementation see [architecture.md](architecture.md); for the
> current status see [localization-progress.md](localization-progress.md).

---

## 1. Objectives — how they sharpened

### Original goal (archive, early May)

The [original implementation plan](archive/scene_graph_vae_implementation_plan.md) and [GVAE-description](archive/GVAE-description.md) defined a **standalone VAE** that compresses a 3D outdoor scene graph into **two** KL-regularised latent volumes (`Z_mid`, `Z_coarse`) for the **first two levels** of a hierarchical diffusion model (~500 m DALES tiles).

Core intent:

- Graph in → dense spatial latents out
- Diffusion conditions on layout, not raw graphs
- Reconstruction + occupancy + KL as training signals
- Four-stage curriculum to stabilize coarsening before full joint training

### Evolved goal (late May)

| Shift | From | To |
|-------|------|-----|
| **Latent hierarchy** | 2 volumes (mid + coarse) | **3 volumes** (`Z_fine`, `Z_mid`, `Z_coarse`) at 64×64×8 / 32×32×8 / 16×16×4 |
| **Fine level** | “Layer A / Z_fine pending” in archive | **S0 coarsening** (instances → fine supernodes) + `occ_fine` caches — finest level must **pin instance layout** (~8 m cells) |
| **DDM handoff criteria** | Val loss + decoder miou | **Probe-driven gates**: `Z(p_gt)` must encode position (Probe B), peak at object sites (Probe C), and support **Z-only readout** at known layout slots — not only joint `(h, Z)` decode |

The product question moved from *“can we compress a graph?”* to *“can `Z` alone carry layout that a downstream DDM can use?”*

---

## 2. Technical solutions — chronological arc

### Phase A — Foundation (May 5–19)

**Commits:** scene graph + coarsening scaffold → model elements → losses → LiDAR occupancy wiring.

- **Scene graph:** nodes `(p, r, s)`, proximity edges, per-scene `[-1,1]³` normalization
- **Coarsening:** FPS + ball-query supernodes; progressive dims `[72, 144, 288]` with PointROPE R-GAT
- **B policy:** ground/vegetation/fence stay on instance graph but are **excluded from coarsening**
- **Occupancy:** LiDAR voxel caches (initially mid/coarse)

### Phase B — Training stability & honest decode (May 20–21)

Key commits: `6efad21`, `783e6d1`.

- **GroupNorm** replaces BatchNorm (batch=1 scenes broke train/val parity)
- **h-predicted decoder anchors** replace GT `(p, r)` blending — metrics reflect what `Z` encodes at inference
- **Soft semantic KL** on supernode mixture labels (not CE on probabilities)
- **Layout metrics:** `inst_pos_err_mid`, occ IoU, Z-only pos err, soft mIoU
- **Training hygiene:** finite-loss gating for `best.pth`, grad clip, tanh-bounded positions

*Insight:* High decoder miou with GT anchors was **misleading** — semantics came from `h`, not from spatial structure in `Z`.

### Phase C — Architecture realignment (May 21)

Commit `7f87238`:

- **Three latent grids** + `occ_fine` sidecars
- **Hard FPS coarsening** as default (soft MinCutPool optional)
- **Single-stage training** (150 ep, LR decay @ ep 41) replaces 4-stage curriculum
- **Performance:** AMP, cached graphs, chunked splat, `(C,H,W,D)` Conv3d-native layout
- **`best.pth`** by val loss instead of per-stage checkpoints

U-Net fix `c2398c9`: `pool_kernel=(2,2,1)` — full 2×2×2 pooling collapsed the thin Z dimension.

### Phase D — Z localization push (May 22–25)

**Fine coarsening (S0)** merged (`2050200`, `a18690c`): instance → fine supernodes.

**Grid occupancy** (`361c51d`, `4d7c218`):

- `OccGridHead` — full-grid BCE on `Z` vs LiDAR caches
- Dropped query-point occupancy readout
- Soft CE semantics; fixed soft mIoU (per-supernode, not global pool)

**Latent probes** (`utils/probe_latent.py`): anchor ablation (A), linear `Z→p` (B), norm ratio at GT vs empty (C).

**Z-only decoder** (`0911410`):

- Dual reconstruction: h+Z deformable decoder + **Z-only MLP** at supernode slots (+ train jitter)
- `LAMBDA_RECON_H` / `LAMBDA_RECON_ZONLY`; console `zpos=`

**Norm contrastive** (`9dd00f3`): hinge on `‖Z(p_gt)‖` vs `‖Z(empty)‖` (fine/mid).

### Phase E — Ablation-driven config lock-in (May 25)

| Run | Config highlight | Outcome |
|-----|------------------|---------|
| `20260525_145547` | h+Z + Z-only, depth-3, no norm contrast | Strong baseline: `pos≈0.18`, `zpos≈0.08`; Probe C fine **~0.45** (fail) |
| `20260525_155231` | depth-1 + norm contrast + λ_z=1.2 | **Regression:** fine `zpos≈0.48` |
| `ablate_depth1_only` | depth-1 only | Confirms depth-1 breaks fine Z readout |
| `ablate_norm_only` | norm contrast only @ 60 ep | Fine `zpos≈0.09` — norm contrast harmless |
| **`norm_depth3_150`** | depth-3 + norm contrast + λ_z=1.2 | **Production candidate** |

---

## 3. Current architecture (summary)

```
G_L (instances) → S0 → G_fine → S1 → G_mid → S2 → G_coarse
                      ↓              ↓              ↓
                   Z_fine         Z_mid         Z_coarse
                      ↓              ↓              ↓
              h+Z decoder    h+Z decoder    h+Z decoder
              Z-only decoder Z-only decoder
              OccGridHead    OccGridHead    OccGridHead
```

**Losses:**

$$\mathcal{L} = \lambda_h \mathcal{L}_{\text{recon\_h}} + \lambda_z \mathcal{L}_{\text{recon\_zonly}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \sum_l \lambda_{\text{grid},l} \mathcal{L}_{\text{occ}} + \lambda_{\text{norm}} \mathcal{L}_{\text{norm\_contrast}}$$

**Defaults (`config.py`):** `UNET_DEPTH_FINE=3`, `LAMBDA_NORM_CONTRAST_*=0.1`, `LAMBDA_RECON_ZONLY=1.2`, `USE_Z_ONLY_DECODER=True`.

See [architecture.md](architecture.md) for full detail.

---

## 4. Obstacles encountered

### 4.1 “Good metrics, bad latents”

Early probes (e.g. `checkpoint/20260521_163301`) showed:

- Decoder miou ~96% but **linear `Z(p)→p` error ~0.68**
- **Probe C ratio ~0.45** — empty voxels had stronger `‖Z‖` than object centres
- Anchor ablation: reading `Z` at GT positions *hurt* vs learned anchors

**Root cause:** Splat + U-Net spread energy; reconstruction trained `(h, Z)` jointly; nothing forced voxel `(i,j,k)` to mean “object here.”

### 4.2 Metric bugs masked progress

- Global-pool soft mIoU → fixed to per-supernode soft IoU (`fb5cf76`)
- h-decoder `pos_err` conflated localization with slot readout → split `pos` vs `zpos`
- Query-based occ vs grid IoU misaligned → grid-only supervision (`4d7c218`)

### 4.3 Fine occ IoU plateau (~50%)

Not a metric bug: model over-predicts occupied voxels (~2× GT rate). IoU ceiling ≈ `gt_rate/pred_rate` ≈ 50%. Mid occ ~64% is healthier.

### 4.4 Failed experiments

- **`UNET_DEPTH_FINE = 1`:** Intended to reduce blur after splat; **destroyed** fine Z-only readout (`zpos` ~0.55). Splat writes peaks; shallow U-Net cannot refine `Z_fine` for readout.
- Combined run `155231` looked like norm contrast might hurt; ablations isolated depth-1 as the cause.

### 4.5 Engineering friction

- BatchNorm + batch=1 → train/val divergence
- U-Net pooling on thin Z axis → latent collapse (`pool_kernel=(2,2,1)` fix)
- Checkpoint/config U-Net depth mismatch breaks probe loading
- Val loss not comparable when norm-contrast term is added (+~0.7–1.0)

### 4.6 Data & scale

- 29 train / 11 val scenes
- Fixed val split undocumented (TODO G1)
- B policy affects what enters `Z` vs occ GT

---

## 5. Where we stand today

### Best checkpoint: `norm_depth3_150` (ep 145)

**Console (val):** fine `pos=0.18`, `zpos=0.08`, `smiou=84%`, `occ=50%`; mid `inst=0.15`, `zpos=0.07`, `occ=64%`.

**Probes:**

| Probe | Fine | Mid | Gate |
|-------|------|-----|------|
| A `mix_0_pos` | **0.18** | 0.15 | h localizes without GT anchor |
| B `linear_pos` | **0.14** | 0.11 | was ~0.68 pre-localization work |
| C `norm_ratio` | **1.51** | 1.41 | target > 1.0 — **pass** |

Coarse Probe C **~0.75** — acceptable for scene envelope, not instance pinning.

Prior reference `20260525_145547` had similar console layout metrics but Probe C fine ~0.45.

---

## 6. Potential shortcomings

1. **Z-only readout samples at GT supernode positions** — correct for DDM “known slots”; `zpos` does not prove peak-finding without h-anchor or grid-argmax readout (TODO Z5).

2. **Fine occ IoU ~50%** — structural over-prediction; may limit geometric fidelity for diffusion.

3. **Small dataset** — 40 tiles; probe gains may not generalize.

4. **Coarse `Z` background-dominated** — Probe C fails at coarse; scene semantics weak historically.

5. **Val loss as selection criterion** — misleading with norm contrast; use probe gates for promotion.

6. **No multi-tile world frame** — per-tile normalization; stitching deferred.

7. **Road edges, cross-level Z consistency, soft density** — not implemented.

---

## 7. Future improvements

### Near-term

| Item | ID | Rationale |
|------|-----|-----------|
| Automate probes on checkpoint save | A4, Z2 | Stop manual post-hoc analysis |
| Z-only @ h-predicted anchors | Z5 | Stricter than GT-slot `zpos` |
| Instance-level probes | Z3 | Probes use fine supernodes, not raw instances |
| Occ visual sanity + pred_rate/recall | A3, A5 | Understand 50% IoU plateau |
| Document val split | G1 | Reproducibility |

### Medium-term

| Item | ID | Rationale |
|------|-----|-----------|
| Soft density target on fine | Z1 | Complement grid occ for centroid pinning |
| Centre/offset field | agent plan 2C | Strongest pinning if occ + norm insufficient |
| Dataset expansion | G2 | 29 train scenes is thin |

### Deferred

- Multi-tile shared coordinates
- Road-edge reconstruction loss
- Cross-level Z consistency
- Full diffusion cascade (XCube L4+)
- Soft coarsening ablation unless blocked

See [TODO.md](../TODO.md) for active backlog.

---

## 8. Git timeline (condensed)

```
May 5–11   Scaffold, SceneGraph, coarsening
May 18–19  Losses, occupancy, progressive dims
May 20     GroupNorm, B policy, h-anchors, metrics overhaul
May 21     3-level latents, single-stage train, AMP, docs reorganize
May 21–22  Configurable coarsening, fine S0, grid occ, probes
May 22     U-Net pool_kernel fix, dead code removal
May 25     Grid-only occ refactor, Z-only decoder, norm contrast
May 25     Ablation cycle → norm_depth3_150 validated, depth-1 rejected
May 25     Documentation refresh (ba6ce19)
```

Representative commits: `7f87238`, `783e6d1`, `361c51d`, `4d7c218`, `0911410`, `9dd00f3`.

---

## 9. Narrative in one paragraph

GVAE started as a **two-volume graph compressor** with staged training and GT-anchored decode metrics that looked good but lied about `Z`. Over ~3 weeks it became a **three-volume variational encoder** with fine coarsening, grid occupancy, dual decoders, and probe-driven evaluation. The hardest problem — making **`Z` spatially meaningful for DDM** — was attacked through sharper splatting, full-grid occ, Z-only readout, and norm contrastive loss. A failed shallow-U-Net experiment proved that **write-path capacity matters as much as readout**. The current candidate (`norm_depth3_150`) passes Probe C on fine/mid while matching the best layout metrics; fine occ IoU and coarse semantics remain open before full diffusion handoff.

---

## 10. Related docs

| File | Purpose |
|------|---------|
| [archive/next_step-202605-handoff.md](archive/next_step-202605-handoff.md) | Earlier handoff (pre–Z-only / pre–grid-occ); partially superseded by this doc |
| [architecture.md](architecture.md) | Current implementation reference |
| [training.md](training.md) | How to train, metrics, probes |
| [../archive/](archive/) | Superseded two-volume / four-stage design |

*Last updated: 2026-05-25.*
