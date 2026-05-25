# Architecture

This page explains **how the Scene Graph VAE works**, in order, without assuming you already know VAEs or scene graphs.

---

## The big picture

**Input:** One outdoor scene — a list of objects (position, size, class) and which objects are near each other.

**Output:** Three 3D latent volumes (`Z_fine`, `Z_mid`, `Z_coarse`) that encode layout at three resolutions. A diffusion model can later sample or edit scenes using these volumes.

**Training trick:** The model must not only *produce* good `Z` volumes — it must also *read them back* to reconstruct object attributes. That readout is what keeps `Z` meaningful.

```
Scene graph  →  encode  →  Z_fine, Z_mid, Z_coarse
                              ↓
                         decode (two paths)
                              ↓
                    predicted class, position, size
                              ↓
                         compare to ground truth → loss
```

---

## Input: scene graph + occupancy

Each training scene is a JSON file (see [data.md](data.md)). For every object (node):

| Field | Symbol | Meaning |
|-------|--------|---------|
| Position | `p` | Object center in scene-normalised coordinates `[-1, 1]³` |
| Footprint | `r` | Semi-axes of an axis-aligned box (how big the object is) |
| Semantics | `s` | One-hot vector over 15 classes (car, tree, pole, …) |
| Label | — | Human-readable class name |

**Edges** connect objects whose centers are within a small distance (`EDGE_PROXIMITY`).

**Occupancy caches** (`occ_fine`, `occ_mid`, `occ_coarse`) are binary 3D grids from LiDAR — “was there a point here?” They supervise a separate occupancy head on each `Z` volume.

---

## B policy: background vs objects

Some classes are **non-instantiable** background: ground, vegetation, fence.

- They stay on the **instance graph** so nearby objects get context.
- They are **not coarsened** — they never become supernodes or feed the splat → U-Net path.

Config: `COARSEN_EXCLUDE_NON_INSTANTIABLE=True`.

---

## Encoder: from graph to three `Z` volumes

The encoder runs a **four-level chain**. Each level has graph neural network (GNN) message passing, then (except the last) coarsening.

```mermaid
flowchart TD
    GL["G_L — instance graph\none node per object"]
    C0["Coarsen 0 → fine supernodes"]
    GF["G_fine"]
    C1["Coarsen 1 → mid supernodes"]
    GM["G_mid"]
    C2["Coarsen 2 → coarse supernodes"]
    GC["G_coarse"]
    ZF["Splat + U-Net → Z_fine"]
    ZM["Splat + U-Net → Z_mid"]
    ZC["Splat + U-Net → Z_coarse"]
    GL --> C0 --> GF --> C1 --> GM --> C2 --> GC
    GF --> ZF
    GM --> ZM
    GC --> ZC
```

### Step 1 — Instance graph (`G_L`)

Every object is a node. An **R-GAT** (relational graph attention) with **PointROPE** updates node embeddings `h` using edges and 3D positions. Feature width: `D_INSTANCE = 72`.

### Step 2 — Coarsening (three times)

**Coarsening** merges many nodes into fewer **supernodes** using Farthest Point Sampling (FPS):

| Step | From → To | Default keep ratio |
|------|-----------|-------------------|
| S0 | instances → fine | 20% (`REDUCTION_RATIO_LEVELS[0]`) |
| S1 | fine → mid | 20% |
| S2 | mid → coarse | 20% |

Each supernode gets pooled position, mixed semantics, and a bounding-box footprint. New edges are built with **ball query** (radius grows at coarser levels).

**Hard assignment** (default): each node belongs to exactly one supernode (Voronoi-style).

**Soft assignment** (optional): fuzzy membership + extra pool regularisation losses.

### Step 3 — Splatting

Each supernode’s embedding is **splatted** onto a fixed 3D voxel grid as an anisotropic Gaussian (stretched by `r`). Think of painting a soft blob of features at each object location.

- Fine grid uses a **tighter** Gaussian (`SPLAT_TRUNCATION_SIGMA_FINE = 1.0` vs `2.0` on mid/coarse).
- Fine splat can be **capped** to ~1 voxel radius (`SPLAT_FINE_VOXEL_CAP`) so peaks stay sharp.

### Step 4 — 3D U-Net → variational `Z`

A **3D U-Net** refines the splatted grid and outputs Gaussian parameters `μ` and `log σ²`. We sample:

`Z = μ + σ ⊙ ε`   (standard VAE reparameterisation)

| Output | Grid (H×W×D) | Channels | U-Net depth |
|--------|----------------|----------|-------------|
| `Z_fine` | 64×64×8 | 72 | **1** (shallow — less blur) |
| `Z_mid` | 32×32×8 | 144 | 3 |
| `Z_coarse` | 16×16×4 | 288 | 2 |

Grids are **fixed independent volumes**, not subdivisions of each other.

**GroupNorm** is used inside the U-Net (not BatchNorm) because we often train with one scene per batch element.

---

## Two decoders (why two?)

During training, two decoders reconstruct supernode attributes from `(h, Z)`. They answer different questions:

| Decoder | Uses | Question it answers |
|---------|------|---------------------|
| **h + Z** (`SceneGraphDecoder`) | node embedding `h` **and** volume `Z` | “Given structure **and** latents, where is each object and what is it?” |
| **Z-only** (`ZOnlyDecoder`) | volume `Z` **only** | “If DDM only has `Z` and **known layout slots**, what is at each slot?” |

### h + Z decoder (full readout)

1. Predict an anchor box `(p, r)` from **`h`** (not from ground truth).
2. Place 27 sample points on a 3×3×3 grid inside that box (+ learned offsets).
3. Bilinear-sample **`Z`** at those points; cross-attend; predict `ŝ`, `p̂`, `r̂`.

`DECODER_GT_ANCHOR_MIX = 0.0` in normal training — anchors come from the model, not from labels. (Probe ablations can blend in GT anchors by raising this value.)

This path drives **`pos_err_fine`** / **`pos_err_mid`** — true **localization** metrics.

### Z-only decoder (DDM-aligned readout)

1. Sample **`Z` at ground-truth supernode positions `p`** (with small random jitter during training).
2. A small MLP predicts `ŝ`, `p̂`, `r̂` — **no `h`**.

**Important nuance:** Using GT `p` as the *sample location* does **not** remove position from the loss — the MLP still outputs `p̂` and we penalise `‖p̂ − p‖`. What GT `p` provides is **where to look in the volume**, not the answer. This matches the DDM contract: layout slots are known; `Z` fills in appearance and local geometry at those slots.

Jitter (`Z_ONLY_QUERY_JITTER = 0.05`) forces `Z` to be informative in a small neighborhood, not only at a single voxel.

This path drives **`pos_err_zonly_*`** (console: `zpos=`). For **global** localization evidence, also check h-decoder metrics and offline probes (see [training.md](training.md#latent-probes)).

---

## Occupancy head

Each `Z` level has an **`OccGridHead`**: a small conv network that predicts occupancy logits for **every voxel** in the grid. Loss is binary cross-entropy against the LiDAR cache (with automatic class imbalance weighting).

This replaces older designs that sampled random query points for occupancy.

---

## Loss function

Total loss per branch (fine / mid / coarse) combines:

$$\mathcal{L} = \lambda_h \mathcal{L}_{\text{recon\_h}} + \lambda_z \mathcal{L}_{\text{recon\_zonly}} + \lambda_{\text{KL}}(t)\,\mathcal{L}_{\text{KL}} + \lambda_{\text{grid}}\,\mathcal{L}_{\text{occ\_grid}} + \lambda_{\text{norm}}\,\mathcal{L}_{\text{norm\_contrast}} + \lambda_{\text{pool}}\,\mathcal{L}_{\text{pool}}$$

| Term | Default weight | What it does |
|------|----------------|--------------|
| **Recon h** | `LAMBDA_RECON_H = 1.0` | Soft CE on class + MSE on `p`/`r` via h+Z decoder |
| **Recon zonly** | `LAMBDA_RECON_ZONLY = 1.2` | Same targets via Z-only decoder (slightly favoured for DDM) |
| **KL** | cyclical → `LAMBDA_KL_MAX = 1e-3` | Regularise `μ, σ` toward standard normal |
| **Occ grid** | `LAMBDA_OCC_GRID_* = 1.0` | Voxel occupancy BCE (`OccGridHead`) |
| **Norm contrast** | `0.1` fine & mid | Hinge: push `‖Z(p_gt)‖` above `‖Z(empty voxel)‖` (Probe C alignment) |
| **Pool** | soft mode only | Keep coarsening assignments compact and separated |

**Reconstruction** at each supernode uses soft cross-entropy for semantics and MSE for position and footprint (`LAMBDA_SEM`, `LAMBDA_POS`).

**KL annealing** cycles over training (Fu et al., 2019) so the model repeatedly explores then regularises.

---

## Validation metrics (what the console shows)

| Metric | Decoder | Good direction | Plain meaning |
|--------|---------|----------------|---------------|
| `pos_err_fine` | h + Z | lower | Fine supernode position error |
| `zpos` (`pos_err_zonly_fine`) | Z-only | lower | Position error when reading `Z` at GT slot |
| `soft_miou_fine` | h + Z | higher | Semantic overlap on fine supernodes |
| `occ_iou_fine` | OccGridHead | higher | Predicted vs LiDAR occupancy (fine grid) |
| `inst_pos_err_mid` | h + Z chain | lower | Instance position via S0→S1→mid decode |
| `pos_err_mid` | h + Z | lower | Mid supernode position error |
| `zpos` mid (`pos_err_zonly_mid`) | Z-only | lower | Z-only mid slot readout |
| `soft_miou_mid` | h + Z | higher | Semantics on mid supernodes |
| `occ_iou_mid` | OccGridHead | higher | Occupancy IoU on mid grid |

Set `LOG_FULL_METRICS=True` for extra TensorBoard scalars (hard mIoU, coarse level, KL breakdown, …).

---

## Code layout

```
gvae/
├── models/       encoder, decoder, coarsening, splatting, unet3d, occ_grid_head, gvae
├── losses/       gvae_loss, metrics, diagnostics
├── probes/       offline latent probe library
├── data/         scene_graph, occupancy, voxelize, graph_masks
train.py          training loop
config.py         all hyperparameters
utils/            build_scene_graph, probe_latent, visualize_supernodes, diagnostics
```

---

## References

- FPS + ball-query: PointNet++ (Qi et al., NeurIPS 2017)
- Pool losses: MinCutPool (Bianchi et al., ICML 2020)
- PointROPE: LitePT (Yue et al., arXiv 2512.13689)
- Cyclical KL: Fu et al. (ACL 2019)
