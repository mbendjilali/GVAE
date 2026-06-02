# Data pipeline

How raw LiDAR becomes training files the GVAE can load.

---

## Overview

```
LAZ point cloud
      ↓
utils/build_scene_graph.py
      ↓
scene JSON  +  three occupancy .npy sidecars
      ↓
SceneGraph.from_json()  (in train.py)
      ↓
GVAE forward pass
```

Each **scene** is one tile (typically ~500 m). Coordinates are normalised per scene to fit inside `[-1, 1]³`.

---

## Scene graph JSON

Built per tile from LiDAR. Files live under `data/graphs/train/` and `data/graphs/test/`.

### Top-level contents

- **`instances`**: list of objects, each with `position`, `radius`, `label`
- **`normalization`**: `centroid` and `scale` used for the `[-1, 1]³` mapping
- Optional metadata (tile id, point counts, …)

### Tensors loaded at training time

| Tensor | Shape | Description |
|--------|-------|-------------|
| `p` | `(N, 3)` | Normalised object centers |
| `r` | `(N, 3)` | Footprint semi-axes (axis-aligned extent) |
| `s` | `(N, C)` | One-hot semantics (`C = 15`) |
| `label` | `(N,)` | Class index |
| `edge_index` | `(2, E)` | Undirected proximity edges |
| `coarsen_mask` | `(N,)` | `True` = instantiable → may enter FPS coarsening |
| `occ_fine` | `(H, W, D)` | LiDAR occupancy, fine grid |
| `occ_mid` | `(H, W, D)` | LiDAR occupancy, mid grid |
| `occ_coarse` | `(H, W, D)` | LiDAR occupancy, coarse grid |

Edges are built in normalised space: two nodes connect if their centers are closer than `EDGE_PROXIMITY` (default `0.03`).

---

## Semantic classes (15)

Defined in `config.SEMANTIC_CLASSES`:

`ground`, `vegetation`, `car`, `powerline`, `fence`, `tree`, `pickup`, `van_truck`, `heavy_duty`, `utility_pole`, `light_pole`, `traffic_pole`, `habitat`, `complex`, `annex`

### Non-instantiable (B policy)

`ground`, `vegetation`, `fence` — large background regions.

- Present on the **instance graph** for context.
- **Excluded from coarsening** when `COARSEN_EXCLUDE_NON_INSTANTIABLE=True` (default).
- Also filtered out of occupancy caches when `OCC_FILTER_NON_INSTANTIABLE=True` (object-centric occ GT).

---

## Occupancy caches

Occupancy is **real LiDAR voxelisation**, not bounding-box fill.

| Level | Grid (H×W×D) | Sidecar suffix |
|-------|----------------|----------------|
| Fine | 64×64×8 | `{stem}_occ_fine.npy` |
| Mid | 32×32×8 | `{stem}_occ_mid.npy` |
| Coarse | 16×16×4 | `{stem}_occ_coarse.npy` |

These grids match `config.GRID_FINE`, `GRID_MID`, `GRID_COARSE` and the `OccGridHead` output shapes.

### Build settings

| Config | Default | Meaning |
|--------|---------|---------|
| `OCC_MAX_POINTS` | `500_000` | Subsample LiDAR when voxelising |
| `OCC_REQUIRE_CACHE` | `True` | Crash early if a sidecar is missing |
| `OCC_FILTER_NON_INSTANTIABLE` | `True` | Drop ground/vegetation/fence from occ GT |

Training supervises occupancy with **full-grid BCE** on `OccGridHead` (not random query points). Probes may sample query locations from these grids for diagnostics.

---

## Building graphs from LAZ

```bash
python utils/build_scene_graph.py --help
```

Point the script at your LAZ / tile root (see script help for exact arguments). It writes JSON + the three `.npy` sidecars per scene.

### Train / test split

Organise outputs into:

```
data/graphs/train/   # training scenes (29 tiles)
data/graphs/test/    # held-out validation scenes (11 tiles)
```

**Fixed validation split** (tile IDs = JSON stems without `.json`):

| Split | Count | Tile IDs |
|-------|------:|----------|
| **train** | 29 | `5080_54435`, `5085_54320`, `5095_54440`, `5095_54455`, `5100_54495`, `5105_54405`, `5105_54460`, `5110_54320`, `5110_54460`, `5110_54475`, `5110_54495`, `5115_54480`, `5130_54355`, `5135_54495`, `5140_54445`, `5145_54340`, `5145_54405`, `5145_54460`, `5145_54470`, `5145_54480`, `5150_54340`, `5160_54330`, `5165_54390`, `5165_54395`, `5180_54435`, `5180_54485`, `5185_54390`, `5185_54485`, `5190_54400` |
| **test** (val) | 11 | `5080_54400`, `5080_54470`, `5100_54440`, `5100_54490`, `5120_54445`, `5135_54430`, `5135_54435`, `5140_54390`, `5150_54325`, `5155_54335`, `5175_54395` |

`train.py` loads `data/graphs/train/` for training and `data/graphs/test/` for validation (despite the folder name, this is the **val** split). Graphs with **no coarsenable instances** are skipped by the dataset loader.

---

## Normalisation

Per scene:

1. Subtract the scene **centroid**.
2. Scale by the longest axis so the scene fits in `[-1, 1]³`.

The same transform is stored in JSON and reused when voxelising LiDAR, so graph nodes and occupancy voxels stay **aligned**.

---

## If you change grid shapes

If you edit `GRID_FINE`, `GRID_MID`, or `GRID_COARSE` in `config.py`, you must **rebuild all occupancy caches** (TODO G3). Old `.npy` files will have the wrong shape.

---

## Related docs

- Model use of this data: [architecture.md](architecture.md)
- Training commands: [training.md](training.md)
- Reconstruction BEV checks: `utils/visualize_recon.py` (loads `best.pth` + val graph)
- Localization experiment recap: [localization-progress.md](localization-progress.md)
