# Data pipeline

## Overview

```
LAZ point cloud  →  utils/build_scene_graph.py  →  scene JSON + occ caches
                                                      ↓
                                              SceneGraph.from_json()
                                                      ↓
                                              train.py / GVAE
```

---

## Scene graph JSON

Built per tile from LiDAR. Each file under `data/graphs/{train,test}/` contains:

- **`instances`**: list of objects with `position`, `radius`, `label`
- **`normalization`**: `centroid` and `scale` used to map world coords → `[-1, 1]³`
- Optional metadata (tile id, point counts)

Node attributes loaded by `SceneGraph`:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `p` | `(N, 3)` | Normalised centroids |
| `r` | `(N, 3)` | Footprint semi-axes |
| `s` | `(N, C)` | One-hot semantics |
| `label` | `(N,)` | Class index |
| `edge_index` | `(2, E)` | Proximity edges |
| `coarsen_mask` | `(N,)` | B policy: instantiable → coarsen |

---

## Semantic classes (15)

Defined in `config.SEMANTIC_CLASSES`: ground, vegetation, car, powerline, fence, tree, pickup, van_truck, heavy_duty, utility_pole, light_pole, traffic_pole, habitat, complex, annex.

**Non-instantiable** (B policy): `ground`, `vegetation`, `fence` — present on `G_L`, excluded from FPS coarsening when `COARSEN_EXCLUDE_NON_INSTANTIABLE=True`.

---

## Occupancy caches

Ground truth for `L_occ` — **LiDAR voxelisation**, not bounding-box proxies.

| Sidecar | Grid | Suffix |
|---------|------|--------|
| Mid | 16×16×8 | `{stem}_occ_mid.npy` |
| Coarse | 8×8×4 | `{stem}_occ_coarse.npy` |

- Built alongside JSON by `utils/build_scene_graph.py`
- `OCC_FILTER_NON_INSTANTIABLE=True` — object-centric voxels (no ground mass in occ GT)
- `OCC_MAX_POINTS=500_000` — subsample when building caches
- `OCC_REQUIRE_CACHE=True` — training fails fast if sidecars missing

Training queries (`OCC_QUERY_POINTS=2048`, `OCC_POS_RATIO=0.5`) sample occupied vs empty voxels from these grids.

---

## Building graphs from LAZ

```bash
python utils/build_scene_graph.py --help   # see script for tile paths / args
```

After build, split tiles into train/test:

```
data/graphs/train/   # ~29 scenes
data/graphs/test/    # ~11 scenes (fixed split — document tile IDs in TODO G2)
```

Graphs with zero coarsenable instances are skipped by the dataset loader.

---

## Normalisation

Per-scene: translate to centroid, scale by longest axis to fit `[-1, 1]³`. The same transform is stored in JSON and applied when voxelising LiDAR so graph nodes and occ caches stay aligned.

---

## Planned (PR2)

- **`Z_fine` grid** and `{stem}_occ_fine.npy` sidecars
- Instance-only encoder path on coarsenable nodes

See [TODO.md](../TODO.md) Layer A section.
