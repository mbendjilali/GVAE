---
license: mit
language:
  - en
pretty_name: DALES 2
size_categories:
  - 1G<n<10G
tags:
  - lidar
  - point-cloud
  - aerial
  - 3d-scene-understanding
  - instance-segmentation
  - semantic-segmentation
  - scene-graph
  - remote-sensing
task_categories:
  - other
---

# DALES 2

## Dataset summary

**DALES 2** is a renovated large-scale **aerial LiDAR** benchmark for **3D scene understanding**, built on the geographic coverage of the original [DALES](https://udayton.edu/blogs/artssciences/2020-stories/dales-large-scale-aerial-lidar.php) release with **re-annotated** semantic and **instance** labels, refined taxonomy, and companion **scene-graph** structure (objects and relations) described in the USM3D workshop paper at CVPR 2026.

This Hub repository distributes the **annotated point clouds** (typically LAZ/LAS per tile). **Graph JSON** and the interactive **[graph editing tool](https://github.com/mbendjilali/dales-2)** live in the companion code repository (see [Scene graph tool](#scene-graph-tool)).

### What is new vs. DALES Objects

- Human-in-the-loop revision of instances and semantics (e.g. powerline granularity per span, tree instances promoted from vegetation, fence as non-instantiable stuff).
- **15-class** semantic taxonomy with per-point **classification** and cross-class **unique instance IDs** within each tile (see [Class IDs](#class-ids)).
- Optional use with **relational graphs** (edges such as adjacent, near, support, extension) when generated with the official pipeline.

## Supported tasks

- Semantic and instance segmentation on aerial LiDAR  
- 3D detection / panoptic-style parsing in urban / infrastructure scenes  
- Scene graph generation and reasoning (with external graph assets and tools)

## Data modality

- **3D point clouds** (LAZ or LAS), aerial / large-footprint tiles  

## Class IDs

| ID | Class        | Notes (high level)        |
|---:|--------------|---------------------------|
| 0  | Ground       | Stuff                     |
| 1  | Vegetation   | Stuff                     |
| 2  | Car          | Vehicle instance          |
| 3  | Powerline    | Conductor instance        |
| 4  | Fence        | Stuff                     |
| 5  | Tree         | Tree instance             |
| 6  | Pick-up      | Vehicle instance          |
| 7  | Van & Truck  | Vehicle instance          |
| 8  | Heavy-duty   | Vehicle instance          |
| 9  | Utility pole | Pole instance             |
| 10 | Light pole   | Pole instance             |
| 11 | Traffic pole | Pole instance             |
| 12 | Habitat      | Building instance         |
| 13 | Complex      | Building instance         |
| 14 | Annex        | Building instance         |

**Instance field:** Each point carries an **instance** attribute. **Non-stuff** objects use distinct positive instance IDs; **stuff** classes use instance **0** in the released convention. Instance IDs are **unique across semantic classes within a tile** (one ID does not appear on two different classes in the same file).

## Dataset structure (repository layout)

Layout may evolve between releases; a typical layout is:

- One file per spatial **tile**, e.g. `*.laz` / `*.las` at the repository root or under a `tiles/` (or similar) prefix.  
- Optional sidecar assets (metadata, splits, checksums) as you add them on the Hub.

After uploading, configure **Dataset > Files and versions** so the Hub (and `datasets`) can see supported patterns if you add a `load_dataset` script later. Until then, use **`huggingface_hub`** to download files (below).

## Download

### huggingface_hub (recommended for raw files)

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mbendjilali/DALES-2",
    repo_type="dataset",
    local_dir="dales2",
)
```

### Git LFS

```bash
git lfs install
git clone https://huggingface.co/datasets/mbendjilali/DALES-2
```

## Scene graph tool

Scene graphs (`graph_*.json`, `graph_*_edges.json`, `geom_*.json`) and the **web editor** are maintained in the **[Github](https://github.com/mbendjilali/dales-2)** codebase. Documentation, schema, and build instructions are in the project README shipped with that repository.

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{bendjilali2026dales2,
  title     = {{DALES} 2: A Renovated Aerial {LiDAR} Benchmark for 3D Scene Understanding},
  author    = {Bendjilali, Moussa and Peyran, Claire and Velumani, Kaaviya and Mauri, Antoine and Luminari, Nicola and Alliez, Pierre},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year      = {2026},
  note      = {USM3D workshop},
}
```

(Update `pages` or volume fields when official workshop proceedings are available.)

## License

This dataset is released under the **MIT License**, consistent with the Hub dataset metadata.

## Acknowledgments

DALES 2 builds on the original **DALES** aerial LiDAR data and the **DALES Objects** instance benchmark. Please cite the original DALES and DALES Objects papers when comparing to prior work on that geography.

## Contact

For questions about the release or benchmark, open a discussion on this dataset page or contact the authors via the affiliations listed in the paper.
