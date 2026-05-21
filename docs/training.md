# Training

## Environment

**Recommended:** conda env from `environment.yml` (CUDA 12.4 example):

```bash
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
export PIP_FIND_LINKS=https://data.pyg.org/whl/torch-2.6.0+cu124.html
mamba env create -f environment.yml
conda activate gvae
pip install tensorboard   # if not already present
```

Alternative: `pip install -r requirements.txt` with matching PyG wheel links (see comments in that file).

Set GPU in `config.py`: `CUDA_DEVICE = 1` (or `None` for CPU).

---

## Data layout

Training expects split directories:

```
data/graphs/
├── train/*.json
└── test/*.json
```

Each JSON needs matching occupancy sidecars (built by `utils/build_scene_graph.py`):

- `{scene}_occ_mid.npy`
- `{scene}_occ_coarse.npy`

See [data.md](data.md) for the build pipeline.

---

## Run training

```bash
python train.py
```

- Checkpoints and logs: `checkpoint/{timestamp}/`
- TensorBoard: `tensorboard --logdir checkpoint`

Four stages run sequentially (40 epochs each by default). Stage boundaries and LR are in `config.py`.

---

## Checkpoints

| File | Use |
|------|-----|
| `stage{N}_best.pth` | **Use these** — lowest val total loss in stage N |
| `final.pth` | Last epoch; often worse than stage best |

**PR1 baseline:** `checkpoint/20260520_163246/stage4_best.pth` (S4 ep 15, val 0.664).

PR1 checkpoints are **incompatible** with pre-PR1 weights (decoder API changed for Z-only anchors).

Loading for inference or continued training:

```python
import torch
from gvae.models.gvae import GVAE

model = GVAE(stage=4)
model.load_state_dict(torch.load("checkpoint/.../stage4_best.pth", map_location="cpu"))
model.eval()
```

---

## Interpreting metrics

Primary val metrics (console + TensorBoard):

| Metric | Good direction | Notes |
|--------|----------------|-------|
| `inst_pos_err_mid` | ↓ | Layout proxy: instance → supernode decode |
| `pos_err_mid` | ↓ | Z-only supernode positions (honest, harder than GT-anchored) |
| `occ_iou_mid` | ↑ | ~73% plateau observed; check precision too |
| `occ_precision_mid` | ↑ | High prec + low recall → conservative blob |
| `soft_miou_mid` | ↑ (slow) | ~10% expected on merged supernode labels; not a go/no-go metric |

Loss totals after PR1 use soft semantic KL — **not comparable** to pre-PR1 runs (~0.4–0.7 vs ~2–5).

---

## Key config knobs

| Knob | Default | Effect |
|------|---------|--------|
| `DECODER_GT_ANCHOR_MIX` | `0.0` | Z-only decoder anchors |
| `REDUCTION_RATIO` | `0.03` | Supernode count; lower = more merging |
| `GRAD_CLIP_NORM` | `1.0` | Gradient clipping (0 = off) |
| `NUM_EPOCHS_STAGE*` | 40 each | Consider shortening S1/S4 (see TODO.md B1/B2) |
| `LOG_FULL_METRICS` | `False` | Extended debug metrics |
| `LAMBDA_*` | see config | Loss balance |

---

## Stability notes

- **GroupNorm** in U-Net (not BatchNorm) — required for batch=1.
- **`S.detach()`** on coarsening feature path — prevents NaN in stage 2+.
- Empty `edge_index` warnings on sparse coarse graphs are benign.
- Training skips graphs with zero coarsenable nodes.

---

## Utilities

| Script | Purpose |
|--------|---------|
| `utils/build_scene_graph.py` | LAZ → JSON + occ caches |
| `utils/visualize_supernodes.py` | Coarsening visualisation |
| `utils/diagnose_nan_losses.py` | Per-graph loss breakdown |
| `utils/smoke_test.py` | Quick forward pass |

---

## Next work

See [TODO.md](../TODO.md) for Layer A (PR2), diagnostics, and ablations.
