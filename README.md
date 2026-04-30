# Scene Graph Variational Autoencoder (GVAE)

## Overview

The GVAE is a standalone module that compresses a structured 3D outdoor scene graph into two dense spatial volumes of latent features:

- a **coarse volume** matched to diffusion level 1
- a **medium volume** matched to diffusion level 2

Both volumes are KL-regularised toward a standard Gaussian, so the diffusion model can treat them as a layout prior and condition on them directly.

---

## Input

A scene graph where every node carries:

| Attribute | Description |
|---|---|
| Position | 3D centroid coordinates |
| Footprint | Semi-axes of the volumetric extent |
| Semantic label | One-hot class vector |
| Node type | Instance, road segment, junction, etc. |

Edges encode **physical proximity** and **road connectivity**.

---

## Encoder — Three-Level Sequential Chain

### Level 1 — Instance graph (G_L)

The instance graph passes through a shallow 2–3 layer **Relational Graph Attention Network (R-GAT)**. Attention uses **PointROPE**: a parameter-free rotary positional embedding that divides the feature dimension equally across the x, y, z axes and applies standard 1D rotary embeddings independently per axis. This makes every attention operation implicitly aware of metric 3D geometry between nodes, with no learned positional parameters.

### Coarsening: G_L → G_{L-1}

**Farthest Point Sampling (FPS)** on 3D node positions produces region-level supernodes. Spatial contiguity is guaranteed by construction — nearby instances are always grouped together.

Each supernode inherits:
- **Position**: weighted average of member centroids
- **Footprint**: bounding box of the cluster
- **Semantics**: soft average of member labels

The embedding passed upward is the **pooled output of the instance-level R-GAT** — not the raw supernode attributes. This preserves local relational context across levels.

### Level 2 — Region graph (G_{L-1})

The region encoder is initialised from the concatenation of inherited embeddings and the supernode's own geometric attributes, projected to the working dimension `d`. A second shallow R-GAT with PointROPE refines these embeddings over the coarser graph topology.

### Coarsening: G_{L-1} → G_1

A second FPS step produces scene-level supernodes — a handful of large spatial regions covering the full scene. The same embedding inheritance applies.

### Level 3 — Scene graph (G_1)

Because the node count is now small (~10–30), a **full exact multi-head attention** with PointROPE is applied over all supernodes simultaneously. This is the only level where full-graph communication is both necessary and computationally affordable. At finer levels, the fixed-hop R-GAT receptive field is sufficient because coarsening automatically expands the physical extent covered by each hop.

---

## Splatting and Latent Volume Production

> Applies independently to the **region** and **scene** levels.

Each node scatters its embedding across the target voxel grid via **anisotropic Gaussian splatting**: the kernel bandwidth is set directly from the node's footprint semi-axes. The kernel is truncated at ±2σ per axis and accumulated via sparse `scatter_add` operations.

The splatted volume feeds a **3D U-Net** that:
1. propagates information from node-occupied cells into empty background cells
2. smooths the features into a continuous, coherent spatial field

The U-Net is **shallower at the coarse level** (large-footprint supernodes → dense coverage) and **deeper at the region level** (smaller footprints → more inpainting needed).

A final convolutional head produces `(μ, log σ²)` per voxel. A reparameterised sample `Z = μ + σ ⊙ ε` forms the latent volume.

| Output | Resolution | Voxels |
|---|---|---|
| `Z^G_coarse` | 8 × 8 × 4 | 256 |
| `Z^G_mid` | 16 × 16 × 8 | 2 048 |

---

## Decoder — Training Signal Only

The decoder reads spatial information back from the latent volume to each node using **deformable cross-attention**:

1. 27 reference points are placed on a 3×3×3 grid within the node's footprint bounding box
2. Learned offsets derived from the node embedding shift these points
3. Latent features at the reference points are aggregated via cross-attention

This is the structural inverse of splatting: where splatting distributes a node's embedding outward to voxels, the decoder aggregates voxel features back to each node over a footprint-bounded neighbourhood.

Three MLP heads predict: **semantic distribution**, **centroid position**, and **footprint**.

**Edge reconstruction** uses a position-consistency margin loss rather than binary BCE on the adjacency matrix, avoiding the severe class imbalance that arises from sparse scene graphs.

**Occupancy loss**: query points are sampled in 3D space, the latent volume is queried via cross-attention, and binary occupancy is predicted. Ground truth comes from the actual 3D scene voxelised at the target resolution — not from a footprint proxy. Query points are sampled half inside, half outside occupied voxels.

---

## Training — Four Stages

| Stage | What is active | Purpose |
|---|---|---|
| 1 | Coarsening modules only | Establish stable, spatially coherent supernodes |
| 2 | + Instance R-GAT, first coarsening, region encoder, mid pipeline and decoder | Train the medium-resolution branch |
| 3 | + Second coarsening, scene encoder, coarse pipeline and decoder | Train the coarse-resolution branch |
| 4 | Full network, reduced learning rate | Joint fine-tuning |

The KL regularisation weight follows a **cyclical annealing schedule** throughout: it ramps from zero, holds, then repeats. This prevents posterior collapse while maintaining a useful variational bottleneck.

---

## Inference

The GVAE is completely independent of the diffusion model. At inference, the encoder chain produces `Z^G_coarse` and `Z^G_mid` directly from the scene graph. These are consumed by diffusion levels 1 and 2 as layout conditioning signals. All finer diffusion levels are conditioned exclusively on the cascade of coarser diffusion outputs — the scene graph plays no role beyond level 2.

> The graph controls the general spatial layout, and then steps aside.
