# config.py — Central configuration for the Scene Graph VAE

# ─── Data ─────────────────────────────────────────────────────────────────────
GRAPH_DATA_DIR = 'data/graphs'   # folder containing scene JSON files (expects train/ and test/ subdirs)

#  Semantic classes 
# From the plan: "label" node attribute — integer for each class.
SEMANTIC_CLASSES = [
    'ground',
    'vegetation',
    'car',
    'powerline',
    'fence',
    'tree',
    'pickup',
    'van_truck',
    'heavy_duty',
    'utility_pole',
    'light_pole',
    'traffic_pole',
    'habitat',
    'complex',
    'annex',
]
NUM_CLASSES = len(SEMANTIC_CLASSES)   # = 15

# Non-instantiable classes (large background regions: ground, vegetation, fence)
NON_INSTANTIABLE_CLASSES = {'ground', 'vegetation', 'fence'}

# B policy: keep them at instance G_L (R-GAT + edges), exclude from coarsening / Z paths
REMOVE_NON_INSTANTIABLE = False
COARSEN_EXCLUDE_NON_INSTANTIABLE = True

# Proximitty edge construction: 
# Connect nodes within this distance in normalised space
# It is the distance in normalised [-1,1]³ space below which we consider two objects to be "close" and connect them with an edge in the graph.
EDGE_PROXIMITY = 0.03

# ─── Feature dimensions (progressive across graph levels) ─────────────────────
# GNN / splat width — each divisible by 6 (PointROPE)
D_MODEL_LEVELS = [72, 144, 288]
D_INSTANCE, D_REGION, D_SCENE = D_MODEL_LEVELS
D_NUM_HEADS = [8, 8, 8]       # GATv2 heads per level (d % heads == 0)

# Z volume channel width (U-Net I/O); may exceed D_MODEL_LEVELS via splat→1×1 proj
# Each dim should be divisible by LATENT_GRAPH_ATTN_HEADS (8)
D_LATENT_LEVELS = [96, 192, 384]
D_FINE_LATENT, D_MID_LATENT, D_COARSE_LATENT = D_LATENT_LEVELS

# ─── Voxel grid resolutions ( cubic, PR2 / Layer A ) ─────────────────────────
# Fine / mid / coarse latent volumes — matched to diffusion hierarchy levels 3 / 2 / 1
GRID_FINE = (64, 64, 8)     # Z^G_fine
GRID_MID = (32, 32, 8)      # Z^G_mid
GRID_COARSE = (16, 16, 4)   # Z^G_coarse

# ─── Graph coarsening ─────────────────────────────────────────────────────────
# Assignment: "hard" = FPS + Voronoi one-hot (default baseline, no trainable coarsening)
#             "soft" = FPS + softmax(-dist/T) with learnable per-level temperature
COARSEN_ASSIGNMENT = "hard"
REDUCTION_RATIO_LEVELS = [0.2, 0.2, 0.2]  # fraction kept per step [instance→fine, fine→mid, mid→coarse]
SOFTMAX_TEMPERATURE = 1.0       # initial T for soft mode
COARSEN_DETACH_FEATURES = True  # soft mode: detach S on p/s/h pooling (pool loss still trains T)

# Pool loss (soft mode only; USE_POOL_LOSS ignored when COARSEN_ASSIGNMENT == "hard")
USE_POOL_LOSS = False
LAMBDA_POOL    = 0.1
LAMBDA_CUT     = 1.0
LAMBDA_ORTHO   = 3.0
LAMBDA_SPATIAL = 5.0

# Ball-query radius (normalised [-1,1]³) after each coarsening step.
# Supernodes are farther apart; use larger radii at coarser levels so E>0.
# [fine graph, region (mid graph), scene (coarse graph)]
BALL_QUERY_RADIUS_LEVELS = [0.05, 0.1, 0.2]
MAX_NUM_NEIGHBORS = 32  # max neighbors per node in ball-query to limit memory

# Occupancy caches: object-only voxels (aligned with object-centric Z, no ground in occ GT)
OCC_FILTER_NON_INSTANTIABLE = True

# ─── Splatting ────────────────────────────────────────────────────────────────
SPLAT_TRUNCATION_SIGMA = 2.0   # truncate Gaussian kernel at ±2σ (mid / coarse)
SPLAT_TRUNCATION_SIGMA_FINE = 1.0   # finer support for instance-level Z_fine (Phase 2A)
SPLAT_FINE_VOXEL_CAP = True         # cap fine splat box to ~N voxel spacings
SPLAT_FINE_VOXEL_RADIUS = 1.0       # max truncation in units of fine-grid voxel spacing
# Floor trunc box per axis (× voxel spacing) so thin r_z still hits nearest voxel
SPLAT_MIN_TRUNC_VOXEL_FRAC = 0.55
SPLAT_SUBTRACT_SPATIAL_MEAN = True   # remove per-channel DC before U-Net (reduces center blob)
SPLAT_EPS = 1e-6               # denominator stabiliser in scatter normalisation
SPLAT_DENSE_MAX_PAIRS = 32_000_000  # use dense N×V path below this pair count
SPLAT_NODE_CHUNK      = 64          # nodes per chunk in chunked-dense path (large scenes)

# ─── 3D U-Net ─────────────────────────────────────────────────────────────────
# Latent volumes use (C, H, W, D) everywhere — matches Conv3d without permutes.
UNET_USE_CHECKPOINT = False     # gradient checkpoint on large grids (≥ UNET_CHECKPOINT_MIN_SIDE)
UNET_CHECKPOINT_MIN_SIDE = 32
UNET_CHANNELS_LAST = True       # channels_last_3d for cudnn conv (avoids layout copies)
UNET_DEPTH_FINE = 3             # depth-1 ablation regressed fine zpos; keep deep fine U-Net
UNET_DEPTH_MID = 3
UNET_DEPTH_COARSE = 2
UNET_ALIGN_CORNERS = True          # match grid_sample (decoder) vs False upsample bias

# ─── Deformable cross-attention decoder ───────────────────────────────────────
NUM_REF_POINTS = 27            # P = 3×3×3 reference points per node
USE_ANCHOR_MLP = True          # 2-layer MLP for p/r anchors (else single Linear)
ANCHOR_MLP_HIDDEN = 0          # 0 → hidden dim = level latent dim d
# Shared trunk on deformable cross-attn z_pred before s/p/r heads (SceneGraphDecoder)
USE_Z_PRED_READOUT_MLP = True
Z_PRED_READOUT_MLP_HIDDEN = 0  # 0 → hidden dim = d
# Position head: "clamp" = linear + clip to [-1,1]; "tanh" saturates before |p|→1
POSITION_BOUND = "clamp"
# Z-only / deformable p: Δp from query (anchor or GT slot), not absolute from biased Z
POSITION_RESIDUAL = True

# ─── Latent graph VAE (honest Z → graph_hat, no h at decode) ─────────────────
# graph → encode → Z → LatentGraphDecoder → graph_hat; metrics on recon_* only.
LATENT_GRAPH_VAE_MODE = False
LAMBDA_RECON_LATENT = 1.0       # reconstruction loss on Z-only slot decoder
LATENT_GRAPH_MAX_SLOTS = 512    # max supernodes per level (queries)
LATENT_GRAPH_ATTN_HEADS = 8
LATENT_GRAPH_SPATIAL_TEMP = 0.05   # lower → sharper softmax over voxels
LATENT_GRAPH_Z_NORM_BIAS = True    # bias spatial logits toward high-||Z|| voxels
LAMBDA_POS_LATENT = 2.0            # position MSE weight in latent-graph recon
LAMBDA_SEM_LATENT = 1.0            # sem/size on honest graph_hat (Z sampled at p_hat)
LAMBDA_SEM_AT_GT = 0.0             # aux sem/size loss: Z@p_gt, does not change graph_hat forward
LAMBDA_SIZE_AT_GT = 0.0            # aux footprint at p_gt (0 = same weight as LAMBDA_SEM_AT_GT path)
LAMBDA_SLOT_SPREAD = 0.5           # penalise distinct slots predicting the same point
LATENT_TRAIN_Z_SAMPLE_AT_GT = False  # deprecated; use LAMBDA_SEM_AT_GT aux instead
LATENT_LAYOUT_HEAD = True          # 1×1 conv → layout volume for peak loss (not full ||Z||)
LATENT_GRAPH_SLOT_SPREAD_SIGMA = 0.12  # min separation scale in normalised space

# ─── Z spatial layout (diffusion-ready latent peaks at supernode sites) ───────
LAMBDA_Z_PEAK_FINE = 0.0         # KL( Gaussian(p_gt) ‖ softmax(||Z||) ) in k-NN window
LAMBDA_Z_PEAK_MID = 0.0
LAMBDA_Z_PEAK_COARSE = 0.0
LATENT_PEAK_NEIGHBORS = 128      # voxels per supernode in peak loss / metrics
LATENT_PEAK_SIGMA = 0.06         # target Gaussian σ in [-1,1] space (~2 fine voxels)
LATENT_PEAK_TEMP = 0.15          # temperature on ||Z|| logits
LATENT_DECODE_QUERY_WEIGHT = 1.0 # slot-query bias inside k-NN decode windows
LATENT_DECODE_MODE = "vgae"      # vgae | query | nms_slots | nms
LATENT_NMS_SUPPRESS_SIGMA = 0.08 # NMS suppression radius in [-1,1] space
LATENT_POS_MATCHED = False       # if True: Hungarian pos loss (hides index mis-alignment)
LATENT_GT_WINDOW_CURRICULUM = False  # query mode only: blend GT k-NN windows → query
LATENT_GT_WINDOW_ANNEAL_EPOCHS = 60  # epochs to go mix 1→0 (0 = query-only forward in train)
LATENT_GT_WINDOW_MIX = 1.0       # runtime; set each epoch by train loop
LAMBDA_QUERY_COVER = 0.0         # min dist slot-query window → p_gt[i]
LAMBDA_INDEX_PEAK_DISTILL = 0.0    # MSE(p_hat[i], oracle peak at k-NN(p_gt[i])) per slot
LAMBDA_LAYOUT_DISTILL = 0.0      # deprecated alias → LAMBDA_INDEX_PEAK_DISTILL

# ─── Z-only decoder (DDM-aligned readout from Z alone) ────────────────────────
USE_Z_ONLY_DECODER = True
# If True, recon_fine['p'] is replaced by ZOnlyDecoder @ p_anchor (hzpos = pos in logs).
# If False (experiment A: --zonly-aux-loss-only), Z-only paths are loss-only; deformable mlp_p → pos.
Z_ONLY_PATCH_DEFORMABLE_POSITION = True
LAMBDA_RECON_H = 1.5           # h+Z deformable decoder (semantics/r; p from deformable or Z@anchor)
LAMBDA_RECON_ZONLY = 1.0       # Z-only at GT slots (DDM readout)
LAMBDA_RECON_HZONLY = 0.8      # Z-only at h-predicted anchors (targets hzpos)
Z_ONLY_QUERY_JITTER = 0.05     # uniform noise on query points in train (0 = sample at p_gt)

# Auxiliary anchor regression: ||bound(mlp_p_anchor(h)) - p_gt||² (fine / mid)
LAMBDA_ANCHOR_FINE = 1.0
LAMBDA_ANCHOR_MID = 0.5
LAMBDA_ANCHOR_COARSE = 0.0
# Anchor footprint: footprint_loss on softplus(mlp_r_anchor(h)) vs r_gt
LAMBDA_ANCHOR_R_FINE = 0.5
LAMBDA_ANCHOR_R_MID = 0.25
LAMBDA_ANCHOR_R_COARSE = 0.0

# Teacher-forcing curriculum on h-decoder reference boxes (training only; val uses mix=0)
ANCHOR_MIX_CURRICULUM = True
ANCHOR_MIX_START = 1.0         # epoch 0: sample Z on GT anchors
ANCHOR_MIX_END = 0.0           # after anneal: h-predicted anchors only
ANCHOR_MIX_ANNEAL_EPOCHS = 40  # linear blend over epochs 0 .. N-1, then hold END

# ─── Z norm contrastive (Probe C: ||Z(p_gt)|| > ||Z(empty)||) ─────────────────
LAMBDA_NORM_CONTRAST_FINE = 0.1
LAMBDA_NORM_CONTRAST_MID = 0.1
LAMBDA_NORM_CONTRAST_COARSE = 0.1
NORM_CONTRAST_MARGIN = 0.0     # hinge: relu(||Z_empty|| - ||Z_gt|| + margin)
NORM_CONTRAST_EMPTY_POINTS = 256

# ─── Loss weights ─────────────────────────────────────────────────────────────
# Total per branch: λ_h·L_recon_h + λ_z·L_recon_zonly + λ_KL·L_KL + …
LAMBDA_KL_MAX   = 1e-3         # β — maximum KL weight after annealing ramp

LAMBDA_SEM      = 1.0          # semantic soft-CE weight (h+Z recon)
LAMBDA_SEM_ZONLY = 2.0         # stronger CE on Z-only / hz paths (zsmiou)
LAMBDA_POS      = 1.0          # position MSE weight (in L_recon)
LAMBDA_SIZE     = 1.0          # footprint loss on h+Z recon (see footprint_loss)
LAMBDA_SIZE_ZONLY = 0.5        # footprint on Z-only paths (single-point readout; softer)
SIZE_LOG_EPS    = 1e-6         # matches encoder log(r + eps)
SIZE_LOG_HUBER_BETA = 0.25     # Smooth-L1 in log-space (~25% relative scale knee)

# Set at train startup: len(train_dataset) * NUM_EPOCHS (graph forwards)
KL_TOTAL_STEPS = 0

# ─── Cyclical KL annealing (Fu et al., 2019) ──────────────────────────────────
KL_ANNEAL_CYCLES  = 4          # number of ramp-hold cycles over full training
KL_ANNEAL_RATIO   = 0.5        # fraction of each cycle spent ramping (vs. holding)

# ─── 3D U-Net normalization (batch=1 scenes → GroupNorm, not BatchNorm) ───────
UNET_NUM_GROUPS = 8

# ─── Device ───────────────────────────────────────────────────────────────────
CUDA_DEVICE = 0

# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE      = 3e-4
LEARNING_RATE_LATE = 1e-4   # LR after LR_DECAY_EPOCH
LR_DECAY_EPOCH     = 120     # epochs 1..40 at LEARNING_RATE, then LEARNING_RATE_LATE
BATCH_SIZE         = 2
NUM_EPOCHS         = 150

# Performance (training throughput)
USE_AMP                 = True
SEQUENTIAL_BACKWARD     = True   # fine→mid→coarse backward separately (disabled under AMP)
DATALOADER_NUM_WORKERS  = 2
DATALOADER_PIN_MEMORY   = True

# Decoder — sampling anchors from h (not GT); DECODER_GT_ANCHOR_MIX>0 only for probe ablations
DECODER_GT_ANCHOR_MIX = 0.0

# Validation metrics
SOFT_MIOU_EPS = 1e-6          # min soft class mass to include in soft mIoU mean
LOG_FULL_METRICS = True      # if True, log extended debug metrics to TensorBoard

# Training stability
GRAD_CLIP_NORM = 1.0          # max grad norm before optimizer.step (0 = disabled)

# ─── Occupancy (point-cloud voxelisation) ─────────────────────────────────────
OCC_CACHE_SUFFIX_FINE   = '_occ_fine.npy'
OCC_CACHE_SUFFIX_MID    = '_occ_mid.npy'
OCC_CACHE_SUFFIX_COARSE = '_occ_coarse.npy'
OCC_MAX_POINTS          = 500_000            # subsample LiDAR when building caches
OCC_REQUIRE_CACHE       = True               # raise if caches missing at load time


def apply_latent_levels(fine: int, mid: int, coarse: int) -> None:
    """Set Z^G_* channel dims (call from train CLI before building the model)."""
    levels = [fine, mid, coarse]
    heads = LATENT_GRAPH_ATTN_HEADS
    for d in levels:
        if d % heads != 0:
            raise ValueError(
                f"latent dim {d} must be divisible by LATENT_GRAPH_ATTN_HEADS={heads}",
            )
    global D_LATENT_LEVELS, D_FINE_LATENT, D_MID_LATENT, D_COARSE_LATENT
    D_LATENT_LEVELS = levels
    D_FINE_LATENT, D_MID_LATENT, D_COARSE_LATENT = levels
