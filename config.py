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
# [instance G_L, region G_{L-1}, scene G_1] — each divisible by 6 (PointROPE)
D_MODEL_LEVELS = [72, 144, 288]
D_INSTANCE, D_REGION, D_SCENE = D_MODEL_LEVELS
D_MID_LATENT = D_REGION       # Z^G_mid  channel dim
D_COARSE_LATENT = D_SCENE     # Z^G_coarse channel dim
D_NUM_HEADS = [8, 8, 8]       # GATv2 heads per level (d % heads == 0)

# ─── Voxel grid resolutions ───────────────────────────────────────────────────
# Mid latent volume  Z^G_mid  (diffusion level 2)
GRID_MID = (16, 16, 8)        # H × W × D  →  2048 voxels

# Coarse latent volume  Z^G_coarse  (diffusion level 1)
GRID_COARSE = (8, 8, 4)       # H × W × D  →  256 voxels

# ─── Graph coarsening ─────────────────────────────────────────────────────────
REDUCTION_RATIO = 0.25  # keep 25% of nodes at each coarsening step

# Ball-query radius (normalised [-1,1]³) after each coarsening step.
# Supernodes are farther apart; use larger radii at coarser levels so E>0.
# [region (mid graph), scene (coarse graph)]
BALL_QUERY_RADIUS_LEVELS = [0.1, 0.2]
# Legacy alias (region / first coarsening step)
BALL_QUERY_RADIUS = BALL_QUERY_RADIUS_LEVELS[0]
MAX_NUM_NEIGHBORS = 32  # max neighbors per node in ball-query to limit memory

# Occupancy caches: object-only voxels (aligned with object-centric Z, no ground in occ GT)
OCC_FILTER_NON_INSTANTIABLE = True

# ─── Splatting ────────────────────────────────────────────────────────────────
SPLAT_TRUNCATION_SIGMA = 2.0   # truncate Gaussian kernel at ±2σ
SPLAT_EPS = 1e-6               # denominator stabiliser in scatter normalisation

# ─── Deformable cross-attention decoder ───────────────────────────────────────
NUM_REF_POINTS = 27            # P = 3×3×3 reference points per node

# ─── Loss weights ─────────────────────────────────────────────────────────────
# Total loss = L_recon + λ_KL * L_KL + λ_pool * L_pool + λ_occ * L_occ
# but we add some scaling factor to control how much each component loss contributes to the total loss
LAMBDA_KL_MAX   = 1e-3         # β — maximum KL weight after annealing ramp
LAMBDA_OCC      = 1.0          # occupancy BCE weight
LAMBDA_POOL     = 0.1          # MinCutPool + spatial compactness weight
LAMBDA_EDGE     = 1.0          # proximity edge margin loss weight
LAMBDA_SEM      = 1.0          # semantic CE weight
LAMBDA_POS      = 1.0          # position MSE weight
LAMBDA_FOOTPRINT = 1.0         # footprint MSE weight

# ─── Cyclical KL annealing (Fu et al., 2019) ──────────────────────────────────
KL_ANNEAL_CYCLES  = 4          # number of ramp-hold cycles over full training
KL_ANNEAL_RATIO   = 0.5        # fraction of each cycle spent ramping (vs. holding)

# ─── 3D U-Net normalization (batch=1 scenes → GroupNorm, not BatchNorm) ───────
UNET_NUM_GROUPS = 8

# ─── Device ───────────────────────────────────────────────────────────────────
CUDA_DEVICE = 1

# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE      = 3e-4
LEARNING_RATE_FINETUNE = 1e-4  # reduced LR for stage 4 joint fine-tune
BATCH_SIZE         = 8
NUM_EPOCHS_STAGE1  = 40        # coarsening only
NUM_EPOCHS_STAGE2  = 40        # + mid branch
NUM_EPOCHS_STAGE3  = 40        # + coarse branch
NUM_EPOCHS_STAGE4  = 40        # joint fine-tune
LOG_COARSE_PREVIEW_AT_STAGE2 = True  # log coarse-branch losses (not in total) during stage 2

# ─── R-GAT ────────────────────────────────────────────────────────────────────
RGAT_HEADS        = 8
RGAT_LAYERS_FINE  = 3          # instance & region levels
RGAT_LAYERS_COARSE = 2         # coarsest level (+ GPS exact attention)

# ─── Road edge reconstruction ─────────────────────────────────────────────────
ROAD_EDGE_NEG_RATIO = 5        # negative sampling ratio for road edges

# ─── Occupancy (point-cloud voxelisation) ─────────────────────────────────────
OCC_CACHE_SUFFIX_MID    = '_occ_mid.npy'     # sidecar next to scene JSON
OCC_CACHE_SUFFIX_COARSE = '_occ_coarse.npy'
OCC_MAX_POINTS          = 500_000            # subsample LiDAR when building caches
OCC_REQUIRE_CACHE       = True               # raise if caches missing at load time
OCC_QUERY_POINTS        = 2048               # query points per scene during training
OCC_POS_RATIO           = 0.5                # fraction sampled from occupied voxels
