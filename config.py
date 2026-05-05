# config.py — Central configuration for the Scene Graph VAE

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

# Proximitty edge construction: 
# Connect nodes within this distance in normalised space
# It is the distance in normalised [-1,1]³ space below which we consider two objects to be "close" and connect them with an edge in the graph.
EDGE_PROXIMITY = 0.3

# ─── Feature dimension ────────────────────────────────────────────────────────
# Must be divisible by 6 (required by PointROPE)
D_MODEL = 192

# ─── Voxel grid resolutions ───────────────────────────────────────────────────
# Coarse latent volume  Z^G_coarse  (diffusion level 1)
GRID_COARSE = (8, 8, 4)       # H × W × D  →  256 voxels

# Mid latent volume  Z^G_mid  (diffusion level 2)
GRID_MID = (16, 16, 8)        # H × W × D  →  2048 voxels

# ─── Graph coarsening ─────────────────────────────────────────────────────────
# Reduction ratio r: each FPS step keeps ⌊|V| / r⌋ nodes
REDUCTION_RATIO = 4

# Ball-query radius (in normalised [-1,1]³ space) for edge construction
# after each coarsening step
BALL_QUERY_RADIUS = 0.3
MAX_NUM_NEIGHBORS = 32  # max neighbors per node in ball-query to limit memory

# ─── Splatting ────────────────────────────────────────────────────────────────
SPLAT_TRUNCATION_SIGMA = 2.0   # truncate Gaussian kernel at ±2σ
SPLAT_EPS = 1e-6               # denominator stabiliser in scatter normalisation

# ─── Deformable cross-attention decoder ───────────────────────────────────────
NUM_REF_POINTS = 27            # P = 3×3×3 reference points per node

# ─── Loss weights ─────────────────────────────────────────────────────────────
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

# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE      = 3e-4
LEARNING_RATE_FINETUNE = 1e-4  # reduced LR for stage 4 joint fine-tune
BATCH_SIZE         = 8
NUM_EPOCHS_STAGE1  = 20        # coarsening only
NUM_EPOCHS_STAGE2  = 40        # + mid branch
NUM_EPOCHS_STAGE3  = 40        # + coarse branch
NUM_EPOCHS_STAGE4  = 60        # joint fine-tune

# ─── R-GAT ────────────────────────────────────────────────────────────────────
RGAT_HEADS        = 8
RGAT_LAYERS_FINE  = 3          # instance & region levels
RGAT_LAYERS_COARSE = 2         # coarsest level (+ GPS exact attention)

# ─── Road edge reconstruction ─────────────────────────────────────────────────
ROAD_EDGE_NEG_RATIO = 5        # negative sampling ratio for road edges

# ─── Occupancy sampling ───────────────────────────────────────────────────────
OCC_QUERY_POINTS  = 2048       # total query points per scene during training
OCC_POS_RATIO     = 0.5        # fraction sampled from inside occupied voxels
