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

# Classes that do not form meaningful individual instances (large background regions)
# Set REMOVE_NON_INSTANTIABLE = True to exclude them at load time (JSON files stay intact)
REMOVE_NON_INSTANTIABLE  = True
NON_INSTANTIABLE_CLASSES  = {'ground', 
                             'vegetation', 
                             'fence'}

# Proximitty edge construction: 
# Connect nodes within this distance in normalised space
# It is the distance in normalised [-1,1]³ space below which we consider two objects to be "close" and connect them with an edge in the graph.
EDGE_PROXIMITY = 0.03

# ─── Feature dimension ────────────────────────────────────────────────────────
# Must be divisible by 6 (required by PointROPE)
D_MODEL = 192

# ─── Voxel grid resolutions ───────────────────────────────────────────────────
# Mid latent volume  Z^G_mid  (diffusion level 2)
GRID_MID = (16, 16, 8)        # H × W × D  →  2048 voxels

# Coarse latent volume  Z^G_coarse  (diffusion level 1)
GRID_COARSE = (8, 8, 4)       # H × W × D  →  256 voxels

# ─── Graph coarsening ─────────────────────────────────────────────────────────
REDUCTION_RATIO = 0.03  # keep 3% of nodes at each coarsening step
#SOFTMAX_TEMPERATURE = 0.1  # temperature for soft assignment softmax — lower = sharper (→ collapse), higher = softer (→ uniform); start at 0.1 and increase if collapse persists

# Ball-query radius (in normalised [-1,1]³ space) for edge construction
# after each coarsening step
BALL_QUERY_RADIUS = 0.03
MAX_NUM_NEIGHBORS = 32  # max neighbors per node in ball-query to limit memory

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

# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE      = 3e-4
LEARNING_RATE_FINETUNE = 1e-4  # reduced LR for stage 4 joint fine-tune
BATCH_SIZE         = 8
NUM_EPOCHS_STAGE1  = 40        # coarsening only
NUM_EPOCHS_STAGE2  = 1        # + mid branch
NUM_EPOCHS_STAGE3  = 1        # + coarse branch
NUM_EPOCHS_STAGE4  = 1        # joint fine-tune

# ─── R-GAT ────────────────────────────────────────────────────────────────────
RGAT_HEADS        = 8
RGAT_LAYERS_FINE  = 3          # instance & region levels
RGAT_LAYERS_COARSE = 2         # coarsest level (+ GPS exact attention)

# ─── Road edge reconstruction ─────────────────────────────────────────────────
ROAD_EDGE_NEG_RATIO = 5        # negative sampling ratio for road edges

# ─── Occupancy sampling ───────────────────────────────────────────────────────
OCC_QUERY_POINTS  = 2048       # total query points per scene during training
OCC_POS_RATIO     = 0.5        # fraction sampled from inside occupied voxels
