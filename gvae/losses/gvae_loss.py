# gvae/losses/gvae_loss.py
# All loss terms: recon, voxel-wise KL, occupancy, pool + cyclical KL schedule
# TODO: implement (todo id: losses)

import torch
import torch.nn.functional as F
import config
from gvae.data.occupancy import sample_occupancy_queries

def kl_weight(step):
    total_steps = (config.NUM_EPOCHS_STAGE1 + config.NUM_EPOCHS_STAGE2 +
                   config.NUM_EPOCHS_STAGE3 + config.NUM_EPOCHS_STAGE4)  # approximate total
    cycle_len = max(1, total_steps // config.KL_ANNEAL_CYCLES)
    ramp_len  = int(cycle_len * config.KL_ANNEAL_RATIO)
    pos_in_cycle = step % cycle_len
    # linear ramp up from 0 to LAMBDA_KL_MAX over ramp_len steps, then hold
    return config.LAMBDA_KL_MAX * min(1.0, pos_in_cycle / max(1, ramp_len))


def reconstruction_loss(recon, p_true, r_true, s_true, edge_index):
    # recon is a dict with keys "p", "r", "s" containing the predicted values at the reference points
    # p_true, r_true, s_true are the ground truth values for the nodes in the original graph
    delta = config.BALL_QUERY_RADIUS  # margin for edge loss

    # semantic: cross-entropy loss
    L_sem = F.cross_entropy(recon['s'], s_true)

    # position and radius: MSE
    L_pos = F.mse_loss(recon['p'], p_true)
    L_size = F.mse_loss(recon['r'], r_true)

    # edge loss: connected pairs should be close, non-connected far
    p_pred = recon['p'] # (N, 3)
    # distance between connected nodes
    d_pos = torch.norm(p_pred[edge_index[0]] - p_pred[edge_index[1]], dim=1) # (E,)
    L_edge_pos = torch.clamp(d_pos - delta, min=0).mean() 

    # sample random non-edge (same count as edges)
    N = p_true.shape[0]
    num_edges = edge_index.shape[1]
    # random pairs as negative samples
    neg = torch.randint(0, N, (2, num_edges), device=p_true.device)
    d_neg = torch.norm(p_pred[neg[0]] - p_pred[neg[1]], dim=1)
    L_edge_neg = torch.clamp(delta - d_neg, min=0).mean()

    return (config.LAMBDA_SEM * L_sem
            + config.LAMBDA_POS * L_pos
            + config.LAMBDA_POS * L_size
            + config.LAMBDA_EDGE * L_edge_pos
            + config.LAMBDA_EDGE * L_edge_neg)
    

def KL_loss(mu, logvar):
    # voxel-wise KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # normalise by number of voxels
    return kl.sum() / mu.numel()


def loss_pool(S, edge_index, p, N_nodes):
    # S: soft assignment matrix (N, M) where N is number of original nodes and M is number of supernodes
    # edge_index: (2, E) edges in the original graph
    
    M = S.shape[1] # number of supernodes

    # 1. cut loss
    deg = torch.zeros(N_nodes, device=S.device) # degree of each node
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=S.device)) # count degree from edge_index
    D_S = (deg.unsqueeze(1) * S)
    SAS = S.T @ D_S # approximate: uses D instead of A to simplicity
    # trace of S^T A S - use edge contributions
    StS = S[edge_index[0]] * S[edge_index[1]] # (E, M) contribution of each edge to each supernode
    tr_SAS = StS.sum() # sum contributions of all edges to get trace
    tr_DDS = (D_S * S).sum() # sum of D_ii * S[i,j]^2
    cut_loss = - tr_SAS / (tr_DDS + 1e-6)

    # 2. othogonality loss
    StS_mat = S.T @ S # (M, M) should be close to diagonal
    StS_norm = StS_mat / (StS_mat.norm() + 1e-6) # normalise for stability
    I_norm = torch.eye(M, device=S.device) / (M ** 0.5) # normalised identity matrix
    ortho_loss = (StS_norm - I_norm).norm() 

    # 3. spatial compactness loss
    p_super = (S.T @ p) / (S.sum(dim=0).unsqueeze(1) + 1e-6) # (M, 3) supernode positions
    diff = p.unsqueeze(1) - p_super.unsqueeze(0) # (N, M, 3) difference between each node and each supernode
    dist_sq = (diff ** 2).sum(dim=2) # (N, M) squared distance
    spatial_loss = (S * dist_sq).sum() / (S.sum() + 1e-6) # weighted average distance of nodes to their assigned supernodes

    return cut_loss + ortho_loss + spatial_loss

def loss_occupancy(occ_readout, z, occ_grid):
    """Supervise latent z against point-voxelised occupancy (not graph bounding boxes)."""
    q, labels = sample_occupancy_queries(occ_grid)
    logits = occ_readout(q, z)
    return F.binary_cross_entropy_with_logits(logits, labels)

# total loss
def compute_loss(outputs, graph, step, stage=4):
    # unpack ground truth
    p, r, s, edge_index, label = graph.p, graph.r, graph.s, graph.edge_index, graph.label

    # pool loss — always active: coarsening is trained from stage 1
    # S1: original → mid-level  → use original edge_index and p
    # S2: mid → coarse-level    → use mid-level edge_index and p
    L_pool = (loss_pool(outputs['S1'], edge_index, p, p.shape[0]) +
              loss_pool(outputs['S2'], outputs['edge_index_lm1'], outputs['p_lm1'], outputs['p_lm1'].shape[0]))

    if stage == 1:
        # only coarsening is trained — no point computing recon/KL/occ
        total = config.LAMBDA_POOL * L_pool
        return total, {'pool': L_pool, 'lambda_kl': 0.0}

    # stage 2+: mid branch (encoder + decoder) is active
    lambda_kl = kl_weight(step)
    L_recon = reconstruction_loss(outputs['recon_mid'], outputs['p_lm1'], outputs['r_lm1'], outputs['s_lm1'], outputs['edge_index_lm1'])
    L_KL    = KL_loss(outputs['mu_mid'], outputs['logvar_mid'])
    L_occ   = loss_occupancy(outputs['occ_readout'], outputs['z_mid'], graph.occ_mid)

    if stage >= 3:
        # coarse branch is also active — add its contributions
        L_recon = L_recon + reconstruction_loss(outputs['recon_coarse'], outputs['p_1'], outputs['r_1'], outputs['s_1'], outputs['edge_index_1'])
        L_KL    = L_KL    + KL_loss(outputs['mu_coarse'], outputs['logvar_coarse'])
        L_occ   = L_occ   + loss_occupancy(outputs['occ_readout'], outputs['z_coarse'], graph.occ_coarse)

    total = L_recon + lambda_kl * L_KL + config.LAMBDA_POOL * L_pool + config.LAMBDA_OCC * L_occ

    return total, {'recon': L_recon,
                   'KL':    L_KL,
                   'pool':  L_pool,
                   'occ':   L_occ,
                   'lambda_kl': lambda_kl}

