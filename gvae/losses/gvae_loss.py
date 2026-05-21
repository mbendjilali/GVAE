# gvae/losses/gvae_loss.py
# All loss terms: recon, voxel-wise KL, occupancy, pool + cyclical KL schedule
# TODO: implement (todo id: losses)

import warnings

import torch
import torch.nn.functional as F
import config
from gvae.data.graph_masks import pool_subgraph
from gvae.data.occupancy import sample_occupancy_queries

def kl_weight(step):
    total_steps = (config.NUM_EPOCHS_STAGE1 + config.NUM_EPOCHS_STAGE2 +
                   config.NUM_EPOCHS_STAGE3 + config.NUM_EPOCHS_STAGE4)  # approximate total
    cycle_len = max(1, total_steps // config.KL_ANNEAL_CYCLES)
    ramp_len  = int(cycle_len * config.KL_ANNEAL_RATIO)
    pos_in_cycle = step % cycle_len
    # linear ramp up from 0 to LAMBDA_KL_MAX over ramp_len steps, then hold
    return config.LAMBDA_KL_MAX * min(1.0, pos_in_cycle / max(1, ramp_len))


def reconstruction_loss(recon, p_true, r_true, s_true, edge_index, edge_margin=None):
    # recon is a dict with keys "p", "r", "s" containing the predicted values at the reference points
    # p_true, r_true, s_true are the ground truth values for the nodes in the original graph
    delta = edge_margin if edge_margin is not None else config.BALL_QUERY_RADIUS

    # semantic: cross-entropy loss
    L_sem = F.cross_entropy(recon['s'], s_true)

    # position and radius: MSE
    L_pos = F.mse_loss(recon['p'], p_true)
    L_size = F.mse_loss(recon['r'], r_true)

    p_pred = recon['p']  # (N, 3)
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        warnings.warn("reconstruction_loss: empty edge_index", stacklevel=2)
        L_edge_pos = p_pred.new_zeros(())
        L_edge_neg = p_pred.new_zeros(())
    else:
        d_pos = torch.norm(p_pred[edge_index[0]] - p_pred[edge_index[1]], dim=1)
        L_edge_pos = torch.clamp(d_pos - delta, min=0).mean()
        N = p_true.shape[0]
        neg = torch.randint(0, N, (2, num_edges), device=p_pred.device)
        d_neg = torch.norm(p_pred[neg[0]] - p_pred[neg[1]], dim=1)
        L_edge_neg = torch.clamp(delta - d_neg, min=0).mean()

    return (config.LAMBDA_SEM * L_sem
            + config.LAMBDA_POS * L_pos
            + config.LAMBDA_POS * L_size
            + config.LAMBDA_EDGE * L_edge_pos
            + config.LAMBDA_EDGE * L_edge_neg)
    

def KL_loss(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum() / mu.numel()


def loss_pool(S, edge_index, p, N_nodes):
    M = S.shape[1]

    num_edges = edge_index.shape[1]
    deg = torch.zeros(N_nodes, device=S.device)
    if num_edges == 0:
        warnings.warn("loss_pool: empty edge_index", stacklevel=2)
    else:
        deg.scatter_add_(0, edge_index[0], torch.ones(num_edges, device=S.device))
    D_S = (deg.unsqueeze(1) * S)
    if num_edges > 0:
        StS = S[edge_index[0]] * S[edge_index[1]]
        tr_SAS = StS.sum()
    else:
        tr_SAS = S.new_zeros(())
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

    total = config.LAMBDA_CUT * cut_loss + config.LAMBDA_ORTHO * ortho_loss + config.LAMBDA_SPATIAL * spatial_loss
    return total, cut_loss, ortho_loss, spatial_loss

def loss_occupancy(occ_readout, z, occ_grid):
    """Supervise latent z against point-voxelised occupancy (not graph bounding boxes)."""
    q, labels = sample_occupancy_queries(occ_grid)
    logits = occ_readout(q, z)
    return F.binary_cross_entropy_with_logits(logits, labels)

# total loss
def compute_loss(outputs, graph, step, stage=4):
    # unpack ground truth
    p, r, s, edge_index, label = graph.p, graph.r, graph.s, graph.edge_index, graph.label

    # pool loss — S1 only on coarsenable instance nodes (B policy)
    ei_pool, p_pool = pool_subgraph(edge_index, p, graph.coarsen_mask)
    n_pool = p_pool.shape[0]
    if n_pool > 0 and outputs['S1'].numel() > 0:
        L_pool_s1, L_cut_s1, L_ortho_s1, L_spatial_s1 = loss_pool(outputs['S1'], ei_pool, p_pool, n_pool)
    else:
        L_pool_s1 = L_cut_s1 = L_ortho_s1 = L_spatial_s1 = p.new_zeros(())
    L_pool_s2, L_cut_s2, L_ortho_s2, L_spatial_s2 = loss_pool(
        outputs['S2'], outputs['edge_index_lm1'], outputs['p_lm1'], outputs['p_lm1'].shape[0],
    )
    L_pool    = L_pool_s1    + L_pool_s2
    L_cut     = L_cut_s1     + L_cut_s2
    L_ortho   = L_ortho_s1   + L_ortho_s2
    L_spatial = L_spatial_s1 + L_spatial_s2

    if stage == 1:
        # only coarsening is trained — no point computing recon/KL/occ
        total = config.LAMBDA_POOL * L_pool
        return total, {'pool': L_pool, 'pool_cut': L_cut, 'pool_ortho': L_ortho, 'pool_spatial': L_spatial, 'lambda_kl': 0.0}

    # stage 2+: mid branch (encoder + decoder) is active
    lambda_kl = kl_weight(step)
    L_recon = reconstruction_loss(
        outputs['recon_mid'], outputs['p_lm1'], outputs['r_lm1'], outputs['s_lm1'],
        outputs['edge_index_lm1'], edge_margin=config.BALL_QUERY_RADIUS_LEVELS[0],
    )
    L_KL    = KL_loss(outputs['mu_mid'], outputs['logvar_mid'])
    L_occ   = loss_occupancy(outputs['occ_readout_mid'], outputs['z_mid'], graph.occ_mid)

    extras = {}
    if stage == 2 and config.LOG_COARSE_PREVIEW_AT_STAGE2:
        extras['recon_coarse_preview'] = reconstruction_loss(
            outputs['recon_coarse'], outputs['p_1'], outputs['r_1'], outputs['s_1'],
            outputs['edge_index_1'], edge_margin=config.BALL_QUERY_RADIUS_LEVELS[1],
        )
        extras['KL_coarse_preview'] = KL_loss(outputs['mu_coarse'], outputs['logvar_coarse'])
        extras['occ_coarse_preview'] = loss_occupancy(
            outputs['occ_readout_coarse'], outputs['z_coarse'], graph.occ_coarse,
        )

    if stage >= 3:
        L_recon = L_recon + reconstruction_loss(
            outputs['recon_coarse'], outputs['p_1'], outputs['r_1'], outputs['s_1'],
            outputs['edge_index_1'], edge_margin=config.BALL_QUERY_RADIUS_LEVELS[1],
        )
        L_KL = L_KL + KL_loss(outputs['mu_coarse'], outputs['logvar_coarse'])
        L_occ = L_occ + loss_occupancy(
            outputs['occ_readout_coarse'], outputs['z_coarse'], graph.occ_coarse,
        )

    total = L_recon + lambda_kl * L_KL + config.LAMBDA_POOL * L_pool + config.LAMBDA_OCC * L_occ

    components = {'recon': L_recon, 'KL': L_KL, 'pool': L_pool, 'pool_cut': L_cut, 'pool_ortho': L_ortho, 'pool_spatial': L_spatial, 'occ': L_occ, 'lambda_kl': lambda_kl}
    components.update(extras)
    return total, components