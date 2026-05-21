# gvae/losses/gvae_loss.py
# Loss terms: recon, voxel-wise KL, occupancy + cyclical KL schedule

import warnings

import torch
import torch.nn.functional as F
import config
from gvae.data.graph_masks import pool_subgraph
from gvae.data.occupancy import occ_query_count, sample_occupancy_queries

def kl_weight(step):
    total_steps = config.NUM_EPOCHS
    cycle_len = max(1, total_steps // config.KL_ANNEAL_CYCLES)
    ramp_len  = int(cycle_len * config.KL_ANNEAL_RATIO)
    pos_in_cycle = step % cycle_len
    return config.LAMBDA_KL_MAX * min(1.0, pos_in_cycle / max(1, ramp_len))


def soft_semantic_loss(pred_probs: torch.Tensor, true_soft: torch.Tensor) -> torch.Tensor:
    """KL(pred || true) for soft supernode semantic targets (rows sum to 1)."""
    true = true_soft / true_soft.sum(dim=1, keepdim=True).clamp(min=config.SOFT_MIOU_EPS)
    log_pred = pred_probs.clamp(min=config.SOFT_MIOU_EPS).log()
    return F.kl_div(log_pred, true, reduction='batchmean')


def reconstruction_loss(recon, p_true, r_true, s_true, edge_index, edge_margin=None):
    delta = edge_margin if edge_margin is not None else config.BALL_QUERY_RADIUS

    L_sem = soft_semantic_loss(recon['s'], s_true)

    L_pos = F.mse_loss(recon['p'], p_true)
    L_size = F.mse_loss(recon['r'], r_true)

    p_pred = recon['p']
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


def loss_occupancy(occ_readout, z, occ_grid):
    """Supervise latent z against point-voxelised occupancy (not graph bounding boxes)."""
    n_q = occ_query_count(occ_grid)
    q, labels = sample_occupancy_queries(occ_grid, n_queries=n_q)
    logits = occ_readout(q, z)
    return F.binary_cross_entropy_with_logits(logits, labels)


def loss_pool(S, edge_index, p, N_nodes):
    """MinCut-style pool regularisation on soft assignment S."""
    M = S.shape[1]
    num_edges = edge_index.shape[1]
    deg = torch.zeros(N_nodes, device=S.device)
    if num_edges == 0:
        warnings.warn("loss_pool: empty edge_index", stacklevel=2)
    else:
        deg.scatter_add_(0, edge_index[0], torch.ones(num_edges, device=S.device))
    D_S = deg.unsqueeze(1) * S
    if num_edges > 0:
        tr_SAS = (S[edge_index[0]] * S[edge_index[1]]).sum()
    else:
        tr_SAS = S.new_zeros(())
    tr_DDS = (D_S * S).sum()
    cut_loss = -tr_SAS / (tr_DDS + 1e-6)

    StS_mat = S.T @ S
    StS_norm = StS_mat / (StS_mat.norm() + 1e-6)
    I_norm = torch.eye(M, device=S.device) / (M ** 0.5)
    ortho_loss = (StS_norm - I_norm).norm()

    p_super = (S.T @ p) / (S.sum(dim=0).unsqueeze(1) + 1e-6)
    diff = p.unsqueeze(1) - p_super.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=2)
    spatial_loss = (S * dist_sq).sum() / (S.sum() + 1e-6)

    total = (
        config.LAMBDA_CUT * cut_loss
        + config.LAMBDA_ORTHO * ortho_loss
        + config.LAMBDA_SPATIAL * spatial_loss
    )
    return total, cut_loss, ortho_loss, spatial_loss


def compute_pool_loss(outputs, graph) -> tuple[torch.Tensor, dict]:
    """Pool loss on S1 (coarsenable instances) and S2 (mid graph)."""
    p = graph.p
    zero = p.new_zeros(())
    ei_pool, p_pool = pool_subgraph(graph.edge_index, p, graph.coarsen_mask)
    n_pool = p_pool.shape[0]

    if n_pool > 0 and outputs['S1'].numel() > 0:
        L_pool_s1, L_cut_s1, L_ortho_s1, L_spatial_s1 = loss_pool(
            outputs['S1'], ei_pool, p_pool, n_pool,
        )
    else:
        L_pool_s1 = L_cut_s1 = L_ortho_s1 = L_spatial_s1 = zero

    L_pool_s2, L_cut_s2, L_ortho_s2, L_spatial_s2 = loss_pool(
        outputs['S2'], outputs['edge_index_lm1'], outputs['p_lm1'], outputs['p_lm1'].shape[0],
    )
    L_pool = L_pool_s1 + L_pool_s2
    parts = {
        'pool': L_pool,
        'pool_cut': L_cut_s1 + L_cut_s2,
        'pool_ortho': L_ortho_s1 + L_ortho_s2,
        'pool_spatial': L_spatial_s1 + L_spatial_s2,
    }
    return config.LAMBDA_POOL * L_pool, parts


def _maybe_branch_loss(
    branches,
    recon,
    p_true,
    r_true,
    s_true,
    edge_index,
    edge_margin,
    mu,
    logvar,
    occ_readout,
    z,
    occ_grid,
    name: str,
    lambda_kl: float,
):
    if recon is None or p_true.numel() == 0:
        return
    L_recon = reconstruction_loss(
        recon, p_true, r_true, s_true, edge_index, edge_margin=edge_margin,
    )
    L_kl = KL_loss(mu, logvar)
    L_occ = loss_occupancy(occ_readout, z, occ_grid)
    total = L_recon + lambda_kl * L_kl + config.LAMBDA_OCC * L_occ
    branches.append((
        name,
        total,
        {'recon': L_recon, 'KL': L_kl, 'occ': L_occ},
    ))


def compute_branch_losses(outputs, graph, step, stage=1):
    """
    Per-branch losses for sequential backward (fine → mid → coarse).

    Returns:
        branches: list of (name, total_loss, partial_components)
        lambda_kl: float
    """
    lambda_kl = kl_weight(step)
    branches = []

    _maybe_branch_loss(
        branches,
        outputs.get('recon_fine'),
        outputs['p_inst'], outputs['r_inst'], outputs['s_inst'],
        outputs['edge_index_inst'],
        config.EDGE_PROXIMITY,
        outputs['mu_fine'], outputs['logvar_fine'],
        outputs['occ_readout_fine'], outputs['z_fine'], graph.occ_fine,
        'fine', lambda_kl,
    )
    _maybe_branch_loss(
        branches,
        outputs.get('recon_mid'),
        outputs['p_lm1'], outputs['r_lm1'], outputs['s_lm1'],
        outputs['edge_index_lm1'],
        config.BALL_QUERY_RADIUS_LEVELS[0],
        outputs['mu_mid'], outputs['logvar_mid'],
        outputs['occ_readout_mid'], outputs['z_mid'], graph.occ_mid,
        'mid', lambda_kl,
    )
    _maybe_branch_loss(
        branches,
        outputs.get('recon_coarse'),
        outputs['p_1'], outputs['r_1'], outputs['s_1'],
        outputs['edge_index_1'],
        config.BALL_QUERY_RADIUS_LEVELS[1],
        outputs['mu_coarse'], outputs['logvar_coarse'],
        outputs['occ_readout_coarse'], outputs['z_coarse'], graph.occ_coarse,
        'coarse', lambda_kl,
    )

    if config.USE_POOL_LOSS and config.COARSEN_ASSIGNMENT == "soft":
        L_pool, pool_parts = compute_pool_loss(outputs, graph)
        branches.append(('pool', L_pool, pool_parts))

    return branches, lambda_kl


def compute_loss(outputs, graph, step, stage=1):
    p = graph.p
    branches, lambda_kl = compute_branch_losses(outputs, graph, step, stage=stage)
    zero = p.new_zeros(())
    L_recon = L_KL = L_occ = zero
    L_pool = zero
    pool_extras = {}

    for _, _, parts in branches:
        L_recon = L_recon + parts.get('recon', zero)
        L_KL = L_KL + parts.get('KL', zero)
        L_occ = L_occ + parts.get('occ', zero)
        if 'pool' in parts:
            L_pool = L_pool + parts['pool']
            pool_extras = {k: v for k, v in parts.items() if k.startswith('pool')}

    if branches:
        total = sum(branch[1] for branch in branches)
    else:
        total = zero

    components = {'recon': L_recon, 'KL': L_KL, 'occ': L_occ, 'lambda_kl': lambda_kl}
    if pool_extras:
        components['pool'] = L_pool
        components.update(pool_extras)
    return total, components
