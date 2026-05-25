# gvae/losses/gvae_loss.py
# Loss terms: recon, voxel-wise KL, grid occupancy + cyclical KL schedule

import torch
import torch.nn.functional as F
import config
from gvae.data.graph_masks import pool_subgraph
from gvae.data.occupancy import loss_occ_grid


def kl_weight(step):
    total_steps = config.KL_TOTAL_STEPS or config.NUM_EPOCHS
    cycle_len = max(1, total_steps // config.KL_ANNEAL_CYCLES)
    ramp_len = int(cycle_len * config.KL_ANNEAL_RATIO)
    pos_in_cycle = step % cycle_len
    return config.LAMBDA_KL_MAX * min(1.0, pos_in_cycle / max(1, ramp_len))


def soft_cross_entropy_loss(pred_probs: torch.Tensor, true_soft: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy: -mean_n sum_c t_c log p_c (one-hot targets => standard CE)."""
    true = true_soft / true_soft.sum(dim=1, keepdim=True).clamp(min=config.SOFT_MIOU_EPS)
    log_pred = pred_probs.clamp(min=config.SOFT_MIOU_EPS).log()
    return -(true * log_pred).sum(dim=1).mean()


# Backward-compatible alias for metrics / diagnostics
soft_semantic_loss = soft_cross_entropy_loss


def reconstruction_loss(recon, p_true, r_true, s_true):
    """Decode-from-Z reconstruction: semantics + position + footprint."""
    L_sem = soft_cross_entropy_loss(recon['s'], s_true)
    L_pos = F.mse_loss(recon['p'], p_true)
    L_size = F.mse_loss(recon['r'], r_true)
    return (
        config.LAMBDA_SEM * L_sem
        + config.LAMBDA_POS * L_pos
        + config.LAMBDA_POS * L_size
    )


def KL_loss(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum() / mu.numel()


def loss_pool(S, edge_index, p, N_nodes):
    """MinCut-style pool regularisation on soft assignment S."""
    M = S.shape[1]
    num_edges = edge_index.shape[1]
    deg = torch.zeros(N_nodes, device=S.device)
    if num_edges > 0:
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
    """Pool loss on S0 (instances→fine), S1 (fine→mid), S2 (mid→coarse)."""
    p = graph.p
    zero = p.new_zeros(())
    ei_pool, p_pool = pool_subgraph(graph.edge_index, p, graph.coarsen_mask)
    n_pool = p_pool.shape[0]

    if n_pool > 0 and outputs['S0'].numel() > 0:
        L_pool_s0, L_cut_s0, L_ortho_s0, L_spatial_s0 = loss_pool(
            outputs['S0'], ei_pool, p_pool, n_pool,
        )
    else:
        L_pool_s0 = L_cut_s0 = L_ortho_s0 = L_spatial_s0 = zero

    if outputs['p_fine'].numel() > 0 and outputs['S1'].numel() > 0:
        L_pool_s1, L_cut_s1, L_ortho_s1, L_spatial_s1 = loss_pool(
            outputs['S1'], outputs['edge_index_fine'], outputs['p_fine'],
            outputs['p_fine'].shape[0],
        )
    else:
        L_pool_s1 = L_cut_s1 = L_ortho_s1 = L_spatial_s1 = zero

    L_pool_s2, L_cut_s2, L_ortho_s2, L_spatial_s2 = loss_pool(
        outputs['S2'], outputs['edge_index_lm1'], outputs['p_lm1'], outputs['p_lm1'].shape[0],
    )
    L_pool = L_pool_s0 + L_pool_s1 + L_pool_s2
    parts = {
        'pool': L_pool,
        'pool_cut': L_cut_s0 + L_cut_s1 + L_cut_s2,
        'pool_ortho': L_ortho_s0 + L_ortho_s1 + L_ortho_s2,
        'pool_spatial': L_spatial_s0 + L_spatial_s1 + L_spatial_s2,
    }
    return config.LAMBDA_POOL * L_pool, parts


def _lambda_occ_grid(name: str) -> float:
    return {
        'fine': config.LAMBDA_OCC_GRID_FINE,
        'mid': config.LAMBDA_OCC_GRID_MID,
        'coarse': config.LAMBDA_OCC_GRID_COARSE,
    }[name]


def _maybe_branch_loss(
    branches,
    recon,
    p_true,
    r_true,
    s_true,
    mu,
    logvar,
    occ_grid_head,
    z,
    occ_grid,
    name: str,
    lambda_kl: float,
):
    if recon is None or p_true.numel() == 0:
        return
    L_recon = reconstruction_loss(recon, p_true, r_true, s_true)
    L_kl = KL_loss(mu, logvar)
    lambda_grid = _lambda_occ_grid(name)
    parts = {'recon': L_recon, 'KL': L_kl}
    total = L_recon + lambda_kl * L_kl
    if lambda_grid > 0 and occ_grid_head is not None:
        L_occ = loss_occ_grid(occ_grid_head(z), occ_grid)
        parts['occ'] = L_occ
        total = total + lambda_grid * L_occ
    branches.append((name, total, parts))


def compute_branch_losses(outputs, graph, step):
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
        outputs['p_fine'], outputs['r_fine'], outputs['s_fine'],
        outputs['mu_fine'], outputs['logvar_fine'],
        outputs['occ_grid_head_fine'],
        outputs['z_fine'], graph.occ_fine,
        'fine', lambda_kl,
    )
    _maybe_branch_loss(
        branches,
        outputs.get('recon_mid'),
        outputs['p_lm1'], outputs['r_lm1'], outputs['s_lm1'],
        outputs['mu_mid'], outputs['logvar_mid'],
        outputs['occ_grid_head_mid'],
        outputs['z_mid'], graph.occ_mid,
        'mid', lambda_kl,
    )
    _maybe_branch_loss(
        branches,
        outputs.get('recon_coarse'),
        outputs['p_1'], outputs['r_1'], outputs['s_1'],
        outputs['mu_coarse'], outputs['logvar_coarse'],
        outputs['occ_grid_head_coarse'],
        outputs['z_coarse'], graph.occ_coarse,
        'coarse', lambda_kl,
    )

    if config.USE_POOL_LOSS and config.COARSEN_ASSIGNMENT == "soft":
        L_pool, pool_parts = compute_pool_loss(outputs, graph)
        branches.append(('pool', L_pool, pool_parts))

    return branches, lambda_kl


def compute_loss(outputs, graph, step):
    p = graph.p
    branches, lambda_kl = compute_branch_losses(outputs, graph, step)
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
