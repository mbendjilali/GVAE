# gvae/losses/gvae_loss.py
# Loss terms: recon, voxel-wise KL, occupancy + cyclical KL schedule

import warnings

import torch
import torch.nn.functional as F
import config
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
    return branches, lambda_kl


def compute_loss(outputs, graph, step, stage=1):
    p = graph.p
    branches, lambda_kl = compute_branch_losses(outputs, graph, step, stage=stage)
    zero = p.new_zeros(())
    L_recon = zero
    L_KL = zero
    L_occ = zero

    for _, _, parts in branches:
        L_recon = L_recon + parts['recon']
        L_KL = L_KL + parts['KL']
        L_occ = L_occ + parts['occ']

    if branches:
        total = sum(branch[1] for branch in branches)
    else:
        total = zero

    components = {'recon': L_recon, 'KL': L_KL, 'occ': L_occ, 'lambda_kl': lambda_kl}
    return total, components
