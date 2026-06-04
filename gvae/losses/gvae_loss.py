# gvae/losses/gvae_loss.py
# Loss terms: recon, voxel-wise KL, grid occupancy + cyclical KL schedule

import torch
import torch.nn.functional as F
import config
from gvae.data.graph_masks import pool_subgraph
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


def footprint_loss(r_pred: torch.Tensor, r_true: torch.Tensor) -> torch.Tensor:
    """Log-space Smooth-L1 on footprint semi-axes (stable vs raw log-MSE)."""
    if r_true.numel() == 0:
        return r_true.new_zeros(())
    eps = config.SIZE_LOG_EPS
    log_pred = torch.log(r_pred.clamp(min=eps))
    log_true = torch.log(r_true.clamp(min=eps))
    return F.smooth_l1_loss(
        log_pred, log_true, beta=config.SIZE_LOG_HUBER_BETA,
    )


def reconstruction_loss(
    recon,
    p_true,
    r_true,
    s_true,
    *,
    size_weight: float | None = None,
    sem_weight: float | None = None,
):
    """Decode-from-Z reconstruction: semantics + position + footprint."""
    w_size = config.LAMBDA_SIZE if size_weight is None else size_weight
    w_sem = config.LAMBDA_SEM if sem_weight is None else sem_weight
    L_sem = soft_cross_entropy_loss(recon['s'], s_true)
    L_pos = F.mse_loss(recon['p'], p_true)
    L_size = footprint_loss(recon['r'], r_true)
    return (
        w_sem * L_sem
        + config.LAMBDA_POS * L_pos
        + w_size * L_size
    )


def anchor_loss(p_anchor: torch.Tensor, p_gt: torch.Tensor) -> torch.Tensor:
    """MSE on h-predicted anchor positions vs GT supernode centres."""
    if p_gt.numel() == 0 or p_anchor.numel() == 0:
        return p_gt.new_zeros(())
    return F.mse_loss(p_anchor, p_gt)


def anchor_footprint_loss(r_anchor: torch.Tensor, r_gt: torch.Tensor) -> torch.Tensor:
    """Log-MSE on h-predicted anchor footprints vs GT supernode semi-axes."""
    return footprint_loss(r_anchor, r_gt)


def decoder_gt_anchor_mix_for_epoch(epoch: int) -> float:
    """Linear curriculum for DECODER_GT_ANCHOR_MIX (epoch is 0-indexed)."""
    if not config.ANCHOR_MIX_CURRICULUM:
        return config.DECODER_GT_ANCHOR_MIX
    n = max(1, config.ANCHOR_MIX_ANNEAL_EPOCHS)
    t = min(1.0, epoch / n)
    return config.ANCHOR_MIX_START + t * (config.ANCHOR_MIX_END - config.ANCHOR_MIX_START)


def _lambda_anchor(name: str) -> float:
    return {
        'fine': config.LAMBDA_ANCHOR_FINE,
        'mid': config.LAMBDA_ANCHOR_MID,
        'coarse': config.LAMBDA_ANCHOR_COARSE,
    }[name]


def _lambda_anchor_r(name: str) -> float:
    return {
        'fine': config.LAMBDA_ANCHOR_R_FINE,
        'mid': config.LAMBDA_ANCHOR_R_MID,
        'coarse': config.LAMBDA_ANCHOR_R_COARSE,
    }[name]


def KL_loss(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum() / mu.numel()


def _lambda_norm_contrast(name: str) -> float:
    return {
        'fine': config.LAMBDA_NORM_CONTRAST_FINE,
        'mid': config.LAMBDA_NORM_CONTRAST_MID,
        'coarse': config.LAMBDA_NORM_CONTRAST_COARSE,
    }[name]


def norm_contrastive_loss(
    z: torch.Tensor,
    p_gt: torch.Tensor,
    occ_grid: torch.Tensor,
) -> torch.Tensor:
    """Hinge loss pushing ||Z(p_gt)|| above ||Z(empty occ)|| (Probe C alignment)."""
    from gvae.models.decoder import sample_volume
    from gvae.data.voxelize import sample_occupancy_queries

    if p_gt.numel() == 0:
        return z.new_zeros(())

    z_gt = sample_volume(z, p_gt.unsqueeze(1)).reshape(p_gt.shape[0], -1)
    norm_gt = z_gt.norm(dim=1)

    n_query = config.NORM_CONTRAST_EMPTY_POINTS
    q, labels = sample_occupancy_queries(occ_grid, n_queries=n_query, pos_ratio=0.0)
    empty_pts = q[labels < 0.5]
    if empty_pts.numel() == 0:
        return z.new_zeros(())

    if empty_pts.shape[0] > n_query:
        pick = torch.randint(0, empty_pts.shape[0], (n_query,), device=z.device)
        empty_pts = empty_pts[pick]

    z_empty = sample_volume(z, empty_pts.unsqueeze(1)).reshape(empty_pts.shape[0], -1)
    norm_empty = z_empty.norm(dim=1)
    margin = config.NORM_CONTRAST_MARGIN
    return F.relu(norm_empty.unsqueeze(0) - norm_gt.unsqueeze(1) + margin).mean()


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


def _maybe_branch_loss(
    branches,
    recon,
    recon_zonly,
    recon_hzonly,
    p_true,
    r_true,
    s_true,
    p_anchor,
    r_anchor,
    mu,
    logvar,
    z,
    occ_grid,
    name: str,
    lambda_kl: float,
):
    if p_true.numel() == 0:
        return
    has_latent = (
        recon is not None
        and config.LATENT_GRAPH_VAE_MODE
        and config.LAMBDA_RECON_LATENT > 0
    )
    has_h = (
        recon is not None
        and not config.LATENT_GRAPH_VAE_MODE
        and config.LAMBDA_RECON_H > 0
    )
    has_z = recon_zonly is not None and config.USE_Z_ONLY_DECODER and config.LAMBDA_RECON_ZONLY > 0
    has_hz = (
        recon_hzonly is not None
        and config.USE_Z_ONLY_DECODER
        and config.LAMBDA_RECON_HZONLY > 0
    )
    lambda_anchor = 0.0 if config.LATENT_GRAPH_VAE_MODE else _lambda_anchor(name)
    lambda_anchor_r = 0.0 if config.LATENT_GRAPH_VAE_MODE else _lambda_anchor_r(name)
    if (
        not has_latent and not has_h and not has_z and not has_hz
        and lambda_anchor <= 0
        and lambda_anchor_r <= 0
    ):
        return

    parts: dict = {}
    total = mu.new_zeros(())

    w_size_z = config.LAMBDA_SIZE_ZONLY
    w_sem_z = config.LAMBDA_SEM_ZONLY

    if has_latent:
        L_recon = reconstruction_loss(recon, p_true, r_true, s_true)
        parts['recon_latent'] = L_recon
        total = total + config.LAMBDA_RECON_LATENT * L_recon

    if has_h:
        L_recon_h = reconstruction_loss(recon, p_true, r_true, s_true)
        parts['recon'] = L_recon_h
        total = total + config.LAMBDA_RECON_H * L_recon_h

    if has_z:
        L_recon_z = reconstruction_loss(
            recon_zonly, p_true, r_true, s_true,
            size_weight=w_size_z, sem_weight=w_sem_z,
        )
        parts['recon_zonly'] = L_recon_z
        total = total + config.LAMBDA_RECON_ZONLY * L_recon_z

    if has_hz:
        L_recon_hz = reconstruction_loss(
            recon_hzonly, p_true, r_true, s_true,
            size_weight=w_size_z, sem_weight=w_sem_z,
        )
        parts['recon_hzonly'] = L_recon_hz
        total = total + config.LAMBDA_RECON_HZONLY * L_recon_hz

    if lambda_anchor > 0 and p_anchor is not None and p_anchor.numel() > 0:
        L_anchor = anchor_loss(p_anchor, p_true)
        parts['anchor'] = L_anchor
        total = total + lambda_anchor * L_anchor

    if lambda_anchor_r > 0 and r_anchor is not None and r_anchor.numel() > 0:
        L_anchor_r = anchor_footprint_loss(r_anchor, r_true)
        parts['anchor_r'] = L_anchor_r
        total = total + lambda_anchor_r * L_anchor_r

    L_kl = KL_loss(mu, logvar)
    parts['KL'] = L_kl
    total = total + lambda_kl * L_kl

    lambda_norm = _lambda_norm_contrast(name)
    if lambda_norm > 0 and occ_grid.numel() > 0:
        L_norm = norm_contrastive_loss(z, p_true, occ_grid)
        parts['norm_contrast'] = L_norm
        total = total + lambda_norm * L_norm

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
        outputs.get('recon_fine_zonly'),
        outputs.get('recon_fine_zonly_hanchor'),
        outputs['p_fine'], outputs['r_fine'], outputs['s_fine'],
        outputs.get('p_anchor_fine'),
        outputs.get('r_anchor_fine'),
        outputs['mu_fine'], outputs['logvar_fine'],
        outputs['z_fine'], graph.occ_fine,
        'fine', lambda_kl,
    )
    _maybe_branch_loss(
        branches,
        outputs.get('recon_mid'),
        outputs.get('recon_mid_zonly'),
        outputs.get('recon_mid_zonly_hanchor'),
        outputs['p_lm1'], outputs['r_lm1'], outputs['s_lm1'],
        outputs.get('p_anchor_mid'),
        outputs.get('r_anchor_mid'),
        outputs['mu_mid'], outputs['logvar_mid'],
        outputs['z_mid'], graph.occ_mid,
        'mid', lambda_kl,
    )
    _maybe_branch_loss(
        branches,
        outputs.get('recon_coarse'),
        outputs.get('recon_coarse_zonly'),
        outputs.get('recon_coarse_zonly_hanchor'),
        outputs['p_1'], outputs['r_1'], outputs['s_1'],
        outputs.get('p_anchor_coarse'),
        outputs.get('r_anchor_coarse'),
        outputs['mu_coarse'], outputs['logvar_coarse'],
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
    L_recon = L_recon_zonly = L_recon_hzonly = L_KL = L_norm = L_anchor = L_anchor_r = zero
    L_pool = zero
    pool_extras = {}

    for _, _, parts in branches:
        L_recon = L_recon + parts.get('recon', zero)
        L_recon_zonly = L_recon_zonly + parts.get('recon_zonly', zero)
        L_recon_hzonly = L_recon_hzonly + parts.get('recon_hzonly', zero)
        L_anchor = L_anchor + parts.get('anchor', zero)
        L_anchor_r = L_anchor_r + parts.get('anchor_r', zero)
        L_KL = L_KL + parts.get('KL', zero)
        L_norm = L_norm + parts.get('norm_contrast', zero)
        if 'pool' in parts:
            L_pool = L_pool + parts['pool']
            pool_extras = {k: v for k, v in parts.items() if k.startswith('pool')}

    if branches:
        total = sum(branch[1] for branch in branches)
    else:
        total = zero

    components = {
        'recon': L_recon,
        'recon_zonly': L_recon_zonly,
        'recon_hzonly': L_recon_hzonly,
        'anchor': L_anchor,
        'anchor_r': L_anchor_r,
        'KL': L_KL,
        'norm_contrast': L_norm,
        'lambda_kl': lambda_kl,
    }
    if pool_extras:
        components['pool'] = L_pool
        components.update(pool_extras)
    return total, components
