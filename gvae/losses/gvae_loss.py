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


def matched_position_loss(p_pred: torch.Tensor, p_true: torch.Tensor) -> torch.Tensor:
    """MSE after optimal bipartite match (supernode order ≠ peak order)."""
    if p_true.numel() == 0 or p_pred.numel() == 0:
        return p_true.new_zeros(())
    if p_pred.shape[0] != p_true.shape[0]:
        return F.mse_loss(p_pred, p_true)
    from scipy.optimize import linear_sum_assignment

    with torch.no_grad():
        cost = torch.cdist(p_pred, p_true, p=2).cpu().numpy()
        row, col = linear_sum_assignment(cost)
    row_t = torch.as_tensor(row, device=p_pred.device, dtype=torch.long)
    col_t = torch.as_tensor(col, device=p_true.device, dtype=torch.long)
    return F.mse_loss(p_pred[row_t], p_true[col_t])


def query_window_cover_loss(
    queries: torch.Tensor,
    tokens: torch.Tensor,
    centers: torch.Tensor,
    p_gt: torch.Tensor,
) -> torch.Tensor:
    """Min dist from each p_gt to nearest voxel in the slot query k-NN window."""
    from gvae.models.latent_layout import peak_windows_from_queries

    if p_gt.numel() == 0:
        return p_gt.new_zeros(())
    k = min(config.LATENT_PEAK_NEIGHBORS, centers.shape[0])
    topi, _ = peak_windows_from_queries(queries, tokens, k)
    local_c = centers[topi]
    return torch.cdist(p_gt.unsqueeze(1), local_c, p=2).min(dim=1).values.mean()


def reconstruction_loss(
    recon,
    p_true,
    r_true,
    s_true,
    *,
    size_weight: float | None = None,
    sem_weight: float | None = None,
    pos_weight: float | None = None,
    matched_pos: bool = False,
):
    """Decode-from-Z reconstruction: semantics + position + footprint."""
    w_size = config.LAMBDA_SIZE if size_weight is None else size_weight
    w_sem = config.LAMBDA_SEM if sem_weight is None else sem_weight
    w_pos = config.LAMBDA_POS if pos_weight is None else pos_weight
    L_sem = soft_cross_entropy_loss(recon['s'], s_true)
    if matched_pos and config.LATENT_POS_MATCHED:
        L_pos = matched_position_loss(recon['p'], p_true)
    else:
        L_pos = F.mse_loss(recon['p'], p_true)
    L_size = footprint_loss(recon['r'], r_true)
    return (
        w_sem * L_sem
        + w_pos * L_pos
        + w_size * L_size
    )


def slot_spread_loss(p_pred: torch.Tensor) -> torch.Tensor:
    """Penalise supernodes collapsing to the same predicted position."""
    if p_pred.shape[0] < 2:
        return p_pred.new_zeros(())
    dist2 = torch.cdist(p_pred, p_pred, p=2).pow(2)
    n = p_pred.shape[0]
    mask = ~torch.eye(n, device=p_pred.device, dtype=torch.bool)
    sigma = config.LATENT_GRAPH_SLOT_SPREAD_SIGMA
    return torch.exp(-dist2[mask] / (2.0 * sigma * sigma)).mean()


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


def latent_gt_window_mix_for_epoch(epoch: int) -> float:
    """Train-only: 1 = k-NN(p_gt[i]) windows; 0 = slot-query windows (same as val)."""
    if not config.LATENT_GRAPH_VAE_MODE:
        return 0.0
    if not config.LATENT_GT_WINDOW_CURRICULUM:
        return 0.0
    n = max(1, config.LATENT_GT_WINDOW_ANNEAL_EPOCHS)
    return max(0.0, 1.0 - min(1.0, epoch / n))


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


def _lambda_z_peak(name: str) -> float:
    return {
        'fine': config.LAMBDA_Z_PEAK_FINE,
        'mid': config.LAMBDA_Z_PEAK_MID,
        'coarse': config.LAMBDA_Z_PEAK_COARSE,
    }[name]


def latent_peak_loss(
    z: torch.Tensor,
    p_gt: torch.Tensor,
    layout: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Train ||Z|| to peak at each supernode: per-node KL to a Gaussian target over
    the k nearest voxels (differentiable layout signal for diffusion).
    """
    from gvae.models.latent_layout import _peak_k, layout_magnitude, peak_windows_around_gt
    from gvae.models.splatting import make_voxel_centers

    if p_gt.numel() == 0:
        return z.new_zeros(())

    _, H, W, D = z.shape
    centers = make_voxel_centers((H, W, D), z.device)
    mag = layout_magnitude(z, layout)
    k = _peak_k((H, W, D), p_gt.shape[0])
    windows = peak_windows_around_gt(centers, p_gt, k)
    local_centers = centers[windows]
    topd = (local_centers - p_gt.unsqueeze(1)).pow(2).sum(dim=-1)
    sigma2 = config.LATENT_PEAK_SIGMA ** 2
    temp = max(config.LATENT_PEAK_TEMP, 1e-4)
    target = F.softmax(-topd / (2.0 * sigma2), dim=-1)
    log_pred = F.log_softmax(mag[windows] / temp, dim=-1)
    return -(target * log_pred).sum(dim=-1).mean()


def latent_sem_at_gt_aux_loss(
    z: torch.Tensor,
    p_gt: torch.Tensor,
    r_gt: torch.Tensor,
    s_gt: torch.Tensor,
    decoder,
) -> torch.Tensor:
    """Train Z semantics at GT sites without changing graph_hat (Z@p_hat)."""
    if p_gt.numel() == 0:
        return z.new_zeros(())
    aux = decoder.readout_at(z, p_gt)
    w_sem = config.LAMBDA_SEM_AT_GT
    w_size = config.LAMBDA_SIZE_AT_GT if config.LAMBDA_SIZE_AT_GT > 0 else w_sem
    return (
        w_sem * soft_cross_entropy_loss(aux['s'], s_gt)
        + w_size * footprint_loss(aux['r'], r_gt)
    )


def latent_index_peak_distill_loss(
    z: torch.Tensor,
    p_gt: torch.Tensor,
    p_dec: torch.Tensor,
    layout: torch.Tensor | None = None,
) -> torch.Tensor:
    """Index-aligned: p_hat[i] → oracle layout peak at k-NN(p_gt[i]) (deploy order)."""
    from gvae.models.latent_layout import index_aligned_peak_positions

    if p_gt.numel() == 0 or p_dec.numel() == 0:
        return z.new_zeros(())
    with torch.no_grad():
        p_tgt = index_aligned_peak_positions(z, p_gt, layout=layout)
    return F.mse_loss(p_dec, p_tgt)


def latent_layout_distill_loss(
    z: torch.Tensor,
    p_gt: torch.Tensor,
    p_dec: torch.Tensor,
    layout: torch.Tensor | None = None,
) -> torch.Tensor:
    return latent_index_peak_distill_loss(z, p_gt, p_dec, layout=layout)


@torch.no_grad()
def latent_peak_position_error(
    z: torch.Tensor,
    p_gt: torch.Tensor,
    layout: torch.Tensor | None = None,
) -> float:
    """RMSE of oracle layout-map peaks vs p_gt."""
    from gvae.models.latent_layout import peak_positions_at_gt

    if p_gt.numel() == 0:
        return float('nan')
    p_hat = peak_positions_at_gt(z, p_gt, layout=layout)
    return F.mse_loss(p_hat, p_gt, reduction='mean').sqrt().item()


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
    layout: torch.Tensor | None = None,
    latent_decoder=None,
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
        L_recon = reconstruction_loss(
            recon, p_true, r_true, s_true,
            pos_weight=config.LAMBDA_POS_LATENT,
            sem_weight=config.LAMBDA_SEM_LATENT,
            matched_pos=config.LATENT_POS_MATCHED,
        )
        parts['recon_latent'] = L_recon
        total = total + config.LAMBDA_RECON_LATENT * L_recon
        if config.LAMBDA_SLOT_SPREAD > 0 and name == 'fine':
            L_spread = slot_spread_loss(recon['p'])
            parts['slot_spread'] = L_spread
            total = total + config.LAMBDA_SLOT_SPREAD * L_spread
        lambda_idx = (
            config.LAMBDA_INDEX_PEAK_DISTILL
            if config.LAMBDA_INDEX_PEAK_DISTILL > 0
            else config.LAMBDA_LAYOUT_DISTILL
        )
        if lambda_idx > 0:
            L_distill = latent_index_peak_distill_loss(
                z, p_true, recon['p'], layout=layout,
            )
            parts['index_peak_distill'] = L_distill
            total = total + lambda_idx * L_distill
        if config.LAMBDA_SEM_AT_GT > 0 and latent_decoder is not None:
            L_aux = latent_sem_at_gt_aux_loss(
                z, p_true, r_true, s_true, latent_decoder,
            )
            parts['sem_at_gt'] = L_aux
            total = total + L_aux
        if (
            config.LAMBDA_QUERY_COVER > 0
            and latent_decoder is not None
            and config.LATENT_DECODE_MODE in ("query", "nms_slots")
            and name == 'fine'
        ):
            from gvae.models.splatting import make_voxel_centers

            feat = latent_decoder.z_proj(z.unsqueeze(0))
            tokens = feat.flatten(2).transpose(1, 2).squeeze(0)
            centers = make_voxel_centers(z.shape[1:], z.device)
            slot_ids = torch.arange(p_true.shape[0], device=z.device, dtype=torch.long)
            queries = latent_decoder.slot_query(slot_ids)
            L_cover = query_window_cover_loss(queries, tokens, centers, p_true)
            parts['query_cover'] = L_cover
            total = total + config.LAMBDA_QUERY_COVER * L_cover

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

    lambda_peak = _lambda_z_peak(name)
    if lambda_peak > 0 and p_true.numel() > 0:
        L_peak = latent_peak_loss(z, p_true, layout=layout)
        parts['z_peak'] = L_peak
        total = total + lambda_peak * L_peak

    branches.append((name, total, parts))


def compute_branch_losses(outputs, graph, step, model=None):
    """
    Per-branch losses for sequential backward (fine → mid → coarse).

    Returns:
        branches: list of (name, total_loss, partial_components)
        lambda_kl: float
    """
    lambda_kl = kl_weight(step)
    branches = []
    dec_fine = dec_mid = dec_coarse = None
    if model is not None and config.LATENT_GRAPH_VAE_MODE:
        dec_fine = getattr(model, 'latent_decoder_fine', None)
        dec_mid = getattr(model, 'latent_decoder_mid', None)
        dec_coarse = getattr(model, 'latent_decoder_coarse', None)

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
        layout=outputs.get('layout_fine'),
        latent_decoder=dec_fine,
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
        layout=outputs.get('layout_mid'),
        latent_decoder=dec_mid,
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
        layout=outputs.get('layout_coarse'),
        latent_decoder=dec_coarse,
    )

    if config.USE_POOL_LOSS and config.COARSEN_ASSIGNMENT == "soft":
        L_pool, pool_parts = compute_pool_loss(outputs, graph)
        branches.append(('pool', L_pool, pool_parts))

    return branches, lambda_kl


def compute_loss(outputs, graph, step, model=None):
    p = graph.p
    branches, lambda_kl = compute_branch_losses(outputs, graph, step, model=model)
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
