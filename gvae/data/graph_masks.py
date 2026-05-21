# gvae/data/graph_masks.py — masks for B policy (instance GAT, object-only coarsening)

import torch

import config


def is_instantiable_label(label: str) -> bool:
    return label not in config.NON_INSTANTIABLE_CLASSES


def coarsen_mask_from_labels(labels: list[str]) -> torch.Tensor:
    """True = node participates in FPS coarsening."""
    return torch.tensor(
        [is_instantiable_label(lbl) for lbl in labels],
        dtype=torch.bool,
    )


def subgraph_edge_index(
    edge_index: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep edges with both endpoints in node_mask; reindex to local 0..N-1."""
    device = edge_index.device
    idx = node_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    old_to_new = torch.full((node_mask.shape[0],), -1, dtype=torch.long, device=device)
    old_to_new[idx] = torch.arange(idx.numel(), device=device)

    src, dst = edge_index[0], edge_index[1]
    keep = node_mask[src] & node_mask[dst]
    if keep.sum() == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    return torch.stack([old_to_new[src[keep]], old_to_new[dst[keep]]], dim=0)


def pool_subgraph(
    edge_index: torch.Tensor,
    positions: torch.Tensor,
    coarsen_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edges and positions for MinCutPool on coarsenable instance nodes only."""
    coarsen_mask = coarsen_mask.to(positions.device)
    idx = coarsen_mask.nonzero(as_tuple=True)[0]
    p_sub = positions[idx]
    ei_sub = subgraph_edge_index(edge_index, coarsen_mask)
    return ei_sub, p_sub
