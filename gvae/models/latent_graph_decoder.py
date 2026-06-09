# gvae/models/latent_graph_decoder.py
# Honest graph_hat from Z: VGAE-style per-slot readout (default) or legacy spatial modes.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from gvae.models.decoder import position_from_logits, sample_volume
from gvae.models.latent_layout import (
    _peak_k,
    layout_magnitude,
    nms_peaks_assigned_to_slots,
    peak_windows_around_gt,
    peak_windows_from_queries,
    positions_in_windows,
    slot_window_centroids,
)
from gvae.models.splatting import make_voxel_centers


class LatentGraphDecoder(nn.Module):
    """
    Decode supernode (p, r, s) from Z (+ optional layout head).

    LATENT_DECODE_MODE:
      vgae      — per-slot softmax over all voxels (VGAE: slot i → node i)
      query     — per-slot k-NN on layout (legacy)
      nms_slots — NMS peaks + Hungarian assign (legacy)
      nms       — global NMS only (legacy)
    """

    def __init__(self, d: int, *, max_slots: int | None = None):
        super().__init__()
        self.d = d
        self.max_slots = max_slots if max_slots is not None else config.LATENT_GRAPH_MAX_SLOTS

        self.z_proj = nn.Conv3d(d, d, kernel_size=1)
        self.slot_query = nn.Embedding(self.max_slots, d)
        self.readout = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
        )
        self.mlp_s = nn.Linear(d, config.NUM_CLASSES)
        self.mlp_r = nn.Linear(d, 3)
        self.mlp_p = nn.Linear(d, 3)
        self.softplus = nn.Softplus()

    def _slot_global_readout(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        layout_mag: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per supernode i: softmax over full grid → position + feature (VGAE decoder)."""
        scale = math.sqrt(queries.shape[-1]) * max(config.LATENT_GRAPH_SPATIAL_TEMP, 1e-4)
        logits = queries @ tokens.T
        if config.LATENT_GRAPH_Z_NORM_BIAS:
            logits = logits + layout_mag.unsqueeze(0)
        logits = logits / scale
        weights = F.softmax(logits.float(), dim=-1).to(tokens.dtype)
        p_coarse = (weights.unsqueeze(-1) * centers).sum(dim=1)
        feat = weights @ tokens
        return p_coarse, feat

    def _decode_positions_query(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        layout_mag: torch.Tensor,
        centers: torch.Tensor,
        grid_shape: tuple[int, int, int],
        n_nodes: int,
    ) -> torch.Tensor:
        k = _peak_k(grid_shape, n_nodes)
        topi, q_logits = peak_windows_from_queries(queries, tokens, k)
        return positions_in_windows(centers, layout_mag, topi, extra_logits=q_logits)

    def _decode_positions_gt(
        self,
        layout_mag: torch.Tensor,
        centers: torch.Tensor,
        p_gt: torch.Tensor,
        grid_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        k = _peak_k(grid_shape, p_gt.shape[0])
        windows = peak_windows_around_gt(centers, p_gt, k)
        return positions_in_windows(centers, layout_mag, windows)

    def _decode_positions(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        layout_mag: torch.Tensor,
        centers: torch.Tensor,
        grid_shape: tuple[int, int, int],
        n_nodes: int,
        p_gt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = config.LATENT_DECODE_MODE
        k = _peak_k(grid_shape, n_nodes)
        if mode == "query":
            p_query = self._decode_positions_query(
                queries, tokens, layout_mag, centers, grid_shape, n_nodes,
            )
            mix = (
                config.LATENT_GT_WINDOW_MIX
                if self.training and p_gt is not None and p_gt.shape[0] == n_nodes
                else 0.0
            )
            if mix <= 0.0:
                return p_query
            if mix >= 1.0:
                return self._decode_positions_gt(layout_mag, centers, p_gt, grid_shape)
            p_gtwin = self._decode_positions_gt(layout_mag, centers, p_gt, grid_shape)
            return (mix * p_gtwin + (1.0 - mix) * p_query).clamp(-1.0, 1.0)
        if mode == "nms_slots":
            centroids = slot_window_centroids(queries, tokens, centers, k)
            return nms_peaks_assigned_to_slots(
                layout_mag, centers, centroids, n_nodes,
            )
        if mode == "nms":
            from gvae.models.latent_layout import nms_peak_positions
            return nms_peak_positions(layout_mag, centers, n_nodes)
        raise ValueError(f"Unknown LATENT_DECODE_MODE={mode!r}")

    def forward(
        self,
        Z: torch.Tensor,
        n_nodes: int,
        p_gt: torch.Tensor | None = None,
        layout: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # vgae: no p_gt; query+curriculum may use p_gt for window blend only
        if n_nodes <= 0:
            empty = Z.new_zeros(0, 3)
            empty_s = Z.new_zeros(0, config.NUM_CLASSES)
            return {'s': empty_s, 'p': empty, 'r': empty}

        if n_nodes > self.max_slots:
            raise ValueError(
                f"n_nodes={n_nodes} exceeds LATENT_GRAPH_MAX_SLOTS={self.max_slots}",
            )

        feat = self.z_proj(Z.unsqueeze(0))
        _, _, H, W, D = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).squeeze(0)
        layout_mag = layout_magnitude(Z, layout)
        centers = make_voxel_centers((H, W, D), Z.device)

        slot_ids = torch.arange(n_nodes, device=Z.device, dtype=torch.long)
        queries = self.slot_query(slot_ids)

        if config.LATENT_DECODE_MODE == "vgae":
            p_coarse, feat_at = self._slot_global_readout(
                queries, tokens, layout_mag, centers,
            )
            feat_at = self.readout(feat_at)
            p_hat = position_from_logits(self.mlp_p(feat_at), p_coarse)
            return {
                's': torch.softmax(self.mlp_s(feat_at), dim=1),
                'p': p_hat,
                'r': self.softplus(self.mlp_r(feat_at)),
            }

        p_hat = self._decode_positions(
            queries, tokens, layout_mag, centers, (H, W, D), n_nodes, p_gt=p_gt,
        )
        z_samp = sample_volume(Z, p_hat.unsqueeze(1)).reshape(n_nodes, -1)
        feat_at = self.readout(z_samp)

        return {
            's': torch.softmax(self.mlp_s(feat_at), dim=1),
            'p': p_hat,
            'r': self.softplus(self.mlp_r(feat_at)),
        }

    def readout_at(self, Z: torch.Tensor, p: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sem/size readout at arbitrary points (aux loss at p_gt only)."""
        if p.numel() == 0:
            empty_s = Z.new_zeros(0, config.NUM_CLASSES)
            return {'s': empty_s, 'r': Z.new_zeros(0, 3)}
        z_samp = sample_volume(Z, p.unsqueeze(1)).reshape(p.shape[0], -1)
        feat_at = self.readout(z_samp)
        return {
            's': torch.softmax(self.mlp_s(feat_at), dim=1),
            'r': self.softplus(self.mlp_r(feat_at)),
        }
