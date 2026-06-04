# gvae/models/gvae.py
# Top-level GVAE: three-level encoder chain + fine / mid / coarse decoder branches

import torch
import torch.nn as nn

import config
from gvae.models.encoder import SceneGraphEncoder
from gvae.models.decoder import SceneGraphDecoder, ZOnlyDecoder
from gvae.models.latent_graph_decoder import LatentGraphDecoder

def _branch_readouts(
    decoder: SceneGraphDecoder,
    zonly_decoder: ZOnlyDecoder | None,
    *,
    h,
    z,
    p_gt,
    r_gt,
) -> tuple[dict, torch.Tensor, dict | None, dict | None]:
    """Deformable h+Z recon; optional Z-only @ GT / @ anchor (aux losses and/or position patch)."""
    p_anchor, r_anchor = decoder.predict_anchors(h)
    recon = decoder(h=h, Z=z, p_gt=p_gt, r_gt=r_gt)
    recon_zonly = None
    recon_zonly_hanchor = None
    if zonly_decoder is not None:
        recon_zonly = zonly_decoder.forward_at_gt(z, p_gt)
        recon_zonly_hanchor = zonly_decoder.forward(z, p_anchor)
        if config.Z_ONLY_PATCH_DEFORMABLE_POSITION:
            recon['p'] = recon_zonly_hanchor['p']
    return recon, p_anchor, r_anchor, recon_zonly, recon_zonly_hanchor


def _latent_branch_readout(
    decoder: LatentGraphDecoder,
    *,
    z,
    n_nodes: int,
) -> dict:
    """Z → graph_hat; n_nodes sets slot count only (no GT positions)."""
    return decoder(z, n_nodes)


class GVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SceneGraphEncoder()
        self.latent_mode = config.LATENT_GRAPH_VAE_MODE

        self.decoder_fine = None
        self.decoder_mid = None
        self.decoder_coarse = None
        self.zonly_decoder_fine = None
        self.zonly_decoder_mid = None
        self.zonly_decoder_coarse = None
        self.latent_decoder_fine = None
        self.latent_decoder_mid = None
        self.latent_decoder_coarse = None

        if self.latent_mode:
            self.latent_decoder_fine = LatentGraphDecoder(config.D_FINE_LATENT)
            self.latent_decoder_mid = LatentGraphDecoder(config.D_MID_LATENT)
            self.latent_decoder_coarse = LatentGraphDecoder(config.D_COARSE_LATENT)
        else:
            self.decoder_fine = SceneGraphDecoder(config.D_FINE_LATENT)
            self.decoder_mid = SceneGraphDecoder(config.D_MID_LATENT)
            self.decoder_coarse = SceneGraphDecoder(config.D_COARSE_LATENT)
            if config.USE_Z_ONLY_DECODER:
                self.zonly_decoder_fine = ZOnlyDecoder(config.D_FINE_LATENT)
                self.zonly_decoder_mid = ZOnlyDecoder(config.D_MID_LATENT)
                self.zonly_decoder_coarse = ZOnlyDecoder(config.D_COARSE_LATENT)

    def forward(self, graph):
        enc = self.encoder(graph)

        device_ref = enc['p_fine']
        out = {
            'mu_fine': enc['mu_fine'],
            'logvar_fine': enc['logvar_fine'],
            'mu_mid': enc['mu_mid'],
            'logvar_mid': enc['logvar_mid'],
            'mu_coarse': enc['mu_coarse'],
            'logvar_coarse': enc['logvar_coarse'],
            'z_fine': enc['z_fine'],
            'z_mid': enc['z_mid'],
            'z_coarse': enc['z_coarse'],
            'h_fine': enc['h_fine'],
            'h_lm1': enc['h_lm1'],
            'h_1': enc['h_1'],
            'p_fine': enc['p_fine'],
            'r_fine': enc['r_fine'],
            's_fine': enc['s_fine'],
            'edge_index_fine': enc['edge_index_fine'],
            'p_lm1': enc['p_lm1'],
            'r_lm1': enc['r_lm1'],
            's_lm1': enc['s_lm1'],
            'p_1': enc['p_1'],
            'r_1': enc['r_1'],
            's_1': enc['s_1'],
            'S0': enc['S0'],
            'S1': enc['S1'],
            'S2': enc['S2'],
            'edge_index_lm1': enc['edge_index_lm1'],
            'edge_index_1': enc['edge_index_1'],
            'recon_fine': None,
            'recon_mid': None,
            'recon_coarse': None,
            'recon_fine_zonly': None,
            'recon_mid_zonly': None,
            'recon_coarse_zonly': None,
            'recon_fine_zonly_hanchor': None,
            'recon_mid_zonly_hanchor': None,
            'recon_coarse_zonly_hanchor': None,
            'p_anchor_fine': device_ref.new_zeros(0, 3),
            'r_anchor_fine': device_ref.new_zeros(0, 3),
            'p_anchor_mid': device_ref.new_zeros(0, 3),
            'r_anchor_mid': device_ref.new_zeros(0, 3),
            'p_anchor_coarse': device_ref.new_zeros(0, 3),
            'r_anchor_coarse': device_ref.new_zeros(0, 3),
            'latent_graph_vae': self.latent_mode,
        }

        if self.latent_mode:
            n_fine = enc['p_fine'].shape[0]
            if n_fine > 0:
                out['recon_fine'] = _latent_branch_readout(
                    self.latent_decoder_fine, z=enc['z_fine'], n_nodes=n_fine,
                )
            n_mid = enc['p_lm1'].shape[0]
            if n_mid > 0:
                out['recon_mid'] = _latent_branch_readout(
                    self.latent_decoder_mid, z=enc['z_mid'], n_nodes=n_mid,
                )
            n_coarse = enc['p_1'].shape[0]
            if n_coarse > 0:
                out['recon_coarse'] = _latent_branch_readout(
                    self.latent_decoder_coarse, z=enc['z_coarse'], n_nodes=n_coarse,
                )
            return out

        if enc['h_fine'].numel() > 0:
            recon, p_anchor, r_anchor, z_gt, z_anc = _branch_readouts(
                self.decoder_fine,
                self.zonly_decoder_fine,
                h=enc['h_fine'],
                z=enc['z_fine'],
                p_gt=enc['p_fine'],
                r_gt=enc['r_fine'],
            )
            out['recon_fine'] = recon
            out['p_anchor_fine'] = p_anchor
            out['r_anchor_fine'] = r_anchor
            out['recon_fine_zonly'] = z_gt
            out['recon_fine_zonly_hanchor'] = z_anc

        if enc['h_lm1'].numel() > 0:
            recon, p_anchor, r_anchor, z_gt, z_anc = _branch_readouts(
                self.decoder_mid,
                self.zonly_decoder_mid,
                h=enc['h_lm1'],
                z=enc['z_mid'],
                p_gt=enc['p_lm1'],
                r_gt=enc['r_lm1'],
            )
            out['recon_mid'] = recon
            out['p_anchor_mid'] = p_anchor
            out['r_anchor_mid'] = r_anchor
            out['recon_mid_zonly'] = z_gt
            out['recon_mid_zonly_hanchor'] = z_anc

        if enc['h_1'].numel() > 0:
            recon, p_anchor, r_anchor, z_gt, z_anc = _branch_readouts(
                self.decoder_coarse,
                self.zonly_decoder_coarse,
                h=enc['h_1'],
                z=enc['z_coarse'],
                p_gt=enc['p_1'],
                r_gt=enc['r_1'],
            )
            out['recon_coarse'] = recon
            out['p_anchor_coarse'] = p_anchor
            out['r_anchor_coarse'] = r_anchor
            out['recon_coarse_zonly'] = z_gt
            out['recon_coarse_zonly_hanchor'] = z_anc

        return out
