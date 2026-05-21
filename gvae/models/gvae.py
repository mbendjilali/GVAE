# gvae/models/gvae.py
# Top-level GVAE: three-level encoder chain + fine / mid / coarse decoder branches

import torch.nn as nn

import config
from gvae.models.encoder import SceneGraphEncoder
from gvae.models.decoder import SceneGraphDecoder
from gvae.data.occupancy import OccupancyReadout


class GVAE(nn.Module):
    def __init__(self, stage: int = 2):
        super().__init__()
        self.stage = stage
        self.encoder = SceneGraphEncoder()
        self.decoder_fine = SceneGraphDecoder(config.D_FINE_LATENT)
        self.decoder_mid = SceneGraphDecoder(config.D_MID_LATENT)
        self.decoder_coarse = SceneGraphDecoder(config.D_COARSE_LATENT)
        self.occ_readout_fine = OccupancyReadout(config.D_FINE_LATENT)
        self.occ_readout_mid = OccupancyReadout(config.D_MID_LATENT)
        self.occ_readout_coarse = OccupancyReadout(config.D_COARSE_LATENT)

    def forward(self, graph, stage: int | None = None):
        stage = self.stage if stage is None else stage
        enc = self.encoder(graph, stage=stage)

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
            'h_inst': enc['h_inst'],
            'h_lm1': enc['h_lm1'],
            'h_1': enc['h_1'],
            'p_inst': enc['p_inst'],
            'r_inst': enc['r_inst'],
            's_inst': enc['s_inst'],
            'edge_index_inst': enc['edge_index_inst'],
            'p_lm1': enc['p_lm1'],
            'r_lm1': enc['r_lm1'],
            's_lm1': enc['s_lm1'],
            'p_1': enc['p_1'],
            'r_1': enc['r_1'],
            's_1': enc['s_1'],
            'S1': enc['S1'],
            'S2': enc['S2'],
            'edge_index_lm1': enc['edge_index_lm1'],
            'edge_index_1': enc['edge_index_1'],
            'occ_readout_fine': self.occ_readout_fine,
            'occ_readout_mid': self.occ_readout_mid,
            'occ_readout_coarse': self.occ_readout_coarse,
            'recon_fine': None,
            'recon_mid': None,
            'recon_coarse': None,
        }

        if stage >= 1 and enc['h_inst'].numel() > 0:
            out['recon_fine'] = self.decoder_fine(
                h=enc['h_inst'],
                Z=enc['z_fine'],
                p_gt=enc['p_inst'],
                r_gt=enc['r_inst'],
            )

        if stage >= 1 and enc['h_lm1'].numel() > 0:
            out['recon_mid'] = self.decoder_mid(
                h=enc['h_lm1'],
                Z=enc['z_mid'],
                p_gt=enc['p_lm1'],
                r_gt=enc['r_lm1'],
            )

        if stage >= 1 and enc['h_1'].numel() > 0:
            out['recon_coarse'] = self.decoder_coarse(
                h=enc['h_1'],
                Z=enc['z_coarse'],
                p_gt=enc['p_1'],
                r_gt=enc['r_1'],
            )

        return out
