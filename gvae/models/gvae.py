# gvae/models/gvae.py
# Top-level GVAE: three-level encoder chain + two decoder branches

import torch.nn as nn

import config
from gvae.models.encoder import SceneGraphEncoder
from gvae.models.decoder import SceneGraphDecoder
from gvae.data.occupancy import OccupancyReadout


class GVAE(nn.Module):
    def __init__(self, stage: int = 4):
        super().__init__()
        self.stage = stage
        self.encoder = SceneGraphEncoder()
        self.decoder_mid = SceneGraphDecoder(config.D_MID_LATENT)
        self.decoder_coarse = SceneGraphDecoder(config.D_COARSE_LATENT)
        self.occ_readout_mid = OccupancyReadout(config.D_MID_LATENT)
        self.occ_readout_coarse = OccupancyReadout(config.D_COARSE_LATENT)

    def forward(self, graph):
        enc = self.encoder(graph)

        dec_mid = self.decoder_mid(
            h=enc['h_lm1'],
            p=enc['p_lm1'],
            r=enc['r_lm1'],
            Z=enc['z_mid'],
        )
        dec_coarse = self.decoder_coarse(
            h=enc['h_1'],
            p=enc['p_1'],
            r=enc['r_1'],
            Z=enc['z_coarse'],
        )

        return {
            'mu_mid': enc['mu_mid'],
            'logvar_mid': enc['logvar_mid'],
            'mu_coarse': enc['mu_coarse'],
            'logvar_coarse': enc['logvar_coarse'],
            'z_mid': enc['z_mid'],
            'z_coarse': enc['z_coarse'],
            'recon_mid': dec_mid,
            'recon_coarse': dec_coarse,
            'p_lm1': enc['p_lm1'], 'r_lm1': enc['r_lm1'], 's_lm1': enc['s_lm1'],
            'p_1': enc['p_1'], 'r_1': enc['r_1'], 's_1': enc['s_1'],
            'S1': enc['S1'],
            'S2': enc['S2'],
            'edge_index_lm1': enc['edge_index_lm1'],
            'edge_index_1': enc['edge_index_1'],
            'occ_readout_mid': self.occ_readout_mid,
            'occ_readout_coarse': self.occ_readout_coarse,
        }
