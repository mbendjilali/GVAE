# gvae/models/gvae.py
# Top-level GVAE: three-level encoder chain + two decoder branches
# Returns Z^G_coarse, Z^G_mid and all reconstructions

import torch
import torch.nn as nn

import config
from gvae.models.encoder import SceneGraphEncoder
from gvae.models.decoder import SceneGraphDecoder

class GVAE(nn.Module):
    def __init__(self, stage: int = 4):
        super().__init__()
        self.stage = stage # to control which parts are frozen during training
        self.encoder = SceneGraphEncoder()
        self.decoder = SceneGraphDecoder()

    def forward(self, graph):
        # encode input graph to get latent codes and intermediate features
        enc = self.encoder(graph)

        # decode mid level 
        dec_mid = self.decoder(
            h = enc['h_lm1'],
            p = enc['p_lm1'],
            r = enc['r_lm1'],
            Z = enc['z_mid']
        )

        # decode coarse level
        dec_coarse = self.decoder(
            h = enc['h_1'],
            p = enc['p_1'],
            r = enc['r_1'],
            Z = enc['z_coarse']
        )

        return {
            # latent distributions (for KL loss)
            "mu_mid": enc['mu_mid'],
            "logvar_mid": enc['logvar_mid'],
            "mu_coarse": enc['mu_coarse'],
            "logvar_coarse": enc['logvar_coarse'],

            # reconstructions (for reconstruction loss)
            "recon_mid": dec_mid, # dict with keys "s_lm1", "p_lm1", "r_lm1"
            "recon_coarse": dec_coarse, # dict with keys "s_1", "p_1", "r_1"

            # soft assignment matrices (for pool regularization loss)
            'S1': enc['S1'],
            'S2': enc['S2']
        }
