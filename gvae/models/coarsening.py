# gvae/models/coarsening.py
# FPS + ball-query (hard) coarsening + soft-S MLP + supernode attribute derivation
# TODO: implement (todo id: coarsening)


import torch
import torch.nn as nn
from torch_cluster import fps
from torch_geometric.nn import radius_graph
import config

class FPSCoarsening(nn.Module):
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio

        # MLP to compute soft cluster assignment S from node features (learnable part)
        self.mlp_S = nn.Sequential(
            nn.Linear(1, 16),  # input: 1 distance value → 16 hidden values
            nn.ReLU(),
            nn.Linear(16, 1),  #  16 hidden values → 1 output value per node
        )
    def forward(self, p, r, s, h):
        """
        p: (N, 3) node positions
        r: (N, 3) node radii
        s: (N, config.NUM_CLASSES) node semantic one-hot vectors
        h: (N, d) node features from the encoder (R-GAT layer)

        Returns:
        p_coarse: (M, 3) coarsened node positions
        r_coarse: (M, 3) coarsened node radii
        s_coarse: (M, config.NUM_CLASSES) coarsened node semantic one-hot vectors
        h_coarse: (M, d) coarsened node features
        """

        # 1. FPS to select M representative nodes (supernodes)
        # fps() needs a "batch" vector telling it which graph each node belongs to.
        # Since we process one graph at a time, all nodes belong to batch 0.
        batch = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)  # single batch

        # select seed node indices using FPS
        # ratio = fraction of nodes to keep, e.g. 0.25 to keep 25% of nodes
        seed_indices = fps(p, batch=batch, ratio=self.ratio)

        # 2. Hard assignment 
        # Get the 3D positions of the seed nodes
        p_seeds = p[seed_indices]  # (M, 3) positions of the M seed nodes
        # compute pairwise Euclidean distances from all nodes to all seed nodes
        # dists[i,j] = distance from node i to seed node j
        dist = torch.cdist(p, p_seeds)  # (N, M)
        # for each node, find the index of the closest seed
        hard_assign = dist.argmin(dim=1) # (N,) index of closest seed for each node
        # hard_assign[i] = j means node i is assigned to seed node j
        
        # 3. Soft assignment
        M = seed_indices.shape[0] # number of supernodes
        N = p.shape[0] # number of original nodes

        # normalised distance 
        # we normalise distances by the seed's size (r) so the distance is relative to the object scale.
        r_seeds = r[seed_indices]  # (M, 3) radii of seed nodes
        # For each node i and seed j: distance = ||p_i - p_seed_j|| / mean(r_seed_j)
        dist_normalised = dist / (r_seeds.mean(dim=1).unsqueeze(0) + 1e-6)  # (N, M) normalised distance

        # compute soft assignment weights using the MLP on the distances
        scores = self.mlp_S(dist_normalised.reshape(-1, 1)).reshape(N, M)  # (N, M) raw scores from MLP
        S = torch.softmax(scores, dim=1)  # (N, M) soft assignment weights
        # S[i, j] = probability that node i belongs to supernode j

        # 4. Supernode attributes
        # position
        # p̄_j = sum_i(S[i,j] * p_i) / sum_i(S[i,j])
        # In matrix form: S.T @ p  /  S.sum(dim=0)
        # @ matrix multiplication operator
        p_super = (S.T @ p) / (S.sum(dim=0, keepdim=True).T + 1e-6)  # (M, 3) weighted average position

        # size (radius)
        r_super = torch.zeros(K, 3, device=p.device)  # (M, 3) supernode radii
        for j in range(M):
            members = (hard_assign == j)  # boolean mask of nodes assigned to supernode j
            if members.sum() > 0:
                p_members = p[members]  # (num_members, 3) positions of member nodes
                lower_corner = p_members.min(dim=0).values  # (3,) lower corner of bounding box
                upper_corner = p_members.max(dim=0).values  # (3,) upper corner of bounding box
                r_super[j] = (upper_corner - lower_corner) / 2  # (3,) radius = half-extent of bounding box
            
        # semantic class (one-hot): semantic mix for each supernode
        S_col_sum = S.sum(dim=0) # (M,) sum of soft assignment weights for each supernode
        s_super = (S.T @ s) / (S_col_sum.unsqueeze(1) + 1e-6)  # (M, NUM_CLASSES) weighted average semantic vector

        # 5. New edges between supernodes
        # connect nearby supernodes using ball-query with radius config.BALL_QUERY_RADIUS
        edge_index_super = radius_graph(p_super, 
                                  r=config.BALL_QUERY_RADIUS, 
                                  loop=False,
                                  max_num_neighbors=config.MAX_NUM_NEIGHBORS)  # (2, E_super)
        
        # 6. Return the coarsened graph data
        return {'p': p_super, # (M, 3) supernode positions
                'r': r_super, # (M, 3) supernode radius
                's': s_super, # (M, NUM_CLASSES) supernode semantic vectors
                'edge_index': edge_index_super, # (2, E_super) supernode edges
                'S': S, # (N, M) soft assignment matrix
                'hard_assign': hard_assign, # (N,) hard assignment indices
                'h_pooled': S.T @ h / (S_col_sum.unsqueeze(1) + 1e-6) # (M, d) pooled node features for supernodes
                }






