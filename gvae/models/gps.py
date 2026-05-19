# gvae/models/gps.py
# GPS layer: R-GAT local stream + exact attention (coarsest level); PointROPE


import torch, torch.nn as nn
from torch_geometric.nn import GATv2Conv

# PointROPE positional encoding function
def apply_PointROPE(x, pos):
    # x: [N, d], pos: [N, 3]
    # Apply PointROPE encoding to pos and concatenate with x

    d = x.shape[1]
    chunk = d//6
    chunks = x.chunk(6, dim=1)  # Split x into 6 chunks

    # rotation for X (chunks 0 and 1)
    theta = pos[:, 0] # one angle per node
    cos_t = theta.cos().unsqueeze(1)  # [N, 1]
    sin_t = theta.sin().unsqueeze(1)  # [N, 1]
    a, b = chunks[0], chunks[1]
    rot_a = a * cos_t - b * sin_t
    rot_b = a * sin_t + b * cos_t

    # rotation for Y (chunks 2 and 3)
    theta = pos[:, 1] # one angle per node
    cos_t = theta.cos().unsqueeze(1)  # [N, 1]
    sin_t = theta.sin().unsqueeze(1)  # [N, 1]
    c, d = chunks[2], chunks[3]
    rot_c = c * cos_t - d * sin_t
    rot_d = c * sin_t + d * cos_t

    # rotation for Z (chunks 4 and 5)
    theta = pos[:, 2] # one angle per node
    cos_t = theta.cos().unsqueeze(1)  # [N, 1]
    sin_t = theta.sin().unsqueeze(1)  # [N, 1]
    e, f = chunks[4], chunks[5]
    rot_e = e * cos_t - f * sin_t
    rot_f = e * sin_t + f * cos_t

    # concatenate all rotated chunks and return 
    rot_x = torch.cat([rot_a, rot_b, rot_c, rot_d, rot_e, rot_f], dim=1)  # [N, d]
    return rot_x

class GPSLayer(nn.Module):
    def __init__(self, d, num_heads, use_global_attention:bool):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.use_global_attention = use_global_attention

        # R-GAT local stream
        self.rgat = GATv2Conv(in_channels = d,
                              out_channels = d // num_heads,
                              heads = num_heads, 
                              edge_dim =1, # edge feature of dimension 1 (distance)
                              concat = True) 
        
        # for global stream: Q, K, V projectoins
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        self.W_O = nn.Linear(d, d)  # output projection after global attention

        # feed forward network (2 layers MLP to combine local and global features)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),  # expand dimension
            nn.ReLU(),
            nn.Linear(d * 4, d)   # project back to d
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, h, edge_index, edge_attr, pos):
        """
        h: (N, d) node features
        edge_index: (2, E) edge indices for R-GAT
        edge_attr: (E, 1) edge attributes (e.g. distances) for R-GAT
        pos: (N, 3) node positions for PointROPE

        Returns:
        h: (N, d) updated node features after GPS layer
        """

        # Local stream: R-GAT
        h_local = self.rgat(h, edge_index, edge_attr)  # (N, d)

        # global stream: exact attention (only at coarsest level)
        h_global = torch.zeros_like(h)  # default to zero if not used

        if self.use_global_attention:
            # compute Q, K, V
            Q = self.W_Q(h)  # (N, d)
            K = self.W_K(h)  # (N, d)
            V = self.W_V(h)  # (N, d)

            # apply PointROPE to Q and K
            Q = apply_PointROPE(Q, pos)  # (N, d)
            K = apply_PointROPE(K, pos)  # (N, d)

            # scaled dot-product attention
            scale = self.d ** 0.5
            attn = (Q @ K.T) / scale  # (N, N) attention scores
            attn = torch.softmax(attn, dim=1)  # (N, N) attention weights
            
            h_global = self.W_O(attn @ V)  # (N, d) global context vector

        # combine local and global features
        h = self.norm1(h + h_local + h_global)  # residual connection + layer norm
        h = self.norm2(h + self.ffn(h))  

        return h
    
