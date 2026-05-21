import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

import config
from gvae.models.gvae import GVAE
from gvae.data.scene_graph import SceneGraph
from gvae.losses.gvae_loss import compute_loss

GRAPH_PATH = os.path.join(config.GRAPH_DATA_DIR, 'test', '5080_54400.json')

# load the graph
print("Loading graph...")
graph = SceneGraph.from_json(GRAPH_PATH)
print(f" {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

# model
print("Creating model...")
model = GVAE(stage=1)  
model.train()

# forward pass
print("Running forward pass...")
outputs = model(graph)
print("keys:", list(outputs.keys()))

# loss + backward pass
print("Computing loss...")
total_loss, loss_dict = compute_loss(outputs, graph, step=0)
print(f"  total loss: {total_loss.item():.4f}")
for k, v in loss_dict.items():
    if hasattr(v, 'item'):
        print(f"    {k}: {v.item():.4f}")

print("Backward pass...")
total_loss.backward()
print("  gradients OK")

print("\n Smoke test passed!")
