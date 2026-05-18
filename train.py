# train.py — Stage-wise training loop for the Scene Graph VAE
# Stage 1: coarsening only
# Stage 2: + mid encoder branch
# Stage 3: + coarse encoder branch
# Stage 4: joint fine-tune (all modules, reduced LR)
# TODO: implement (todo id: train)

import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader

import config
from gvae.models.gvae import GVAE
from gvae.data.scene_graph import SceneGraph
from gvae.losses.gvae_loss import compute_loss
from gvae.losses.metrics import compute_metrics

class SceneGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        # find all .json files in the folder
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.json') 
        ]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # load the scene graph from the JSON file
        return SceneGraph.from_json(self.files[idx])


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(modules):
    # modules: a list of nn.Module to unfreeze
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True

def set_stage(model, stage):
    freeze_all(model)  # start by freezing everything
    enc = model.encoder
    dec = model.decoder

    if stage >= 1:
        # coarsening modules only
        unfreeze([enc.coarsen_1, enc.coarsen_2])

    if stage >= 2:
        # add mid branch
        unfreeze([enc.input_proj_L, enc.gps_L, enc.splat_mid,
                  enc.input_proj_L1, enc.gps_L1, enc.unet_mid,
                  dec])

    if stage >= 3:
        # add coarse branch
        unfreeze([enc.input_proj_1, enc.gps_1,
                  enc.splat_coarse, enc.unet_coarse])

def validate(model, loader, stage, step_counter, device):
    model.eval()

    val_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for batch in loader:
            for graph in batch:
                graph = graph.to(device)
                outputs = model(graph)
                loss, _ = compute_loss(outputs, graph, step_counter[0])
                val_loss += loss.item()

                m = compute_metrics(outputs, stage)
                if m:
                    all_metrics.append(m)

                step_counter[0] += 1

    avg_loss = val_loss / len(loader)

    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    model.train()  # switch back to training mode
    return avg_loss, avg_metrics


def train_stage(model, loader, val_loader, num_epochs, stage, step_counter, lr, device, ckpt_dir):
    set_stage(model, stage)
    # only pass unfrozen params to the optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr  # use the lr argument, not hardcoded config value
    )

    best_loss = float('inf')  # track best (lowest) average loss seen so far

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in loader: # batch is a list of graph objects due to collate_fn in dataloader
            # we need to loop over the list of graphs and process them one by one, since our model and loss are not designed for batching multiple graphs together
            optimizer.zero_grad()  # clear gradients before processing the batch
            for graph in batch:
                graph = graph.to(device)  # move graph tensors to GPU
                outputs = model(graph)  # forward pass for each graph in the batch
                loss, _ = compute_loss(outputs, graph, step_counter[0])  # compute loss for each graph
                loss.backward()  # accumulate gradients for each graph
                epoch_loss += loss.item()  # accumulate loss for logging
                step_counter[0] += 1  # increment global step counter for each graph

            optimizer.step()  # update weights once per batch, after all graphs

        avg_loss = epoch_loss / len(loader)

        # validation 
        avg_val_loss, avg_metrics = validate(model, val_loader, stage, step_counter, device)
        print(f"Stage {stage} Epoch {epoch+1}/{num_epochs}, train_loss: {avg_loss:.4f}, val_loss: {avg_val_loss:.4f}")
        if avg_metrics:
            print(f"mid  → acc={avg_metrics['acc_mid']:.1%}  pos_err={avg_metrics['pos_err_mid']:.4f}  size_err={avg_metrics['size_err_mid']:.4f}")
            print(f"coarse→ acc={avg_metrics['acc_coarse']:.1%}  pos_err={avg_metrics['pos_err_coarse']:.4f}  size_err={avg_metrics['size_err_coarse']:.4f}")

        # save best model for this stage based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"stage{stage}_best.pth"))
            print(f"  → New best validation loss {best_loss:.4f}, model saved.")
 
    return step_counter


def main(ckpt_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset and create dataloader
    train_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'train'))
    val_dataset = SceneGraphDataset(os.path.join(config.GRAPH_DATA_DIR, 'val'))

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)  # collate_fn to return list of graphs
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

    # Create model
    model = GVAE().to(device)

    step_counter = [0]  # global step counter across all stages

    # Stage 1
    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE1,
                stage=1, step_counter=step_counter, lr=config.LEARNING_RATE, device=device, ckpt_dir=ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "stage1.pth"))

    # Stage 2
    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE2,
                stage=2, step_counter=step_counter, lr=config.LEARNING_RATE, device=device, ckpt_dir=ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "stage2.pth"))

    # Stage 3
    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE3,
                stage=3, step_counter=step_counter, lr=config.LEARNING_RATE, device=device, ckpt_dir=ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "stage3.pth"))

    # Stage 4 — reduced LR for joint fine-tune
    train_stage(model, train_dataloader, val_dataloader, config.NUM_EPOCHS_STAGE4,
                stage=4, step_counter=step_counter, lr=config.LEARNING_RATE_FINETUNE, device=device, ckpt_dir=ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pth"))

    print(f"Training complete. Final model saved to {ckpt_dir}/final.pth")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20260513_142301
    ckpt_dir  = os.path.join("checkpoint", timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    main(ckpt_dir)