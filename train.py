# train.py — Stage-wise training loop for the Scene Graph VAE
# Stage 1: coarsening only
# Stage 2: + mid encoder branch
# Stage 3: + coarse encoder branch
# Stage 4: joint fine-tune (all modules, reduced LR)
# TODO: implement (todo id: train)

import os
import torch
from torch.utils.data import DataLoader

import config
from gvae.models.gvae import GVAE
from gvae.data.scene_graph import SceneGraph
from gvae.losses.gvae_loss import compute_loss

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


def train_stage(model, loader, num_epochs, stage, step_counter, lr):
    set_stage(model, stage)
    # only pass unfrozen params to the optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr  # use the lr argument, not hardcoded config value
    )

    best_loss = float('inf')  # track best (lowest) average loss seen so far

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for graph in loader:
            optimizer.zero_grad()  # clear old gradients

            outputs = model(graph)  # forward pass
            loss, breakdown = compute_loss(outputs, graph, step_counter[0])

            loss.backward()   # backpropagate to compute gradients
            optimizer.step()  # update weights

            epoch_loss += loss.item()
            step_counter[0] += 1  # increment global step counter

        avg_loss = epoch_loss / len(loader)
        print(f"Stage {stage} Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # save best model for this stage
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"checkpoint/gvae/stage{stage}_best.pth")
            print(f"  → New best loss {best_loss:.4f}, model saved.")

    return step_counter


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset and create dataloader
    dataset = SceneGraphDataset(config.GRAPH_DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Create model
    model = GVAE().to(device)

    step_counter = [0]  # global step counter across all stages

    # Stage 1
    train_stage(model, dataloader, config.NUM_EPOCHS_STAGE1,
                stage=1, step_counter=step_counter, lr=config.LEARNING_RATE)
    torch.save(model.state_dict(), "checkpoint/gvae/stage1.pth")

    # Stage 2
    train_stage(model, dataloader, config.NUM_EPOCHS_STAGE2,
                stage=2, step_counter=step_counter, lr=config.LEARNING_RATE)
    torch.save(model.state_dict(), "checkpoint/gvae/stage2.pth")

    # Stage 3
    train_stage(model, dataloader, config.NUM_EPOCHS_STAGE3,
                stage=3, step_counter=step_counter, lr=config.LEARNING_RATE)
    torch.save(model.state_dict(), "checkpoint/gvae/stage3.pth")

    # Stage 4 — reduced LR for joint fine-tune
    train_stage(model, dataloader, config.NUM_EPOCHS_STAGE4,
                stage=4, step_counter=step_counter, lr=config.LEARNING_RATE_FINETUNE)
    torch.save(model.state_dict(), "checkpoint/gvae/final.pth")

    print("Training complete. Final model saved to checkpoint/gvae/final.pth")


if __name__ == "__main__":
    os.makedirs("checkpoint/gvae", exist_ok=True)
    main()