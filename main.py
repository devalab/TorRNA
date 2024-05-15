import os
import torch
import numpy as np

from tqdm import tqdm
import wandb

from data.dataloader import get_train_test_dataloaders

from models.transformer import TorsionalAnglesTransformerDecoder

import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
# from utils import get_indices_of_last_node_from_batchs

import argparse

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('-w', '--wandb', default="disabled")
parser.add_argument('-ti', '--trainingiters', type=int, default=100)
parser.add_argument('-lr', '--lr', type=float, default=2e-4)
parser.add_argument('-ed', '--embeddim', type=int, default=640)
parser.add_argument('-hd', '--hiddendim', type=int, default=256)
parser.add_argument('-nh', '--numheads', type=int, default=4)
parser.add_argument('-nl', '--numlayers', type=int, default=3)
parser.add_argument('-do', '--dropout', type=float, default=0.2)
parser.add_argument('-tol', '--earlystoppingtolerance', type=int, default=5)
args = parser.parse_args()

wandb.login()
run = wandb.init(
        project="ToRNA",
        name=f"{args.numlayers}l_{args.embeddim}_{args.hiddendim}_{args.dropout}do_{args.numheads}nh_{args.lr}lr",
        mode=args.wandb
        )

train_iters = args.trainingiters

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_loader, test_loader = get_train_test_dataloaders()
train_loader, val_loader, test_loader = get_train_test_dataloaders(root="./data/temp_dataset/",
                            pdbs_path="./data/all_torrna_pdbs",
                            perfect_pdb_files_train_val_test_path="./data/torrna_train_val_test.pkl")

model = TorsionalAnglesTransformerDecoder(embed_dim=args.embeddim, hidden_dim=args.hiddendim, num_heads=args.numheads, num_layers=args.numlayers, dropout=args.dropout)
model.to(device)

# for each_batch in train_loader:
#     rna_fm_embeddings, torsional_angles, padding_mask = each_batch
#     rna_fm_embeddings = rna_fm_embeddings.to(device)
#     torsional_angles = torsional_angles.to(device)
#     padding_mask = padding_mask.to(device)

#     print(f"rna_fm_embeddings: {rna_fm_embeddings.shape}")
#     print(f"torsional_angles: {torsional_angles.shape}")
#     print(f"padding_mask: {padding_mask.shape}")

#     output = model(rna_fm_embeddings, padding_mask)

#     print(f"Model output: {output.shape}")

# exit(1)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=8, min_lr=1.e-7)

class EarlyStopping:
    def __init__(self, tolerance=5):
        self.tolerance = tolerance
        self.min_loss = 1e9
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss > self.min_loss:
            self.counter +=1
        else:
            self.min_loss = val_loss
            self.counter = 0

        if self.counter >= self.tolerance:  
            self.early_stop = True
            return True
        else:
            return False
early_stopper = EarlyStopping(tolerance=args.earlystoppingtolerance)

def masked_MSELoss(output, target):
    nan_mask = ~torch.isnan(target)
    target = torch.nan_to_num(target)
    output = output * nan_mask
    return nn.MSELoss(reduction='sum')(output, target)
loss_fn = masked_MSELoss

best_val_loss = 1e10
train_tqdm = tqdm(range(train_iters))
for train_iter in train_tqdm:
    model.train()
    
    all_losses = list()
    for batch in train_loader:
    
        rna_fm_embeddings, torsional_angles, initial_embeddings, padding_mask = batch
        rna_fm_embeddings = rna_fm_embeddings.to(device)
        torsional_angles = torsional_angles.to(device)
        initial_embeddings = initial_embeddings.to(device)
        padding_mask = padding_mask.to(device)

        optimizer.zero_grad()

        output = model(rna_fm_embeddings, padding_mask, initial_embeddings)

        # print(f"\nrna_fm_embeddings: {rna_fm_embeddings.shape}")
        # print(f"initial_embeddings: {initial_embeddings.shape}")
        # print(f"torsional_angles: {torsional_angles.shape}")
        # print(f"padding_mask: {padding_mask.shape}")
        # print(f"Model output: {output.shape}\n")

        loss = loss_fn(output, torsional_angles)
        loss.backward()

        orig_grad_norm = clip_grad_norm_(model.parameters(), 100., error_if_nonfinite=True)
        optimizer.step()

        train_tqdm.set_description(f"TIter: {train_iter} Loss: {loss.item():.2f}")
        all_losses.append(loss.item())

    wandb.log({"train-loss": sum(all_losses)/len(all_losses)})
    torch.save(model.state_dict(), f"./checkpoints/most_recent_model_rna_transformer_{args.lr}_{args.embeddim}_{args.hiddendim}_{args.numheads}_{args.numlayers}_{args.dropout}_{args.earlystoppingtolerance}.pkl")

    if train_iter % 5 == 0:
        model.eval()
        all_losses = list()
        for batch in tqdm(val_loader):

            rna_fm_embeddings, torsional_angles, initial_embeddings, padding_mask = batch
            rna_fm_embeddings = rna_fm_embeddings.to(device)
            torsional_angles = torsional_angles.to(device)
            initial_embeddings = initial_embeddings.to(device)
            padding_mask = padding_mask.to(device)

            output = model(rna_fm_embeddings, padding_mask, initial_embeddings)

            loss = loss_fn(output, torsional_angles)

            # print(f"Train iteration :{train_iter} Test Loss: {loss.item():.2f}")
            all_losses.append(loss.item())
        
        cur_val_loss = sum(all_losses)/len(all_losses)
        wandb.log({"val-loss": cur_val_loss})
        print("Val loss:", cur_val_loss)
        scheduler.step(cur_val_loss)
        if early_stopper(cur_val_loss):
            print(f"Ending due to early stopper")
            break
        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            torch.save(model.state_dict(), f"./checkpoints/best_model_rna_transformer_{args.lr}_{args.embeddim}_{args.hiddendim}_{args.numheads}_{args.numlayers}_{args.dropout}_{args.earlystoppingtolerance}.pkl")
