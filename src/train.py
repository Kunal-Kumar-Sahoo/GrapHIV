import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, 
    precision_score, recall_score, roc_auc_score
)
import numpy as np
from tqdm import tqdm
from dataset import MoleculeDataset
from model import MoleculeNet

import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("http://localhost:5000")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0

    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x.float(),
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss / step

def calculate_metrics(preds, labels, epoch, name):
    raise NotImplementedError