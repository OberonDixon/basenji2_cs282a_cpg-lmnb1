"""Contains code for the custom Dataset and train/test loops for CS 182/282A project.

This is for opening HDF5 files of embeddings and labels and training/testing
the linear transformation head, and plotting losses and metrics for every epoch.

Author: Jimin Jung
"""

import torch
from torch.utils.data import Dataset

import h5py
import tqdm
import numpy as np
import matplotlib.pyplot as plt


class H5Dataset(Dataset):
    """Takes in HDF5 files of embeddings and labels for train/validation/test datasets."""

    def __init__(self, embeds_path, targets_path):
        self.data = h5py.File(embeds_path)['embeddings']
        self.targets = h5py.File(targets_path)['128bp_bins']
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]
        x, y = np.transpose(x), np.transpose(y)
        return x, y


def train_loop(dataloader, model, device, loss_fn, optimizer, pearson):
    """Iterates through a chunk of the training dataset with the given device.
    
    Accumulates training loss and the data for Pearson coefficient every chunk
    for calculations of training loss and Pearson correlation coefficient.

    Args:
        dataloader: A PyTorch DataLoader instance with pre-determined batch size.
        model: An instantiated model pre-loaded to the device.
        device: A PyTorch device object to train on (i.e. 'mps' or 'cuda').
        loss_fn: A PyTorch criterion instantiated from torch.nn package.
        optimizer: A PyTorch optimizer instantiated from torch.optim package.
        pearson: An instantiated object of PearsonR.

    Returns:
        A tuple containing training loss and number of batches in the chunk.
    """
    model.train()
    num_batches = len(dataloader)
    train_loss = 0

    for X, y in dataloader:
        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)

        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        numpy_y = torch.permute(y, (0, 2, 1)).numpy(force=True)
        numpy_pred = torch.permute(pred, (0, 2, 1)).numpy(force=True)
        pearson.update_state(numpy_y, numpy_pred)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return train_loss, num_batches
    

def validation_test_loop(dataloader, model, device, loss_fn, pearson):
    """Iterates through a chunk of the validation or test dataset with
    the given device.
    
    Accumulates loss and the data for Pearson coefficient every chunk
    for calculations of loss and Pearson correlation coefficient.

    Args:
        dataloader: A PyTorch DataLoader instance with pre-determined batch size.
        model: An instantiated model pre-loaded to the device.
        device: A PyTorch device object to evaluate on (i.e. 'mps' or 'cuda').
        loss_fn: A PyTorch criterion instantiated from torch.nn package.
        pearson: An instantiated object of PearsonR.

    Returns:
        A tuple containing loss and number of batches in the chunk.
    """
    model.eval()
    num_batches = len(dataloader)
    loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            numpy_y = torch.permute(y, (0, 2, 1)).numpy(force=True)
            numpy_pred = torch.permute(pred, (0, 2, 1)).numpy(force=True)
            pearson.update_state(numpy_y, numpy_pred)

    return loss, num_batches


def plot_loss(loss_list):
    """Plots training and validation loss against epoch.
    
    Args:
        loss_list: A list of tuples of loss values.
    """
    x = np.arange(1, len(loss_list)+1)
    plt.plot(x, loss_list)
    plt.xticks(x)
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs. Epoch")
    plt.legend(["Training", "Validation"])
    plt.show()


def plot_coefficient(coeff_list):
    """Plots training and validation Pearson coefficients.
    
    Args:
        loss_list: A list of tuples of Pearson coefficient values.
    """
    x = np.arange(1, len(coeff_list)+1)
    plt.plot(x, coeff_list)
    plt.xticks(x)
    plt.xlabel("Epoch Number")
    plt.ylabel("Pearson Coefficient")
    plt.title("Training and Validation Pearson Coefficient vs. Epoch")
    plt.legend(["Training", "Validation"])
    plt.show()
    