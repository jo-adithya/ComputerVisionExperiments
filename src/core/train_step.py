"""
Contains functions for core training loop of a PyToch model for one epoch.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Tuple


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Train a PyTorch model for a single epoch.

    Parameters
    ----------
    model: nn.Module
        A PyTorch model to be trained.
    dataloader: DataLoader
        A DataLoader to be used for training the model.
    loss_fn: nn.Module
        A PyTorch loss function to calculate the loss on the training data.
    optimizer: torch.optim.Optimizer
        A PyTorch optimizer to help minimize the loss function
    device: torch.device
        Target device to compute on ("cuda", "cpu", etc.)
    verbose: bool
        If true, logs the gradients of the model params once after each epoch.

    Returns
    -------
    A tuple of training loss and training accuracy metrics.
    In the form of (train_loss, train_accuracy).
    """
    # Setup the model to be on training mode
    model.train()

    # Setup training metrics
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Forward pass
        y_pred = model(X)

        # Calculate the loss
        loss = loss_fn(y_pred, y)

        # Optimize the model params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the metrics
        train_loss += loss.item()
        y_pred_classes = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += torch.sum(y_pred_classes == y).item() / len(y_pred)

    # Log gradients once per epoch
    if verbose:
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"Layer: {name} | Grad Norm: {grad_norm:.6f}")
            else:
                print(f"Layer: {name} | Grad Norm: None")
        print()

    # Calculate the average metrics across all batches
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


__all__ = ["train_step"]
