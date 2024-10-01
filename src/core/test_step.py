"""
Contains functions for core testing loop of a PyToch model for one epoch.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Tuple


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Parameters
    ----------
    model: nn.Module
        A PyTorch model to be tested.
    dataloader: DataLoader
        A DataLoader to be used for testing the model.
    loss_fn: nn.Module
        A PyTorch loss function to calculate the loss on the testing data.
    device: torch.device
        Target device to compute on ("cuda", "cpu", etc.)

    Returns
    -------
    A tuple of testing loss and testing accuracy metrics.
    In the form of (test_loss, test_accuracy).
    """
    # Setup the model to be in testing mode
    model.eval()

    # Setup testing metrics
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Calculate the loss
            loss = loss_fn(y_pred, y)

            # Update the metrics
            test_loss += loss.item()
            y_pred_classes = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += torch.sum(y_pred_classes == y).item() / len(y_pred)

        # Calculate the average metrics across all batches
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


__all__ = ["test_step"]
