"""
Contains functions for core testing loop of a PyToch model.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from typing import Dict, List

from .train_step import train_step
from .test_step import test_step


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Parameters
    ----------
    model: nn.Module
        A PyTorch model to be trained and tested.
    train_dataloader: DataLoader
        A DataLoader to be used for training the model.
    test_dataloader: DataLoader
        A DataLoader to be used for testing the model.
    loss_fn: nn.Module
        A PyTorch loss function to calculate the loss on both datasets.
    optimizer: torch.optim.Optimizer
        A PyTorch optimizer to help minimize the loss function
    device: torch.device
        Target device to compute on ("cuda", "cpu", etc.)
    verbose: bool
        If true, logs the gradients of the model params once after each epoch.

    Returns
    -------
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form:
    {
        train_loss: [...],
        train_acc: [...],
        test_loss: [...],
        test_acc: [...]
    }
    """
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}")
        print(f"---------")

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            verbose=verbose
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc * 100:.1f}%")
        print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc * 100:.1f}%\n")

        # Update the results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


__all__ = ["train"]
