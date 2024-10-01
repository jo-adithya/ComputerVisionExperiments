"""
Contains functionality for creating PyTorch DataLoaders.
"""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from typing import Tuple


NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_path: str,
    test_path: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader]:
    """Creates training and testing DataLoaders.

    Parameters
    ----------
    train_path: str
        Path to the training directory.
    test_path: str
        Path to the testing directory.
    train_transform: torchvision.transforms.Compose
        torchvision transforms to be performed on the training data.
    test_transform: torchvision.transforms.Compose
        torchvision transforms to be performed on the testing data.
    batch_size: int
        Number of samples per batch.
    num_workers: int
        Number of workers per DataLoader.

    Returns
    -------
    A tuple of training dataloader and testing dataloader.
    In the form of (train_dataloader, test_dataloader).
    """
    # Create training and testing datasets
    train_data = datasets.ImageFolder(train_path, transform=train_transform)
    test_data = datasets.ImageFolder(test_path, transform=test_transform)

    # Turn datasets into dataloaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


__all__ = ["create_dataloaders"]
