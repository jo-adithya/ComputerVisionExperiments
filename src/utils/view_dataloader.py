"""
Contains functionality for viewing a batch of PyTorch DataLoaders.
"""

import math

from torch.utils.data import DataLoader


def view_dataloader(
    dataloader: DataLoader,
    ncols: int = 8
):
    """
    View a batch of images in the PyTorch DataLoader.

    Parameters
    ----------
    dataloader: DataLoader
        The dataloader that wants to be viewed.
    ncols: int
        Number of matplotlib plot columns.
    """
    images, labels = next(iter(dataloader))
    nrows = math.ceil(len(images) / ncols)

    for i, image in enumerate(images):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.set_title(dataloader.classes[labels[i]])
        plt.axis(False)


__all__ = ["view_dataloader"]
