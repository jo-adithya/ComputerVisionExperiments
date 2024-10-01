"""
Contains utility function for reading custom images for testing
"""

from pathlib import Path

import torch
import torchvision

from typing import List


def read_custom_images(image_path: str) -> List[torch.float]:
    """Read custom images and turn it into PyTorch tensors.

    Parameters
    ----------
    image_path: str
        Path to the custom images that want to be read.

    Returns
    -------
    List of PyTorch tensors corresponding to the custom images.
    """
    custom_images_path = Path(image_path).rglob("*.*")
    images_arr = []

    for i, image_path in enumerate(custom_images_path):
        image_arr = torchvision.io.read_image(str(image_path))
        normalized_image_arr = image_arr.type(torch.float) / 255.
        images_arr.append(normalized_image_arr)

    return images_arr


__all__ = ["read_custom_images"]
