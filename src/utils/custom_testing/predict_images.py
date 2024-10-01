"""
Contains utility function for predicting custom images
"""

import math

import torch
from torch import nn
from torchvision import transforms

from .read_custom_images import read_custom_images

from typing import List


def predict_images(
    model: nn.Module,
    image_path: str,
    class_names: List[str],
    transform: transforms.Compose,
    plot_predictions: bool = False,
    device: torch.device,
) -> List[str]:
    """Predict custom images.

    Parameters
    ----------
    model: nn.Module
        A PyTorch model to make the predictions.
    image_path: str
        Path to the images that wanted to be predicted.
    class_names: List[str]
        List of class names associated with the model.
    transform: transforms.Compose
        A transformation function to transform the images into a desired format.
    plot_predictions: bool
        If true, plot the predictions and the images.
    device:
        Target device to compute on ("cuda", "cpu", etc.)

    Returns
    -------
    A list of string containing the predicted class names.
    """
    # Load the images
    images_arr = read_custom_images(image_path=image_path)

    # Transform the images
    batch_images = torch.stack([transform(image) for image in images_arr])

    # Make the predictions
    model.to(device)
    model.eval()
    with torch.inference_mode():
        y_pred = model(batch_images.to(device))
        pred_probs = torch.softmax(y_pred, dim=1)
        pred_labels = torch.argmax(pred_probs, dim=1)

    predictions = [class_names[label.cpu()] for label in pred_labels]

    # Plot the predictions
    if plot_predictions:
        import matplotlib.pyplot as plt
        ncols = 2
        nrows = math.ceil(len(images_arr) / ncols)
        for i, image in enumerate(images_arr):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(image.permute(1, 2, 0))
            plt.title(f"{predictions[i]} ({pred_probs[i].max().cpu() * 100:.1f}%)")
            plt.axis(False)

    return predictions


__all__ = ["predict_images"]
