"""
Contains utility function for saving PyTorch model.
"""

from pathlib import Path

import torch
from torch import nn


def save_model(
    model: nn.Module,
    target_path: str,
    model_name: str,
):
    """Saves a PyTorch model to a target directory.

    Parameters
    ----------
    model: nn.Module
        A PyTorch model to be saved.
    target_path: str
        Path for saving the model to.
    model_name: str
        File name for the saved model.
        Should include ".pth" or ".pt" file extension.
    """
    # Create the target directory if not exists
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endsWith(".pth") or model_name.endsWith(".pt"),
        "model_name should ends with '.pt' or '.pth'"
    saved_model_path = target_path / model_name

    # Save the model
    print(f"[INFO] Saving model to: {saved_model_path}")
    torch.save(obj=model.state_dict(), f=saved_model_path)
    print("[INFO] Successfully saved the model.")


__all__ = ["save_model"]
