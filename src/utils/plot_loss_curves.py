"""
Contains utility function for plotting the model loss curves from the result dictionary.
"""

import matplotlib.pyplot as plt

from typing import Dict, List


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plot the model loss curves.

    Parameters
    ----------
    results: Dict[str, List[float]]
        Results dictionary from the training step of a model.
        In form of:
        {
            train_loss: [...],
            train_acc: [...],
            test_loss: [...],
            test_acc: [...]
        }
    """
    train_loss = results["train_loss"]
    train_acc  = results["train_acc"]
    test_loss  = results["test_loss"]
    test_acc   = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss,  label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, test_acc,  label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


__all__ = ["plot_loss_curves"]
