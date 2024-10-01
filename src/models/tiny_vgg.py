"""
Contains PyTorch model class to instantiate a TinyVGG model.
"""

from torch import nn


class TinyVGGConvBlock(nn.Module):
    """Creates the TinyVGG convolutional block architecture.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels (filters).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.model(x)


class TinyVGG(nn.Module):
    """Creates the TinyVGG model architecture.

    Link: https://poloclub.github.io/cnn-explainer/

    Parameters
    ----------
    input_shape: int
        Number of input channels.
    hidden_units: int
        Number of hidden units between layers.
    output_shape: int
        Number of output units (number of classes).
    """
    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
    ):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            TinyVGGConvBlock(in_channels=input_shape, out_channels=hidden_units),
            TinyVGGConvBlock(in_channels=hidden_units, out_channels=hidden_units),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=13 * 13 * hidden_units,
                out_features=output_shape
            )
        )

    def forward(self, x):
        return self.classifier(self.conv_blocks(x))


__all__ = ["TinyVGG"]
