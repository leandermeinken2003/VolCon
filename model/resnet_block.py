"""Define the ResNetBlock."""
#pylint: disable=too-many-instance-attributes, too-few-public-methods

import torch
from torch import nn
from torch.nn import functional as F

class ResNetBlock(nn.Module):
    """ResNetBlock.
    
    Define the resnet block from the ResNet-Architechture, from the "Deep Residual Learning for
    Image Recognition" paper from Microsoft Research.
    """
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            filter_size: int,
    ) -> None:
        """Define ResNetBlock.

        :param input_channels: The number of image channels from the input image.
        :param output_channels: The number of image channels of the output image, and also the
        number of channels used in the Convolutional layers.
        :param filter_size: The size of the filters used.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, filter_size, padding="same",
        )
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, filter_size, padding="same"
        )
        if not output_channels == input_channels:
            self.upsample = nn.Conv2d(
                input_channels, output_channels, 1,
            )
        else:
            self.upsample = None
        self.output_channels = output_channels

    def forward(self, image_x: torch.Tensor) -> torch.Tensor:
        """Compute the operations in this resnet-block."""
        residual = image_x
        if self.upsample:
            residual = self.upsample(residual)
        image_x = self.conv1(image_x)
        image_x = F.relu(image_x)
        image_x = self.conv2(image_x)
        image_x += residual
        return F.relu(image_x)
