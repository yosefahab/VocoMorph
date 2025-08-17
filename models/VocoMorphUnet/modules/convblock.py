import mlx.core as mx
import mlx.nn as nn


class ConvBlock(nn.Module):
    """
    A standard Convolutional Block for the U-Net encoder.

    Consists of Conv1d -> BatchNorm -> ReLU. An optional MaxPool is applied
    outside this block in the Encoder class.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def __call__(self, x: mx.array):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
