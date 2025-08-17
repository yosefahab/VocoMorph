import mlx.core as mx
import mlx.nn as nn


class Decoder(nn.Module):
    """
    Deconvolutional Block for the U-Net decoder.

    This block handles upsampling, convolutions, and the concatenation of
    the skip connection. The key is correctly calculating the channel counts.
    """

    def __init__(
        self,
        prev_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # conv1 processes the upsampled tensor from the previous decoder layer.
        self.conv1 = nn.Conv1d(
            prev_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

        # The input to conv2 is the concatenated tensor, so its input channels
        # must be the sum of the upsampled tensor's channels and the skip connection's channels.
        self.conv2 = nn.Conv1d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def __call__(self, x: mx.array, skip_features: mx.array):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # pad the upsampled tensor if its length doesn't match the skip connection's length.
        # this assumes the channel dimension is at axis=2.
        diff = skip_features.shape[1] - x.shape[1]  # pyright: ignore[reportIndexIssue]
        if diff > 0:
            x = mx.pad(x, [0, 0, 0, diff])

        # concatenate the upsampled tensor with the skip connection.
        x = mx.concatenate([x, skip_features], axis=2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
