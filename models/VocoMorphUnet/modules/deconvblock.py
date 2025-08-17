import mlx.core as mx
import mlx.nn as nn


class DeconvBlock(nn.Module):
    def __init__(
        self, prev_channels, skip_channels, out_channels, kernel_size, padding
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv1d(
            prev_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels + skip_channels,  # This is the correct calculation
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()

    def __call__(self, x, skip_features):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        diff = skip_features.shape[2] - x.shape[2]
        if diff > 0:
            x = mx.pad(x, [diff // 2, diff - diff // 2])

        x = mx.concatenate([x, skip_features], axis=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
