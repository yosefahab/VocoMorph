import torch
import torch.nn as nn


class DeconvBlock(nn.Module):
    """
    Standard Deconvolutional Block for the U-Net decoder, rewritten for smoother upsampling.
    Consists of Upsample -> Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm -> ReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        upscale_factor=2,
        dropout_p=0.2,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode="nearest")

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, skip_features):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        diff = skip_features.size(2) - x.size(2)
        if diff > 0:
            x = nn.functional.pad(x, [diff // 2, diff - diff // 2])

        # apply dropout to skip connection before concatenation
        skip_features = self.skip_dropout(skip_features)
        # concatenate skip connection features along the channel dimension
        x = torch.cat([x, skip_features], dim=1)  # dim=1 for channels

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
