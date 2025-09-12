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
        skip_channels,
        kernel_size,
        padding,
        dropout_p=0.2,
    ):
        super().__init__()
        self.upsample_conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, skip_features):
        x = self.upsample_conv(x)

        # check if output shape of upsampling matches skip connection before concatenation
        assert x.size(2) == skip_features.size(2), (
            f"Spatial dimensions of upsampled tensor {x.size(2)} and skip features {skip_features.size(2)} do not match."
        )

        x = self.bn1(x)
        x = self.relu(x)

        skip_features = self.skip_dropout(skip_features)
        x = torch.cat([x, skip_features], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
