import torch
import torch.nn as nn


class DeconvBlock(nn.Module):
    """
    Standard Deconvolutional Block for the U-Net decoder.
    Consists of ConvTranspose2d (upsampling) -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv1 = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_features):
        x = self.upconv(x)

        diffY = skip_features.size()[2] - x.size()[2]
        diffX = skip_features.size()[3] - x.size()[3]
        x = nn.functional.pad(
            x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

        # concatenate skip connection features along the channel dimension
        x = torch.cat([x, skip_features], dim=1)  # dim=1 for channels

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
