import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for the U-Net encoder.
    Consists of Conv1d -> BatchNorm -> ReLU -> Optional MaxPool.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, padding, apply_pool=True
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = (
            nn.MaxPool1d(kernel_size=2, stride=2) if apply_pool else nn.Identity()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
