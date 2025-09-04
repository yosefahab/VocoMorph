import torch.nn as nn


class DilatedConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, apply_pool=True
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, stride=2) if apply_pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
