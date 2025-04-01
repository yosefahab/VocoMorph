import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv2d(
            config["channels"],
            config["channels"],
            config["kernel_size"],
            stride=config["stride"],
            padding=config["padding"],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
