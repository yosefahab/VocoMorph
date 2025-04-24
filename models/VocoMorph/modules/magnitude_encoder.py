import torch.nn as nn


class SubNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_channels = config["input_channels"]
        self.hidden_channels = config["hidden_channels"]
        self.kernel_size = config["kernel_size"]
        self.padding = "same"

        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.input_channels,
            kernel_size=self.kernel_size,
            padding="same",
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
