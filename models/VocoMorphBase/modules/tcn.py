import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(in_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out) + x


class TCN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.channels = config["channels"]
        self.num_layers = config["num_layers"]
        self.kernel_size = config["kernel_size"]

        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]

        layers = []
        for i in range(self.num_layers):
            layers.append(
                TCNBlock(
                    in_channels=self.channels,
                    kernel_size=self.kernel_size,
                    dilation=2**i,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x

    def calculate_receptive_field(self):
        """Calculate the receptive field of the TCN"""
        # RF = (kernel_size-1) * sum(dilations) + 1
        dilations_sum = sum(2**i for i in range(self.num_layers))
        return (self.kernel_size - 1) * dilations_sum + 1
