import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(
        self, config: dict, in_channels, out_channels, dilation, embedding_dim
    ):
        super().__init__()
        kernel_size = config["kernel_size"]
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels + embedding_dim,
            out_channels,
            kernel_size,  # Adjusted input channels
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, embedding):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        # Repeat embedding along the temporal dimension
        embedding_repeated = embedding.unsqueeze(-1).repeat(1, 1, out.shape[-1])

        # Concatenate with the feature map
        out_concat = torch.cat([out, embedding_repeated], dim=1)

        out = self.conv2(out_concat)
        out = self.norm2(out)
        out = self.relu2(out)
        return out + self.residual(x)


class TCN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.input_channels = config["input_channels"]
        self.hidden_channels = config["hidden_channels"]
        self.output_channels = config["output_channels"]
        self.num_layers = config["num_layers"]
        self.embedding_dim = config["embedding_dim"]
        self.kernel_size = config["kernel_size"]

        layers = []
        in_channels = self.input_channels
        for i in range(self.num_layers):
            out_channels = (
                self.hidden_channels
                if i < self.num_layers - 1
                else self.output_channels
            )
            layers.append(
                TCNBlock(
                    config=config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=2**i,
                    embedding_dim=self.embedding_dim,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x, embedding):
        out = x
        for layer in self.network:
            out = layer(out, embedding)
        return out

    def calculate_receptive_field(self):
        dilations_sum = sum(2**i for i in range(self.num_layers))
        return (self.kernel_size - 1) * dilations_sum + 1
