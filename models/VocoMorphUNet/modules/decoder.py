import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, config, effect_dim):
        super().__init__()
        layers = []
        base_channels = config["base_channels"] * (2 ** (config["num_layers"] - 1))
        out_channels = config["out_channels"]
        for _ in range(config["num_layers"]):
            layers.append(
                nn.ConvTranspose2d(
                    base_channels + effect_dim,
                    base_channels // 2,
                    config["kernel_size"],
                    stride=config["stride"],
                    padding=config["padding"],
                )
            )
            layers.append(nn.ReLU())
            base_channels //= 2
        layers.append(nn.Conv2d(base_channels, out_channels, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, effect_embed):
        """Concatenates effect embedding across channels."""
        x = torch.cat([x, effect_embed], dim=1)  # Concatenate along channel dim
        return self.decoder(x)
