import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config, effect_dim):
        super().__init__()
        layers = []
        in_channels = config["in_channels"] + effect_dim  # Inject effect embedding
        base_channels = config["base_channels"]
        for _ in range(config["num_layers"]):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    base_channels,
                    config["kernel_size"],
                    stride=config["stride"],
                    padding=config["padding"],
                )
            )
            layers.append(nn.ReLU())
            in_channels = base_channels
            base_channels *= 2
        self.encoder = nn.Sequential(*layers)

    def forward(self, x, effect_embed):
        """Concatenates effect embedding across channels."""
        x = torch.cat([x, effect_embed], dim=1)  # Concatenate along channel dim
        return self.encoder(x)
