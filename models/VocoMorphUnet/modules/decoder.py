import torch.nn as nn

from .deconvblock import DeconvBlock
from .film import FiLM


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        embedding_dim,
        skip_channels,
    ):
        super().__init__()
        self.deconv_block = DeconvBlock(
            in_channels, out_channels, skip_channels, kernel_size, padding
        )
        self.film_layer = FiLM(out_channels, embedding_dim)

    def forward(self, x, skip_features, effect_embedding):
        x = self.deconv_block(x, skip_features)
        x = self.film_layer(x, effect_embedding)
        return x
