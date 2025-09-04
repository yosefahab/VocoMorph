import torch.nn as nn

from .convblock import DilatedConvBlock
from .film import FiLM


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        embedding_dim,
        apply_pool,
        dilation,
    ):
        super().__init__()
        self.conv_block = DilatedConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            apply_pool=False,
        )
        self.film_layer = FiLM(out_channels, embedding_dim)
        self.pool = nn.MaxPool1d(2, stride=2) if apply_pool else nn.Identity()

    def forward(self, x, effect_embedding):
        x = self.conv_block(x)
        x_pre_pool = self.film_layer(x, effect_embedding)
        x_post_pool = self.pool(x_pre_pool)
        return x_post_pool, x_pre_pool
