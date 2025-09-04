import torch.nn as nn

from .convblock import ConvBlock
from .effect_cross_att import EffectCrossAttention


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        embedding_dim,
        apply_pool,
        num_heads,
    ):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            apply_pool=False,
        )
        self.attn_mod = EffectCrossAttention(out_channels, embedding_dim, num_heads)
        self.pool = (
            nn.MaxPool1d(kernel_size=2, stride=2) if apply_pool else nn.Identity()
        )

    def forward(self, x, effect_embedding):
        x = self.conv_block(x)

        # x_processed will be used for the skip connection
        x_pre_pool = self.attn_mod(x, effect_embedding)

        # apply pooling after conditioning for the main path
        x_post_pool = self.pool(x_pre_pool)

        return x_post_pool, x_pre_pool
