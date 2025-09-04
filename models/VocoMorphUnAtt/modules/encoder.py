import torch.nn as nn

from .attenblock import AttentionBlock
from .convblock import ConvBlock
from .film import FiLM


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        embedding_dim,
        apply_pool,
        num_attn_heads,
        use_sequence_atten=False,
        use_channel_atten=False,
    ):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            apply_pool=False,
        )
        self.film_layer = FiLM(out_channels, embedding_dim)
        self.pool = (
            nn.MaxPool1d(kernel_size=2, stride=2) if apply_pool else nn.Identity()
        )

        self.attn = AttentionBlock(
            out_channels,
            use_sequence=use_sequence_atten,
            use_channel=use_channel_atten,
            num_heads=num_attn_heads,
        )

    def forward(self, x, effect_embedding):
        # x: (batch, channels, length)
        x = self.conv_block(x)

        x = self.attn(x)
        # x_processed will be used for the skip connection
        x_pre_pool = self.film_layer(x, effect_embedding)

        # apply pooling after FiLM for the main path
        x_post_pool = self.pool(x_pre_pool)

        return x_post_pool, x_pre_pool
