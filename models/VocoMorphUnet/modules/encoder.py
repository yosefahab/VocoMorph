import torch.nn as nn
from .convblock import ConvBlock
from .film import FiLM


# class Encoder(nn.Module):
#     def __init__(
#         self, in_channels, out_channels, kernel_size, padding, embedding_dim, apply_pool
#     ):
#         super().__init__()
#         self.conv_block = ConvBlock(
#             in_channels,
#             out_channels,
#             kernel_size,
#             padding,
#             apply_pool=apply_pool,
#         )
#         self.film_layer = FiLM(out_channels, embedding_dim)
#
#     def forward(self, x, effect_embedding):
#         x = self.conv_block(x)
#         x = self.film_layer(x, effect_embedding)
#         return x


class Encoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, embedding_dim, apply_pool
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
            nn.MaxPool2d(kernel_size=2, stride=2) if apply_pool else nn.Identity()
        )

    def forward(self, x, effect_embedding):
        # Apply conv_block first
        x_processed = self.conv_block(x)

        # Apply FiLM layer to the output of the conv_block
        # This output (x_processed) will be used for the skip connection
        x_pre_pool = self.film_layer(x_processed, effect_embedding)

        # Apply pooling after FiLM for the main path
        x_post_pool = self.pool(x_pre_pool)

        return x_post_pool, x_pre_pool  # Return both pooled and pre-pooled (for skip)
