import mlx.core as mx
import mlx.nn as nn

from .convblock import ConvBlock
from .film import FiLM


class Encoder(nn.Module):
    """
    A full encoder block for the U-Net.

    It applies a ConvBlock, then a FiLM layer for conditioning, and finally
    a pooling layer for downsampling. It returns both the pooled output
    for the main path and the pre-pooled tensor for the skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        embedding_dim: int,
        apply_pool: bool,
    ):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, padding)
        self.film_layer = FiLM(out_channels, embedding_dim)
        # Apply MaxPool1d if apply_pool is True, otherwise use Identity.
        self.pool = (
            nn.MaxPool1d(kernel_size=2, stride=2) if apply_pool else nn.Identity()
        )

    def __call__(self, x: mx.array, effect_embedding: mx.array):
        x = self.conv_block(x)
        # Apply the FiLM layer before pooling to get the skip connection tensor.
        x_pre_pool = self.film_layer(x, effect_embedding)
        # Apply pooling for the main path.
        x_post_pool = self.pool(x_pre_pool)

        # Return both the pooled tensor (for the next encoder/bottleneck)
        # and the pre-pooled tensor (for the decoder's skip connection).
        return x_post_pool, x_pre_pool
