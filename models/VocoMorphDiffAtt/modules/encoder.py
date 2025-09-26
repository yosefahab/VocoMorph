import torch.nn as nn
from .convblock import ConvBlock
from .cross_attention import CrossAttention1D
from .time_embedding import TimeEmbedding
import torch


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        embed_dim,
        apply_pool,
        use_attn=False,
        time_dim=None,
    ):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size, padding, apply_pool=False
        )
        self.use_attn = use_attn
        if self.use_attn:
            self.attn = CrossAttention1D(out_channels, embed_dim + (time_dim or 0))
        self.pool = (
            nn.MaxPool1d(kernel_size=2, stride=2) if apply_pool else nn.Identity()
        )

    def forward(self, x, effect_embedding, time_embedding=None):
        x = self.conv_block(x)
        cond = effect_embedding
        if time_embedding is not None:
            cond = torch.cat([cond, time_embedding], dim=-1)
        if self.use_attn:
            x_pre_pool = self.attn(x, cond)
        else:
            x_pre_pool = x
        x_post_pool = self.pool(x_pre_pool)
        return x_post_pool, x_pre_pool
