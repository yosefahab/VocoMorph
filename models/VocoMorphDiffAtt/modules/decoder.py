import torch.nn as nn
from .deconvblock import DeconvBlock
from .cross_attention import CrossAttention1D
import torch


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        embedding_dim,
        skip_channels,
        use_attn=False,
        time_dim=None,
    ):
        super().__init__()
        self.deconv_block = DeconvBlock(
            in_channels, out_channels, skip_channels, kernel_size, padding
        )
        self.use_attn = use_attn
        if self.use_attn:
            self.attn = CrossAttention1D(out_channels, embedding_dim + (time_dim or 0))

    def forward(self, x, skip_features, effect_embedding, time_embedding=None):
        x = self.deconv_block(x, skip_features)
        cond = effect_embedding
        if time_embedding is not None:
            cond = torch.cat([cond, time_embedding], dim=-1)
        if self.use_attn:
            x = self.attn(x, cond)
        return x
