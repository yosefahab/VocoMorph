import torch.nn as nn

from .deconvblock import DeconvBlock
from .effect_cross_att import EffectCrossAttention


class Decoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, embedding_dim, num_heads
    ):
        super().__init__()
        self.deconv_block = DeconvBlock(in_channels, out_channels, kernel_size, padding)
        self.attn_mod = EffectCrossAttention(out_channels, embedding_dim, num_heads)

    def forward(self, x, skip_features, effect_embedding):
        x = self.deconv_block(x, skip_features)
        x = self.attn_mod(x, effect_embedding)
        return x
