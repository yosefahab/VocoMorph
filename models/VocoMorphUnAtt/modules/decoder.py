import torch.nn as nn

from .attenblock import AttentionBlock
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
        num_heads,
        use_sequence_atten=False,
        use_channel_atten=False,
    ):
        super().__init__()
        self.deconv_block = DeconvBlock(in_channels, out_channels, kernel_size, padding)
        self.film_layer = FiLM(out_channels, embedding_dim)
        self.attn = AttentionBlock(
            out_channels,
            use_sequence=use_sequence_atten,
            use_channel=use_channel_atten,
            num_heads=num_heads,
        )

    def forward(self, x, skip_features, effect_embedding):
        x = self.deconv_block(x, skip_features)
        x = self.attn(x)
        x = self.film_layer(x, effect_embedding)
        return x
