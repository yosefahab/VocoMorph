import torch.nn as nn

from .modules.decoder import Decoder
from .modules.effect_encoder import EffectEncoder
from .modules.encoder import Encoder


class VocoMorphDilUnet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.chunk_size = config["chunk_size"]

        embedding_dim = config["embedding_dim"]
        self.num_channels = config["num_channels"]
        encoder_filters = config["encoder_filters"]
        bottleneck_filters = config["bottleneck_filters"]
        decoder_filters = config["decoder_filters"]
        kernel_size = config["kernel_size"]
        padding = config["padding"]
        dilation = config["dilation"]

        assert len(encoder_filters) == len(decoder_filters), (
            "Number of Encoder and Decoder blocks must match"
        )

        self.effect_encoder = EffectEncoder(config["num_effects"], embedding_dim)

        self.encoder_blocks = nn.ModuleList()
        for i, out_ch in enumerate(encoder_filters):
            self.encoder_blocks.append(
                Encoder(
                    self.num_channels if i == 0 else encoder_filters[i - 1],
                    out_ch,
                    kernel_size,
                    embedding_dim,
                    apply_pool=True,
                    dilation=dilation,
                )
            )

        self.bottleneck_blocks = nn.ModuleList()
        for i, out_ch in enumerate(bottleneck_filters):
            self.bottleneck_blocks.append(
                Encoder(
                    encoder_filters[-1] if i == 0 else bottleneck_filters[i - 1],
                    out_ch,
                    kernel_size,
                    embedding_dim,
                    apply_pool=False,
                    dilation=dilation,
                )
            )

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_filters)):
            self.decoder_blocks.append(
                Decoder(
                    (bottleneck_filters[-1] if i == 0 else decoder_filters[i - 1]),
                    decoder_filters[i],
                    kernel_size,
                    padding,
                    embedding_dim,
                )
            )

        self.final_conv = nn.Conv1d(
            decoder_filters[-1], self.num_channels, kernel_size=1
        )

        self.num_downsampling_steps = len(encoder_filters)

    def forward(self, x):
        effect_id, audio_chunk = x
        _, C, T = audio_chunk.shape
        assert T == self.chunk_size, (
            f"Expected input chunk size {self.chunk_size}, but got {T} instead."
        )
        assert C == self.num_channels

        # convert effect id to embedding
        effect_embedding = self.effect_encoder(effect_id)

        x = audio_chunk
        skip_connections = []

        for encoder in self.encoder_blocks:
            x, skip = encoder(x, effect_embedding)
            skip_connections.append(skip)

        for bottleneck in self.bottleneck_blocks:
            x, _ = bottleneck(x, effect_embedding)

        for decoder, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder(x, skip, effect_embedding)

        x = self.final_conv(x)
        assert x.shape == audio_chunk.shape, (
            f"model input shape ({audio_chunk.shape}) != output shape ({x.shape})"
        )
        return x
