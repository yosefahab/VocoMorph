import torch
import torch.nn as nn

from .modules.decoder import Decoder
from .modules.encoder import Encoder
from .modules.effect_encoder import EffectEncoder


class VocoMorphUnet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.chunk_size = config["chunk_size"]
        window = torch.hann_window(self.chunk_size)
        self.register_buffer("window", window.view(1, 1, -1))

        self.overlap = config["overlap"]
        self.stride = self.chunk_size - self.overlap

        embedding_dim = config["embedding_dim"]
        self.num_channels = config["num_channels"]
        encoder_filters = config["encoder_filters"]
        bottleneck_filters = config["bottleneck_filters"]
        decoder_filters = config["decoder_filters"]
        kernel_size = config["kernel_size"]
        padding = config["padding"]

        assert len(encoder_filters) == len(decoder_filters), (
            "Number of Encoder and Decoder blocks must match"
        )

        self.effect_encoder = EffectEncoder(config["num_effects"], embedding_dim)

        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_filters)):
            self.encoder_blocks.append(
                Encoder(
                    self.num_channels if i == 0 else encoder_filters[i - 1],
                    encoder_filters[i],
                    kernel_size,
                    padding,
                    embedding_dim,
                    apply_pool=True,
                )
            )

        self.bottleneck_blocks = nn.ModuleList()
        for i in range(len(bottleneck_filters)):
            self.bottleneck_blocks.append(
                Encoder(
                    (encoder_filters[-1] if i == 0 else bottleneck_filters[i - 1]),
                    bottleneck_filters[i],
                    kernel_size,
                    padding,
                    embedding_dim,
                    apply_pool=False,
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
        effect_id, audio = x
        _, C, T = audio.shape
        assert C == self.num_channels

        # convert effect id to embedding
        effect_embedding = self.effect_encoder(effect_id)

        # pre-compute scale & shift once
        output = torch.zeros_like(audio)
        overlap_count = torch.zeros_like(audio)
        for i in range(0, T, self.stride):
            end_idx = min(T, i + self.chunk_size)
            chunk = audio[:, :, i:end_idx]
            assert chunk.shape[-1] % (2 ** len(self.encoder_blocks)) == 0, (
                f"chunk length ({chunk.shape[-1]}) is not divisible by 2^n_levels ({len(self.encoder_blocks)})"
            )

            chunk_audio = self.forward_one(chunk, effect_embedding)

            # overlap add
            window = self.window[..., : end_idx - i]
            output[:, :, i:end_idx] += (
                chunk_audio[:, :, : end_idx - i] * window[..., : end_idx - i]
            )
            overlap_count[:, :, i:end_idx] += window

        output /= overlap_count.clamp(min=1e-6)
        return output

    def forward_one(self, chunk, effect_embedding):
        T = chunk.shape[-1]
        if T < self.chunk_size:
            pad = self.chunk_size - T
            chunk = nn.functional.pad(chunk, (0, pad))

        x = chunk
        skip_connections = []

        for encoder in self.encoder_blocks:
            x, skip = encoder(x, effect_embedding)
            skip_connections.append(skip)

        for bottleneck in self.bottleneck_blocks:
            x, _ = bottleneck(x, effect_embedding)

        for decoder, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder(x, skip, effect_embedding)

        x = self.final_conv(x)
        assert x.shape == chunk.shape, (
            f"model input shape ({chunk.shape}) != output shape ({x.shape})"
        )
        return x
