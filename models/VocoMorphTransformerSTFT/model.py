import torch
import torch.nn as nn

from .modules.stft import STFT
from .modules.decoder import Decoder
from .modules.encoder import Encoder
from .modules.effect_encoder import EffectEncoder


class VocoMorphUnetSTFT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.chunk_size = config["chunk_size"]
        window = torch.hann_window(self.chunk_size)
        self.register_buffer("window", window.view(1, 1, -1))

        self.overlap = config["overlap"]
        self.stride = self.chunk_size - self.overlap

        embedding_dim = config["embedding_dim"]
        num_channels = config["num_channels"]
        encoder_filters = config["encoder_filters"]
        bottleneck_filters = config["bottleneck_filters"]
        decoder_filters = config["decoder_filters"]
        kernel_size = config["kernel_size"]
        padding = config["padding"]

        assert len(encoder_filters) == len(decoder_filters), (
            "Number of Encoder and Decoder blocks must match"
        )

        self.stft = STFT(config["module_stft"])
        self.effect_encoder = EffectEncoder(config["num_effects"], embedding_dim)

        self.encoder_blocks = nn.ModuleList()
        for i in range(len(encoder_filters)):
            self.encoder_blocks.append(
                Encoder(
                    num_channels if i == 0 else encoder_filters[i - 1],
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

        self.final_conv = nn.Conv2d(decoder_filters[-1], num_channels, kernel_size=1)

    def forward(self, x):
        effect_id, audio = x
        _, _, T = audio.shape

        # convert effect id to embedding
        effect_embedding = self.effect_encoder(effect_id)

        # pre-compute scale & shift once
        output = torch.zeros_like(audio)
        overlap_count = torch.zeros_like(audio)
        for i in range(0, T, self.stride):
            end_idx = min(T, i + self.chunk_size)
            chunk = audio[:, :, i:end_idx]

            # pad if needed
            if chunk.shape[-1] < self.chunk_size:
                pad = self.chunk_size - chunk.shape[-1]
                chunk = nn.functional.pad(chunk, (0, pad))

            chunk_audio = self.forward_one(chunk, effect_embedding)

            # overlap add
            output[:, :, i:end_idx] += chunk_audio[:, :, : end_idx - i]

            window = self.window[..., : end_idx - i]
            output[:, :, i:end_idx] += (
                chunk_audio[:, :, : end_idx - i] * window[..., : end_idx - i]
            )
            overlap_count[:, :, i:end_idx] += window

        output /= overlap_count.clamp(min=1e-6)
        return output

    def forward_one(self, chunk, effect_embedding):
        # convert audio to spectrogram
        chunk_stft = self.stft.stft(chunk)

        # get the magnitude component
        magnitude = torch.abs(chunk_stft)
        # get the phase component
        phase = torch.angle(chunk_stft)

        # pass through unet
        x = magnitude
        # assert that the padded dimensions are indeed divisible
        assert x.shape[-1] * x.shape[-2] % (2 ** len(self.encoder_blocks)) == 0, (
            f"input shape {x.shape} is not divisible by 2^{len(self.encoder_blocks)}"
        )

        skip_connections = []
        for encoder in self.encoder_blocks:
            x, skip = encoder(x, effect_embedding)
            skip_connections.append(skip)

        for bottleneck in self.bottleneck_blocks:
            x, _ = bottleneck(x, effect_embedding)

        for decoder, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder(x, skip, effect_embedding)

        x = self.final_conv(x)

        # reconstruct audio using modulated magnitude & original phase
        real = x * torch.cos(phase)
        imag = x * torch.sin(phase)
        modulated_complex = torch.complex(real, imag)

        # convert spectrogram back into audio
        chunk_audio = self.stft.istft(modulated_complex)
        return chunk_audio
