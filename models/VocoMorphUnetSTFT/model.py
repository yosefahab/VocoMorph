import torch
import torch.nn as nn

from .modules.decoder import Decoder
from .modules.effect_encoder import EffectEncoder
from .modules.encoder import Encoder
from .modules.stft import STFT


class VocoMorphUnetSTFT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.chunk_size = config["chunk_size"]
        self.stft_params = config["module_stft"]
        self.stft = STFT(self.stft_params)

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

        self.final_conv = nn.Conv2d(
            decoder_filters[-1], self.num_channels, kernel_size=1
        )

        # check for divisibility after STFT
        # num_downsampling_steps = len(encoder_filters)
        # T_stft = int(
        #     torch.floor(
        #         (torch.tensor(self.chunk_size) - self.stft_params["n_fft"])
        #         / self.stft_params["hop_length"]
        #     )
        #     + 1
        # )
        # assert T_stft % (2**num_downsampling_steps) == 0, (
        #     f"STFT time dimension ({T_stft}) is not divisible by 2^{num_downsampling_steps}"
        # )

    def forward(self, x):
        effect_id, audio = x

        chunk_stft = self.stft.stft(audio)
        magnitude = torch.abs(chunk_stft)  # Shape: [B, C, F, T_stft]
        phase = torch.angle(chunk_stft)

        effect_embedding = self.effect_encoder(effect_id)

        x = magnitude
        skip_connections = []
        for encoder in self.encoder_blocks:
            x, skip = encoder(x, effect_embedding)
            skip_connections.append(skip)

        for bottleneck in self.bottleneck_blocks:
            x, _ = bottleneck(x, effect_embedding)

        for decoder, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder(x, skip, effect_embedding)

        x = self.final_conv(x)

        assert x.shape == magnitude.shape, (
            f"Model output shape ({x.shape}) != magnitude shape ({magnitude.shape})"
        )

        real = x * torch.cos(phase)
        imag = x * torch.sin(phase)
        modulated_complex = torch.complex(real, imag)

        output_audio = self.stft.istft(modulated_complex)
        assert output_audio.shape == audio.shape
        return output_audio
