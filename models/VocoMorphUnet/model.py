import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.stft import STFT
from .modules.decoder import Decoder
from .modules.encoder import Encoder
from .modules.effect_encoder import EffectEncoder


class VocoMorphUnet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.chunk_size = config["chunk_size"]
        self.overlap = config["overlap"]
        self.stride = self.chunk_size - self.overlap

        embedding_dim = config["embedding_dim"]
        num_channels = config["num_channels"]
        encoder_filters = config["encoder_filters"]
        bottleneck_filters = config["bottleneck_filters"]
        decoder_filters = config["decoder_filters"]
        kernel_size = config["kernel_size"]
        padding = config["padding"]

        assert len(encoder_filters) == len(
            decoder_filters
        ), f"Number of Encoder and Decoder blocks must match"

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

        self.num_downsampling_steps = len(encoder_filters)

    def forward(self, x):
        effect_id, audio = x
        _, _, T = audio.shape

        # convert effect id to embedding
        effect_embedding = self.effect_encoder(effect_id)

        # pre-compute scale & shift once
        output = torch.zeros_like(audio)
        for i in range(0, T, self.stride):
            end_idx = min(T, i + self.chunk_size)
            chunk = audio[:, :, i:end_idx]

            # pad if needed
            if chunk.shape[-1] < self.chunk_size:
                pad = self.chunk_size - chunk.shape[-1]
                chunk = nn.functional.pad(chunk, (0, pad))

            # convert audio to spectrogram
            chunk_stft = self.stft.stft(chunk)

            # get the magnitude component
            magnitude = torch.abs(chunk_stft)
            # get the phase component
            phase = torch.angle(chunk_stft)

            # pass through unet
            magnitude = self.forward_one(magnitude, effect_embedding)

            # reconstruct audio using modulated magnitude & original phase
            real = magnitude * torch.cos(phase)
            imag = magnitude * torch.sin(phase)
            modulated_complex = torch.complex(real, imag)

            # convert spectrogram back into audio
            chunk_audio = self.stft.istft(modulated_complex)

            # overlap add
            output[:, :, i:end_idx] += chunk_audio[:, :, : end_idx - i]
        return output

    def pad_spectrogram_for_unet(self, spectrogram):
        """
        Pads the spectrogram to ensure its height and width are divisible by 2^N,
        where N is the number of downsampling steps in the U-Net.
        Returns the padded spectrogram and the original height/width for cropping later.
        """
        original_h, original_w = spectrogram.shape[-2:]
        divisible_by = 2**self.num_downsampling_steps

        # calculate target dimensions
        target_h = ((original_h + divisible_by - 1) // divisible_by) * divisible_by
        target_w = ((original_w + divisible_by - 1) // divisible_by) * divisible_by

        # calculate padding amounts for height and width
        pad_h = target_h - original_h
        pad_w = target_w - original_w

        # pad on the right and bottom sides. 'reflect' mode is often good for image-like data.
        # (pad_left, pad_right, pad_top, pad_bottom)
        padded_spectrogram = F.pad(spectrogram, (0, pad_w, 0, pad_h), mode="reflect")

        return padded_spectrogram, original_h, original_w

    def forward_one(self, spectrogram, effect_embedding):
        """
        Performs a single forward pass through the U-Net for one spectrogram chunk.
        Handles padding the input spectrogram and cropping the output spectrogram.
        """
        # store the original spectrogram shape for later cropping
        original_spectrogram_shape = spectrogram.shape

        # pad the spectrogram to make its dimensions divisible by 2^N
        x, original_h, original_w = self.pad_spectrogram_for_unet(spectrogram)

        # assert that the padded dimensions are indeed divisible
        assert (
            x.shape[-2] % (2**self.num_downsampling_steps) == 0
        ), f"Padded height {x.shape[-2]} is not divisible by 2^{self.num_downsampling_steps}"
        assert (
            x.shape[-1] % (2**self.num_downsampling_steps) == 0
        ), f"Padded width {x.shape[-1]} is not divisible by 2^{self.num_downsampling_steps}"

        skip_connections = []

        for i, block in enumerate(self.encoder_blocks):
            x_post_pool, x_pre_pool = block(x, effect_embedding)
            # store the pre-pooled output for skip connection
            skip_connections.append(x_pre_pool)
            x = x_post_pool

        for i, block in enumerate(self.bottleneck_blocks):
            # unpack, but discard the 'pre-pool' from bottleneck
            x, _ = block(x, effect_embedding)

        # iterate through decoder blocks, using skip connections in reverse order
        for i, block in enumerate(self.decoder_blocks):
            # skip_connections[-1 - i] correctly gets the corresponding skip connection
            x = block(x, skip_connections[-1 - i], effect_embedding)

        x = self.final_conv(x)

        # crop the output back to the original spectrogram dimensions
        x = x[:, :, :original_h, :original_w]

        assert (
            x.shape == original_spectrogram_shape
        ), f"Output shape {x.shape} does not match original input spectrogram shape {original_spectrogram_shape}"
        return x
