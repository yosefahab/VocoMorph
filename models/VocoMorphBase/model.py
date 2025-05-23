import torch
import torch.nn as nn

from .modules.stft import STFT
from .modules.film import FiLM
from .modules.magnitude_encoder import SubNet
from .modules.effect_encoder import EffectEncoder
from .modules.magnitude_encoder import SubNet


class VocoMorph(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.chunk_size = config["chunk_size"]
        self.overlap = config["overlap"]
        self.stride = self.chunk_size - self.overlap

        self.effect_encoder = EffectEncoder(config["module_effect_encoder"])
        self.stft = STFT(config["module_stft"])
        self.residual_blocks = nn.ModuleList(
            [SubNet(config["module_subnet"]) for _ in range(config["num_blocks"])]
        )

        self.film = FiLM(config["module_film"])
        self.activation = nn.ReLU()

    def forward(self, x):
        effect_id, audio = x
        B, C, T = audio.shape

        # convert effect id to embedding
        embedding = self.effect_encoder(effect_id)

        # pre-compute scale & shift once
        scale, shift = self.film(embedding)
        # repeat across F, TT
        scale = scale.view(B, C, -1, 1)
        shift = shift.view(B, C, -1, 1)

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

            # residual blocks
            for rb in self.residual_blocks:
                residual = magnitude
                magnitude = rb(magnitude)
                # apply film
                magnitude = magnitude * scale + shift
                # activate with residual
                magnitude = self.activation(magnitude + residual)

            # reconstruct audio using modulated magnitude & original phase
            real = magnitude * torch.cos(phase)
            imag = magnitude * torch.sin(phase)
            modulated_complex = torch.complex(real, imag)

            # convert spectrogram back into audio
            chunk_audio = self.stft.istft(modulated_complex)

            # overlap add
            output[:, :, i:end_idx] += chunk_audio[:, :, : end_idx - i]
        return output
