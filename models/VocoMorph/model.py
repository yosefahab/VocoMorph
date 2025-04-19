import torch
import torch.nn as nn

from .modules.stft import STFT
from .modules.film import FiLM
from .modules.effect_encoder import EffectEncoder


class VocoMorph(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.chunk_size = config["chunk_size"]
        self.overlap = config["overlap"]

        self.effect_encoder = EffectEncoder(config["module_effect_encoder"])
        self.stft = STFT(config["module_stft"])
        self.film = FiLM(config["module_film"])

    def forward(self, x):
        effect_id, audio = x
        # audio shape: (B, C, T)
        T = audio.shape[-1]

        embedding = self.effect_encoder(effect_id)

        output = torch.zeros_like(audio)
        stride = self.chunk_size - self.overlap

        for i in range(0, T, stride):
            # (B, C, chunk_size)
            chunk = audio[:, :, i : i + self.chunk_size]
            assert (
                chunk.shape[-1] == self.chunk_size
            ), f"Unexpected chunk size: {chunk.shape[-1]}"

            chunk_stft = self.stft.stft(chunk)
            magnitude = torch.abs(chunk_stft)

            modulated_mag = self.film(magnitude, embedding)

            # reconstruct complex STFT using original phase
            phase = torch.angle(chunk_stft)
            real = modulated_mag * torch.cos(phase)
            imag = modulated_mag * torch.sin(phase)
            modulated_complex = torch.complex(real, imag)

            chunk_audio = self.stft.istft(modulated_complex)
            # overlap add
            end_idx = min(T, i + self.chunk_size)
            output[:, :, i:end_idx] += chunk_audio[:, :, : end_idx - i]
        return output
