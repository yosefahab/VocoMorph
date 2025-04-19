import torch
import torch.nn as nn

from .modules.stft import STFT
from .modules.rnn_film import RNNFiLM
from .modules.effect_encoder import EffectEncoder


class VocoMorph(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.chunk_size = config["chunk_size"]
        self.overlap = config["overlap"]

        self.film = RNNFiLM(config["module_rnn_film"])
        self.stft = STFT(config["module_stft"])
        self.effect_encoder = EffectEncoder(config["module_effect_encoder"])

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

            # pad chunk
            if chunk.shape[-1] < self.chunk_size:
                pad_len = self.chunk_size - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

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
            output[:, :, i : i + self.chunk_size] += chunk_audio[
                :, :, : min(chunk_audio.shape[-1], output.shape[-1] - i)
            ]
