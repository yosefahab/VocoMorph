import torch
import torch.nn as nn

from .modules.encoder import Encoder
from .modules.decoder import Decoder
from .modules.bottleneck import Bottleneck
from .modules.stft import STFTModule


class VocoMorphUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stft_module = STFTModule(config["module_stft"])
        self.effect_embedding = nn.Embedding(
            len(config["effects"]), config["effect_dim"]
        )
        self.encoder = Encoder(config["module_encoder"], config["effect_dim"])
        self.bottleneck = Bottleneck(config["module_bottleneck"])
        self.decoder = Decoder(config["module_decoder"], config["effect_dim"])
        self.final_conv = nn.Conv2d(
            config["module_decoder"]["out_channels"],
            config["module_decoder"]["out_channels"],
            config["module_final_conv"]["kernel_size"],
        )

    def forward(self, x):
        """Processes raw waveform and returns modified waveform."""

        effect_id, x = x
        print(x.shape)  # torch.Size([1, 2, 160_000])
        # convert ID to embedding vector
        effect_embed = self.effect_embedding(effect_id.long())
        print(effect_embed.shape)  # torch.Size([1, 1, 16])

        # convert to spectrogram
        spec = self.stft_module.stft(x)

        # split magnitude & phase
        mag, phase = spec.abs(), torch.atan2(spec.imag, spec.real)

        # repeat effect_embed along the second dimension
        effect_embed = effect_embed.repeat(1, mag.shape[1], 1)
        print(effect_embed.shape)  # torch.Size([1, 2, 16])
        # Reshape effect_embed to match the last two dimensions of mag
        effect_embed = effect_embed.unsqueeze(-1).unsqueeze(-1)  # Add two singleton dimensions, making it [1, 2, 16, 1, 1]
        print(effect_embed.shape)  # torch.Size([1, 2, 16])
        effect_embed = effect_embed.repeat(1, 1, 1, mag.shape[2], mag.shape[3])  # Repeat to match [1, 2, 16, 513, 626]
        print(effect_embed.shape)  # torch.Size([1, 2, 16])

        # ensure mag and phase have the same shape
        assert (
            mag.shape == phase.shape
        ), "Magnitude and phase tensors must have the same shape"

        # U-Net with effect conditioning
        print(mag.shape)  # torch.Size([1, 2, 513, 626])
        mag = self.encoder(mag, effect_embed)  # crash here
        print(mag.shape)
        mag = self.bottleneck(mag)
        print(mag.shape)
        mag = self.decoder(mag, effect_embed)
        print(mag.shape)
        mag = self.final_conv(mag)
        print(mag.shape)

        # ensure mag and phase are synchronized in shape
        assert mag.shape == phase.shape, "Shapes must match before recombination"

        # recombine magnitude & phase
        spec_out = torch.polar(mag, phase)

        # convert back to waveform
        return self.stft_module.istft(spec_out)
