import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_fft = config["n_fft"]
        self.hop_length = config["hop_length"]
        self.win_length = config["win_length"]
        self.output_length = config["output_length"]
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def stft(self, x):
        B, C, T = x.shape
        x = x.view(B * C, T)
        stft_output = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,  # pyright: ignore[reportArgumentType]
            return_complex=True,
        )
        _, F, TT = stft_output.shape
        return stft_output.view(B, C, F, TT)

    def istft(self, x):
        B, C, F, TT = x.shape
        x = x.view(B * C, F, TT)
        istft_output = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False,
        )
        return istft_output.view(B, C, -1)
