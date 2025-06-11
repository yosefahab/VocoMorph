import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_fft = config["n_fft"]
        self.hop_length = config["hop_length"]
        self.win_length = config["win_length"]
        self.output_length = config["output_length"]
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft + 2)[1:-1],
        )

    def stft(self, x):
        B, C, T = x.shape
        x = x.view(B * C, T)
        stft_output = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
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
            length=self.output_length,
            window=self.window,
            center=False,
            return_complex=False,
        )
        assert self.output_length == istft_output.shape[-1], (
            f"Shape mismatch, expected output wave of length {self.output_length} but got {istft_output.shape[-1]}"
        )
        return istft_output.view(B, C, self.output_length)
