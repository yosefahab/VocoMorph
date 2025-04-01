import torch
import torch.nn as nn


class STFTModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_fft = config["n_fft"]
        self.hop_length = config["hop_length"]
        self.win_length = config["win_length"]
        self.window = nn.Parameter(
            torch.hann_window(self.win_length), requires_grad=False
        )

    def stft(self, x):
        """
        Apply STFT to a batched input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T)

        Returns:
            torch.Tensor: STFT output of shape (B, C, F, TT)
        """
        B, C, T = x.shape
        x = x.view(B * C, T)
        stft_output = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        _, F, TT = stft_output.shape
        return stft_output.view(B, C, F, TT)

    def istft(self, x):
        """
        Apply inverse STFT to a batched input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, F, TT)

        Returns:
            torch.Tensor: Reconstructed waveform of shape (B, C, T)
        """
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
        T = istft_output.shape[-1]
        return istft_output.view(B, C, T)
