from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio.transforms as T


@dataclass(slots=True)
class STFT:
    n_fft: int
    win_length: int
    hop_length: int

    stft_fn: Callable = field(init=False)

    def __post_init__(self):
        self.stft_fn = lambda x: torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            window=torch.hann_window(self.win_length).to(x.device),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.view(x.shape[0], -1)
        return self.stft_fn(x)


@dataclass(slots=True)
class STFTLoss:
    n_fft: int
    win_length: int
    hop_length: int
    alpha: float = 1.0  # weight for magnitude loss
    beta: float = 1.0  # weight for phase loss
    stft_fn: STFT = field(init=False)

    def __post_init__(self):
        self.stft_fn = STFT(self.n_fft, self.win_length, self.hop_length)

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor):
        logits_stft = self.stft_fn(logits)
        targets_stft = self.stft_fn(targets)

        mag_loss = F.l1_loss(
            torch.abs(logits_stft), torch.abs(targets_stft), reduction="mean"
        )
        phase_loss = F.l1_loss(
            torch.real(logits_stft), torch.real(targets_stft), reduction="mean"
        ) + F.l1_loss(
            torch.imag(logits_stft), torch.imag(targets_stft), reduction="mean"
        )

        return self.alpha * mag_loss + self.beta * phase_loss


@dataclass(slots=True)
class MultiResolutionSTFTLoss:
    resolutions: List[Tuple[int, int, int]]
    alpha: float = 1.0  # weight for magnitude loss
    beta: float = 1.0  # weight for phase loss

    stft_losses: List[STFTLoss] = field(init=False)

    def __post_init__(self):
        self.stft_losses = [
            STFTLoss(n_fft, win_length, hop_length, self.alpha, self.beta)
            for n_fft, win_length, hop_length in self.resolutions
        ]

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor):
        assert logits.shape == targets.shape
        return sum(stft_loss(logits, targets) for stft_loss in self.stft_losses)


@dataclass(slots=True)
class MelSpecLoss:
    sample_rate: int
    n_mels: int
    n_fft: int
    hop_length: int
    win_length: int

    stft_fn: STFT = field(init=False)
    mel_fn: T.MelScale = field(init=False)

    def __post_init__(self):
        self.stft_fn = STFT(self.n_fft, self.win_length, self.hop_length)
        self.mel_fn = T.MelScale(
            sample_rate=self.sample_rate,
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
        )

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor):
        self.mel_fn.to(logits.device)
        logits_mel = self.mel_fn(torch.abs(self.stft_fn(logits)))
        targets_mel = self.mel_fn(torch.abs(self.stft_fn(targets)))
        return F.l1_loss(logits_mel, targets_mel, reduction="mean")


@dataclass(slots=True)
class VocalModulationLoss:
    alpha: float
    beta: float
    gamma: float
    delta: float

    sample_rate: int
    n_mels: int

    n_fft: int
    hop_length: int
    win_length: int

    stft_fn: Callable = field(init=False)
    mel_fn: T.MelScale = field(init=False)

    def __post_init__(self):
        self.stft_fn = STFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        self.mel_fn = T.MelScale(
            sample_rate=self.sample_rate,
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
        )

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the combined loss for vocal modulation."""
        self.mel_fn.to(logits.device)
        logits_stft = self.stft_fn(logits)
        targets_stft = self.stft_fn(targets)

        mag_loss = F.l1_loss(torch.abs(logits_stft), torch.abs(targets_stft))
        phase_real_loss = F.l1_loss(torch.real(logits_stft), torch.real(targets_stft))
        phase_imag_loss = F.l1_loss(torch.imag(logits_stft), torch.imag(targets_stft))
        time_loss = F.l1_loss(logits, targets)

        # convert to magnitude spectrogram
        logits_mel = self.mel_fn(torch.abs(logits_stft))
        targets_mel = self.mel_fn(torch.abs(targets_stft))
        mel_loss = F.l1_loss(logits_mel, targets_mel)

        return (
            self.alpha * mag_loss
            + self.beta * (phase_real_loss + phase_imag_loss)
            + self.gamma * time_loss
            + self.delta * mel_loss
        )


@dataclass(slots=True)
class SISNRLoss:
    eps: float = 1e-8

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss.
        Args:
        - preds: (B, T)
        - targets: (B, T)
        Returns:
            Negative SI-SNR as a loss (scalar)
        """
        preds = preds - preds.mean(dim=-1, keepdim=True)
        targets = targets - targets.mean(dim=-1, keepdim=True)

        dot = torch.sum(preds * targets, dim=-1, keepdim=True)
        target_energy = torch.sum(targets**2, dim=-1, keepdim=True) + self.eps

        scale = dot / target_energy
        projection = scale * targets

        noise = preds - projection
        ratio = torch.sum(projection**2, dim=-1) / (
            torch.sum(noise**2, dim=-1) + self.eps
        )
        si_snr = 10 * torch.log10(ratio + self.eps)

        return -si_snr.mean()


@dataclass(slots=True)
class EnergyLoss:
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        output_energy = torch.mean(logits**2, dim=-1)
        target_energy = torch.mean(targets**2, dim=-1)
        energy_loss = F.mse_loss(output_energy, target_energy, reduction="mean")
        return energy_loss
