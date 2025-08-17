import mlx.core as mx
from typing import Tuple, List
from dataclasses import dataclass, field


def _stft(
    x: mx.array,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> mx.array:
    """
    Reimplements a simplified version of torch.stft using mlx.
    """
    if x.ndim == 3:
        x = mx.reshape(x, (x.shape[0], -1))

    # create a hann window
    window = mx.array(
        [0.5 * (1 - mx.cos(2 * mx.pi * mx.arange(win_length) / (win_length - 1)))],
        dtype=mx.float32,
    )

    # pad the signal
    pad_width = n_fft // 2
    padded_x = mx.pad(x, ((0, 0), (pad_width, pad_width)))

    # frame the signal by creating a strided view
    num_frames = (padded_x.shape[1] - win_length) // hop_length + 1

    # manually calculate strides in bytes
    batch_stride = padded_x.shape[1] * padded_x.itemsize
    frame_stride = hop_length * padded_x.itemsize
    sample_stride = padded_x.itemsize

    frames = mx.as_strided(
        padded_x,
        (x.shape[0], num_frames, win_length),
        (batch_stride, frame_stride, sample_stride),
    )

    # apply window
    frames = frames * window

    # compute fft
    stft_result = mx.fft.fft(frames, axis=-1)

    return stft_result[:, :, : n_fft // 2 + 1]


def _mel_fbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> mx.array:
    """
    Creates a mel-scale filter bank matrix.
    """
    min_mel = 0.0
    max_mel = 2595.0 * mx.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = mx.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = mx.floor((n_fft / sample_rate) * hz_points)

    fbank = mx.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = int(bin_points[m - 1].item())
        f_m = int(bin_points[m].item())
        f_m_plus = int(bin_points[m + 1].item())

        if f_m_minus != f_m:
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (
                    bin_points[m] - bin_points[m - 1]
                )
        if f_m != f_m_plus:
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (
                    bin_points[m + 1] - bin_points[m]
                )

    return fbank


@dataclass(slots=True)
class STFTLoss:
    n_fft: int
    win_length: int
    hop_length: int
    alpha: float = 1.0  # weight for magnitude loss
    beta: float = 1.0  # weight for phase loss

    def __call__(self, logits: mx.array, targets: mx.array):
        logits_stft = _stft(logits, self.n_fft, self.hop_length, self.win_length)
        targets_stft = _stft(targets, self.n_fft, self.hop_length, self.win_length)

        mag_loss = mx.mean(mx.abs(mx.abs(logits_stft) - mx.abs(targets_stft)))
        phase_loss = mx.mean(
            mx.abs(mx.real(logits_stft) - mx.real(targets_stft))
        ) + mx.mean(mx.abs(mx.imag(logits_stft) - mx.imag(targets_stft)))

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

    def __call__(self, logits: mx.array, targets: mx.array):
        assert logits.shape == targets.shape
        total_loss = sum(stft_loss(logits, targets) for stft_loss in self.stft_losses)
        return total_loss


@dataclass(slots=True)
class MelSpecLoss:
    sample_rate: int
    n_mels: int
    n_fft: int
    hop_length: int
    win_length: int

    mel_fbank: mx.array = field(init=False)

    def __post_init__(self):
        self.mel_fbank = _mel_fbank(self.sample_rate, self.n_fft, self.n_mels)

    def __call__(self, logits: mx.array, targets: mx.array):
        logits_stft = _stft(logits, self.n_fft, self.hop_length, self.win_length)
        targets_stft = _stft(targets, self.n_fft, self.hop_length, self.win_length)

        logits_mel = mx.matmul(mx.abs(logits_stft), self.mel_fbank.T)
        targets_mel = mx.matmul(mx.abs(targets_stft), self.mel_fbank.T)

        return mx.mean(mx.abs(logits_mel - targets_mel))


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

    mel_fbank: mx.array = field(init=False)

    def __post_init__(self):
        self.mel_fbank = _mel_fbank(self.sample_rate, self.n_fft, self.n_mels)

    def __call__(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Computes the combined loss for vocal modulation."""
        logits_stft = _stft(logits, self.n_fft, self.hop_length, self.win_length)
        targets_stft = _stft(targets, self.n_fft, self.hop_length, self.win_length)

        mag_loss = mx.mean(mx.abs(mx.abs(logits_stft) - mx.abs(targets_stft)))
        phase_real_loss = mx.mean(mx.abs(mx.real(logits_stft) - mx.real(targets_stft)))
        phase_imag_loss = mx.mean(mx.abs(mx.imag(logits_stft) - mx.imag(targets_stft)))
        time_loss = mx.mean(mx.abs(logits - targets))

        # convert to magnitude spectrogram
        logits_mel = mx.matmul(mx.abs(logits_stft), self.mel_fbank.T)
        targets_mel = mx.matmul(mx.abs(targets_stft), self.mel_fbank.T)
        mel_loss = mx.mean(mx.abs(logits_mel - targets_mel))

        return (
            self.alpha * mag_loss
            + self.beta * (phase_real_loss + phase_imag_loss)
            + self.gamma * time_loss
            + self.delta * mel_loss
        )


@dataclass(slots=True)
class SISNRLoss:
    eps: float = 1e-8

    def __call__(self, preds: mx.array, targets: mx.array) -> mx.array:
        """
        Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss.
        """
        preds = preds - mx.mean(preds, axis=-1, keepdims=True)
        targets = targets - mx.mean(targets, axis=-1, keepdims=True)

        dot = mx.sum(preds * targets, axis=-1, keepdims=True)
        target_energy = mx.sum(targets**2, axis=-1, keepdims=True) + self.eps

        scale = dot / target_energy
        projection = scale * targets

        noise = preds - projection
        ratio = mx.sum(projection**2, axis=-1) / (mx.sum(noise**2, axis=-1) + self.eps)
        # The key change: We ensure the ratio is non-negative before taking the logarithm.
        si_snr = 10 * mx.log10(mx.maximum(ratio, self.eps))

        return -mx.mean(si_snr)


@dataclass(slots=True)
class EnergyLoss:
    def __call__(self, logits: mx.array, targets: mx.array) -> mx.array:
        output_energy = mx.mean(logits**2, axis=-1)
        target_energy = mx.mean(targets**2, axis=-1)
        energy_loss = mx.mean(mx.abs(output_energy - target_energy))
        return energy_loss
