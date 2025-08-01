from pathlib import Path
import torch

from src.dataset.modulation.utils import plot_tensors, plot_waves
from src.utils.audio import load_audio


def stft(x, n_fft, hop_length, win_length, window):
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)

    stft_output = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    _, F, TT = stft_output.shape
    return stft_output.view(1, F, TT)


def compare_audios_stft(config: dict):
    sr = config["sample_rate"]
    channels = config["channels"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    win_length = config["win_length"]
    window = torch.hann_window(win_length)

    waves = [
        "data/timit/TEST/DR4/MGMM0/SA2.WAV",
        "data/timit/modulated/00001_e0.wav",
        "data/timit/modulated/00001_e1.wav",
        "data/timit/modulated/00001_e2.wav",
        "data/timit/modulated/00001_e3.wav",
        "data/timit/modulated/00001_e4.wav",
    ]
    audios = [load_audio(Path(w), sr=sr, channels=channels)[0] for w in waves]
    specs = [stft(a, n_fft, hop_length, win_length, window) for a in audios]
    plot_waves([(a, sr) for a in audios])
    plot_tensors(specs)
