import torch
from src.logging.logger import get_logger
from src.modulation.utils import plot_waves, plot_tensors
from src.utils import load_audio, save_audio
from src.modulation.effects import *
from src.modulation.filters import *
from src.modulation.synthesis import *
from src.modulation.transformations import *
from src.dataset.generate_dataset import apply_effects

logger = get_logger(__name__)


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


def compare_audios(config: dict):
    logger.info("Running comparison test")
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
    audios = [load_audio(w, sr=sr, channels=channels)[0] for w in waves]
    specs = [stft(a, n_fft, hop_length, win_length, window) for a in audios]
    plot_waves([(a, sr) for a in audios])
    plot_tensors(specs)


def main(config: dict):
    logger.info(f"Running tests from {__name__}")
    compare_audios(config)


def test_apply_effects(config: dict):
    sr = config["sample_rate"]
    channels = config["channels"]

    w = "data/TIMIT/TEST/DR1/FAKS0/SA2.WAV"
    audio, sr = load_audio(w, sr=sr, channels=channels)

    for i, a in enumerate(apply_effects(audio, sr, config["effects"])):
        save_audio(a, sr, f"audio_e{i}")
