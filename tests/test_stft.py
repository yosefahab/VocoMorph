from pathlib import Path

import numpy as np
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

    model_dir = Path(os.environ["DATA_ROOT"])

    eid = 3

    model_output_path = model_dir.joinpath(f"output/SI455/SI455_{eid}.wav")
    model_output = load_audio(model_output_path, sr=sr, channels=channels)[0]

    test_waves = np.load(model_dir.joinpath("timit/augmented/test/0302.npz"))
    raw_wave = test_waves["raw"]
    effect_wave = test_waves[f"wave_{eid}"]

    audios = [raw_wave, effect_wave, model_output]
    specs = [stft(a, n_fft, hop_length, win_length, window) for a in audios]
    plot_waves([(a, sr) for a in audios])
    plot_tensors(specs)


if __name__ == "__main__":
    import os

    from src.utils.parsers import parse_yaml

    model_dir = Path(os.environ["PROJECT_ROOT"]).joinpath("models", "VocoMorphUnet")
    assert model_dir.exists(), f"{model_dir} does not exist"
    yaml_path = model_dir.joinpath("config.yaml")
    yaml_dict = parse_yaml(yaml_path)
    config = yaml_dict["config"]
    compare_audios_stft(config["data"])
