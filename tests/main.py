import math

import torch

from src.dataset.generate_dataset import apply_effects
from src.modulation.effects import *
from src.modulation.filters import *
from src.modulation.synthesis import *
from src.modulation.transformations import *
from src.modulation.utils import plot_tensors, plot_waves
from src.trainer.custom.criterions import *
from src.utils.audio import load_audio, save_audio
from src.utils.logger import get_logger

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
    test_loss_functions(config)


def test_apply_effects(config: dict):
    sr = config["sample_rate"]
    channels = config["channels"]

    w = "data/TIMIT/TEST/DR1/FAKS0/SA2.WAV"
    audio, sr = load_audio(w, sr=sr, channels=channels)

    for i, a in enumerate(apply_effects(audio, sr, config["effects"])):
        save_audio(a, sr, f"audio_e{i}")


def test_loss_functions(config):
    logger.info("Running loss functions test")
    # Define common parameters based on your config
    batch_size = 4
    sample_rate = config["sample_rate"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    win_length = config["win_length"]
    n_mels = config["n_mels"]
    audio_length = config["frame_length"]  # Corresponds to your chunk_size
    num_channels = config["channels"]  # Mono audio

    print("Loss Function Test Suite")
    print(
        f"Audio Length: {audio_length}, Batch Size: {batch_size}, Channels: {num_channels}\n"
    )

    # Initialize Loss Functions
    stft_loss_fn = STFTLoss(n_fft, win_length, hop_length)
    multi_res_stft_loss_fn = MultiResolutionSTFTLoss(
        resolutions=[[1024, 1024, 256], [2048, 2048, 512], [512, 512, 128]],
        alpha=1.0,
        beta=0.5,
    )
    mel_spec_loss_fn = MelSpecLoss(sample_rate, n_mels, n_fft, hop_length, win_length)
    vocal_modulation_loss_fn = VocalModulationLoss(
        alpha=0.5,
        beta=0.2,
        gamma=0.1,
        delta=1.0,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    si_snr_loss_fn = SISNRLoss()
    energy_loss_fn = EnergyLoss()

    loss_fns = {
        "STFTLoss": stft_loss_fn,
        "MultiResolutionSTFTLoss": multi_res_stft_loss_fn,
        "MelSpecLoss": mel_spec_loss_fn,
        "VocalModulationLoss": vocal_modulation_loss_fn,
        "SISNRLoss": si_snr_loss_fn,
        "EnergyLoss": energy_loss_fn,
    }

    # Create Synthetic Data
    # Target: A sine wave, representing a clean audio signal
    t = torch.linspace(0, (audio_length - 1) / sample_rate, audio_length, device="cpu")
    targets = 0.5 * torch.sin(2 * math.pi * 440 * t).unsqueeze(0).repeat(
        batch_size, num_channels, 1
    )

    print(
        f"Target signal min/max: {targets.min():.4f}/{targets.max():.4f}, mean: {targets.mean():.4f}, std: {targets.std():.4f}\n"
    )

    # Test Scenarios

    # Scenario 1: Perfect Match
    print("Scenario 1: Perfect Match (logits == targets)")
    logits_perfect = targets.clone()
    for name, loss_fn in loss_fns.items():
        loss = loss_fn(logits_perfect, targets)
        print(f"  {name}: {loss.item():.6f}")
    print("-" * 50 + "\n")

    # Scenario 2: Zero Output (Simulating "cheating")
    print("Scenario 2: Zero Output (logits are all zeros)")
    logits_zero = torch.zeros_like(targets)
    print(
        f"  Logits min/max: {logits_zero.min():.4f}/{logits_zero.max():.4f}, mean: {logits_zero.mean():.4f}, std: {logits_zero.std():.4f}"
    )
    for name, loss_fn in loss_fns.items():
        loss = loss_fn(logits_zero, targets)
        print(f"  {name}: {loss.item():.6f}")
    print(
        "\n  Observation: L1/L2 losses are low, SI-SNR is high (bad) as it tries to maximize match.\n  EnergyLoss shows significant difference.\n"
    )
    print("-" * 50 + "\n")

    # Scenario 3: Low Amplitude Output (Simulating "cheating" with quiet waves)
    print("Scenario 3: Low Amplitude Output (logits = 0.01 * targets)")
    logits_low_amp = 0.01 * targets
    print(
        f"  Logits min/max: {logits_low_amp.min():.4f}/{logits_low_amp.max():.4f}, mean: {logits_low_amp.mean():.4f}, std: {logits_low_amp.std():.4f}"
    )
    for name, loss_fn in loss_fns.items():
        loss = loss_fn(logits_low_amp, targets)
        print(f"  {name}: {loss.item():.6f}")
    print(
        "\n  Observation: L1/L2 losses are much lower than random, SI-SNR is very low (good) due to scale-invariance. \n  EnergyLoss shows difference unless output is scaled appropriately.\n"
    )
    print("-" * 50 + "\n")

    # Scenario 4: Random Noise Output
    print("Scenario 4: Random Noise Output")
    logits_noise = torch.randn_like(targets) * 0.1  # Small random noise
    print(
        f"  Logits min/max: {logits_noise.min():.4f}/{logits_noise.max():.4f}, mean: {logits_noise.mean():.4f}, std: {logits_noise.std():.4f}"
    )
    for name, loss_fn in loss_fns.items():
        loss = loss_fn(logits_noise, targets)
        print(f"  {name}: {loss.item():.6f}")
    print(
        "\n  Observation: All losses are high as noise is uncorrelated and has different energy/spectrum.\n"
    )
    print("-" * 50 + "\n")

    # Scenario 5: SI-SNR perfect match (scaled version)
    print("Scenario 5: SI-SNR Perfect Match (logits = 0.5 * targets)")
    logits_scaled = 0.5 * targets
    print(
        f"  Logits min/max: {logits_scaled.min():.4f}/{logits_scaled.max():.4f}, mean: {logits_scaled.mean():.4f}, std: {logits_scaled.std():.4f}"
    )
    for name, loss_fn in loss_fns.items():
        loss = loss_fn(logits_scaled, targets)
        print(f"  {name}: {loss.item():.6f}")
    print(
        "\n  Observation: SI-SNR is very low (good) as it's scale invariant. Other losses are non-zero.\n"
    )
    print("-" * 50 + "\n")
