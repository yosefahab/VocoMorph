"""functions to synthesize audio"""

import numpy as np
from scipy import signal

from .common import apply_distortion


def generate_carrier(
    shape, sr: int, wave_type: str = "square", freq: float = 100.0
) -> np.ndarray:
    """
    generates a carrier wave for vocoding
    Args:
    - wave_type: ["sawtooth", "square", "noise", "sine"]
    """
    C, T = shape
    t = np.arange(T) / sr
    if wave_type == "sawtooth":
        noise = signal.sawtooth(2 * np.pi * freq * t)
    elif wave_type == "square":
        noise = signal.square(2 * np.pi * freq * t)
    elif wave_type == "noise":
        noise = np.random.uniform(-1, 1, shape)
    else:
        noise = np.sin(2 * np.pi * freq * t)  # default to sine wave

    return np.stack([noise] * C)


def generate_kick(
    sr: int,
    num_samples: int,
    base_freq: float = 40,
    pitch_drop: float = 0.2,
    distortion: float = 0.8,
    lfo_rate: float = 3.0,
    lfo_depth: float = 0.3,
):
    """
    Generates a synthetic kick drum with a growling effect for sidechain.
    The duration of the kick is determined by the number of samples.
    """
    t = np.linspace(0, num_samples / sr, num_samples, endpoint=False)

    # pitch sweep for the "thump"
    freq = base_freq * (1 - pitch_drop * t)
    kick_wave = np.sin(2 * np.pi * freq * t)

    # amplitude envelope (punchy decay)
    envelope = np.exp(-5 * t)
    kick_wave *= envelope

    # LFO modulation for growl
    lfo = np.sin(2 * np.pi * lfo_rate * t) * lfo_depth
    kick_wave *= 1 + lfo

    # distortion
    kick_wave = apply_distortion(kick_wave, (1 + distortion * 10))

    # normalize
    kick_wave /= np.max(np.abs(kick_wave))

    return kick_wave


def generate_noise(
    shape,
    noise_level: float = 0.02,
    noise_type: str = "white",
    dtype=np.float32,
) -> np.ndarray:
    """
    Generates noise to add to an audio signal.
    Args:
    - shape: returned array shape
    - noise_level: The intensity of the noise (scaled between 0 and 1).
    - noise_type: Type of noise ("white", "pink", "brown").
    Returns:
        np.ndarray: Noise signal of the same shape as the input audio.
    """
    assert noise_type in ["white", "pink", "brown"]

    if noise_type == "white":
        noise = np.random.normal(0, 1, shape)

    elif noise_type == "pink":
        white = np.random.normal(0, 1, shape)
        b = np.array(
            [0.02109238, 0.07113478, 0.68873558]
        )  # filter coefficients for pink noise
        a = np.array([1, -1.3199866, 0.49603215])
        pink = signal.lfilter(b, a, white)
        noise = pink / np.max(np.abs(pink))

    else:
        brown = np.cumsum(np.random.normal(0, 1, shape))
        noise = brown / np.max(np.abs(brown))  # normalize to keep levels consistent

    return (noise * noise_level).astype(dtype)
