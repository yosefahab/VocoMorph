"""functions to alter audio perception"""

import numpy as np

from .common import apply_distortion
from .filters import apply_bandpass, apply_lowpass
from .synthesis import generate_carrier, generate_noise
from .transformations import (
    apply_compression,
    normalize_audio,
    apply_pitch_shift,
    identity_transform,
)


def apply_bit_crush(audio: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """
    Compresses/reduces fidelity of audio
    """
    quantization_levels = 2**bit_depth
    return np.round(audio * quantization_levels) / quantization_levels


def apply_chorus(
    audio: np.ndarray,
    sr: int,
    rate: float = 4.0,
    depth: float = 0.003,
    delay: float = 0.03,
    voices: int = 3,
    mix: float = 0.5,
) -> np.ndarray:
    """
    applies a chorus effect to each channel independently
    """
    C, T = audio.shape
    t = np.arange(T) / sr
    chorus_signal = np.zeros_like(audio)

    for i in range(voices):
        phase_shift = (i / voices) * 2 * np.pi
        mod = np.sin(2 * np.pi * rate * t + phase_shift) * depth
        delay_samples = ((delay + mod) * sr).astype(int)

        for c in range(C):
            delayed_audio = np.interp(
                np.arange(T) - delay_samples,
                np.arange(T),
                audio[c],
                left=0,
                right=0,
            )
            chorus_signal[c] += delayed_audio / voices

    return (1 - mix) * audio + mix * chorus_signal


def apply_reverb(audio: np.ndarray, sr: int, delay=70, decay=0.3) -> np.ndarray:
    """
    apply reverb effect to the audio by applying a simple convolution reverb filter
    """
    C, T = audio.shape
    delay_samples = int((delay / 1_000) * sr)
    impulse = np.zeros(delay_samples + 1)
    impulse[0] = 1
    impulse[delay_samples] = decay

    output = np.zeros_like(audio)
    for c in range(C):
        y_reverb = np.convolve(audio[c], impulse, mode="full")[:T]
        output[c] = normalize_audio(audio[c], y_reverb)

    return output


def apply_echo(
    audio: np.ndarray,
    sr: int,
    delay: int = 500,
    decay: float = 0.15,
    num_echoes: int = 3,
) -> np.ndarray:
    """
    Apply an echo effect with progressively fading echoes.

    Args:
    - audio: Input audio signal (1D NumPy array)
    - sr: Sample rate of the audio
    - delay: Time (in milliseconds) between echoes
    - decay: Echo decay factor (0.0 to 1.0, lower values mean faster fade)
    - num_echoes: Number of echoes to apply
    Returns:
        Audio with echo effect
    """
    delay_samples = int((delay / 1_000) * sr)  # convert delay to sample count
    output = np.copy(audio)  # start with the original audio

    # add multiple fading echoes
    for n in range(1, num_echoes + 1):
        echo_offset = n * delay_samples  # compute delay for this echo
        if echo_offset < len(audio):
            output[echo_offset:] += audio[:-echo_offset] * (
                decay**n
            )  # decay factor applied progressively

    output = normalize_audio(audio, output)
    return output


def apply_tremolo(
    audio: np.ndarray, sr: int, rate: float = 30, depth: float = 0.4
) -> np.ndarray:
    """
    applies a tremolo (amplitude modulation) effect to the audio signal
    """
    _, T = audio.shape
    t = np.linspace(0, T / sr, num=T)
    tremolo = 1 + depth * np.sin(2 * np.pi * rate * t)
    return (audio * tremolo).astype(audio.dtype)


def apply_vocoder(
    audio: np.ndarray,
    carrier: np.ndarray,
    sr: int,
    num_bands: int = 8,
    min_freq: float = 100,
    max_freq: float = 4000,
) -> np.ndarray:
    """
    Applies a basic vocoder effect by modulating a carrier wave with the input audio's frequency envelopes.
    """
    assert audio.shape == carrier.shape, (
        "Audio and carrier must have the same shape (C, T)"
    )
    band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands + 1)
    output = np.zeros_like(audio, dtype=np.float32)

    for i in range(num_bands):
        low, high = band_edges[i], band_edges[i + 1]
        filtered_audio = apply_bandpass(audio, sr, low, high, order=2)
        envelope = np.abs(filtered_audio)
        envelope = apply_lowpass(envelope, sr, cutoff=10, order=1)
        carrier_band = apply_bandpass(carrier, sr, low, high, order=2)
        output += envelope * carrier_band

    return output


def apply_sidechain(
    audio: np.ndarray, kick: np.ndarray, reduction: float = 0.5
) -> np.ndarray:
    """
    Applies sidechain compression based on kick drum
    """
    env = np.convolve(np.abs(kick), np.ones(100) / 100, mode="same")
    gain = 1.0 - (env / np.max(env)) * reduction
    return audio * gain


def apply_modulation(
    audio: np.ndarray,
    sr: int,
    mod_freq: float = 19.0,
    mod_depth: float = 0.2,
    mod_type: str = "am",
) -> np.ndarray:
    """
    applies amplitude or ring modulation to an audio signal
    """
    C, T = audio.shape
    t = np.arange(T) / sr
    sin_wave = np.sin(2 * np.pi * mod_freq * t).astype(audio.dtype)
    modulator = 1.0 + mod_depth * sin_wave if mod_type == "am" else sin_wave
    modulated_audio = audio * modulator

    if mod_type == "am":
        for c in range(C):
            modulated_audio[c] = normalize_audio(audio[c], modulated_audio[c])

    return modulated_audio


def apply_direction_effect(
    audio: np.ndarray, direction: str = "center", speed: float = 5.0
) -> np.ndarray:
    """
    applies directional effects to a stereo audio signal
    """
    assert audio.ndim == 2, "Expected stereo audio, found mono"

    _, T = audio.shape
    left, right = audio[0], audio[1]

    if direction == "left":
        right *= 0.2
    elif direction == "right":
        left *= 0.2
    elif direction == "pingpong":
        t = np.linspace(0, np.pi * speed, T)
        pan_curve = (np.sin(t) + 1) / 2
        left *= 1 - pan_curve
        right *= pan_curve

    left = normalize_audio(audio[0], left)
    right = normalize_audio(audio[1], right)
    return np.vstack((left, right))


def apply_radio_effect(audio: np.ndarray, sr: int) -> np.ndarray:
    # imitate AM radio frequency response
    mod_audio = apply_bandpass(audio, sr, low=400.0, high=2500.0)

    # simulate low bit-depth compression (reducing fidelity)
    mod_audio = apply_bit_crush(mod_audio, bit_depth=6)

    # mimick analog signal degradation
    mod_audio = apply_distortion(mod_audio, 10.0)

    # add amplitude modulation (signal wobble)
    mod_audio = apply_modulation(
        mod_audio, sr, mod_freq=50.0, mod_depth=0.1, mod_type="am"
    )

    # add random white noise (radio static)
    noise = generate_noise(mod_audio.shape, 0.009, "white", mod_audio.dtype)
    mod_audio = mod_audio + noise

    audio = normalize_audio(audio, mod_audio)

    return audio


def apply_scifi_effect(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = apply_pitch_shift(audio, sr, -1)
    audio = apply_modulation(audio, sr, mod_freq=80, mod_type="am")
    audio = apply_bit_crush(audio, bit_depth=16)
    audio = apply_chorus(audio, sr, delay=0.005)
    audio = apply_distortion(audio, gain=2)
    audio = apply_tremolo(audio, sr, rate=30)
    return audio


def apply_robotic_effect(audio: np.ndarray, sr: int) -> np.ndarray:
    carrier = generate_carrier(audio.shape, sr, wave_type="square", freq=150)
    audio = apply_vocoder(audio, carrier, sr, num_bands=32, max_freq=4000)
    return audio


def apply_autotuner_effect(audio: np.ndarray, sr: int) -> np.ndarray:
    carrier = generate_carrier(audio.shape, sr, wave_type="sawtooth", freq=300)
    audio = apply_vocoder(audio, carrier, sr, num_bands=50, max_freq=5000)
    audio = apply_compression(audio, threshold=-15, ratio=3.0)
    audio = apply_reverb(
        audio, sr, delay=50, decay=0.25
    )  # slightly shorter delay, gentle decay
    return audio


def apply_ghost_effect(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = apply_pitch_shift(audio, sr, 3)

    carrier = generate_noise(
        audio.shape, noise_type="white", noise_level=0.001, dtype=audio.dtype
    )
    mod_audio = apply_vocoder(audio, carrier, sr, num_bands=32, max_freq=6000)

    echoed = apply_echo(mod_audio, sr, delay=400, decay=0.4)
    mod_audio = 0.3 * mod_audio + 0.7 * echoed
    mod_audio = apply_reverb(mod_audio, sr, decay=0.7)

    return mod_audio


def apply_monster_effect(audio: np.ndarray, sr: int) -> np.ndarray:
    low_pitch = apply_pitch_shift(audio, sr, -12)
    mid_pitch = apply_pitch_shift(audio, sr, -6)
    high_pitch = apply_pitch_shift(audio, sr, -3)
    mod_audio = (low_pitch + mid_pitch * 0.6 + high_pitch * 0.3) / 2.0

    # non linear distortion
    mod_audio = np.sign(mod_audio) * np.sqrt(np.abs(mod_audio))

    mod_audio = apply_lowpass(mod_audio, cutoff=1500, sr=sr)

    mod_audio = apply_tremolo(mod_audio, sr, 30)

    audio = normalize_audio(audio, mod_audio)

    return audio
