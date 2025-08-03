# """functions to alter audio perception"""
#
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .common import apply_distortion
from .filters import apply_bandpass, apply_lowpass
from .synthesis import generate_carrier, generate_noise
from .transformations import (
    apply_compression,
    apply_gain,
    apply_identity_transform,
    apply_modulation,
    apply_pitch_shift,
    mix_lufs,
    normalize_lufs,
    normalize_rms,
    reverse_audio,
)


def apply_bit_crush(audio: NDArray, sr: int = -1, bit_depth: int = 8) -> NDArray:
    """
    Compresses/reduces fidelity of audio
    """
    quantization_levels = 2**bit_depth
    return np.round(audio * quantization_levels) / quantization_levels


def apply_chorus(
    audio: NDArray,
    sr: int,
    rate: float = 4.0,
    depth: float = 0.003,
    delay: float = 0.03,
    voices: int = 3,
    mix: float = 0.5,
) -> NDArray:
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


def apply_reverb(
    audio: NDArray,
    sr: int,
    pre_delay=70,
    decay=0.3,
    reverberance=0.5,
    wet_gain_db: float = 0.0,
    dry_gain_db: float = 0.0,
    room_size: float = 0.8,
    damping: float = 0.5,
) -> NDArray:
    """
    Apply reverb effect to the audio by applying a simple convolution reverb filter.

    Parameters:
        audio (NDArray): Audio array of shape (channels, samples)
        sr (int): Sample rate
        pre_delay (float): Silence before reverb start (ms)
        decay (float): Per-echo decay factor
        reverberance (float): Amount of reverb (0.0–1.0), controls number of echoes
        wet_gain_db (float): Gain applied to the reverb tail (dB)
        dry_gain_db (float): Gain applied to the original signal (dB)
        room_size (float): Scales echo density and length (0.0–1.0)
        damping (float): High-frequency attenuation over echo time (0.0–1.0)
    """
    C, T = audio.shape

    pre_delay_samples = int((pre_delay / 1_000) * sr)
    echo_spacing = int(sr * 0.05 * room_size)
    num_echoes = max(1, int(reverberance * room_size * 12))

    impulse_len = pre_delay_samples + echo_spacing * num_echoes + 1
    impulse = np.zeros(impulse_len)
    impulse[pre_delay_samples] = 1.0

    for i in range(1, num_echoes + 1):
        pos = pre_delay_samples + i * echo_spacing
        attenuation = decay**i
        hf_rolloff = 1.0 - damping * (i / num_echoes)
        impulse[pos] = attenuation * hf_rolloff

    wet_gain = 10 ** (wet_gain_db / 20)
    dry_gain = 10 ** (dry_gain_db / 20)

    output = np.zeros_like(audio)
    for c in range(C):
        y_reverb = np.convolve(audio[c], impulse, mode="full")[:T]
        y_reverb = normalize_rms(audio[c], y_reverb)
        output[c] = np.clip(audio[c] * dry_gain + y_reverb * wet_gain, -1.0, 1.0)

    return output


def apply_echo_effect(
    audio: NDArray,
    sr: int,
    delay: int = 500,
    decay: float = 0.15,
    num_echoes: int = 3,
) -> NDArray:
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

    output = normalize_rms(audio, output)
    return output


def apply_tremolo(
    audio: NDArray, sr: int, rate: float = 30, depth: float = 0.4
) -> NDArray:
    """
    applies a tremolo (amplitude modulation) effect to the audio signal
    """
    _, T = audio.shape
    t = np.linspace(0, T / sr, num=T)
    tremolo = 1 + depth * np.sin(2 * np.pi * rate * t)
    return (audio * tremolo).astype(audio.dtype)


def apply_vocoder(
    audio: NDArray,
    sr: int,
    carrier: Optional[NDArray] = None,
    num_bands: int = 8,
    min_freq: float = 100,
    max_freq: float = 4000,
) -> NDArray:
    """
    Applies a basic vocoder effect by modulating a carrier wave with the input audio's frequency envelopes.
    """
    if carrier is None:
        carrier = generate_carrier(audio.shape, sr, wave_type="square", freq=150)

    assert audio.shape == carrier.shape, (
        "Audio and carrier must have the same shape (C, T)"
    )
    band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands + 1)
    output = np.zeros_like(audio, dtype=np.float32)

    for i in range(num_bands):
        low, high = band_edges[i], band_edges[i + 1]
        filtered_audio = apply_bandpass(audio, sr, low, high, order=2)
        envelope = np.abs(filtered_audio)
        envelope = apply_lowpass(envelope, sr, cutoff=50, order=1)
        carrier_band = apply_bandpass(carrier, sr, low, high, order=2)
        output += envelope * carrier_band

    return output


def apply_sidechain(audio: NDArray, kick: NDArray, reduction: float = 0.5) -> NDArray:
    """
    Applies sidechain compression based on kick drum
    """
    env = np.convolve(np.abs(kick), np.ones(100) / 100, mode="same")
    gain = 1.0 - (env / np.max(env)) * reduction
    return audio * gain


def apply_direction_effect(
    audio: NDArray, direction: str = "center", speed: float = 5.0
) -> NDArray:
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

    left = normalize_rms(audio[0], left)
    right = normalize_rms(audio[1], right)
    return np.vstack((left, right))


#
def apply_scifi_effect(audio: NDArray, sr: int) -> NDArray:
    audio = apply_pitch_shift(audio, sr, -1)
    audio = apply_modulation(audio, sr, 20, 0.6, "am")
    audio = apply_bit_crush(audio, bit_depth=16)
    audio = apply_chorus(audio, sr, delay=0.005)
    audio = apply_distortion(audio, gain=2)
    audio = apply_tremolo(audio, sr, rate=30)
    return audio


def apply_autotuner_effect(audio: NDArray, sr: int) -> NDArray:
    carrier = generate_carrier(audio.shape, sr, wave_type="sawtooth", freq=300)
    audio = apply_vocoder(audio, sr, carrier, num_bands=50)
    # audio = apply_compression(audio, threshold=-15, ratio=3.0)
    # # slightly shorter delay, gentle decayreturn
    # audio = apply_reverb(audio, sr, 50, decay=0.25)
    return audio


def apply_robotic_effect(audio: NDArray, sr: int) -> NDArray:
    audio = apply_reverb(audio, sr, 20, 0.3, 0.3, room_size=0.3)
    audio = apply_pitch_shift(audio, sr, 1.68)
    return audio


def apply_radio_effect(audio: NDArray, sr: int) -> NDArray:
    mod_audio = apply_bandpass(audio, sr, low=400.0, high=2000.0)
    mod_audio = apply_bit_crush(mod_audio, bit_depth=8)
    noise = generate_noise(mod_audio.shape, 0.8, "white", mod_audio.dtype)
    audio = mix_lufs([noise, mod_audio], sr)

    return audio


def apply_ghost_effect(audio: NDArray, sr: int) -> NDArray:
    ghost = reverse_audio(audio, sr)
    ghost = apply_reverb(ghost, sr, 150, 0.5, 0.6, 0, -10, 0.85, 0.7)
    ghost = reverse_audio(ghost, sr)
    audio = apply_gain(audio, -22)
    return mix_lufs([audio, ghost], sr)


def apply_demonic_effect(audio: NDArray, sr: int) -> NDArray:
    dtype = audio.dtype
    audio1 = apply_pitch_shift(apply_gain(audio, -3), sr, -1)
    audio = apply_pitch_shift(audio, sr, -1.78)
    audio2 = apply_pitch_shift(apply_gain(audio, +3), sr, -7.72)
    return mix_lufs([audio, audio1, audio2], sr).astype(dtype)
