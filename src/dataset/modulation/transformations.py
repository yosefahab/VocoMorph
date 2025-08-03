"""functions to alter audio structure"""

import librosa
import numpy as np
import pyloudnorm as pyln
from numpy.typing import NDArray


def apply_identity_transform(audio: NDArray, sr: int) -> NDArray:
    return audio


def reverse_audio(audio: NDArray, sr: int) -> NDArray:
    return audio[:, ::-1]


def normalize_lufs(reference_audio: NDArray, new_audio: NDArray, sr: int) -> NDArray:
    meter = pyln.Meter(sr)
    try:
        target_lufs = meter.integrated_loudness(reference_audio)
        input_lufs = meter.integrated_loudness(new_audio)
    except Exception as _:
        return new_audio

    return pyln.normalize.loudness(new_audio, input_lufs, target_lufs)


def normalize_rms(reference_audio: NDArray, new_audio: NDArray) -> NDArray:
    target_rms = np.sqrt(np.mean(reference_audio**2, axis=-1, keepdims=True))
    current_rms = np.sqrt(np.mean(new_audio**2, axis=-1, keepdims=True))

    if np.any(current_rms == 0) or np.any(target_rms == 0):
        return new_audio

    normalized = new_audio * (target_rms / current_rms)
    return np.clip(normalized, -1, 1)


def mix_lufs(tracks: list[NDArray], sr: int) -> NDArray:
    normalized_tracks = [normalize_lufs(tracks[0], track, sr) for track in tracks[1:]]
    mixed = np.sum(normalized_tracks, axis=0)
    final = normalize_lufs(tracks[0], mixed, sr)
    return np.clip(final, -1, 1)


def convert_to_mono(audio: NDArray) -> NDArray:
    """
    Converts multi-channel audio to mono by averaging across channels.
    """
    if audio.ndim == 2:
        return np.mean(audio, axis=0, keepdims=True)
    return audio


def apply_time_stretch(audio: NDArray, sr: int, rate: float = 0.7) -> NDArray:
    """
    Stretches or compresses the audio in time without altering pitch.
    """
    if audio.ndim == 1:
        return librosa.effects.time_stretch(audio, rate=rate)
    return np.stack(
        [librosa.effects.time_stretch(channel, rate=rate) for channel in audio],
    )


def apply_pitch_shift(audio: NDArray, sr: int, n_steps: float = 6) -> NDArray:
    """
    Shifts pitch by a number of steps (positive or negative).
    """
    return np.stack(
        [
            librosa.effects.pitch_shift(channel, sr=sr, n_steps=n_steps)
            for channel in audio
        ]
    )


def apply_gain(audio: NDArray, gain_db: float) -> NDArray:
    factor = 10 ** (gain_db / 20)
    out = audio * factor
    return np.clip(out, -1.0, 1.0)


def apply_compression(
    audio: NDArray, threshold: float = -20, ratio: float = 4.0
) -> NDArray:
    """
    Applies dynamic range compression to an audio signal.
    """
    threshold = 10 ** (threshold / 20)  # convert dB to linear scale
    gain = np.ones_like(audio)

    above_threshold = np.abs(audio) > threshold
    gain[above_threshold] = (
        threshold + (np.abs(audio[above_threshold]) - threshold) / ratio
    ) / np.abs(audio[above_threshold])

    return audio * gain


def apply_modulation(
    audio: NDArray,
    sr: int,
    mod_freq: float = 19.0,
    mod_depth: float = 0.2,
    mod_type: str = "am",
) -> NDArray:
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
            modulated_audio[c] = normalize_rms(audio[c], modulated_audio[c])

    return modulated_audio
