"""functions to alter audio structure"""

import librosa
import numpy as np
from numpy.typing import NDArray


def identity_transform(audio: NDArray, sr: int) -> NDArray:
    return audio


def normalize_audio(original_audio: NDArray, new_audio: NDArray) -> NDArray:
    """
    Normalizes an audio array of samples to match the RMS (loudness) of the original.
    Ensures output does not exceed [-1, 1] to prevent clipping.
    """
    target_rms = np.sqrt(np.mean(original_audio**2, axis=-1, keepdims=True))
    current_rms = np.sqrt(np.mean(new_audio**2, axis=-1, keepdims=True))

    if np.any(current_rms == 0) or np.any(target_rms == 0):
        return np.zeros_like(new_audio)

    normalized_audio = new_audio * (target_rms / current_rms)
    return np.clip(normalized_audio, -1, 1)


def convert_to_mono(audio: NDArray) -> NDArray:
    """
    Converts multi-channel audio to mono by averaging across channels.
    """
    if audio.ndim == 2:
        return np.mean(audio, axis=0, keepdims=True)
    return audio


def apply_time_stretch(audio: NDArray, _sr: int, rate: float = 0.7) -> NDArray:
    """
    Stretches or compresses the audio in time without altering pitch.
    """
    if audio.ndim == 1:
        return normalize_audio(audio, librosa.effects.time_stretch(audio, rate=rate))
    return np.stack(
        [
            normalize_audio(channel, librosa.effects.time_stretch(channel, rate=rate))
            for channel in audio
        ],
    )


def apply_pitch_shift(audio: NDArray, sr: int, n_steps: float = 6) -> NDArray:
    """
    Shifts pitch by a number of steps (positive or negative).
    """
    return np.stack(
        [
            normalize_audio(
                channel, librosa.effects.pitch_shift(channel, sr=sr, n_steps=n_steps)
            )
            for channel in audio
        ]
    )


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
