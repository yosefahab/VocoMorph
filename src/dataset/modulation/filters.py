"""functions to alter audio frequency content"""

import numpy as np
from scipy.signal import butter, iirnotch, lfilter


def apply_bandpass(
    audio: np.ndarray, sr: int, low: float, high: float, order=2
) -> np.ndarray:
    """Applies a bandpass filter to an audio signal with shape (C, T) and returns float32."""
    assert audio.ndim == 2, "Input audio must have shape (C, T)"
    audio = audio.astype(audio.dtype)
    nyq = sr / 2
    low = max(low, 1) / nyq
    high = min(high, nyq - 1) / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, audio, axis=1).astype(audio.dtype)


def apply_lowpass(
    audio: np.ndarray, sr: int, cutoff: float = 1000, order=5
) -> np.ndarray:
    """Applies a low-pass filter to an audio signal of shape (C, T) and returns float32."""
    assert audio.ndim == 2, "Input audio must have shape (C, T)"
    audio = audio.astype(audio.dtype)
    nyq = sr / 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low")
    return lfilter(b, a, audio, axis=1).astype(audio.dtype)


def apply_highpass(
    audio: np.ndarray, sr: int, cutoff: float = 500, order=5
) -> np.ndarray:
    """Applies a high-pass filter to an audio signal of shape (C, T) and returns float32."""
    assert audio.ndim == 2, "Input audio must have shape (C, T)"
    audio = audio.astype(audio.dtype)
    nyq = sr / 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high")
    return lfilter(b, a, audio, axis=1).astype(audio.dtype)


def apply_notch(
    audio: np.ndarray, sr: int, notch_freq: float = 60, quality_factor=30
) -> np.ndarray:
    """Applies a notch filter to remove a specific frequency from an audio signal of shape (C, T) and returns float32."""
    assert audio.ndim == 2, "Input audio must have shape (C, T)"
    audio = audio.astype(audio.dtype)
    nyq = sr / 2
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor, fs=sr)
    return lfilter(b, a, audio, axis=1).astype(audio.dtype)


def apply_bandstop(
    audio: np.ndarray, sr: int, lowcut: float = 300, highcut: float = 3000, order=5
) -> np.ndarray:
    """Applies a band-stop filter to remove a frequency range from an audio signal of shape (C, T) and returns float32."""
    assert audio.ndim == 2, "Input audio must have shape (C, T)"
    audio = audio.astype(audio.dtype)
    nyq = sr / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="bandstop")
    return lfilter(b, a, audio, axis=1).astype(audio.dtype)
