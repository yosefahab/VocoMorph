import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal.windows import hann

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_audio(
    audio: NDArray,
    sr: int,
    filename: str,
    output_dir: Optional[Path] = None,
    extension: str = ".wav",
):
    """
    Saves the processed audio.
    Ensures correct format, dtype, and shape before saving.
    """

    # ensure valid extension
    assert extension in [".wav", ".flac", ".ogg"], f"Unsupported extension: {extension}"

    if not filename.lower().endswith(extension):
        filename += extension

    # set default output directory
    if output_dir is None:
        output_dir = Path(os.environ["DATA_ROOT"]).joinpath("output")
    else:
        assert output_dir.is_dir(), (
            f"param output_dir ({output_dir}) is not a directory"
        )

    # create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # construct full file path
    filepath = output_dir.joinpath(filename)

    # ensure correct shape (T, C) for multi-channel
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T

    # check for NaN or Inf values
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        logger.error(f"Audio {filepath} contains NaN or Inf values")

    # convert to float32 if necessary
    if audio.dtype != np.float32:
        logger.warning(
            f"Found audio {filename} dtype: {audio.dtype}, expected np.float32"
        )
        audio = audio.astype(np.float32)

    # save the audio file
    # logger.info(f"Saved audio to: {filepath}")
    sf.write(filepath, audio, sr)


def load_audio(
    filepath: Path, sr: Optional[int] = None, channels: int = 1
) -> Tuple[NDArray, int]:
    """
    Loads an audio file with optional resampling and ensures (C, T) shape.
    Args:
    - filepath: Path to the audio file.
    - sr: Target sample rate (optional).
    - channels: Number of output channels (1 = mono, 2 = stereo).
    Returns:
        audio: (C, T) NumPy array where C is 1 or 2.
        sr_out: Sample rate of the loaded audio.
    """
    sr_float = float(sr) if sr is not None else None
    audio, sr_out = librosa.load(filepath, sr=sr_float, mono=False)

    assert audio.size != 0, f"Found empty audio: {filepath}"

    # ensure (C, T) format
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)

    if channels == 1 and audio.shape[0] == 2:
        # Downmix stereo to mono
        audio = np.mean(audio, axis=0, keepdims=True)
    elif channels == 2 and audio.shape[0] == 1:
        # convert mono to stereo if requested
        audio = np.vstack([audio, audio])

    return audio, int(sr_out)


def overlap_add(
    audio: NDArray[np.float32],
    chunk_size: int,
    hop_size: int,
    processor: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    window: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """
    Apply a processor function to overlapping chunks of a 2D audio signal (C, T).

    Parameters:
        audio: 2D float32 NumPy array of the input audio signal with shape (channels, time).
        chunk_size: Number of samples in each frame.
        hop_size: Number of samples to advance for each frame.
        processor: Function applied to each chunk. Must return a chunk of the same shape.
        window: Optional window function. If None, Hann window is used.

    Returns:
        Reconstructed audio signal as a 2D float32 NumPy array with shape (channels, time).
    """
    assert audio.ndim == 2, "Input audio must be a 2D array (channels, time)."

    if window is None:
        window = hann(chunk_size, sym=False).astype(np.float32)

    channels, audio_len = audio.shape
    output = np.zeros((channels, audio_len + chunk_size), dtype=np.float32)

    for i in range(0, audio_len, hop_size):
        # 1. Get the chunk and pad with zeros if necessary
        chunk = audio[:, i : i + chunk_size]
        if chunk.shape[-1] < chunk_size:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[-1])))

        # 2. Apply window and process
        chunk_windowed = chunk * window  # Broadcasting the 1D window across channels
        processed_windowed = processor(chunk_windowed) * window

        # 3. Overlap-add the processed chunk to the output
        output[:, i : i + chunk_size] += processed_windowed

    return output[:, :audio_len]
