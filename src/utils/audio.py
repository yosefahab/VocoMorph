import os
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_audio(
    audio: np.ndarray,
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
    sf.write(filepath, audio, sr)


def load_audio(
    filepath: str, sr: Optional[int] = None, channels: int = 1
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file with optional resampling and ensures (C, T) shape.

    Args:
    - filepath: Path to the audio file.
    - sr: Target sample rate (optional).
    - channels: Number of output channels (1 = mono, 2 = stereo).

    Returns:
    - audio: (C, T) NumPy array where C is 1 or 2.
    - sr_out: Sample rate of the loaded audio.
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
