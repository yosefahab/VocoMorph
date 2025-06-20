import os
import yaml
import torch
import random
import numpy as np
import librosa
import soundfile as sf
from typing import Literal, Optional, Tuple
from src.logging.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 14):
    """Sets the seet of the entire code to always simulate the same results everytime"""
    logger.info(f"Setting seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_type: Literal["cpu", "gpu"] = "gpu") -> torch.device:
    """
    Selects device for executing models on (cpu, cuda, or mps)

    Args:
        device_type: type of device, (cpu or gpu)

    Returns:
        torch.device.
    """
    if device_type == "gpu":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            logger.warning("No GPU detected, using CPU")
            device = "cpu"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    return torch.device(device)


def parse_yaml(yaml_path: str) -> dict:
    """
    Parse and return the contents of a YAML file.

    Args:
        path: Path to the YAML file to be parsed.

    Returns:
        dict: A dictionary containing the parsed contents of the YAML file.
    """
    assert os.path.exists(yaml_path), f"YAML file {yaml_path} doesn't exist"
    with open(yaml_path, "r") as yaml_file:
        config_dict = yaml.full_load(yaml_file)
        return config_dict


def save_audio(
    audio: np.ndarray,
    sr: int,
    filename: str,
    output_dir: Optional[str] = None,
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
        output_dir = os.path.join(os.environ["DATA_ROOT"], "output")

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # construct full file path
    filename = os.path.join(output_dir, filename)

    # ensure correct shape (T, C) for multi-channel
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T

    # check for NaN or Inf values
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        logger.error(f"Audio {filename} contains NaN or Inf values")

    # convert to float32 if necessary
    if audio.dtype != np.float32:
        logger.warning(
            f"Found audio {filename} dtype: {audio.dtype}, expected np.float32"
        )
        audio = audio.astype(np.float32)

    # save the audio file
    sf.write(filename, audio, sr)


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
