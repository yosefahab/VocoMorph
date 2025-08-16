from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from src.utils.audio import load_audio, save_audio
from src.utils.logger import get_logger

logger = get_logger(__name__)


def infer(
    effect_id: int,
    filepath: Path,
    model: nn.Module,
    config: dict,
    output_path: Optional[Path] = None,
):
    logger.info(f"Running inference on: {filepath} with EID: {effect_id}")
    sr = config["data"]["sample_rate"]
    channels = config["data"]["channels"]

    logger.info("Loading audio")
    audio = load_audio(filepath, sr, channels)[0]
    audio_tensor = mx.expand_dims(mx.array(audio), axis=0)
    effect_id_tensor = mx.expand_dims(mx.array(effect_id, dtype=mx.int8), axis=0)

    audio_output = model((effect_id_tensor, audio_tensor)).squeeze(0).numpy()

    output_path = output_path or filepath
    filedir = output_path.parent
    filename = output_path.stem
    filename += f"_{effect_id}.wav"

    logger.info(f"Saving audio to: {output_path}")
    save_audio(audio_output, sr, filename, filedir)
