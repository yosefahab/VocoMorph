from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.utils.audio import load_audio, save_audio
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.utils.types import DeviceType

logger = get_logger(__name__)


def infer(
    effect_id: int,
    filepath: Path,
    model: nn.Module,
    config: dict,
    device_type: DeviceType,
    output_path: Optional[Path] = None,
):
    device = get_device(device_type)
    logger.info(f"Running inference on: {filepath} with EID: {effect_id}")
    sr = config["data"]["sample_rate"]
    channels = config["data"]["channels"]

    logger.info("Loading audio")
    audio = load_audio(filepath, sr, channels)[0]
    audio_tensor = torch.Tensor(audio).unsqueeze(0).to(device)

    effect_id_tensor = torch.tensor(
        effect_id, dtype=torch.long, device=device
    ).unsqueeze(0)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        audio_output = model((effect_id_tensor, audio_tensor)).cpu().squeeze(0).numpy()

    output_path = output_path or filepath
    filedir = output_path.parent
    filename = output_path.stem
    filename += f"_{effect_id}.wav"

    logger.info(f"Saving audio to: {output_path}")
    save_audio(audio_output, sr, filename, filedir)
