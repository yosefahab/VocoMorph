import os
import torch
import torch.nn as nn
from typing import Optional

from .utils import load_audio, save_audio


def infer(
    effect_id: int,
    filepath: str,
    model: nn.Module,
    config: dict,
    output_path: Optional[str] = None,
):
    sr = config["data"]["sample_rate"]
    channels = config["data"]["channels"]
    audio = load_audio(filepath, sr, channels)[0]
    audio_tensor = torch.Tensor(audio).unsqueeze(0)

    effect_id_tensor = torch.tensor(effect_id, dtype=torch.long).unsqueeze(0)
    audio_output = (
        model((effect_id_tensor, audio_tensor)).detach().cpu().squeeze(0).numpy()
    )

    output_path = output_path or filepath
    filedir = os.path.dirname(output_path)
    filename = os.path.splitext(os.path.basename(output_path))[0]
    filename += f"_{effect_id}.wav"

    save_audio(audio_output, sr, filename, filedir)
