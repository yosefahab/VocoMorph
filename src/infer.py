from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from src.utils.audio import load_audio, overlap_add, save_audio
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_model_processor(model, effect_id, device):
    """
    Creates a callable processor for the overlap-add function.
    """

    def processor(chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        # 1. convert the numpy chunk to a pytorch tensor
        audio_tensor = torch.Tensor(chunk).unsqueeze(0).to(device)

        # 2. prepare the effect id tensor
        effect_id_tensor = torch.tensor(
            effect_id, dtype=torch.long, device=device
        ).unsqueeze(0)

        # 3. run inference with the model
        with torch.no_grad():
            processed_tensor = model((effect_id_tensor, audio_tensor))

        # 4. convert the output back to a numpy array
        processed_chunk = processed_tensor.cpu().squeeze(0).numpy()

        return processed_chunk

    return processor


def infer(
    effect_id: int,
    filepath: Path,
    model: nn.Module,
    config: dict,
    device: torch.device,
    output_path: Optional[Path] = None,
):
    logger.info(f"running inference on: {filepath} with eid: {effect_id}")
    sr = config["data"]["sample_rate"]
    channels = config["data"]["channels"]

    # get model-specific chunk size and hop size
    chunk_size: int = model.chunk_size  # pyright: ignore[reportAssignmentType]
    # you will need to define a hop size; often, it's half the chunk size
    hop_size = chunk_size // 2

    logger.info("loading audio")
    audio = load_audio(filepath, sr, channels)[0]

    # setup the model and create the processor
    model = model.to(device)
    model.eval()

    processor = make_model_processor(model, effect_id, device)

    # use overlap-add to process the entire audio file
    audio_output = overlap_add(
        audio=audio, chunk_size=chunk_size, hop_size=hop_size, processor=processor
    )

    output_path = output_path if output_path else filepath
    filedir = output_path if output_path else filepath.parent
    filename = output_path.stem
    filename += f"_{effect_id}.wav"

    save_audio(audio_output, sr, filename, filedir)
