from typing import Literal

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device(device_type: Literal["cpu", "gpu"] = "gpu") -> torch.device:
    """
    Selects device for executing models on (cpu, cuda, or mps)
    Args:
    - device_type: type of device, (cpu or gpu)
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
