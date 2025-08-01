import torch

from src.utils.logger import get_logger
from src.utils.types import DeviceType

logger = get_logger(__name__)


def get_device(device_type: DeviceType = "gpu", local_rank: int = 0) -> torch.device:
    """
    Selects device for executing models on (cpu, cuda, or mps)
    Args:
    - device_type: type of device, (cpu or gpu)
    - local_rank: the device ID if using multi gpu
    Returns:
        torch.device.
    """
    if device_type == "gpu":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = f"cuda:{local_rank}"
        else:
            logger.warning("No GPU detected, using CPU")
            device = "cpu"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    return torch.device(device)
