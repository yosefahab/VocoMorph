import mlx.core as mx
from mlx.core import Device

from src.utils.logger import get_logger
from src.utils.types import DeviceType as _DeviceType

logger = get_logger(__name__)


def set_device(device_type: _DeviceType = "gpu"):
    """
    Selects device for executing models on (cpu or gpu).
    Args:
    - device_str: type of device, a string ("cpu" or "gpu")
    Returns:
        The selected MLX device object.
    """
    if device_type == "cpu":
        device = mx.cpu
    elif device_type == "gpu":
        if mx.metal.is_available():
            device = mx.gpu
        else:
            logger.warning("GPU not available, falling back to CPU.")
            device = mx.cpu

    mx.set_default_device(Device(device))
    logger.info(f"Using device: {mx.default_device()}")
