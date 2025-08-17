import os
import random

import mlx.core as mx
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 14):
    """Sets a deterministic seed for reproducibility across multiple libraries."""
    logger.info(f"Setting seed to: {seed}")
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
