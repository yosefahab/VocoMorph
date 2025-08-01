import os
import random

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 14):
    """Sets a deterministic seed for reproducibility across multiple libraries."""
    logger.info(f"Setting seed to: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
