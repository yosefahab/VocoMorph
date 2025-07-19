import os
import random

import numpy as np
import torch

from src.utils.logger import get_logger

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
