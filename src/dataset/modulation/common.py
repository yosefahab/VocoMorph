import numpy as np
from numpy.typing import NDArray


def apply_distortion(audio: NDArray, gain: float = 3.0) -> NDArray:
    """
    Adds harmonics and alters the timbre.
    """
    return np.tanh(audio * gain)
