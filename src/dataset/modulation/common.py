import numpy as np


def apply_distortion(audio: np.ndarray, gain: float = 3.0) -> np.ndarray:
    """
    Adds harmonics and alters the timbre.
    """
    return np.tanh(audio * gain)


