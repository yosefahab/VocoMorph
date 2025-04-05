import importlib
import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


from src.logging.logger import get_logger


logger = get_logger(__name__)


def plot_waves(signals: List[Tuple[np.ndarray, float]]):
    n = len(signals)
    _, ax = plt.subplots(nrows=n)
    for i, signal in enumerate(signals):
        s, sr = signal
        librosa.display.waveshow(s, sr=sr, ax=ax[i])
        ax[i].set(title=f"Signal-{i}")
    plt.show()


def plot_wave_and_augmented_wave(
    signal: np.ndarray, augmented_signal: np.ndarray, sr: int
):
    _, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    ax[0].set(title="Original signal")
    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
    ax[1].set(title="Augmented signal")
    plt.show()


def call_functions_by_name(function_names: List[str], *args, **kwargs) -> list:
    """
    Dynamically loads and calls functions from the specified module.

    Args:
    - function_names: List of function names to call.
    - args: Positional arguments to pass to each function.
    - kwargs: Keyword arguments to pass to each function.

    Returns:
        functions' results
    """
    results = []
    module = importlib.import_module("src.modulation.effects")

    for name in function_names:
        if not hasattr(module, name):
            logger.critical(f"Effect {name} not implemented in {module}! Terminating")
            exit(1)

        func = getattr(module, name)
        if not callable(func):
            logger.critical(f"{func} is not a callable function")

        try:
            results.append(func(*args, **kwargs))
        except Exception as e:
            logger.error(
                f"An exception occured while calling {func} with args: {args} and kwargs: {kwargs}: {e}"
            )

    return results
