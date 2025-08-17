import importlib
from typing import Callable, List, Tuple

import librosa
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_waves(signals: List[Tuple[NDArray, float]]):
    n = len(signals)
    _, ax = plt.subplots(nrows=n)
    for i, signal in enumerate(signals):
        s, sr = signal
        librosa.display.waveshow(s, sr=sr, ax=ax[i])
        ax[i].set(title=f"Signal-{i}")
    plt.show()


def plot_wave_and_augmented_wave(signal: NDArray, augmented_signal: NDArray, sr: int):
    _, ax = plt.subplots(nrows=2)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    ax[0].set(title="Original signal")
    librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1])
    ax[1].set(title="Augmented signal")
    plt.show()


def get_functions_by_name(function_names: List[str]) -> List[Callable]:
    """
    Dynamically loads and caches function objects from the 'effects' module.

    This function is intended to be called once to pre-load all effect functions.

    Args:
    - function_names: List of function names (strings) to load.

    Returns:
        A list of callable function objects.
    """
    # a cache to store function objects after the first lookup.
    # this cache is local to this function and will not be a global variable.
    functions: dict[str, Callable] = {}

    # load the module once
    module = importlib.import_module("src.dataset.modulation.effects")

    def get_function_by_name(name):
        if not hasattr(module, name):
            logger.critical(f"Effect '{name}' not implemented in {module}! Terminating")
            exit(1)

        func = getattr(module, name)
        if not callable(func):
            logger.critical(f"'{func}' is not a callable function! Terminating")
            exit(1)
        return func

    for name in function_names:
        if name not in functions:
            functions[name] = get_function_by_name(name)

    return list(functions.values())
