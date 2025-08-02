from pathlib import Path

from src.dataset.generate_dataset import apply_effects
from src.dataset.modulation.effects import *
from src.dataset.modulation.filters import *
from src.dataset.modulation.synthesis import *
from src.dataset.modulation.transformations import *
from src.dataset.modulation.utils import get_functions_by_name
from src.utils.audio import load_audio, save_audio
from src.utils.types import DictConfig


def test_effects(config: DictConfig):
    sr = config["sample_rate"]
    channels = config["channels"]

    wave_path = Path("data/TIMIT/TEST/DR1/FAKS0/SA2.WAV")
    audio, sr = load_audio(wave_path, sr=sr, channels=channels)

    effect_funcs = get_functions_by_name(config["effects"])
    for i, a in enumerate(apply_effects(audio, sr, effect_funcs)):
        save_audio(a, sr, f"audio_e{i}")
