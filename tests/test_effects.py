from pathlib import Path

from src.dataset.generate_dataset import apply_effects
from src.dataset.modulation.effects import *
from src.dataset.modulation.filters import *
from src.dataset.modulation.synthesis import *
from src.dataset.modulation.transformations import *
from src.dataset.modulation.utils import get_functions_by_name
from src.utils.audio import load_audio, save_audio


def test_effects_all():
    efx = [
        "apply_robotic_effect",
        "apply_radio_effect",
        "apply_ghost_effect",
        "apply_demonic_effect",
        "apply_identity_transform",
    ]
    effect_funcs = get_functions_by_name(efx)
    wavs = [
        "data/timit/TEST/DR7/MDVC0/SA2.WAV",
        "data/timit/TEST/DR2/FPAS0/SI1272.WAV",
        "data/timit/TEST/DR7/FISB0/SX319.WAV",
    ]
    for w in wavs:
        wave_path = Path(w)
        wave_name = wave_path.stem
        sr = 16_000
        channels = 1
        audio, sr = load_audio(wave_path, sr=sr, channels=channels)
        for i, a in enumerate(apply_effects(audio, sr, effect_funcs)):
            save_audio(a, sr, f"{wave_name}_{efx[i].split('_')[1]}")

    assert True


if __name__ == "__main__":
    test_effects_all()
