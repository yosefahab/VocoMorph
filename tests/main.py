from src.logging.logger import get_logger
from src.utils import load_audio, save_audio
from src.modulation.effects import *
from src.modulation.filters import *
from src.modulation.synthesis import *
from src.modulation.transformations import *
from src.dataset.generate_dataset import apply_effects

logger = get_logger(__name__)


def main(config: dict):
    logger.info(f"Running tests from {__name__}")
    sr = config["sample_rate"]
    channels = config["channels"]

    audio, sr = load_audio(
        "data/TIMIT/TEST/DR1/FAKS0/SA2.WAV", sr=sr, channels=channels
    )

    for i, a in enumerate(apply_effects(audio, sr, config["effects"])):
        save_audio(a, sr, f"audio_e{i}")
