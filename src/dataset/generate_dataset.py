import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Optional, Tuple

from src.logging.logger import get_logger
from src.utils import load_audio, save_audio
from src.modulation.utils import call_functions_by_name


logger = get_logger(__name__)


def create_split_csv(dataset_dir: str, output_csv: str):
    """
    Scans the dataset directory and creates a CSV with raw audio file paths.

    Args:
        dataset_dir: Path to the dataset directory.
        output_csv: Path to save the full dataset CSV.
    """
    if not os.path.exists(dataset_dir):
        logger.critical(f"{dataset_dir} does not exist! Terminating.")
        exit(1)

    logger.info(f"Scanning {dataset_dir} for wave files")
    id = 0
    audio_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                audio_files.append(
                    {"ID": f"{id:05}", "raw_wav_path": os.path.join(root, file)}
                )
                id += 1

    df = pd.DataFrame(audio_files)
    logger.info(f"Found {len(df)} wave files")
    logger.info(f"Saving to {output_csv}")
    df.to_csv(output_csv, index=False)


def split_dataset_csv(
    input_csv: str,
    output_csv: str,
    split_ratio: Tuple[float, float] = (0.8, 0.2),
    shuffle: Optional[bool] = False,
):
    """
    Splits a dataset CSV into train and validation CSVs.

    Args:
        input_csv: Path to the full dataset CSV.
        output_csv: Path to save the validation CSV.
        split_ratio: (train, validation) split ratios.
        shuffle: whether to shuffle the rows before splitting.
    """
    logger.info(f"Splitting {input_csv} into {split_ratio}")
    df = pd.read_csv(input_csv, dtype={"ID": str})
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    train_end = int(len(df) * split_ratio[0])
    valid_end = train_end + int(len(df) * split_ratio[1])

    output_dir = os.path.dirname(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving 1st split {split_ratio[0]} to {input_csv}")
    df.iloc[:train_end].to_csv(input_csv, index=False)

    logger.info(f"Saving 2nd split {split_ratio[1]} to {output_csv}")
    df.iloc[train_end:valid_end].to_csv(output_csv, index=False)


def apply_effects(audio: np.ndarray, sr: int, effects: List[str]):
    """
    Apply audio effects by name.
    """
    return call_functions_by_name(effects, audio=audio, sr=sr)


def _augment_wav(args: Tuple[pd.Series, int, str, List[str]]):
    """Helper function to process a single row for parallel execution."""
    row, sr, output_dir, effects = args
    wav_id = str(row["ID"])
    raw_wav_path = str(row["raw_wav_path"])

    raw_wave, _ = load_audio(raw_wav_path, sr)

    modulated_waves = apply_effects(raw_wave, sr, effects)
    processed_data = []

    for eid, modulated_wave in enumerate(modulated_waves):
        mod_filename = f"{wav_id}_e{eid}.wav"
        mod_wav_path = os.path.join(output_dir, mod_filename)
        save_audio(modulated_wave, sr, mod_filename, output_dir)
        assert len(modulated_wave) == len(raw_wave)

        processed_data.append(
            {
                "ID": str(row["ID"]),
                "effect_id": eid,
                "raw_wav_path": raw_wav_path,
                "modulated_wav_path": mod_wav_path,
            }
        )
    return processed_data


def augment_files(
    input_csv: str,
    sr: int,
    effects: List[str],
    output_dir: str,
    output_csv: str,
):
    """
    Reads a dataset CSV, applies effects in parallel, and saves outputs to a new CSV.

    Args:
        input_csv: Path to train/valid/test CSV.
        sr: Sample rate of the waves.
        effects: List of effects to apply.
        output_dir: Where to save the output waves.
        output_csv: Path to save the new CSV with augmented file information.
    """
    df = pd.read_csv(input_csv, dtype={"ID": str})
    logger.info(f"Augmenting waves from {input_csv} ({len(df)} total)")

    os.makedirs(output_dir, exist_ok=True)

    def generate_augmentation_args(
        df: pd.DataFrame, sr: int, output_dir: str, effects: List[str]
    ):
        """Generates arguments for _augment_wav on demand."""
        for _, row in df.iterrows():
            yield (row, sr, output_dir, effects)

    with Pool() as p:
        args_generator = generate_augmentation_args(df, sr, output_dir, effects)
        results = list(tqdm(p.imap(_augment_wav, args_generator), total=len(df)))

    df_processed = pd.DataFrame([item for sublist in results for item in sublist])

    logger.info(f"Saving augmented data info to {output_csv}")
    df_processed.to_csv(output_csv, index=False)


def create_splits(dataset: str):
    project_root = os.environ["PROJECT_ROOT"]
    metadata_dir = os.path.join(project_root, "data", "metadata")
    dataset_dir = os.path.join(project_root, "data", dataset)
    os.makedirs(metadata_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.exists(split_dir):
            logger.info(f"Creating {split} csv")
            split_csv = os.path.join(metadata_dir, f"{dataset}_{split}.csv")
            create_split_csv(split_dir, split_csv)
        # timit comes with no validation set, so we create one from train set
        elif dataset.lower() == "timit" and split == "valid":
            logger.warning(f"{split_dir} missing, splitting train set instead")
            train_csv = os.path.join(metadata_dir, f"{dataset}_train.csv")
            valid_csv = os.path.join(metadata_dir, f"{dataset}_valid.csv")
            split_dataset_csv(train_csv, valid_csv)


def augment_dataset(dataset: str, config: dict):
    project_root = os.environ["PROJECT_ROOT"]
    metadata_dir = os.path.join(project_root, "data", "metadata")
    output_dir = os.path.join(project_root, "data", dataset, "modulated")
    sr = config["sample_rate"]
    effects = config["effects"]

    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_csv = os.path.join(metadata_dir, f"{dataset}_{split}.csv")
        if not os.path.exists(split_csv):
            logger.warning(f"{split_csv} does not exist, skipping augmentation.")
            continue
        logger.info(f"Augmenting {split}")
        augment_files(split_csv, sr, effects, output_dir, split_csv)


def main(dataset: str, config: dict):
    create_splits(dataset)
    augment_dataset(dataset, config)
