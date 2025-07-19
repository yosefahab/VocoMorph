import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.modulation.utils import call_functions_by_name
from src.utils.audio import load_audio, save_audio
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_split_csv(dataset_dir: Path, output_csv: str):
    """
    Scans the dataset directory and creates a CSV with raw audio file paths.

    Args:
        dataset_dir: Path to the dataset directory.
        output_csv: Path to save the full dataset CSV.
    """
    if not dataset_dir.exists():
        logger.critical(f"{dataset_dir} does not exist! Terminating.")
        exit(1)

    logger.info(f"Scanning {dataset_dir} for wave files")
    id = 0
    audio_files = []
    for root, _, files in os.walk(dataset_dir):
        root = Path(root)
        for file in files:
            if file.lower().endswith(".wav") or file.lower().endswith(".flac"):
                audio_files.append(
                    {"ID": f"{id:07}", "raw_wav_path": root.joinpath(file)}
                )
                id += 1

    df = pd.DataFrame(audio_files)
    logger.info(f"Found {len(df)} wave files")
    logger.info(f"Saving to {output_csv}")
    df.to_csv(output_csv, index=False)


def split_dataset_csv(
    input_csv: Path,
    output_csv: Path,
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
    assert input_csv.exists(), f"{input_csv} does not exist"
    logger.info(f"Splitting {input_csv} into {split_ratio}")
    df = pd.read_csv(input_csv, dtype={"ID": str})
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    train_end = int(len(df) * split_ratio[0])
    valid_end = train_end + int(len(df) * split_ratio[1])

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
    # wav_id = str(row["ID"]).zfill(7)
    wav_id = row["ID"]
    raw_wav_path = str(row["raw_wav_path"])

    raw_wave, _ = load_audio(raw_wav_path, sr)

    modulated_waves = apply_effects(raw_wave, sr, effects)
    processed_data = []

    for eid, modulated_wave in enumerate(modulated_waves):
        mod_filename = f"{wav_id}_e{eid}.wav"
        mod_wav_path = output_dir.joinpath(mod_filename)
        save_audio(modulated_wave, sr, mod_filename, output_dir)
        assert len(modulated_wave) == len(raw_wave)

        processed_data.append(
            {
                "ID": wav_id,
                "effect_id": eid,
                "raw_wav_path": raw_wav_path,
                "modulated_wav_path": mod_wav_path,
            }
        )
    return processed_data


def augment_files(
    input_csv: Path,
    sr: int,
    effects: List[str],
    output_dir: Path,
    output_csv: Path,
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

    output_dir.mkdir(exist_ok=True)

    num_workers = max(1, int(os.cpu_count() or 1 * 0.8))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_augment_wav, (row, sr, output_dir, effects))
            for _, row in df.iterrows()
        ]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    df_processed = pd.DataFrame([item for sublist in results for item in sublist])

    logger.info(f"Saving augmented data info to {output_csv}")
    df_processed.to_csv(output_csv, index=False)


def create_splits(dataset: str):
    data_root = Path(os.environ["DATA_ROOT"])
    datalists_dir = data_root.joinpath("datalists")
    dataset_dir = data_root.joinpath(dataset)
    datalists_dir.mkdir(exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_dir = dataset_dir.joinpath(split)
        if split_dir.exists():
            logger.info(f"Creating {split} csv")
            split_csv = datalists_dir.joinpath(f"{dataset}_{split}.csv")
            create_split_csv(split_dir, split_csv)
        # timit comes with no validation set, so we create one from train set
        elif dataset.lower() == "timit" and split == "valid":
            logger.warning(f"{split_dir} missing, splitting train set instead")
            train_csv = datalists_dir.joinpath(f"{dataset}_train.csv")
            valid_csv = datalists_dir.joinpath(f"{dataset}_valid.csv")
            split_dataset_csv(train_csv, valid_csv)


def augment_dataset(dataset: str, config: dict):
    data_root = Path(os.environ["DATA_ROOT"])
    datalists_dir = data_root.joinpath("datalists")
    output_dir = data_root.joinpath(dataset, "modulated")
    sr = config["sample_rate"]
    effects = config["effects"]

    output_dir.mkdir(exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_csv = datalists_dir.joinpath(f"{dataset}_{split}.csv")
        if not split_csv.exists():
            logger.warning(f"{split_csv} does not exist, skipping augmentation.")
            continue
        logger.info(f"Augmenting {split}")
        augment_files(split_csv, sr, effects, output_dir, split_csv)


def main(dataset: str, config: dict):
    create_splits(dataset)
    augment_dataset(dataset, config)
