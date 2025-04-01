import os
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from src.logging.logger import get_logger
from src.utils import load_audio, save_audio
from src.modulation.utils import call_functions_by_name


logger = get_logger(__name__)


def create_dataset_csv(dataset_dir: str, output_csv: str):
    """
    Scans the dataset directory and creates a CSV with raw audio file paths.

    Args:
        dataset_dir: Path to the dataset directory.
        output_csv: Path to save the full dataset CSV.
    """
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
    input_csv, output_valid_csv=None, split_ratios=(0.8, 0.2), skip_test=False
):
    """
    Splits a dataset CSV into train and validation (optionally test) CSVs.

    Args:
        input_csv: Path to the full dataset CSV.
        output_valid_csv: Path to save the validation CSV (if different from input).
        split_ratios: (train, validation, [test]) split ratios.
        skip_test: If True, only creates train/valid splits.
    """
    logger.info(f"Splitting {input_csv} into {split_ratios}")
    df = pd.read_csv(input_csv).sample(frac=1).reset_index(drop=True)

    train_end = int(len(df) * split_ratios[0])
    valid_end = train_end + int(len(df) * split_ratios[1])

    output_dir = os.path.dirname(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    train_csv = (
        os.path.join(output_dir, "train.csv") if output_valid_csv is None else input_csv
    )
    valid_csv = (
        output_valid_csv if output_valid_csv else os.path.join(output_dir, "valid.csv")
    )

    logger.info(f"Saving train set to {train_csv}")
    df.iloc[:train_end].to_csv(train_csv, index=False)

    logger.info(f"Saving validation set to {valid_csv}")
    df.iloc[train_end:valid_end].to_csv(valid_csv, index=False)

    if not skip_test and valid_end < len(df):
        test_csv = os.path.join(output_dir, "test.csv")
        df.iloc[valid_end:].to_csv(test_csv, index=False)


def apply_effects(audio: np.ndarray, sr: int, effects: List[str]):
    """
    Placeholder: Apply an audio effect based on effect_id.
    """
    return call_functions_by_name(effects, audio=audio, sr=sr)


def augment_wav(args):
    """Helper function to process a single row for parallel execution."""
    row, sr, channels, output_dir, effects = args
    raw_wav_path = str(row["raw_wav_path"])

    raw_wave, _ = load_audio(raw_wav_path, sr, channels)

    raw_filename = os.path.splitext(os.path.basename(raw_wav_path))[0]

    # generate effect variations
    modulated_waves = apply_effects(raw_wave, sr, effects)
    processed_data = []

    for eid, mod_wave in enumerate(modulated_waves):
        mod_filename = f"{raw_filename}_e{eid}.wav"
        mod_wav_path = os.path.join(output_dir, mod_filename)
        save_audio(mod_wave, sr, mod_filename, output_dir)
        assert len(mod_wave) == len(raw_wave)

        processed_data.append(
            {
                "ID": row["ID"],
                "effect_id": eid,
                "raw_wav_path": raw_wav_path,
                "modulated_wav_path": mod_wav_path,
            }
        )

    return processed_data


def augment_files(input_csv: str, sr: int, output_dir: str, effects: List[str]):
    """
    Reads a dataset CSV, applies `num_effects` per file, and saves outputs.

    Args:
        input_csv: Path to train/valid/test CSV.
        sr: Sample rate of the waves.
        output_dir: Where to save the output waves.
    """
    df = pd.read_csv(input_csv, dtype={"ID": str})
    logger.info(f"Augmenting waves from {input_csv} {len(df)} total")

    # define output directories once
    output_dir = os.path.join(os.environ["PROJECT_ROOT"], "data", "modulated")
    os.makedirs(output_dir, exist_ok=True)

    # augment waves in parallel
    with Pool() as p:
        c = [(row, sr, output_dir, effects) for _, row in df.iterrows()]
        r = list(tqdm(p.imap(augment_wav, c), total=len(df)))

    # flatten results and save CSV
    df_processed = pd.DataFrame([item for sublist in r for item in sublist])
    df_processed.to_csv(input_csv, index=False)


def create_splits(dataset: str, config: dict):
    project_root = os.environ["PROJECT_ROOT"]
    dataset_dir = os.path.join(project_root, config["dataset_path"][dataset])
    metadata_dir = os.path.join(project_root, config["datalist_dir"])
    os.makedirs(metadata_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        csv_out_path = os.path.join(metadata_dir, f"{dataset}_{split}.csv")
        split_dir = os.path.join(dataset_dir, split)

        if os.path.exists(split_dir):
            logger.info(f"creating {split} csv")
            create_dataset_csv(split_dir, csv_out_path)
        elif dataset.lower() == "timit" and split == "valid":
            logger.warning(f"{split_dir} missing, splitting train set instead")
            train_csv = os.path.join(metadata_dir, f"{dataset}_train.csv")

            if not os.path.exists(train_csv):
                train_dir = os.path.join(dataset_dir, "train")
                logger.info("creating train csv")
                create_dataset_csv(train_dir, train_csv)

            valid_csv = os.path.join(metadata_dir, f"{dataset}_valid.csv")

            split_dataset_csv(train_csv, output_valid_csv=valid_csv, skip_test=True)


def augment_dataset(dataset: str, config: dict):
    project_root = os.environ["PROJECT_ROOT"]
    metadata_dir = os.path.join(project_root, config["datalist_dir"])
    output_dir = os.path.join(project_root, "data")
    sr = config["sample_rate"]
    effects = config["effects"]

    for split in ["train", "valid", "test"]:
        csv_path = os.path.join(metadata_dir, f"{dataset}_{split}.csv")
        if os.path.exists(csv_path):
            logger.info(f"augmenting {split}")
            augment_files(csv_path, sr, output_dir, effects)


def main(dataset: str, config: dict):
    create_splits(dataset, config)
    augment_dataset(dataset, config)
