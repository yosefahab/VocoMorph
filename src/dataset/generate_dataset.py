import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from src.dataset.modulation.utils import get_functions_by_name
from src.utils.audio import load_audio
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_split_csv(dataset_dir: Path, output_csv: Path):
    """
    Scans the dataset directory and creates a CSV with zero-padded raw audio file paths
    using a single pass and a pandas apply method for efficient formatting.
    Args:
    - dataset_dir: Path to the dataset directory.
    - output_csv: Path to save the full dataset CSV.
    """
    if not dataset_dir.exists():
        logger.critical(f"{dataset_dir} does not exist! Terminating.")
        exit(1)

    logger.info(f"Scanning {dataset_dir} for wave files")

    # collect all file paths and assign integer IDs
    audio_files = []
    id_counter = 0
    for root, _, files in os.walk(dataset_dir):
        root = Path(root)
        for file in files:
            if file.lower().endswith(".wav") or file.lower().endswith(".flac"):
                audio_files.append(
                    {"ID": id_counter, "raw_wav_path": str(root.joinpath(file))}
                )
                id_counter += 1

    df = pd.DataFrame(audio_files)

    # calculate the zer-padding width based on the total number of files
    padding_width = len(str(id_counter - 1)) if id_counter > 0 else 1

    # zero-pad the IDs
    df["ID"] = df["ID"].apply(lambda x: f"{x:0{padding_width}d}")

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
    - input_csv: Path to the full dataset CSV.
    - output_csv: Path to save the validation CSV.
    - split_ratio: (train, validation) split ratios.
    - shuffle: whether to shuffle the rows before splitting.
    """
    assert input_csv.exists(), f"{input_csv} does not exist"
    logger.info(f"Splitting {input_csv} into {split_ratio}")
    # read ID column as string to preserve zero-padding
    df = pd.read_csv(input_csv, dtype={"ID": str})
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    train_end = int(len(df) * split_ratio[0])
    valid_end = train_end + int(len(df) * split_ratio[1])

    logger.info(f"Saving 1st split {split_ratio[0]} to {input_csv}")
    df.iloc[:train_end].to_csv(input_csv, index=False)

    logger.info(f"Saving 2nd split {split_ratio[1]} to {output_csv}")
    df.iloc[train_end:valid_end].to_csv(output_csv, index=False)


def apply_effects(audio: NDArray, sr: int, effects_funcs: List[Callable]):
    """
    Apply audio effects using pre-loaded functions.
    """
    return [ef(audio=audio, sr=sr) for ef in effects_funcs]


def _augment_wav(args: Tuple[str, Path, int, Path, List[Callable]]):
    wav_id, raw_wav_path, sr, output_dir, effects_funcs = args

    raw_wave, _ = load_audio(raw_wav_path, sr)
    modulated_waves = apply_effects(raw_wave, sr, effects_funcs)

    raw_wave_tensor = torch.from_numpy(raw_wave).float()
    modulated_wave_tensors = [torch.from_numpy(w).float() for w in modulated_waves]

    processed_data = []

    # save raw as effect_id = 0
    raw_dir = output_dir.joinpath("0")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir.joinpath(f"{wav_id}.pt")
    torch.save(raw_wave_tensor, raw_path)
    processed_data.append(
        {"ID": wav_id, "effect_id": 0, "tensor_filepath": str(raw_path)}
    )

    # save modulated starting from effect_id = 1
    for eid, modulated_wave_tensor in enumerate(modulated_wave_tensors, start=1):
        effect_dir = output_dir.joinpath(str(eid))
        effect_dir.mkdir(parents=True, exist_ok=True)
        effect_path = effect_dir.joinpath(f"{wav_id}.pt")
        torch.save(modulated_wave_tensor, effect_path)

        processed_data.append(
            {"ID": wav_id, "effect_id": eid, "tensor_filepath": str(effect_path)}
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
    Reads a dataset CSV, applies effects in parallel, and saves full-length tensors to a new CSV.
    Args:
    - input_csv: Path to train/valid/test CSV.
    - sr: Sample rate of the waves.
    - effects: List of effects to apply.
    - output_dir: Where to save the output tensor files.
    - output_csv: Path to save the new CSV with augmented file information.
    """
    df = pd.read_csv(input_csv, dtype={"ID": str})
    logger.info(f"Augmenting waves from {input_csv}")

    if output_dir.exists():
        logger.warning(f"Augmentation directory {output_dir} already exists, removing")
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    # pre-load the function references
    effects_funcs = get_functions_by_name(effects)

    num_workers = max(1, int((os.cpu_count() or 1) * 0.8))
    logger.info(f"Using {num_workers} workers to augment ({len(df)}) files")
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _augment_wav,
                (
                    str(row["ID"]),
                    Path(str(row["raw_wav_path"])),
                    sr,
                    output_dir,
                    effects_funcs,
                ),
            )
            for _, row in df.iterrows()
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.extend(future.result())

    df_processed = pd.DataFrame(results)

    logger.info(f"Saving augmented data info to {output_csv}")
    df_processed.to_csv(output_csv, index=False)


def create_splits(dataset: str):
    data_root = Path(os.environ["DATA_ROOT"])
    dataset_dir = data_root.joinpath(dataset)
    datalists_dir = dataset_dir.joinpath("datalists")
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
    dataset_dir = data_root.joinpath(dataset)
    datalists_dir = dataset_dir.joinpath("datalists")
    output_dir = dataset_dir.joinpath("modulated_tensors")
    sr = config["sample_rate"]
    effects = config["effects"]

    logger.info(f"Preparing to augment dataset '{dataset}'")

    for split in ["train", "valid", "test"]:
        input_csv = datalists_dir.joinpath(f"{dataset}_{split}.csv")
        output_csv = datalists_dir.joinpath(f"{dataset}_{split}_augmented.csv")

        if not input_csv.exists():
            logger.warning(f"{input_csv} does not exist, skipping augmentation.")
            continue

        logger.info(f"Augmenting {split}")
        augment_files(
            input_csv,
            sr,
            effects,
            output_dir.joinpath(split),
            output_csv,
        )


def main(dataset: str, config: dict):
    create_splits(dataset)
    augment_dataset(dataset, config)
