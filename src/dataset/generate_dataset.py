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
    if not dataset_dir.is_dir():
        logger.critical(f"{dataset_dir} does not exist! Terminating.")
        exit(1)

    logger.info(f"Scanning {dataset_dir} for wave files")

    exts = (".wav", ".flac")
    audio_files = [
        {"ID": i, "raw_wav_path": str(p)}
        for i, p in enumerate(dataset_dir.rglob("*"))
        if p.suffix.lower() in exts and p.is_file()
    ]

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
    - input_csv: Path to the full dataset CSV.
    - output_csv: Path to save the validation CSV.
    - split_ratio: (train, validation) split ratios.
    - shuffle: whether to shuffle the rows before splitting.
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


def apply_effects(audio: NDArray, sr: int, effects_funcs: List[Callable]):
    """
    Apply audio effects using pre-loaded functions.
    """
    return [ef(audio=audio, sr=sr) for ef in effects_funcs]


def _augment_wav(args: Tuple[pd.Series, int, Path, List[Callable], int, int]):
    """
    Helper function to process a single row for parallel execution.
    This function now saves a single large tensor per effect.
    """
    row, sr, output_dir, effects_funcs, chunk_size, overlap = args

    wav_id = row["ID"]
    raw_wav_path = Path(str(row["raw_wav_path"]))

    # Load and apply effects using the pre-loaded functions
    raw_wave, _ = load_audio(raw_wav_path, sr)
    modulated_waves = apply_effects(raw_wave, sr, effects_funcs)

    # Convert numpy arrays to torch tensors
    raw_wave_tensor = torch.from_numpy(raw_wave).float()
    modulated_wave_tensors = [torch.from_numpy(w).float() for w in modulated_waves]

    processed_data = []
    stride = chunk_size - overlap

    for eid, modulated_wave_tensor in enumerate(modulated_wave_tensors):
        # Collect all chunks for this effect into a single tensor
        raw_chunks = []
        modulated_chunks = []

        for start in range(0, raw_wave_tensor.shape[-1] - chunk_size + 1, stride):
            end = start + chunk_size
            raw_chunks.append(raw_wave_tensor[:, start:end])
            modulated_chunks.append(modulated_wave_tensor[:, start:end])

        # Concatenate all chunks into a single large tensor
        all_raw_chunks = torch.stack(raw_chunks)
        all_modulated_chunks = torch.stack(modulated_chunks)

        # Create a directory for this effect ID if it doesn't exist
        effect_dir = output_dir.joinpath(str(eid))
        effect_dir.mkdir(parents=True, exist_ok=True)

        # Create a single filename for the large tensor
        filename = f"{wav_id}_{eid}.pt"
        filepath = effect_dir.joinpath(filename)

        # Save both large tensors in a dictionary
        torch.save({"raw": all_raw_chunks, "modulated": all_modulated_chunks}, filepath)

        # Add each chunk to the processed data list, pointing to the same file
        for chunk_idx in range(len(raw_chunks)):
            processed_data.append(
                {
                    "ID": wav_id,
                    "effect_id": eid,
                    "chunk_index": chunk_idx,
                    "tensor_filepath": str(filepath),
                }
            )

    return processed_data


def augment_files(
    input_csv: Path,
    sr: int,
    effects: List[str],
    output_dir: Path,
    output_csv: Path,
    chunk_size: int,
    overlap: int,
):
    """
    Reads a dataset CSV, applies effects in parallel, chunks, and saves outputs as tensors to a new CSV.
    Args:
    - input_csv: Path to train/valid/test CSV.
    - sr: Sample rate of the waves.
    - effects: List of effects to apply.
    - output_dir: Where to save the output tensor files.
    - output_csv: Path to save the new CSV with augmented file information.
    - chunk_size: The size of each audio chunk.
    - overlap: The overlap between chunks.
    """
    df = pd.read_csv(input_csv, dtype={"ID": str})
    logger.info(f"Augmenting waves from {input_csv} ({len(df)} total)")

    if output_dir.exists():
        logger.warning(
            f"Pre-existing augmentation directory exists, removing {output_dir}"
        )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Pre-load the function references before starting the parallel pool
    effects_funcs = get_functions_by_name(effects)

    num_workers = max(1, int(os.cpu_count() or 1 * 0.8))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _augment_wav, (row, sr, output_dir, effects_funcs, chunk_size, overlap)
            )
            for _, row in df.itertuples(index=False)
        ]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    df_processed = pd.DataFrame([item for sublist in results for item in sublist])

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
    chunk_size = config["chunk_size"]
    overlap = config["overlap"]

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
            chunk_size,
            overlap,
        )


def main(dataset: str, config: dict):
    create_splits(dataset)
    augment_dataset(dataset, config)
