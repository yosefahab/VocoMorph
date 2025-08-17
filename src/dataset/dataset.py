import random
from functools import partial
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import mlx.data as dx
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_chunk(sample: Dict, chunk_size: int) -> dict[str, mx.array]:
    """
    Loads raw and augmented waves from an .npz file and extracts a random chunk.
    """
    # decode bytes -> str
    path = sample["data_path"].tobytes().decode("utf-8")
    effect_id = int(sample["effect_id"])

    archive = np.load(path)

    full_raw_wave = mx.array(archive["wave_0"])
    full_target_wave = mx.array(archive[f"wave_{effect_id}"])

    waveform_length: int = full_raw_wave.shape[1]  # pyright: ignore[reportIndexIssue]

    if waveform_length < chunk_size:
        pad_len = chunk_size - waveform_length
        full_raw_wave = mx.pad(full_raw_wave, [(0, 0), (0, pad_len)], mode="constant")
        full_target_wave = mx.pad(
            full_target_wave, [(0, 0), (0, pad_len)], mode="constant"
        )
        start_idx = 0
    else:
        start_idx = random.randint(0, waveform_length - chunk_size)

    raw_chunk = full_raw_wave[:, start_idx : start_idx + chunk_size]
    target_chunk = full_target_wave[:, start_idx : start_idx + chunk_size]

    effect_id_arr = mx.array(effect_id, dtype=mx.int8)

    return {
        "inputs": raw_chunk.tolist(),
        "targets": target_chunk.tolist(),
        "effect_id": effect_id_arr.tolist(),
    }


def get_data_streams(
    dataset_path: Path, splits: List[str], config: dict
) -> Dict[str, dx.Stream]:  # pyright: ignore[reportAttributeAccessIssue]
    """
    Creates mlx.data streams for each split.
    """
    data_streams = {}

    for split in splits:
        datalist_filepath = dataset_path / "datalists" / f"{split}_augmented.csv"

        assert datalist_filepath.exists(), (
            f"Datalist for split {split} doesn't exist: {datalist_filepath}"
        )
        logger.info(f"Loading {split} data from: {datalist_filepath}")

        df = pd.read_csv(datalist_filepath)
        records = df.to_dict("records")

        # encode string paths into bytes (MLX requirement)
        for r in records:
            r["data_path"] = r["data_path"].encode("utf-8")

        stream = (
            dx.buffer_from_vector(records)  # pyright: ignore[reportAttributeAccessIssue]
            .to_stream()
            .sample_transform(partial(load_and_chunk, chunk_size=config["chunk_size"]))
            .batch(batch_size=config["batch_size"][split])
            # .prefetch(prefetch_size=2, num_threads=max((os.cpu_count() or 1) - 2, 1))
        )

        # if split == "train":
        #     stream = stream.shuffle()

        data_streams[split] = stream

    return data_streams
