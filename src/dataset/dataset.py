import os
import pandas as pd
from typing import Dict, List, Tuple
from functools import partial
from multiprocessing import cpu_count

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from src.logging.logger import get_logger
from src.utils import load_audio

logger = get_logger(__name__)


class VocoMorphDataset(Dataset):
    def __init__(self, config: dict, datalist_filepath: str) -> None:
        super().__init__()

        self.fs = config["sample_rate"]
        self.channels = config["channels"]
        self.frame_length = config["frame_length"]
        self.max_length = config["max_length"]
        self.datalist_filepath = datalist_filepath

        self.df = pd.read_csv(datalist_filepath)
        self.n = len(self.df)
        logger.info(f"Dataset records = {self.n}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(
        self, index
    ) -> Tuple[Tuple[int, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns:
        - wave ID
        - effect ID tensor
        - raw wave (C, frame_length)
        - modulated wave (C, frame_length)
        """
        row = self.df.iloc[index]
        id = row["ID"]
        effect_id = row["effect_id"]
        raw_path = row["raw_wav_path"]
        modulated_path = row["modulated_wav_path"]

        raw_wave, _ = load_audio(raw_path, self.fs, self.channels)[: self.max_length]
        modulated_wave, _ = load_audio(modulated_path, self.fs, self.channels)[
            : self.max_length
        ]

        # shape: (C, frame_length)
        raw_wave = torch.tensor(raw_wave)
        modulated_wave = torch.tensor(modulated_wave)
        effect_id = torch.tensor(effect_id, dtype=torch.long).unsqueeze(0)

        return (
            (
                id,
                effect_id,
                raw_wave,
            ),
            modulated_wave,
        )


def collate_fn(batch, max_length):
    first_elems, modulated_waves = zip(*batch)
    ids, effect_ids, raw_waves = zip(*first_elems)

    # shape: (B, 1)
    effect_ids = torch.stack(effect_ids)

    def pad_wave(wave, max_length):
        """pad all waveforms to the max_length"""
        C, T = wave.shape
        pad_amount = max_length - T
        wave = (
            torch.cat([wave, torch.zeros(C, pad_amount)], dim=1)
            if pad_amount > 0
            else wave
        )
        return wave[:max_length]

    # (B, C, T)
    raw_waves = torch.stack([pad_wave(w, max_length) for w in raw_waves])
    modulated_waves = torch.stack([pad_wave(w, max_length) for w in modulated_waves])

    # create a mask where 1 = real data, 0 = padding
    # masks = torch.tensor(
    #     [[1] * w.shape[1] + [0] * (max_length - w.shape[1]) for w in raw_waves],
    #     dtype=torch.bool,
    # )

    return (
        ids,
        effect_ids,
        raw_waves,
        modulated_waves,
        # masks,
    )


def get_dataloaders(splits: List[str], config: dict) -> Dict[str, DataLoader]:
    """
    Args:
        splits: the splits to create dataloaders for (train/valid/test)
        config: dataset configuration dict

    Returns:
        dict containing dataloader for each split
    """
    # create dataset object for each partition
    dataset_name = config["dataset"]
    logger.info(f"Loading dataset {dataset_name}")

    collate_fn_partial = partial(collate_fn, max_length=config["max_length"])
    dataloaders = {}
    for split in splits:
        datalist_filepath = os.path.join(
            os.environ["PROJECT_ROOT"],
            config["datalist_dir"],
            f"{dataset_name}_{split}.csv",
        )
        logger.info(f"Loading data from {datalist_filepath}")
        num_workers = min(config['num_workers'], cpu_count()-2)
        logger.info(f"Using {num_workers} workers for DataLoaders")
        dataset = VocoMorphDataset(config, datalist_filepath=datalist_filepath)
        logger.info(f"Creating dataloader for split: {split}")
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1 if split == "test" else config["batch_size"],
            shuffle=(split == "train"),
            pin_memory=config["pin_memory"],
            num_workers=num_workers,
            drop_last=config["drop_last"],
            collate_fn=collate_fn_partial,
        )
        dataloaders[split] = dataloader

    return dataloaders
