import os
import random
from collections import OrderedDict
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VocoMorphDataset(Dataset):
    def __init__(self, config: dict, datalist_filepath: Path) -> None:
        super().__init__()

        self.logger = get_logger(self.__class__.__name__)
        self.fs = config["sample_rate"]
        self.channels = config["channels"]
        self.chunk_size = config["chunk_size"]

        self.datalist_filepath = datalist_filepath

        # A simple in-memory LRU cache
        self.tensor_cache = OrderedDict()
        self.cache_size = config.get("cache_size", 100)

        self.df = pd.read_csv(datalist_filepath)

        self.n = len(self.df)
        self.logger.info(f"Dataset records = {self.n}")

    def __len__(self) -> int:
        return self.n

    def _load_tensor_cache(self, filepath: str) -> torch.Tensor:
        """Loads a tensor file into the cache."""
        if filepath not in self.tensor_cache:
            if len(self.tensor_cache) >= self.cache_size:
                # remove the least recently used item
                self.tensor_cache.popitem(last=False)
            self.tensor_cache[filepath] = torch.load(filepath)

        # move the accessed sample to the end to mark it as most recently used
        self.tensor_cache.move_to_end(filepath)
        return self.tensor_cache[filepath]

    def _load_tensor(self, filepath: str) -> torch.Tensor:
        return torch.load(filepath)

    def __getitem__(
        self, index
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        id = str(row["ID"])
        effect_id = int(row["effect_id"])

        modulated_path = row["modulated_tensor_path"]
        raw_path = row["raw_tensor_path"]

        full_modulated = self._load_tensor(modulated_path)
        full_raw = self._load_tensor(str(raw_path))

        waveform_length = full_raw.shape[1]
        if waveform_length < self.chunk_size:
            self.logger.warning(f"Waveform {id} is too short ({waveform_length}) for chunk size {self.chunk_size}")
            pad_len = self.chunk_size - waveform_length
            full_raw = torch.nn.functional.pad(full_raw, (0, pad_len))
            full_modulated = torch.nn.functional.pad(full_modulated, (0, pad_len))
            start_idx = 0
        else:
            start_idx = random.randint(0, waveform_length - self.chunk_size)

        raw_chunk = full_raw[:, start_idx : start_idx + self.chunk_size]
        modulated_chunk = full_modulated[:, start_idx : start_idx + self.chunk_size]

        assert raw_chunk.shape[1] == self.chunk_size
        assert modulated_chunk.shape[1] == self.chunk_size

        effect_id = torch.tensor(effect_id, dtype=torch.long)

        return id, effect_id, raw_chunk, modulated_chunk


def collate_fn(batch):
    """
    Collates a batch of tensors, filtering out any 'None' items.
    """
    # filter out any None values that might be returned for short audio files
    batch = [item for item in batch if item is not None]

    first_elems, modulated_waves = zip(*batch)
    ids, effect_ids, raw_waves = zip(*first_elems)

    # shape: (B)
    effect_ids = torch.stack(effect_ids)

    # the tensors are already the correct size, just stack them
    raw_waves = torch.stack(raw_waves)
    modulated_waves = torch.stack(modulated_waves)

    return (ids, effect_ids, raw_waves, modulated_waves)


def get_dataloaders(
    splits: List[str], config: dict, ddp: bool = False
) -> Dict[str, DataLoader]:
    """
    Args:
    - splits: the splits to create dataloaders for (train/valid/test)
    - config: dataset configuration dict
    - is_distributed: whether to support distributed training. This passes a DistributedSampler to the dataloader
    Returns:
        dict containing dataloader for each split
    """
    dataloaders = {}
    DATA_ROOT = Path(os.environ["DATA_ROOT"])
    dataset_name = config["dataset"]
    for split in splits:
        datalist_filepath = DATA_ROOT.joinpath(
            dataset_name, "datalists", config["datalists"][split]["path"]
        )

        assert datalist_filepath.exists(), (
            f"Datalist for split {split} doesn't exist: {datalist_filepath}"
        )
        logger.info(f"Loading {split} data from: {datalist_filepath}")
        default_workers = max(1, cpu_count() - 2)
        num_workers = config.get("num_workers")
        if num_workers is None:
            num_workers = default_workers

        logger.info(f"Using {num_workers} workers for DataLoaders")
        dataset = VocoMorphDataset(config, datalist_filepath=datalist_filepath)
        logger.info(f"Creating dataloader for split: {split}")
        split_batch_size = config["datalists"][split]["batch_size"]
        logger.info(f"Using batch size {split_batch_size} for split: {split}")

        drop_last = config["drop_last"] if split == "train" else False

        sampler = None
        if ddp:
            logger.info(f"Creating DDP sampler for split: {split}")
            sampler = DistributedSampler(
                dataset=dataset, shuffle=(split == "train"), drop_last=drop_last
            )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=split_batch_size,
            shuffle=(split == "train" and not ddp),
            pin_memory=config.get("pin_memory", True),
            num_workers=num_workers,
            drop_last=drop_last,
            # collate_fn=collate_fn,
            sampler=sampler,
        )
        dataloaders[split] = dataloader

    return dataloaders
