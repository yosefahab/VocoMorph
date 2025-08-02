import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict

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

        self.lru_tensor_cache = OrderedDict()
        self.cache_size = config.get("cache_size", 100)

        self.df = pd.read_csv(datalist_filepath)

        self.n = len(self.df)
        self.logger.info(f"Dataset records = {self.n}")

    def __len__(self) -> int:
        return self.n

    def _load_tensor(self, filepath: str):
        """Loads a tensor file into the cache."""
        if filepath not in self.lru_tensor_cache:
            if len(self.lru_tensor_cache) >= self.cache_size:
                self.lru_tensor_cache.popitem(last=False)
            self.lru_tensor_cache[filepath] = torch.load(filepath)

        self.lru_tensor_cache.move_to_end(filepath)
        return self.lru_tensor_cache[filepath]

    def __getitem__(
        self, index
    ) -> Tuple[Tuple[int, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns:
        - wave ID
        - effect ID tensor
        - raw wave chunk (C, chunk_size)
        - modulated wave chunk (C, chunk_size)
        """
        row = self.df.iloc[index]
        id: int = row["ID"]
        effect_id = row["effect_id"]
        tensor_filepath = row["tensor_filepath"]
        chunk_index = row["chunk_index"]

        # get the large tensor, loading it from disk if not in cache
        large_tensor = self._load_tensor(tensor_filepath)
        raw_chunks_tensor = large_tensor["raw"]
        modulated_chunks_tensor = large_tensor["modulated"]

        # slice the large tensor to get the specific chunk
        raw_chunk = raw_chunks_tensor[chunk_index]
        modulated_chunk = modulated_chunks_tensor[chunk_index]

        assert raw_chunk.shape[1] == self.chunk_size
        assert modulated_chunk.shape[1] == self.chunk_size

        effect_id = torch.tensor(effect_id, dtype=torch.long)

        return ((id, effect_id, raw_chunk), modulated_chunk)


def collate_fn(batch):
    """
    Collates a batch of pre-chunked tensors.
    """
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
    dataset_name = config["dataset_name"]
    for split in splits:
        datalist_filepath = DATA_ROOT.joinpath(
            dataset_name, "datalists", config["datalists"][split]["path"]
        )

        assert datalist_filepath.exists(), (
            f"Datalist for split {split} doesn't exist: {datalist_filepath}"
        )
        logger.info(f"Loading {split} data from: {datalist_filepath}")
        num_workers = min(config["num_workers"], max(0, cpu_count() - 2))
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
            pin_memory=config["pin_memory"],
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            sampler=sampler,
        )
        dataloaders[split] = dataloader

    return dataloaders
