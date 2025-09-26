import random
from collections import OrderedDict
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
        self.npz_cache = OrderedDict()
        self.cache_size = config.get("cache_size", 100)
        self.df = pd.read_csv(datalist_filepath)
        self.n = len(self.df)
        self.logger.info(f"Dataset records = {self.n}")

    def __len__(self) -> int:
        return self.n

    def _load_npz_cache(self, filepath: str) -> dict[str, np.ndarray]:
        """Loads a .npz archive into the cache."""
        if filepath not in self.npz_cache:
            if len(self.npz_cache) >= self.cache_size:
                self.npz_cache.popitem(last=False)
            self.npz_cache[filepath] = np.load(filepath)
        self.npz_cache.move_to_end(filepath)
        return self.npz_cache[filepath]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        item_id = str(row["ID"])
        effect_id = int(row["effect_id"])
        data_path = str(row["data_path"])

        npz_archive = self._load_npz_cache(data_path)
        full_raw = torch.from_numpy(npz_archive["raw"]).float()
        full_target = torch.from_numpy(npz_archive[f"wave_{effect_id}"]).float()

        waveform_length = full_raw.shape[1]
        if waveform_length < self.chunk_size:
            pad_len = self.chunk_size - waveform_length
            full_raw = F.pad(full_raw, (0, pad_len))
            full_target = F.pad(full_target, (0, pad_len))
            start_idx = 0
        else:
            start_idx = random.randint(0, waveform_length - self.chunk_size)

        raw_chunk = full_raw[:, start_idx : start_idx + self.chunk_size]
        target_chunk = full_target[:, start_idx : start_idx + self.chunk_size]

        # diffusion timestep t
        t = torch.randint(low=1, high=1000, size=(1,), dtype=torch.long)

        # noise schedule (β_t)
        beta_t = 0.02  # example constant β
        noise = torch.randn_like(target_chunk)
        noisy_chunk = torch.sqrt(1 - beta_t) * target_chunk + torch.sqrt(beta_t) * noise

        effect_id = torch.tensor(effect_id, dtype=torch.long)

        # return noisy input, timestep, and clean target

        return item_id, effect_id, (noisy_chunk, t), noise

    # def __getitem__(
    #     self, index
    # ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     row = self.df.iloc[index]
    #     item_id = str(row["ID"])
    #     effect_id = int(row["effect_id"])
    #     data_path = str(row["data_path"])
    #
    #     npz_archive = self._load_npz_cache(data_path)
    #     full_raw = torch.from_numpy(npz_archive["raw"]).float()
    #     full_target = torch.from_numpy(npz_archive[f"wave_{effect_id}"]).float()
    #
    #     waveform_length = full_raw.shape[1]
    #     if waveform_length < self.chunk_size:
    #         self.logger.warning(
    #             f"Waveform {item_id} is too short ({waveform_length}) for chunk size {self.chunk_size}"
    #         )
    #         pad_len = self.chunk_size - waveform_length
    #         full_raw = F.pad(full_raw, (0, pad_len))
    #         full_target = F.pad(full_target, (0, pad_len))
    #         start_idx = 0
    #     else:
    #         start_idx = random.randint(0, waveform_length - self.chunk_size)
    #
    #     raw_chunk = full_raw[:, start_idx : start_idx + self.chunk_size]
    #     target_chunk = full_target[:, start_idx : start_idx + self.chunk_size]
    #
    #     assert raw_chunk.shape[1] == self.chunk_size
    #     assert target_chunk.shape[1] == self.chunk_size
    #
    #     effect_id = torch.tensor(effect_id, dtype=torch.long)
    #
    #     return item_id, effect_id, raw_chunk, target_chunk


def get_dataloaders(
    dataset_path: Path, splits: list[str], config: dict, ddp: bool = False
) -> dict[str, DataLoader]:
    dataloaders = {}

    for split in splits:
        datalist_filepath = dataset_path.joinpath("datalists", f"{split}_augmented.csv")

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
        split_batch_size = config["batch_size"][split]
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
            sampler=sampler,
        )
        dataloaders[split] = dataloader

    return dataloaders
