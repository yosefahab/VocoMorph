from abc import abstractmethod
import os

import torch

from src.utils.device import set_device
from src.utils.logger import get_logger
from src.utils.types import DeviceType


class TrainingStrategy:
    def __init__(self, device_type: DeviceType):
        self.device_type: DeviceType = device_type
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def wrap_model(
        self, model: torch.nn.Module
    ) -> tuple[torch.nn.Module, torch.device]:
        pass

    @abstractmethod
    def should_log(self) -> bool:
        pass

    @abstractmethod
    def finalize(self):
        pass


class DDPStrategy(TrainingStrategy):
    def __init__(self, device_type: DeviceType = "gpu"):
        super().__init__(device_type)

        backend = "nccl" if device_type == "gpu" else "gloo"
        self.logger.info(
            f"Using DDP strategy with {device_type} device type, and {backend} backend"
        )
        self.device = set_device(self.device_type, self.local_rank)

        if self.device.type == "mps":
            self.logger.error(
                "DDP with MPS is not supported. Use CPU or SingleDeviceStrategy with mps."
            )
            exit(1)

        torch.distributed.init_process_group(backend, "env://")

        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

    def wrap_model(self, model: torch.nn.Module):
        model = model.to(self.device)
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank]
        ), self.device

    def should_log(self) -> bool:
        return self.rank == 0

    def finalize(self):
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


class SingleDeviceStrategy(TrainingStrategy):
    def __init__(self, device_type: DeviceType = "gpu"):
        super().__init__(device_type)
        self.logger.info("Using SingleDevice strategy")

    def wrap_model(self, model: torch.nn.Module):
        device = set_device(self.device_type)
        return model.to(device), device

    def should_log(self) -> bool:
        return True
