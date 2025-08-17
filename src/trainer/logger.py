from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List

from mlx.optimizers.optimizers import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter

from src.utils.logger import get_logger


class ExperimentLogger:
    def __init__(self, log_dir: Path):
        pass

    @abstractmethod
    def log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        pass

    @abstractmethod
    def log_lrs(self, epoch: int, optimizers: List[Optimizer]):
        pass

    @abstractmethod
    def close_writer(self):
        pass


class TensorBoardLogger(ExperimentLogger):
    def __init__(self, log_dir: Path):
        self.logger = get_logger(self.__class__.__name__)
        self.root_dir = log_dir
        self.log_path = self.root_dir.joinpath(datetime.now().strftime("%d%m%Y-%H%M%S"))
        self.log_path.mkdir(parents=True)
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.logger.info(f"TensorBoard logs at: {self.log_path}")

    def log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        self.logger.info("Logging epoch metrics")
        for group in ("Losses", "Metrics"):
            keys = set(train_metrics[group].keys())
            diff = keys ^ set(val_metrics[group].keys())
            assert len(diff) == 0, f"{group} keys mismatch: {diff}"

            for key in keys:
                self.logger.info(
                    f"{group}/{key}: Train={train_metrics[group][key]:.4f} | Val={val_metrics[group][key]:.4f}"
                )
                self.writer.add_scalars(
                    f"{group}/{key}",
                    {
                        "Train": train_metrics[group][key],
                        "Val": val_metrics[group][key],
                    },
                    epoch,
                )

        self.writer.flush()

    def log_lrs(self, epoch: int, optimizers: List[Optimizer]):
        for optimizer in optimizers:
            optimizer_name = optimizer.__class__.__name__
            lr_values = {}

            for group_index, param_group in enumerate(optimizer.param_groups):
                lr_values[f"group_{group_index}"] = param_group["lr"]

            tag = f"LearningRate/{optimizer_name}/ParamGroup"
            self.writer.add_scalars(tag, lr_values, epoch)

        self.writer.flush()

    def close_writer(self):
        self.writer.flush()
        self.writer.close()


class NullLogger(ExperimentLogger):
    def log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        pass

    def close_writer(self):
        pass
