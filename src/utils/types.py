from typing import Literal, Tuple, TypeVar, Dict, Any
import torch.optim as optim


DeviceType = Literal["gpu", "cpu"]


T = TypeVar("T")
DictConfig = Dict[str, Any]
OptimizerScheduler = Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler | None]
