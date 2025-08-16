from typing import Any, Dict, Literal, Tuple, TypeVar

from mlx.optimizers import Optimizer, schedulers

DeviceType = Literal["cpu", "gpu"]

T = TypeVar("T")

DictConfig = Dict[str, Any]

OptimizerScheduler = Tuple[Optimizer, schedulers.Callable | None]
