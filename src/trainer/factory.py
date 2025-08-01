from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List

import torch.nn as nn
import torch.optim as optim
from torcheval import metrics

from src.utils.logger import get_logger
from src.utils.types import DictConfig, OptimizerScheduler, T

logger = get_logger(__name__)


def load_class(category: str, lib_check: type, name: str) -> type[Any]:
    """Load a class from a module dynamically."""
    try:
        if hasattr(lib_check, name):
            logger.info(f"Loading class {name} from {lib_check}")
            return getattr(lib_check, name)
        logger.info(f"Loading custom class {name} from {category}")
        return getattr(import_module(f"src.trainer.custom.{category}"), name)
    except Exception as e:
        logger.exception(f"Error loading class {name} from {lib_check}|{category}: {e}")
        exit(1)


def create_instance(constructor: Callable[..., T], config: DictConfig) -> T:
    kwargs: DictConfig = config.get("params", {})
    return constructor(**kwargs)


def get_instances(
    category: str,
    lib_check: type,
    configs: List[DictConfig],
    constructor: Callable[..., T],
) -> Dict[str, T]:
    return {
        entry["name"]: constructor(
            load_class(category, lib_check, entry["name"]), entry
        )
        for entry in configs
    }


def get_criterions(config: List[DictConfig]) -> Dict[str, nn.Module]:
    return get_instances("criterions", nn, config, create_instance)


def get_metrics(config: List[DictConfig]) -> Dict[str, metrics.Metric]:
    return get_instances("metrics", metrics, config, create_instance)


def get_optimizers_and_schedulers(
    config: List[DictConfig], parameters: Iterable[nn.Parameter]
) -> List[OptimizerScheduler]:
    """
    Creates optimizer instances and their associated scheduler instances.
    Args:
    - config: A list of dictionaries, where each dictionary defines an optimizer and optionally its scheduler.
    - parameters: The model parameters to optimize.
    Returns:
        A list of tuples, each containing 'optimizer' and optional 'scheduler' pairs.
    """
    optimizer_scheduler_pairs: List[OptimizerScheduler] = []
    for entry in config:
        optimizer_name = entry["name"]
        optimizer_cls: type[optim.Optimizer] = load_class(
            "optimizers", optim, optimizer_name
        )
        optimizer_params = entry.get("params", {})

        optimizer = optimizer_cls(parameters, **optimizer_params)
        scheduler: optim.lr_scheduler._LRScheduler | None = None

        scheduler_config: DictConfig | None = entry.get("scheduler")
        if scheduler_config:
            scheduler_name = scheduler_config["name"]
            scheduler_cls: type[optim.lr_scheduler._LRScheduler] = load_class(
                "schedulers", optim.lr_scheduler, scheduler_name
            )
            scheduler_params = scheduler_config.get("params", {})
            scheduler = scheduler_cls(optimizer, **scheduler_params)

        optimizer_scheduler_pairs.append((optimizer, scheduler))
    return optimizer_scheduler_pairs
