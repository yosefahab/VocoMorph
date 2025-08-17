from importlib import import_module
from typing import Any, Callable, Dict, List, TypeVar

import mlx.nn as nn
import mlx.optimizers.optimizers as optim
import mlx.optimizers.schedulers as sched
import torchmetrics as tm

from src.utils.logger import get_logger
from src.utils.types import DictConfig, T

logger = get_logger(__name__)


def load_class(category: str, lib_check: Any, name: str) -> Any:
    """Load a class from a module dynamically."""
    try:
        if hasattr(lib_check, name):
            logger.info(f"loading class {name} from {lib_check}")
            return getattr(lib_check, name)
        # Custom classes
        logger.info(f"loading custom class {name} from {category}")
        return getattr(import_module(f"src.trainer.custom.{category}"), name)
    except Exception as e:
        logger.exception(f"error loading class {name} from {lib_check}|{category}: {e}")
        exit(1)


def create_instance(constructor: Callable[..., T], config: DictConfig) -> T:
    """Creates an instance of a class with optional keyword arguments."""
    kwargs: DictConfig = config.get("params", {})
    return constructor(**kwargs)


def get_instances(
    category: str,
    lib_check: Any,
    configs: List[DictConfig],
    constructor: Callable[..., T],
) -> Dict[str, T]:
    """Dynamically loads and instantiates classes based on a list of configurations."""
    return {
        entry["name"]: constructor(
            load_class(category, lib_check, entry["name"]), entry
        )
        for entry in configs
    }


def get_criterions(config: List[DictConfig]) -> Dict[str, Any]:
    """
    Dynamically loads and instantiates all loss functions from the configuration.
    This version correctly handles both function-based and class-based losses.
    """
    result = {}
    for entry in config:
        loss_fn_or_cls = load_class("criterions", nn.losses, entry["name"])
        params = entry.get("params", {})

        # Check if the object is a class (by seeing if it has a __call__ method
        # and doesn't have a __dict__). This is a safe way to distinguish from
        # a function.
        if isinstance(loss_fn_or_cls, type):
            # It's a class, so we instantiate it.
            result[entry["name"]] = loss_fn_or_cls(**params)
        else:
            # It's a function, so we wrap it in a lambda to pass params.
            result[entry["name"]] = (
                lambda preds, targets, fn=loss_fn_or_cls, params=params: fn(
                    preds, targets, **params
                )
            )

    return result


def get_metrics(config: List[DictConfig]) -> Dict[str, tm.Metric]:
    return get_instances("metrics", tm, config, create_instance)


def get_optimizer(config: List[DictConfig]) -> optim.Optimizer:
    """
    Creates the MLX optimizer instance.
    """
    entry = config[0]
    optimizer_name = entry["name"]
    mlx_optimizer_cls: TypeVar = load_class("optimizers", optim, optimizer_name)
    optimizer_params = entry.get("params", {})
    scheduler_cfg = entry.get("scheduler")
    lr_schedule = get_scheduler(scheduler_cfg) if scheduler_cfg else None
    return mlx_optimizer_cls(**optimizer_params, learning_rate=lr_schedule)  # pyright: ignore[reportCallIssue]


def get_scheduler(config: DictConfig) -> Callable[[int], float]:
    """
    Creates a functional scheduler that computes the learning rate per step.
    """
    scheduler_name = config["name"]
    scheduler_cls = load_class("schedulers", sched, scheduler_name)
    scheduler_params = config.get("params", {})
    return scheduler_cls(**scheduler_params)
