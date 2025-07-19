from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List

import torch
from torcheval import metrics

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_class(category: str, lib_check: type, name: str) -> type:
    """Load a class from a module dynamically."""
    try:
        if hasattr(lib_check, name):
            logger.info(f"Loading class {name} from {lib_check}")
            return getattr(lib_check, name)
        logger.info(f"Loading custom class {name} from {category}")
        return getattr(import_module(f"src.trainer.custom.{category}"), name)
    except Exception as e:
        logger.critical(e)
        logger.critical(f"Parameter details: {category}|{lib_check}|{name}")
        exit(1)


def create_instance(cls: type, config: Dict[str, Any], target: Any = None) -> Any:
    arg_dict: Dict[str, Any] = config.get("params", {})
    return cls(target, **arg_dict) if target else cls(**arg_dict)


def get_instances(
    category: str,
    lib_check: type,
    instance_creator: Callable[[type, Dict[str, Any], Any], Any],
    config_list: List[Dict[str, Any]],
    target: Any = None,
) -> Dict[str, Any]:
    return {
        entry["name"]: instance_creator(
            load_class(category, lib_check, entry["name"]), entry, target
        )
        for entry in config_list
    }


def get_criterions(config: Dict[str, Any]) -> Dict[str, Any]:
    return get_instances(
        "criterions",
        torch.nn,
        create_instance,
        config,
    )


def get_optimizers_and_schedulers(
    config: List[Dict[str, Any]], parameters: Iterable[torch.nn.Parameter]
) -> List[Dict[str, Any]]:
    """
    Creates optimizer instances and their associated scheduler instances.

    Args:
        config: A list of dictionaries, where each dictionary defines an optimizer
                and optionally its scheduler.
        parameters: The model parameters to optimize.

    Returns:
        A list of dictionaries, each containing 'optimizer' and an optional 'scheduler' key.
    """
    optimizer_scheduler_pairs = []
    for entry in config:
        optimizer_name = entry["name"]
        optimizer_cls = load_class("optimizers", torch.optim, optimizer_name)
        optimizer_params = entry.get("params", {})

        optimizer = optimizer_cls(parameters, **optimizer_params)
        scheduler = None

        scheduler_config = entry.get("scheduler")
        if scheduler_config:
            scheduler_name = scheduler_config["name"]
            scheduler_cls = load_class(
                "schedulers", torch.optim.lr_scheduler, scheduler_name
            )
            scheduler_params = scheduler_config.get("params", {})
            scheduler = scheduler_cls(optimizer, **scheduler_params)

        optimizer_scheduler_pairs.append(
            {"optimizer": optimizer, "scheduler": scheduler}
        )
    return optimizer_scheduler_pairs


def get_metrics(config: Dict[str, Any]) -> Dict[str, metrics.Metric]:
    return get_instances("metrics", metrics, create_instance, config)
