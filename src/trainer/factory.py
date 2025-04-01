from importlib import import_module
from typing import List, Any, Callable, Dict, Iterable

import torch
from torcheval import metrics

from src.logging.logger import get_logger

logger = get_logger(__name__)

def load_class(category: str, lib_check: type, name: str) -> type:
    """Load a class from a module dynamically."""
    try:
        if hasattr(lib_check, name):
            logger.info(f"Loading class {name} from {lib_check}")
            return getattr(lib_check, name)
        logger.info(f"Loading class {name} from {category}")
        return getattr(import_module(f"src.trainer.custom.{category}"), name)
    except Exception as e:
        logger.critical(e)
        logger.critical(f"Parameter details: {category}|{lib_check}|{name}")
        exit(1)

def create_instance(cls: type, name: str, config: Dict[str, Any], target: Any = None) -> Any:
    """Create an instance of a class with arguments from config."""
    arg_dict: Dict[str, Any] = config.get(name, {})
    return cls(target, **arg_dict) if target else cls(**arg_dict)

def get_instances(category: str, lib_check: type, instance_creator: Callable[[type, str], Any], config: Dict[str, Any]) -> List[Any]:
    """Create a list of instances from a given category."""
    return [
        instance_creator(load_class(category, lib_check, name), name)
        for name in config["name"]
    ]

def get_criterions(config: Dict[str, Any]) -> List[Any]:
    """Create a list of criterion instances (loss functions)."""
    return get_instances("criterions", torch.nn, lambda cls, name: create_instance(cls, name, config), config)

def get_optimizers(config: Dict[str, Any], parameters: Iterable[torch.nn.Parameter]) -> List[torch.optim.Optimizer]:
    """Create a list of optimizer instances."""
    return get_instances("optimizers", torch.optim, lambda cls, name: create_instance(cls, name, config, parameters), config)

def get_schedulers(config: Dict[str, Any], optimizers: List[torch.optim.Optimizer]) -> List[Any]:
    """Create a list of learning rate schedulers."""
    if len(optimizers) == 1 and len(config["name"]) > 1:
        optimizers = optimizers * len(config["name"])
    
    return [
        create_instance(load_class("schedulers", torch.optim.lr_scheduler, name), name, config, optimizer)
        for name, optimizer in zip(config["name"], optimizers)
    ]

def get_metrics(config: Dict[str, Any]) -> Dict[str, metrics.Metric]:
    """Return a dictionary of metric names to metric class instances."""
    return {
        name: create_instance(load_class("metrics", metrics, name), name, config)
        for name in config["name"]
    }
