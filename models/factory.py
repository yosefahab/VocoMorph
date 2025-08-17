import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import mlx.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_model_instance(model_dir: Path, config: Dict[str, Any]) -> nn.Module:
    """Load and instantiate a model by name."""
    model_name = model_dir.name
    try:
        # ensure the model file exists
        model_path = Path(os.environ["PROJECT_ROOT"]).joinpath(model_dir, "model.py")
        if not model_path.exists():
            logger.critical(f"Model file not found: {model_path}")
            exit(1)

        # dynamically import the model module
        module_path = f"models.{model_name}.model"
        module = import_module(module_path)

        # ensure the class exists within the module
        if not hasattr(module, model_name):
            logger.critical(f"Module {module_path} does not define class {model_name}")
            exit(1)

        # retrieve the class
        model_class = getattr(module, model_name)

        # get config arguments and instantiate the model
        return model_class(config)

    except Exception as e:
        logger.exception(f"Error creating model {model_name}: {e}")
        exit(1)
