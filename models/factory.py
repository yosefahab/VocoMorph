import os
from importlib import import_module
from typing import Any, Dict

import torch
from src.logging.logger import get_logger

logger = get_logger(__name__)


def create_model_instance(model_name: str, config: Dict[str, Any]) -> torch.nn.Module:
    """Load and instantiate a model by name."""
    try:
        # ensure the model file exists
        model_path = f"models/{model_name}/model.py"
        if not os.path.exists(model_path):
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
