import os
from .logger import setup_logging

setup_logging(os.environ["PROJECT_ROOT"])
