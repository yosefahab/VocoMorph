import os
from pathlib import Path

from .logger import setup_logging

setup_logging(Path(os.environ["PROJECT_ROOT"]).joinpath("logs"))
