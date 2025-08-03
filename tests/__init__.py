import os
import sys
from pathlib import Path

if os.environ.get("PROJECT_ROOT", None) is None:
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
    sys.path.append(str(PROJECT_ROOT))  # ensure src/ is accessible

if os.environ.get("DATA_ROOT", None) is None:
    DATA_ROOT = Path(os.environ["PROJECT_ROOT"]).joinpath("data")
    os.environ["DATA_ROOT"] = str(DATA_ROOT)
