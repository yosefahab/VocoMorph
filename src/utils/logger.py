import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List

# define color mappings for log levels
# fmt: off
LOG_COLORS = {
    "DEBUG": "\033[34m",        # Blue
    "INFO": "\033[32m",         # Green
    "WARNING": "\033[33m",      # Yellow
    "ERROR": "\033[31m",        # Red
    "CRITICAL": "\033[91m",     # Bright Red
    "RESET": "\033[0m",
}
# fmt: on


class ColorFormatter(logging.Formatter):
    """Custom log formatter that adds colors based on log level."""

    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, LOG_COLORS["RESET"])
        reset = LOG_COLORS["RESET"]
        log_message = super().format(record)
        return f"{log_color}{log_message}{reset}"


def setup_logging(
    log_dir: Path,
    log_level=logging.INFO,
    log_to_file=True,
    max_file_size=5 * 1024 * 1024,  # 5MB
    backup_count=3,
):
    """
    Sets up logging for the application.
    Args:
    - log_level: The logging level.
    - log_to_file: Whether to log to a file.
    - max_file_size: Max log file size in bytes before rotating (defaults to 5MB).
    - backup_count: Number of backup log files to keep.
    """

    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        ColorFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    )
    handlers: List[logging.Handler] = [console_handler]

    if log_to_file:
        file_handler = RotatingFileHandler(
            log_dir.joinpath("logs.log"),
            maxBytes=max_file_size,
            backupCount=backup_count,
        )
        file_handler.setFormatter(
            logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        )
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)


def get_logger(module_name: str):
    """Returns a logger instance for a given module name."""
    return logging.getLogger(module_name)
