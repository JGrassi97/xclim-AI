# xclim_tools.utils.logging
# =================================================

# Logging utilities for XclimAI modules.
# Provides:
# - `get_logger`: standard logger with optional file output
# - `set_logger_level`: toggles between DEBUG and WARNING levels
# - `StreamToLogger`: redirects stdout/stderr streams to a logger

import logging
import io
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "XclimAI",
    level: Optional[int] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Initialize and return a logger instance.

    Parameters
    ----------
    name : str
        Logger name.
    level : int, optional
        Logging level (e.g., logging.DEBUG).
    log_file : str, optional
        If provided, logs will also be written to this file.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter("[%(levelname)s] %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
            file_handler = logging.FileHandler(path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    if level is not None:
        logger.setLevel(level)

    return logger


def set_logger_level(logger: logging.Logger, verbose: bool) -> None:
    """
    Set logging level based on verbosity flag.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance.
    verbose : bool
        If True, sets level to DEBUG; otherwise WARNING.
    """
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)


class StreamToLogger(io.StringIO):
    """
    Redirects writes from stdout/stderr to a logger.

    Attributes
    ----------
    logger : logging.Logger
        Target logger.
    level : int
        Logging level.
    """
    def __init__(self, logger: logging.Logger, level: int):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, buf: str) -> None:
        for line in buf.strip().splitlines():
            self.logger.log(self.level, line)

    def flush(self) -> None:
        pass  # No-op for compatibility
