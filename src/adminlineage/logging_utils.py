"""Logging helpers for consistent structured logs."""

from __future__ import annotations

import logging
from pathlib import Path

from .utils import ensure_dir


def setup_logger(run_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Create logger that writes to console and run file."""

    run_path = ensure_dir(run_dir)
    logger_name = f"adminlineage.{run_path.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_path / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
