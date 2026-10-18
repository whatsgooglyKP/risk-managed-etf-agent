"""Consistent, human-readable logging to console + a daily rotating file."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("trading_agent")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"run_{datetime.now():%Y-%m-%d}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
