"""Tiny logging helper so every module has a consistent logger."""
from __future__ import annotations

import logging
import os

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return the logger for ``name``; importing this module configures root logging (level from ``LOG_LEVEL``)."""
    return logging.getLogger(name)
