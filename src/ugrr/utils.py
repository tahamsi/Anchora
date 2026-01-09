"""Utility helpers for reproducible training and logging."""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict
from typing import Any, Dict

import numpy as np
import torch


def setup_logging(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger("ugrr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_config(config: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(config, "__dataclass_fields__"):
        payload = asdict(config)
    else:
        payload = dict(config)
    save_json(payload, path)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
