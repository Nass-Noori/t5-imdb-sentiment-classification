"""Small shared utilities: reproducibility, config loading, device selection."""

import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set every RNG we touch so runs are reproducible.

    NOTE: full bitwise reproducibility on GPU also requires
    torch.use_deterministic_algorithms(True) and disabling cudnn benchmarking,
    which costs speed. We set the common seeds here, which is enough for
    "same results across runs on the same machine" in practice.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # yaml parses "1e-5" as a string in some parsers; make sure it's a float
    config["learning_rate"] = float(config["learning_rate"])
    config["warmup_ratio"] = float(config["warmup_ratio"])
    config["val_split"] = float(config["val_split"])
    config["grad_clip_norm"] = float(config["grad_clip_norm"])
    return config


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
