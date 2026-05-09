"""Shared utilities for the ASAG Research Framework."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed all random number generators for reproducibility.

    Seeds Python's built-in `random`, `numpy`, and `torch` (if available)
    from a single integer value. This ensures reproducible results across
    all framework components.

    Args:
        seed: Integer seed value. Must be non-negative.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch is optional; skip if not installed
