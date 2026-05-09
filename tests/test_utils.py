"""Tests for the seed-setting utility."""

import random

import numpy as np
import pytest

from src.utils import set_seed


def test_set_seed_python_random():
    """Seeding produces deterministic Python random output."""
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_numpy():
    """Seeding produces deterministic numpy random output."""
    set_seed(42)
    a = np.random.rand(5).tolist()
    set_seed(42)
    b = np.random.rand(5).tolist()
    assert a == b


def test_set_seed_negative_raises():
    """Negative seed values raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)


def test_set_seed_torch_if_available():
    """Seeding torch (if installed) produces deterministic output."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    set_seed(42)
    a = torch.rand(5).tolist()
    set_seed(42)
    b = torch.rand(5).tolist()
    assert a == b
