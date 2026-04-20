"""Shared pytest fixtures."""
import pytest, torch, numpy as np
from seed import set_seed

@pytest.fixture(autouse=True)
def fix_seed(): set_seed(42)

@pytest.fixture
def small_batch():
    return torch.randn(8,6,64), torch.randint(0,4,(8,)), torch.randn(8,64)
