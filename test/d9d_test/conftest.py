import pytest
import torch


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(123)
