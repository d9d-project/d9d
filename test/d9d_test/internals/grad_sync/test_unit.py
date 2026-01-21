import pytest
import torch
from d9d.internals.grad_sync.bucket import AccumulationCounter
from torch import nn


@pytest.mark.local
def test_accumulation_counter():
    p1 = nn.Parameter(torch.empty(1))
    p2 = nn.Parameter(torch.empty(1))

    counter = AccumulationCounter(require_accumulations=2, parameters=[p1, p2])

    assert not counter.is_ready()

    counter.update(p1)
    assert not counter.is_ready()

    counter.update(p2)
    assert not counter.is_ready()

    counter.update(p1)
    assert not counter.is_ready()

    counter.update(p2)
    assert counter.is_ready()

    counter.reset()
    assert not counter.is_ready()
    counter.update(p1)
    assert not counter.is_ready()
