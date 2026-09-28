import pytest
import torch
from d9d.internals.grad_sync.bucket import AccumulationCounter
from d9d.internals.grad_sync.synchronizer import GradientSynchronizer
from torch import nn


@pytest.mark.local
def test_accumulation_counter_requires_count_set():
    counter = AccumulationCounter(parameters=[nn.Parameter(torch.empty(1))])

    with pytest.raises(RuntimeError, match="was not set"):
        counter.is_ready()


@pytest.mark.local
def test_accumulation_counter():
    p1 = nn.Parameter(torch.empty(1))
    p2 = nn.Parameter(torch.empty(1))

    counter = AccumulationCounter(parameters=[p1, p2])
    counter.set_required_accumulations(2)

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


@pytest.mark.local
def test_synchronizer_wait_requires_bind():
    sync = GradientSynchronizer(param_groups=[[nn.Parameter(torch.empty(1))]], bucket_size_mb=1)

    with pytest.raises(ValueError, match="not bound"):
        sync.wait()

    sync.bind()
    sync.unbind()

    with pytest.raises(ValueError, match="not bound"):
        sync.wait()
