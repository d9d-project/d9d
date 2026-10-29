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


@pytest.mark.local
@pytest.mark.parametrize("require_accumulations", [0, -1])
def test_synchronizer_rejects_accumulation_count_below_one(require_accumulations):
    sync = GradientSynchronizer(param_groups=[[nn.Parameter(torch.empty(1))]], bucket_size_mb=1)

    with pytest.raises(ValueError, match="must be at least 1"):
        sync.set_required_accumulations(require_accumulations)


@pytest.mark.local
def test_synchronizer_rejects_second_bind():
    sync = GradientSynchronizer(param_groups=[[nn.Parameter(torch.empty(1))]], bucket_size_mb=1)
    sync.bind()

    with pytest.raises(ValueError, match="already bound"):
        sync.bind()

    sync.unbind()
    # A bind after unbind is allowed again, e.g. after offload and onload.
    sync.bind()
    sync.unbind()


@pytest.mark.local
@pytest.mark.parametrize("same_group", [True, False], ids=["same_group", "two_groups"])
def test_synchronizer_rejects_shared_parameter(same_group):
    param = nn.Parameter(torch.empty(1))
    param_groups = [[param, param]] if same_group else [[param], [param]]
    sync = GradientSynchronizer(param_groups=param_groups, bucket_size_mb=1)

    with pytest.raises(ValueError, match="more than once"):
        sync.bind()

    # The failed bind leaves the synchronizer unbound.
    with pytest.raises(ValueError, match="not bound"):
        sync.wait()
