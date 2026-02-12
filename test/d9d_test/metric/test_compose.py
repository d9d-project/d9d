import copy

import pytest
import torch
from d9d.core.dist_context import FLAT_DOMAIN, DeviceMeshParameters
from d9d.metric.impl import ComposeMetric, WeightedMeanMetric
from torch.testing import assert_close
from torch.utils._pytree import tree_map  # noqa: PLC2701


@pytest.mark.local
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_local(device: str):
    expect_device = torch.scalar_tensor(0, device=device).device

    mean_a = WeightedMeanMetric()
    mean_b = WeightedMeanMetric()

    metric = ComposeMetric({"a": mean_a, "b": mean_b})

    # to() propagation
    metric.to(device)

    assert mean_a.accumulated_weight.device == expect_device
    assert mean_b.accumulated_weight.device == expect_device

    # cannot update directly
    with pytest.raises(ValueError, match="Cannot update ComposeMetric directly"):
        metric.update(1, 2)

    # update and compute propagation
    mean_a.update(torch.tensor([10.0], device=device), torch.tensor([1.0], device=device))
    mean_b.update(torch.tensor([20.0], device=device), torch.tensor([2.0], device=device))

    expect_state = {"a": torch.tensor(10.0, device=device), "b": torch.tensor(20.0, device=device)}

    assert_close(metric.compute(), expect_state)

    # prepare state
    old_state = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x), metric.state_dict())

    # reset propagation
    metric.reset()
    assert_close(mean_a.accumulated_weight, torch.tensor(0.0, device=device))
    assert_close(mean_b.accumulated_weight, torch.tensor(0.0, device=device))

    # state propagation
    metric.load_state_dict(old_state)
    assert_close(metric.compute(), expect_state)


@pytest.mark.distributed
def test_distributed(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    device = dist_ctx.current_device
    rank = dist_ctx.mesh_for(FLAT_DOMAIN).get_rank()

    # Setup
    expect_device = torch.scalar_tensor(0, device=device).device
    mean_a = WeightedMeanMetric()
    mean_b = WeightedMeanMetric()
    metric = ComposeMetric({"a": mean_a, "b": mean_b})

    # to() propagation
    metric.to(device)
    assert mean_a.accumulated_weight.device == expect_device
    assert mean_b.accumulated_weight.device == expect_device

    # Validation logic (cannot update directly)
    with pytest.raises(ValueError, match="Cannot update ComposeMetric directly"):
        metric.update(1, 2)

    # Update Children (Rank Dependent Data)
    # Child A: Rank i adds value i, weight 1 -> Global Mean = 3.5 (for 8 ranks)
    mean_a.update(torch.tensor([float(rank)], device=device), torch.tensor([1.0], device=device))
    # Child B: Rank i adds value 2*i, weight 1 -> Global Mean = 7.0 (for 8 ranks)
    mean_b.update(torch.tensor([float(rank) * 2], device=device), torch.tensor([1.0], device=device))

    # Verify ComposeMetric delegates sync triggers to children
    metric.sync(dist_ctx)

    # Compute Check (Global Aggregation)
    expect_state = {"a": torch.tensor(3.5, device=device), "b": torch.tensor(7.0, device=device)}
    assert_close(metric.compute(), expect_state)

    # State Dict / Reset / Load
    # Save local state (contains specific rank data, e.g. Rank 0 has val=0)
    old_state = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x), metric.state_dict())

    # Reset propagation
    metric.reset()
    assert_close(mean_a.accumulated_weight, torch.tensor(0.0, device=device))
    assert_close(mean_b.accumulated_weight, torch.tensor(0.0, device=device))

    # Restore state
    metric.load_state_dict(old_state)

    # We must sync again after loading state to re-populate global synced buffers
    metric.sync(dist_ctx)

    assert_close(metric.compute(), expect_state)
