import dataclasses
from collections.abc import Callable
from typing import Any

import pytest
import torch
from d9d.core.dist_context import FLAT_DOMAIN, DeviceMeshParameters
from d9d.core.types import TensorTree
from d9d.metric import Metric
from d9d.metric.impl import ComposeMetric, WeightedMeanMetric
from torch.testing import assert_close
from torch.utils._pytree import tree_map  # noqa: PLC2701


@dataclasses.dataclass
class ParamsPerRank:
    args: tuple[Any, ...] | None = None
    kwargs: dict[str, Any] | None = None


@dataclasses.dataclass
class MetricStep:
    expect: TensorTree
    params: list[ParamsPerRank]


@dataclasses.dataclass
class MetricCase:
    factory_fn: Callable[[], Metric]
    initial_expect: TensorTree
    steps: list[MetricStep]


@pytest.mark.parametrize(
    "case",
    [
        # Case: Default WM (Homogeneous updates then zero-weight updates)
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    expect=torch.tensor(1.5),
                    params=[
                        ParamsPerRank(args=(torch.tensor(1.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(1.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(1.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(1.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(2.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(2.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(2.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(2.0), torch.tensor(1.0))),
                    ]
                ),
                MetricStep(
                    expect=torch.tensor(1.5),
                    params=[
                        ParamsPerRank(args=(torch.tensor(11342.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(1231.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(123123123.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(12.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(12.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(121.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(1212.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(2121.0), torch.tensor(0.0))),
                    ],
                ),
            ]
        ),
        # Case: Ranks with uneven data (simulating data imbalance or masking)
        # Ranks 0-3 provide Value 10, Weight 1.
        # Ranks 4-7 provide Value 100, Weight 0 (should be ignored).
        # Expected Mean: 10.0
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    expect=torch.tensor(10.0),
                    params=[
                        # Active Ranks
                        ParamsPerRank(args=(torch.tensor(10.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(10.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(10.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(10.0), torch.tensor(1.0))),
                        # Inactive Ranks (weights=0)
                        ParamsPerRank(args=(torch.tensor(100.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(100.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(100.0), torch.tensor(0.0))),
                        ParamsPerRank(args=(torch.tensor(100.0), torch.tensor(0.0))),
                    ]
                )
            ]
        ),
        # Case: Step-wise accumulation
        # Step 1: All ranks contribute 0. Mean 0.
        # Step 2: All ranks contribute 20. Global state: sum_val=(0*8 + 20*8)=160, sum_w=(8+8)=16. Mean=10.
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    expect=torch.tensor(0.0),
                    params=[ParamsPerRank(args=(torch.tensor(0.0), torch.tensor(1.0))) for _ in range(8)]
                ),
                MetricStep(
                    expect=torch.tensor(10.0),
                    params=[ParamsPerRank(args=(torch.tensor(20.0), torch.tensor(1.0))) for _ in range(8)]
                )
            ]
        ),
        # Case: Rank-dependent values (Verification of actual reduction)
        # Rank i contributes Value=i, Weight=1.
        # Sum = 0+1+2+3+4+5+6+7 = 28. Total Weight = 8. Mean = 3.5.
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    expect=torch.tensor(3.5),
                    params=[
                        ParamsPerRank(args=(torch.tensor(0.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(1.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(2.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(3.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(4.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(5.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(6.0), torch.tensor(1.0))),
                        ParamsPerRank(args=(torch.tensor(7.0), torch.tensor(1.0))),
                    ]
                )
            ]
        )
    ]
)
@pytest.mark.distributed
def test_metrics(dist_ctx_factory, case: MetricCase):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    this_rank = dist_ctx.mesh_for(FLAT_DOMAIN).get_rank()
    # do 2 roundabouts to test reset works
    metric = case.factory_fn()
    metric.to("cuda")
    for repeat_i in range(4):
        init_expect = tree_map(lambda x: x.to("cuda"), case.initial_expect)
        assert_close(metric.compute(), init_expect, equal_nan=True)
        for step in case.steps:
            params = step.params[this_rank]
            if params.args is None:
                args = []
            else:
                args = tree_map(lambda x: x.to("cuda"), params.args)
            if params.kwargs is None:
                kwargs = {}
            else:
                kwargs = tree_map(lambda x: x.to("cuda"), params.kwargs)
            metric.update(*args, **kwargs)

            if repeat_i % 2 == 0:
                state = metric.state_dict()
                metric = case.factory_fn()
                metric.load_state_dict(state)

            metric.trigger_sync(dist_ctx)
            metric.wait_sync(dist_ctx)

            expect = tree_map(lambda x: x.to("cuda"), step.expect)
            assert_close(metric.compute(), expect, equal_nan=True)
        metric.reset()


@pytest.mark.distributed
def test_compose_metric_distributed(dist_ctx_factory):
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
    mean_a.update(
        torch.tensor([float(rank)], device=device),
        torch.tensor([1.0], device=device)
    )
    # Child B: Rank i adds value 2*i, weight 1 -> Global Mean = 7.0 (for 8 ranks)
    mean_b.update(
        torch.tensor([float(rank) * 2], device=device),
        torch.tensor([1.0], device=device)
    )

    # Verify ComposeMetric delegates sync triggers to children
    metric.trigger_sync(dist_ctx)
    metric.wait_sync(dist_ctx)

    # Compute Check (Global Aggregation)
    expect_state = {
        "a": torch.tensor(3.5, device=device),
        "b": torch.tensor(7.0, device=device)
    }
    assert_close(metric.compute(), expect_state)

    # State Dict / Reset / Load
    # Save local state (contains specific rank data, e.g. Rank 0 has val=0)
    old_state = tree_map(lambda x: x.clone(), metric.state_dict())

    # Reset propagation
    metric.reset()
    assert_close(mean_a.accumulated_weight, torch.tensor(0.0, device=device))
    assert_close(mean_b.accumulated_weight, torch.tensor(0.0, device=device))

    # Restore state
    metric.load_state_dict(old_state)

    # We must sync again after loading state to re-populate global synced buffers
    metric.trigger_sync(dist_ctx)
    metric.wait_sync(dist_ctx)

    assert_close(metric.compute(), expect_state)
