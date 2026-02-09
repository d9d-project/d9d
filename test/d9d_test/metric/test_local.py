import copy
import dataclasses
from collections.abc import Callable
from typing import Any

import pytest
import torch
from d9d.core.types import TensorTree
from d9d.metric import Metric
from d9d.metric.impl import ComposeMetric, WeightedMeanMetric
from torch.testing import assert_close
from torch.utils._pytree import tree_map  # noqa: PLC2701


@dataclasses.dataclass
class MetricStep:
    expect: TensorTree
    args: tuple[Any, ...] | None = None
    kwargs: dict[str, Any] | None = None


@dataclasses.dataclass
class MetricCase:
    factory_fn: Callable[[], Metric]
    initial_expect: TensorTree
    steps: list[MetricStep]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        # Case: Default WM
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    args=(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0])),
                    expect=torch.tensor(1.5)
                ),
                MetricStep(
                    args=(torch.tensor(4.0), torch.tensor(0.0)),
                    expect=torch.tensor(1.5)
                ),
            ]
        ),
        # Case: Zero weights (Masking/Padding scenario) WM
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    # Initial state: 10/1 = 10
                    args=(torch.tensor([10.0]), torch.tensor([1.0])),
                    expect=torch.tensor(10.0)
                ),
                MetricStep(
                    # Update with weight 0. Should ignore value 500 completely.
                    # State: 10/1 = 10
                    args=(torch.tensor([500.0]), torch.tensor([0.0])),
                    expect=torch.tensor(10.0)
                ),
                MetricStep(
                    # Valid update.
                    # State: (10 + 20) / (1 + 1) = 15
                    args=(torch.tensor([20.0]), torch.tensor([1.0])),
                    expect=torch.tensor(15.0)
                ),
            ]
        ),

        # Case: Multi-dimensional tensors (Batch handling) WM
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    # 2x2 Batch. Sum=10, WeightSum=4 -> 2.5
                    args=(
                            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                            torch.tensor([[1.0, 1.0], [1.0, 1.0]])
                    ),
                    expect=torch.tensor(2.5)
                ),
                MetricStep(
                    # Add heavy weighted items.
                    # Previous Num: 10, Denom: 4.
                    # Update Num: (10*8) + (10*8) = 160. Update Denom: 8+8 = 16.
                    # Total Num: 170. Total Denom: 20. Result: 8.5
                    args=(
                            torch.tensor([10.0, 10.0]),
                            torch.tensor([8.0, 8.0])
                    ),
                    expect=torch.tensor(8.5)
                )
            ]
        ),

        # Case: Empty/Zero-sized tensors (Edge case logic) WM
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    # Update with empty tensors -> No change (remains NaN)
                    args=(torch.tensor([]), torch.tensor([])),
                    expect=torch.tensor(torch.nan)
                ),
                MetricStep(
                    # First valid update
                    args=(torch.tensor([5.0]), torch.tensor([1.0])),
                    expect=torch.tensor(5.0)
                ),
            ]
        ),
    ]
)
@pytest.mark.local
def test_metrics(device: str, case: MetricCase):
    # do 2 roundabouts to test reset works
    metric = case.factory_fn()
    metric.to(device)
    for repeat_i in range(4):
        init_expect = tree_map(lambda x: x.to(device), case.initial_expect)
        assert_close(metric.compute(), init_expect, equal_nan=True)
        for step in case.steps:
            if step.args is None:
                args = []
            else:
                args = tree_map(lambda x: x.to(device), step.args)
            if step.kwargs is None:
                kwargs = {}
            else:
                kwargs = tree_map(lambda x: x.to(device), step.kwargs)
            metric.update(*args, **kwargs)

            if repeat_i % 2 == 0:
                state = metric.state_dict()
                metric = case.factory_fn()
                metric.load_state_dict(state)

            expect = tree_map(lambda x: x.to(device), step.expect)
            assert_close(metric.compute(), expect, equal_nan=True)
        metric.reset()


@pytest.mark.local
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compose_metric(device: str):
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
    mean_a.update(
        torch.tensor([10.0], device=device),
        torch.tensor([1.0], device=device)
    )
    mean_b.update(
        torch.tensor([20.0], device=device),
        torch.tensor([2.0], device=device)
    )

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
