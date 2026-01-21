import dataclasses
from collections.abc import Callable
from typing import Any

import pytest
import torch
from d9d.core.dist_context import FLAT_DOMAIN
from d9d.core.types import TensorTree
from d9d.metric import Metric
from d9d.metric.impl import WeightedMeanMetric
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
def test_metrics(dist_ctx_dpr8, case: MetricCase):
    this_rank = dist_ctx_dpr8.mesh_for(FLAT_DOMAIN).get_rank()
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

            metric.trigger_sync(dist_ctx_dpr8)
            metric.wait_sync(dist_ctx_dpr8)

            expect = tree_map(lambda x: x.to("cuda"), step.expect)
            assert_close(metric.compute(), expect, equal_nan=True)
        metric.reset()
