import dataclasses
from collections.abc import Callable
from typing import Any

from d9d.core.dist_context import FLAT_DOMAIN, DeviceMeshParameters
from d9d.core.types import TensorTree
from d9d.metric import Metric
from torch.testing import assert_close
from torch.utils._pytree import tree_map  # noqa: PLC2701


@dataclasses.dataclass
class MetricParams:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@dataclasses.dataclass
class MetricStep:
    params_per_rank: list[MetricParams]
    expect: TensorTree


@dataclasses.dataclass
class MetricCase:
    factory_fn: Callable[[], Metric]
    initial_expect: TensorTree
    steps: list[MetricStep]


def assert_metric_local(case: MetricCase, device: str, atol: float | None = None, rtol: float | None = None):
    metric = case.factory_fn()
    metric.to(device)
    for repeat_i in range(4):
        init_expect = tree_map(lambda x: x.to(device), case.initial_expect)
        assert_close(metric.compute(), init_expect, equal_nan=True, atol=atol, rtol=rtol)
        for step in case.steps:
            params = step.params_per_rank[0]
            if params.args is None:
                args = []
            else:
                args = tree_map(lambda x: x.to(device), params.args)
            if params.kwargs is None:
                kwargs = {}
            else:
                kwargs = tree_map(lambda x: x.to(device), params.kwargs)
            metric.update(*args, **kwargs)

            if repeat_i % 2 == 0:
                state = metric.state_dict()
                metric = case.factory_fn()
                metric.to(device)
                metric.load_state_dict(state)

            expect = tree_map(lambda x: x.to(device), step.expect)
            assert_close(metric.compute(), expect, equal_nan=True, atol=atol, rtol=rtol)
        metric.reset()


def assert_metric_distributed(
        dist_ctx_factory, case: MetricCase,
        atol: float | None = None,
        rtol: float | None = None
):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    this_rank = dist_ctx.mesh_for(FLAT_DOMAIN).get_rank()
    # do 2 roundabouts to test reset works
    metric = case.factory_fn()
    metric.to("cuda")
    for repeat_i in range(4):
        init_expect = tree_map(lambda x: x.to("cuda"), case.initial_expect)
        assert_close(metric.compute(), init_expect, equal_nan=True, atol=atol, rtol=rtol)
        for step in case.steps:
            params = step.params_per_rank[this_rank]
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
                metric.to("cuda")
                metric.load_state_dict(state)

            metric.sync(dist_ctx)

            expect = tree_map(lambda x: x.to("cuda"), step.expect)
            assert_close(metric.compute(), expect, equal_nan=True, atol=atol, rtol=rtol)
        metric.reset()
