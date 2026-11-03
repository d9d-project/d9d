from collections.abc import Callable
from typing import Any

from d9d.dist_context import DATA_SHARD_DIMS
from d9d.parallel.runtime import ParameterRuntime
from torch import nn


def make_runtime(bucket_size_mb: int = 32, **kwargs: Any) -> ParameterRuntime:
    """Builds a runtime that gathers along the data shard dimensions of d9d meshes."""
    return ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mb=bucket_size_mb, **kwargs)


def step_with_runtime(module: nn.Module, closure: Callable[[], None]) -> ParameterRuntime:
    """Runs a forward and backward pass of a module as one segment and reduces its gradients.

    The returned runtime stays bound, so the sharded parameters keep their gradients until ``unbind()``.
    """
    runtime = make_runtime()
    runtime.register_stage(0, module)
    runtime.bind()
    runtime.zero_grad()
    with runtime.segment_weights(0, 0, writes_grads=True):
        closure()
    plan = runtime.reduction_plan(0, final=True)
    for bucket_ids in plan.after_segment.values():
        for bucket_id in bucket_ids:
            runtime.reduce(bucket_id, plan.final)
    runtime.wait()
    return runtime
