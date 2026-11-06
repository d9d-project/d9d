from collections.abc import Callable
from typing import Any

from d9d.dist_context import DATA_SHARD_DIMS, DENSE_DOMAIN, DeviceMeshParameters, DistributedContext
from d9d.parallel import distribute_parameters
from d9d.parallel.runtime import ParameterRuntime
from torch import nn
from torch.distributed.tensor import Replicate, Shard

MESHES = [
    DeviceMeshParameters(data_parallel_shard=8),
    DeviceMeshParameters(data_parallel_replicate=8),
    DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4),
]


def place_hsdp(module: nn.Module, dist_ctx: DistributedContext):
    """Places the parameters of a module for HSDP on the dense mesh, as a ParameterRuntime expects them."""
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate", "dp_cp_shard", "cp_replicate"]
    dims = tuple(name for name in mesh.mesh_dim_names or () if mesh[name].size() > 1)
    if dims:
        distribute_parameters(module, mesh[dims], tuple(Shard(0) if x == "dp_cp_shard" else Replicate() for x in dims))


def make_runtime(bucket_size_mib: int = 32, **kwargs: Any) -> ParameterRuntime:
    """Builds a runtime that gathers along the data shard dimensions of d9d meshes."""
    return ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mib=bucket_size_mib, **kwargs)


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
            runtime.reduce(plan, bucket_id)
    runtime.wait()
    return runtime
