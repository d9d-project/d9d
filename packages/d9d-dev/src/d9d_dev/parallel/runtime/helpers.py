from collections.abc import Callable

import torch
from d9d.dist_context import DENSE_DOMAIN, DenseDim, DenseDomainConfig, DeviceMeshParameters, DistributedContext
from d9d.parallel.runtime import BoundParameters, ParameterRuntime
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard, distribute_tensor

# FSDP, DDP and HSDP.
MESHES = [
    DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8)),
    DeviceMeshParameters(data_parallel=8),
    DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=4)),
]


def distribute_parameters(module: nn.Module, mesh: DeviceMesh, placements: tuple[Placement, ...]):
    """Turns the parameters of a module into ``DTensor`` objects without hooks. A tied parameter stays one."""
    distributed: dict[int, nn.Parameter] = {}
    for submodule in module.modules():
        for name, param in list(submodule.named_parameters(recurse=False, remove_duplicate=False)):
            if id(param) not in distributed:
                data = distribute_tensor(param.data, mesh, placements)
                distributed[id(param)] = nn.Parameter(data, requires_grad=param.requires_grad)
            submodule.register_parameter(name, distributed[id(param)])


def place_hsdp(module: nn.Module, dist_ctx: DistributedContext):
    """Places the parameters of a module for HSDP on the dense mesh, as a ``ParameterRuntime`` expects them."""
    mesh = dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_REPLICATE, DenseDim.DP_CP_SHARD, DenseDim.CP_REPLICATE]
    dims = tuple(name for name in mesh.mesh_dim_names or () if mesh[name].size() > 1)
    if dims:
        placements = tuple(Shard(0) if name == DenseDim.DP_CP_SHARD else Replicate() for name in dims)
        distribute_parameters(module, mesh[dims], placements)


def make_runtime(training: bool = True) -> ParameterRuntime:
    """Builds a runtime that gathers along the data shard dimension of the dense mesh."""
    return ParameterRuntime(gather_dims={DenseDim.DP_CP_SHARD}, bucket_size_mib=32, training=training)


def run_one_region(bound: BoundParameters, module: nn.Module, closure: Callable[[], None]):
    """Runs the forward and backward pass of a module without other groups and finishes the gradients of the step."""
    parameters = bound.parameters(module)
    parameters.record_order((module,))
    with parameters.backward(final=True).region(0):
        closure()
    bound.wait()


def local_part(param: nn.Parameter, full: torch.Tensor) -> torch.Tensor:
    """Returns the part of a full tensor that the rank holds for a parameter."""
    if isinstance(param, DTensor):
        return distribute_tensor(full, param.device_mesh, param.placements).to_local()
    return full


def local_grad(param: nn.Parameter) -> torch.Tensor:
    """Returns the local gradient of a parameter."""
    grad = param.grad
    assert grad is not None
    return grad.to_local() if isinstance(grad, DTensor) else grad
