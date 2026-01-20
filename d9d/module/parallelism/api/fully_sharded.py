from typing import Any

from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard


def _force_fsdp_grad_reduction_policy(module: FSDPModule):
    module.set_force_sum_reduction_for_comms(enable=True)
    module.set_gradient_divide_factor(1.0)
    module.set_requires_all_reduce(False)


def parallelize_fsdp(
        module: nn.Module,
        mesh: DeviceMesh,
        *args: Any,
        **kwargs: Any
):
    """
    Applies Fully Sharded Data Parallel (FSDP) with forced gradient summation.

    This function wraps the provided module with PyTorch's ``fully_shard`` API using
    the specified device mesh. Unlike standard FSDP usage, this function explicitly
    configures the module to sum gradients across the mesh
    instead of averaging them and disables internal all-sum-reduce hooks.
    This is intended for d9d to handle gradient normalization and reduction across replicas externally.

    Args:
        module: The module to shard.
        mesh: The device mesh over which to shard the module.
        *args: Additional positional arguments passed to ``fully_shard``.
        **kwargs: Additional keyword arguments passed to ``fully_shard``.
    """

    fully_shard(module, *args, mesh=mesh, **kwargs)
    if not isinstance(module, FSDPModule):
        raise RuntimeError("Torch FSDP did not convert the module into FSDPModule")
    _force_fsdp_grad_reduction_policy(module)
