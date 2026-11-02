from typing import Any

from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard


def _force_fsdp_grad_reduction_policy(module: FSDPModule):
    module.set_force_sum_reduction_for_comms(enable=True)
    module.set_gradient_divide_factor(1.0)
    module.set_requires_all_reduce(False)


def parallelize_fsdp(module: nn.Module, mesh: DeviceMesh, *args: Any, **kwargs: Any):
    """Applies Fully Sharded Data Parallel (FSDP) with gradient summation.

    The module is sharded with PyTorch's ``fully_shard`` over ``mesh``. Unlike default FSDP, gradients
    are summed across the mesh instead of averaged, and FSDP does not all-reduce them across
    replicas. d9d normalizes and reduces gradients across replicas itself.

    Args:
        module: The module to shard.
        mesh: The device mesh over which to shard the module.
        *args: Additional positional arguments passed to ``fully_shard``.
        **kwargs: Additional keyword arguments passed to ``fully_shard``.

    Raises:
        ValueError: If the mesh does not have exactly one dimension.
        RuntimeError: If ``fully_shard`` did not convert the module to an ``FSDPModule``.
    """
    if mesh.ndim != 1:
        raise ValueError(
            f"mesh.ndim ({mesh.ndim}) must be 1 for FSDP. "
            f"For HSDP, use parallelize_hsdp() or apply parallelize_replicate() to the other dimensions first."
        )

    fully_shard(module, *args, mesh=mesh, **kwargs)
    if not isinstance(module, FSDPModule):
        raise RuntimeError(f"fully_shard() did not convert the module ({type(module).__name__}) to FSDPModule.")
    _force_fsdp_grad_reduction_policy(module)
