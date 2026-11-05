from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard

from .placement import check_data_shard_dim, distribute_parameters


def parallelize_fsdp(module: nn.Module, mesh: DeviceMesh):
    """Shards the parameters of a module for Fully Sharded Data Parallel (FSDP).

    Each parameter becomes a ``DTensor`` with ``Shard(0)`` over ``mesh``. The ``ParameterRuntime`` gathers the
    parameters before the segments that use them and reduce-scatters their gradients.

    Args:
        module: The module to shard.
        mesh: The device mesh to shard over. Its only dimension must be in ``DATA_SHARD_DIMS``.

    Raises:
        ValueError: If the mesh does not have exactly one named dimension, or the dimension is not a data shard
            dimension.
    """
    names = mesh.mesh_dim_names
    if mesh.ndim != 1 or names is None:
        raise ValueError(
            f"mesh.ndim ({mesh.ndim}) must be 1, with a named dimension, for FSDP. For HSDP, use parallelize_hsdp()."
        )
    check_data_shard_dim(names[0])
    distribute_parameters(module, mesh, (Shard(0),))
