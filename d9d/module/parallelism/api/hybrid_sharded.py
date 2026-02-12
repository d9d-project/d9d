from typing import Any

from torch import nn
from torch.distributed import DeviceMesh

from .fully_sharded import parallelize_fsdp
from .replicate_parallel import parallelize_replicate


def parallelize_hsdp(
    module: nn.Module, mesh: DeviceMesh, shard_dim: str = "dp_cp_shard", *fsdp_args: Any, **fsdp_kwargs: Any
):
    """
    Applies Hybrid Sharded Data Parallelism (HSDP) to a module.

    This function decomposes the provided device mesh into sharding dimensions
    and replication dimensions. It applies replication parallelism
    across the replication dimensions and Fully Sharded Data Parallelism (FSDP)
    across the specified shard dimension.

    Args:
        module: The module to parallelize.
        mesh: The device mesh over which to distribute the module.
        shard_dim: The name of the mesh dimension used for FSDP sharding. Any
            dimension in the mesh not matching this name will be treated as a
            replication dimension.
        *fsdp_args: Positional arguments passed to the underlying FSDP parallelizer.
        **fsdp_kwargs: Keyword arguments passed to the underlying FSDP parallelizer.

    Raises:
        ValueError: If the device mesh does not have named dimensions.
    """

    replicate_dims = mesh.mesh_dim_names

    if replicate_dims is None:
        raise ValueError("Cannot use with unnamed device meshes")

    replicate_dims = tuple(x for x in replicate_dims if x != shard_dim and mesh[x].size() > 1)

    if len(replicate_dims) > 0:
        parallelize_replicate(module, mesh[replicate_dims])

    if mesh[shard_dim].size() != 1:
        parallelize_fsdp(module, mesh[shard_dim], *fsdp_args, **fsdp_kwargs)
