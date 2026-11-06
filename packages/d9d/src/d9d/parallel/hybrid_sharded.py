from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard

from .placement import check_data_shard_dim, distribute_parameters


def parallelize_hsdp(module: nn.Module, mesh: DeviceMesh, shard_dim: str = "dp_cp_shard"):
    """Places the parameters of a module for Hybrid Sharded Data Parallel (HSDP).

    The parameters are sharded along ``shard_dim`` and replicated along every other mesh dimension. Dimensions of
    size 1 are skipped. The ``ParameterRuntime`` gathers the shards and reduces the gradients.

    Args:
        module: The module to parallelize.
        mesh: The device mesh to distribute the module over.
        shard_dim: The name of the mesh dimension to shard along. It must be in ``DATA_SHARD_DIMS``. All other
            dimensions are replication dimensions.

    Raises:
        ValueError: If the device mesh has no dimension names, if it has no ``shard_dim``, or if ``shard_dim`` is not a
            data shard dimension.
    """
    names = mesh.mesh_dim_names
    if names is None:
        raise ValueError("parallelize_hsdp() requires a device mesh with named dimensions.")
    check_data_shard_dim(shard_dim)
    if shard_dim not in names:
        # Without the dimension, the parameters would silently become full replicas.
        raise ValueError(f"The mesh ({names}) has no dimension ({shard_dim}). Pass the shard dimension of the mesh.")

    # A sub-mesh must keep the order of the mesh dimensions, so Shard(0) goes to the position of shard_dim.
    dims = tuple(x for x in names if mesh[x].size() > 1)
    if not dims:
        return
    placements = tuple(Shard(0) if x == shard_dim else Replicate() for x in dims)
    distribute_parameters(module, mesh[dims], placements)
