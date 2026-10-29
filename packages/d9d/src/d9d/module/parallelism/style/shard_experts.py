from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import ParallelStyle

from d9d.module.block.moe import GroupedLinear, MoELayer


class ShardMoESparseExpertsParallel(ParallelStyle):
    """Parallel style that shards the experts of an ``MoELayer`` along one mesh dimension.

    Every ``GroupedLinear`` weight in the layer is sharded along the expert dimension on the shard
    mesh dimension and replicated on the other mesh dimensions. If the shard dimension has more than
    one rank, the style also enables the layer's distributed token dispatch over that dimension.
    """

    def __init__(self, shard_dim_name: str):
        """Constructs the ``ShardMoESparseExpertsParallel`` object.

        Args:
            shard_dim_name: The name of the mesh dimension to shard the experts along.
        """
        self._shard_dim_name = shard_dim_name

    def _partition_experts(self, module_name: str, mod: nn.Module, device_mesh: DeviceMesh):
        if not isinstance(mod, GroupedLinear):
            raise TypeError(f"module ({type(mod).__name__}) must be a GroupedLinear for expert sharding.")

        mesh_dim_names = device_mesh.mesh_dim_names

        if mesh_dim_names is None:
            raise ValueError("Expert sharding requires a device mesh with named dimensions.")

        placements = [Shard(0) if dim_name == self._shard_dim_name else Replicate() for dim_name in mesh_dim_names]
        weight = nn.Parameter(
            distribute_tensor(mod.weight, device_mesh, placements), requires_grad=mod.weight.requires_grad
        )
        mod.weight = weight

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, MoELayer):
            raise TypeError(f"module ({type(module).__name__}) must be a MoELayer for ShardMoESparseExpertsParallel.")

        if device_mesh[self._shard_dim_name].size() > 1:
            module.enable_distributed_communicator(device_mesh.get_group(self._shard_dim_name))

        for submod in module.modules():
            if isinstance(submod, GroupedLinear):
                distribute_module(submod, device_mesh, self._partition_experts)

        return module
