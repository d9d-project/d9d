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
    """
    Parallel style that shards MoE experts across a specific mesh dimension.

    This style is designed for ``MoELayer`` instances using ``GroupedLinear`` for experts.
    It splits the experts across the specified
    dimension of the device mesh (Expert Parallelism). Other dimensions in the
    mesh treat the parameters as Replicated.

    It also initializes the necessary distributed communication groups within the
    MoE layer to handle token dispatching.
    """

    def __init__(self, shard_dim_name: str):
        self._shard_dim_name = shard_dim_name

    def _partition_experts(self, module_name: str, mod: nn.Module, device_mesh: DeviceMesh):
        if not isinstance(mod, GroupedLinear):
            raise TypeError("This plan should be applied only on GroupedLinear")

        mesh_dim_names = device_mesh.mesh_dim_names

        if mesh_dim_names is None:
            raise ValueError("This plan should be applied only on named DeviceMeshes")

        placements = [
            Shard(0) if dim_name == self._shard_dim_name else Replicate()
            for dim_name
            in mesh_dim_names
        ]
        weight = nn.Parameter(
            distribute_tensor(mod.weight, device_mesh, placements),
            requires_grad=mod.weight.requires_grad
        )
        mod.weight = weight

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, MoELayer):
            raise TypeError("This plan should be applied only on MoELayer")

        module.enable_distributed_communicator(device_mesh.get_group(self._shard_dim_name))

        for submod in module.modules():
            if isinstance(submod, GroupedLinear):
                distribute_module(submod, device_mesh, self._partition_experts)

        return module
