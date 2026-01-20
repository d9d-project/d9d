from torch.distributed import DeviceMesh
from torch.distributed.tensor import Partial, Replicate
from torch.distributed.tensor.parallel import parallelize_module

from d9d.module.block.moe import MoELayer
from d9d.module.parallelism.style import ShardMoESparseExpertsParallel, ToLocalParallel


def parallelize_expert_parallel(
        module: MoELayer,
        mesh_experts: DeviceMesh,
        expert_shard_dim: str = "ep_shard"
):
    """
    Applies Expert Parallelism to a MoE layer.

    This function configures the provided Mixture of Experts layer for distributed
    execution.

    It partitions the sparse experts across the specified dimension
    of the device mesh (Expert Parallelism) and replicates along other dims.

    Simultaneously, it configures the router to be fully replicated across
    the mesh.

    Args:
        module: The MoE layer instance to parallelize.
        mesh_experts: The device mesh containing the expert parallel resources.
        expert_shard_dim: The name of the mesh dimension where experts should be sharded.
    """

    parallelize_module(module, mesh_experts, ShardMoESparseExpertsParallel(shard_dim_name=expert_shard_dim))
    parallelize_module(module.router, mesh_experts, ToLocalParallel(
        param_placement=tuple(Replicate() for _ in range(mesh_experts.ndim)),
        grad_placement=tuple(Partial("sum") for _ in range(mesh_experts.ndim))
    ))
