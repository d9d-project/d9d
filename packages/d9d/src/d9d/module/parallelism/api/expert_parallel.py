from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import parallelize_module

from d9d.module.block.moe import MoELayer
from d9d.module.parallelism.style import ShardMoESparseExpertsParallel, ToLocalParallel


def parallelize_expert_parallel(module: MoELayer, mesh_experts: DeviceMesh, expert_shard_dim: str = "ep_shard"):
    """Applies expert parallelism to an MoE layer.

    The sparse experts are sharded along ``expert_shard_dim`` and replicated along the other mesh
    dimensions. The router and the shared expert (if any) are replicated across the whole mesh.

    Args:
        module: The MoE layer to parallelize.
        mesh_experts: The device mesh to distribute the layer over.
        expert_shard_dim: The name of the mesh dimension to shard the experts along.
    """
    parallelize_module(module, mesh_experts, ShardMoESparseExpertsParallel(shard_dim_name=expert_shard_dim))
    parallelize_module(
        module.router,
        mesh_experts,
        ToLocalParallel(
            param_placement=tuple(Replicate() for _ in range(mesh_experts.ndim)),
            grad_placement=tuple(Replicate() for _ in range(mesh_experts.ndim)),
        ),
    )
    if module.shared_expert is not None:
        parallelize_module(
            module.shared_expert,
            mesh_experts,
            ToLocalParallel(
                param_placement=tuple(Replicate() for _ in range(mesh_experts.ndim)),
                grad_placement=tuple(Replicate() for _ in range(mesh_experts.ndim)),
            ),
        )
