from d9d.core.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.module.block.head import TaskHead
from d9d.module.parallelism.api import parallelize_hsdp


def parallelize_task_head(head: TaskHead, dist_context: DistributedContext) -> None:
    """Applies Hybrid Sharded Data Parallelism (HSDP) to a task head.

    Sharding is uniform across every built-in head — each is HSDP on the dense mesh — so this
    single function covers all heads. A head needing different sharding is the point to branch.

    Args:
        head: The task head to parallelize.
        dist_context: The distributed context containing device meshes and topology info.
    """
    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)
    parallelize_hsdp(head, mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])
