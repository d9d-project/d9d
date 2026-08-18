from d9d.core.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.module.block.head import ClassificationHead, EmbeddingHead, SplitLanguageModellingHead
from d9d.module.parallelism.api import parallelize_hsdp


def parallelize_causal_lm_head(head: SplitLanguageModellingHead, dist_context: DistributedContext) -> None:
    """Applies Hybrid Sharded Data Parallelism (HSDP) to a causal language modeling head.

    Args:
        head: The language modeling head to parallelize.
        dist_context: The distributed context containing device meshes and topology info.
    """
    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)
    parallelize_hsdp(head, mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])


def parallelize_classification_head(head: ClassificationHead, dist_context: DistributedContext) -> None:
    """Applies Hybrid Sharded Data Parallelism (HSDP) to a classification head.

    Args:
        head: The classification head to parallelize.
        dist_context: The distributed context containing device meshes and topology info.
    """
    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)
    parallelize_hsdp(head, mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])


def parallelize_embedding_head(head: EmbeddingHead, dist_context: DistributedContext) -> None:
    """Applies Hybrid Sharded Data Parallelism (HSDP) to an embedding head.

    Args:
        head: The embedding head to parallelize.
        dist_context: The distributed context containing device meshes and topology info.
    """
    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)
    parallelize_hsdp(head, mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])
