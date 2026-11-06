"""Horizontal parallelism for the Qwen3 Dense model."""

from d9d.dist_context import BASE_DOMAIN, DENSE_DOMAIN, BaseDim, DenseDim, DistributedContext
from d9d.parallel import parallelize_hsdp
from d9d.pipelining.api import PipelineStageInfo

from .model import Qwen3DenseModel


def parallelize_qwen3_dense_model(dist_context: DistributedContext, model: Qwen3DenseModel, stage: PipelineStageInfo):
    """Parallelizes a Qwen3 Dense backbone within one pipeline stage.

    Applies Hybrid Sharded Data Parallel (HSDP) to the embeddings, norms, attention and MLP
    modules. Tensor parallelism and context parallelism are not supported yet.

    Args:
        dist_context: The distributed context.
        model: The Qwen3 Dense backbone to parallelize.
        stage: The current pipeline stage.

    Raises:
        ValueError: If tensor parallelism or context parallelism is enabled.
    """
    if dist_context.dim_size(BASE_DOMAIN, BaseDim.TP) > 1:
        raise ValueError("Tensor parallelism is not supported for this model yet. Set tensor_parallel to 1.")
    if dist_context.dim_size(BASE_DOMAIN, BaseDim.CP) > 1:
        raise ValueError("Context parallelism is not supported for this model yet. Set context_parallel to 1.")

    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)[DenseDim.DP_REPLICATE, DenseDim.DP_CP_SHARD, DenseDim.CP_REPLICATE]

    if stage.is_current_stage_first:
        parallelize_hsdp(model.embed_tokens, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)

    if stage.is_current_stage_last:
        parallelize_hsdp(model.norm, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)

    for layer in model.layers.values():
        parallelize_hsdp(layer.mlp, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)

        parallelize_hsdp(layer.self_attn, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)
        parallelize_hsdp(layer.input_layernorm, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)
        parallelize_hsdp(layer.post_attention_layernorm, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)
