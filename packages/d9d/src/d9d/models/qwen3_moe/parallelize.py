"""Horizontal parallelism for the Qwen3 MoE model."""

from d9d.dist_context import BASE_DOMAIN, DENSE_DOMAIN, EXPERT_DOMAIN, BaseDim, DenseDim, DistributedContext, ExpertDim
from d9d.nn.moe import parallelize_expert_parallel
from d9d.parallel import parallelize_hsdp
from d9d.pipelining.api import PipelineStageInfo

from .model import Qwen3MoEModel


def parallelize_qwen3_moe_model(dist_context: DistributedContext, model: Qwen3MoEModel, stage: PipelineStageInfo):
    """Parallelizes a Qwen3 MoE backbone within one pipeline stage.

    Applies Hybrid Sharded Data Parallel (HSDP) to the embeddings, norms and attention modules,
    and expert parallelism to the MoE layers. Tensor parallelism and context parallelism are not
    supported yet.

    Args:
        dist_context: The distributed context.
        model: The Qwen3 MoE backbone to parallelize.
        stage: The current pipeline stage.

    Raises:
        ValueError: If tensor parallelism or context parallelism is enabled, or if the mesh configuration has no
            expert section.
    """
    if dist_context.dim_size(BASE_DOMAIN, BaseDim.TP) > 1:
        raise ValueError("Tensor parallelism is not supported for this model yet. Set tensor_parallel to 1.")
    if dist_context.dim_size(BASE_DOMAIN, BaseDim.CP) > 1:
        raise ValueError("Context parallelism is not supported for this model yet. Set context_parallel to 1.")

    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)[DenseDim.DP_REPLICATE, DenseDim.DP_CP_SHARD, DenseDim.CP_REPLICATE]
    expert_mesh = dist_context.mesh_for(EXPERT_DOMAIN)

    if stage.is_current_stage_first:
        parallelize_hsdp(model.embed_tokens, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)

    if stage.is_current_stage_last:
        parallelize_hsdp(model.norm, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)

    for layer in model.layers.values():
        parallelize_expert_parallel(
            layer.mlp,
            mesh_experts=expert_mesh[ExpertDim.EP_REPLICATE, ExpertDim.EP_SHARD],
            expert_shard_dim=ExpertDim.EP_SHARD,
        )

        parallelize_hsdp(layer.self_attn, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)
        parallelize_hsdp(layer.input_layernorm, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)
        parallelize_hsdp(layer.post_attention_layernorm, mesh=dense_mesh, shard_dim=DenseDim.DP_CP_SHARD)
