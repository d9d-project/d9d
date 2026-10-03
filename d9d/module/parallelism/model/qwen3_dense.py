from d9d.core.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.module.model.qwen3_dense import Qwen3DenseModel
from d9d.module.parallelism.api import parallelize_hsdp
from d9d.pipelining.api import PipelineStageInfo


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
    dims = dist_context.mesh_params
    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)

    if dims.has_tensor_parallel:
        raise ValueError("Tensor parallelism is not supported for this model yet. Set tensor_parallel to 1.")
    if dims.has_context_parallel_replicate or dims.has_context_parallel_shard:
        raise ValueError(
            "Context parallelism is not supported for this model yet. "
            "Set context_parallel_shard and context_parallel_replicate to 1."
        )

    if stage.is_current_stage_first:
        parallelize_hsdp(model.embed_tokens, mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])

    if stage.is_current_stage_last:
        parallelize_hsdp(
            model.norm,
            mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"],
        )

    for layer in model.layers.values():
        parallelize_hsdp(layer.mlp, mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])

        parallelize_hsdp(
            layer.self_attn,
            mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"],
        )
        parallelize_hsdp(
            layer.input_layernorm,
            mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"],
        )
        parallelize_hsdp(
            layer.post_attention_layernorm,
            mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"],
        )
