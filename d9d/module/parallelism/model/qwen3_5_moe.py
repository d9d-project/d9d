from d9d.core.dist_context import DENSE_DOMAIN, EXPERT_DOMAIN, DistributedContext
from d9d.module.model.qwen3_5_moe import (
    Qwen3p5MoEFullAttentionLayer,
    Qwen3p5MoEModel,
    Qwen3p5MoEVisionModel,
)
from d9d.module.parallelism.api import parallelize_expert_parallel, parallelize_hsdp
from d9d.pipelining.api import PipelineStageInfo


def parallelize_qwen3p5_moe_model(dist_context: DistributedContext, model: Qwen3p5MoEModel, stage: PipelineStageInfo):
    """Parallelizes the base Qwen3.5 MoE model components.

    This function configures the model layers for distributed execution within a pipeline
    stage. It applies Hybrid Sharded Data Parallelism (HSDP) to dense components (embeddings,
    norms, attention mixers) and Expert Parallelism (EP) to the Mixture-of-Experts (MLP) layers.

    Current usage constraints:
    *   Tensor Parallelism is not supported (we may implement it later).
    *   Context Parallelism is not supported (we will implement it later).

    Args:
        dist_context: The distributed context.
        model: The Qwen3.5 MoE base model to parallelize.
        stage: Information about the current pipeline stage.

    Raises:
        ValueError: If Tensor Parallel or Context Parallel is enabled in the context.
    """
    dims = dist_context.mesh_params
    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)
    expert_mesh = dist_context.mesh_for(EXPERT_DOMAIN)

    if dims.has_tensor_parallel:
        raise ValueError("Tensor Parallel currently is not supported for this model.")
    if dims.has_context_parallel_replicate or dims.has_context_parallel_shard:
        raise ValueError("Context Parallel currently is not supported for this model.")

    if stage.is_current_stage_first:
        parallelize_hsdp(model.embed_tokens, mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])

    if stage.is_current_stage_last:
        parallelize_hsdp(
            model.norm,
            mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"],
        )

    for layer in model.layers.values():
        parallelize_expert_parallel(layer.mlp, mesh_experts=expert_mesh["ep_replicate", "ep_shard"])

        mixer = layer.self_attn if isinstance(layer, Qwen3p5MoEFullAttentionLayer) else layer.linear_attn
        parallelize_hsdp(
            mixer,
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


def parallelize_qwen3p5_moe_vision(dist_context: DistributedContext, model: Qwen3p5MoEVisionModel) -> None:
    """Applies HSDP to the Qwen3.5 vision encoder, submodule by submodule.

    The encoder lives on the first pipeline stage only, so a provider calls this alongside
    ``parallelize_qwen3p5_moe_model`` when that stage is current.

    The rotary module is deliberately skipped: it holds only a deterministic non-persistent buffer
    with no gradients to synchronize, and wrapping it would turn that buffer into a ``DTensor``,
    which breaks the ``aten.index`` lookup inside its forward. This mirrors how the backbone's rope
    provider is handled.

    Args:
        dist_context: The distributed context containing device meshes and topology info.
        model: The vision encoder to parallelize.
    """
    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)
    mesh = dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"]

    parallelize_hsdp(model.patch_embed, mesh=mesh)
    parallelize_hsdp(model.pos_embed, mesh=mesh)

    for block in model.blocks:
        parallelize_hsdp(block, mesh=mesh)

    parallelize_hsdp(model.merger, mesh=mesh)
