from d9d.core.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.module.model.qwen3_5 import (
    Qwen3p5FullAttentionLayer,
    Qwen3p5Model,
    Qwen3p5VisionModel,
)
from d9d.module.parallelism.api import parallelize_hsdp
from d9d.pipelining.api import PipelineStageInfo


def parallelize_qwen3p5_model(dist_context: DistributedContext, model: Qwen3p5Model, stage: PipelineStageInfo):
    """Parallelizes the base Qwen3.5 dense model components.

    This function configures the model layers for distributed execution within a pipeline
    stage. It applies Hybrid Sharded Data Parallelism (HSDP) to every component (embeddings,
    norms, attention mixers, and MLP layers).

    Current usage constraints:
    *   Tensor Parallelism is not supported (we may implement it later).
    *   Context Parallelism is not supported (we will implement it later).

    Args:
        dist_context: The distributed context.
        model: The Qwen3.5 dense base model to parallelize.
        stage: Information about the current pipeline stage.

    Raises:
        ValueError: If Tensor Parallel or Context Parallel is enabled in the context.
    """
    dims = dist_context.mesh_params
    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)

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
        parallelize_hsdp(layer.mlp, mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])

        mixer = layer.self_attn if isinstance(layer, Qwen3p5FullAttentionLayer) else layer.linear_attn
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


def parallelize_qwen3p5_vision(dist_context: DistributedContext, model: Qwen3p5VisionModel) -> None:
    """Applies HSDP to the Qwen3.5 vision encoder, submodule by submodule.

    The encoder lives on the first pipeline stage only, so a provider calls this alongside
    ``parallelize_qwen3p5_model`` when that stage is current.

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
