from d9d.core.dist_context import DENSE_DOMAIN, EXPERT_DOMAIN, DistributedContext
from d9d.module.model.qwen3_moe import Qwen3MoEForCausalLM, Qwen3MoEModel
from d9d.module.parallelism.api import parallelize_expert_parallel, parallelize_hsdp
from d9d.pipelining.api import PipelineStageInfo


def parallelize_qwen3_moe_model(
        dist_context: DistributedContext,
        model: Qwen3MoEModel,
        stage: PipelineStageInfo
):
    """
    Parallelizes the base Qwen3 MoE model components.

    This function configures the model layers for distributed execution within a pipeline
    stage. It applies Hybrid Sharded Data Parallelism (HSDP) to dense components (embeddings,
    norms, attention) and Expert Parallelism (EP) to the Mixture-of-Experts (MLP) layers.

    Current usage constraints:
    *   Tensor Parallelism is not supported (we may implement it later).
    *   Context Parallelism is not supported (we will implement it later).

    Args:
        dist_context: The distributed context.
        model: The Qwen3 MoE base model to parallelize.
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
        parallelize_hsdp(
            model.embed_tokens,
            mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"]
        )

    if stage.is_current_stage_last:
        parallelize_hsdp(
            model.norm,
            mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"],
        )

    for layer in model.layers.values():
        parallelize_expert_parallel(
            layer.mlp,
            mesh_experts=expert_mesh["ep_replicate", "ep_shard"]
        )

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


def parallelize_qwen3_moe_for_causal_lm(
        dist_context: DistributedContext,
        model: Qwen3MoEForCausalLM,
        stage: PipelineStageInfo
):
    """
    Parallelizes the Qwen3 MoE Causal LM model.

    This function delegates backbone parallelization to ``parallelize_qwen3_moe_model``
    and additionally configures the language model head with Hybrid Sharded Data
    Parallelism (HSDP).

    Args:
        dist_context: The distributed context containing device meshes and topology info.
        model: The Qwen3 MoE Causal LM model to parallelize.
        stage: Information about the current pipeline stage.
    """

    dense_mesh = dist_context.mesh_for(DENSE_DOMAIN)

    parallelize_qwen3_moe_model(dist_context, model.model, stage)

    if stage.is_current_stage_last:
        parallelize_hsdp(
            model.lm_head,
            mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"],
        )
