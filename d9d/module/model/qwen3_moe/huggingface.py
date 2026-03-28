from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperPrefixScope
from d9d.model_state.mapper.leaf import (
    ModelStateMapperIdentity,
    ModelStateMapperRename,
    ModelStateMapperStackTensors,
    ModelStateMapperUnstackTensors,
)

from .params import (
    Qwen3MoEForCausalLMParameters,
    Qwen3MoEForClassificationParameters,
    Qwen3MoELayerParameters,
    Qwen3MoEParameters,
)


def _mapper_from_huggingface_qwen3_moe_layer(params: Qwen3MoELayerParameters) -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            *(
                ModelStateMapperStackTensors(
                    source_names=[
                        f"mlp.experts.{expert_i}.{proj_type}.weight" for expert_i in range(params.num_experts)
                    ],
                    target_name=f"mlp.grouped_experts.{proj_type}.weight",
                    stack_dim=0,
                    transpose_dims=(-1, -2),
                )
                for proj_type in ("down_proj", "gate_proj", "up_proj")
            ),
            ModelStateMapperRename("mlp.gate.weight", "mlp.router.gate.weight"),
            *(
                ModelStateMapperIdentity(f"{param_name}.weight")
                for param_name in (
                    "input_layernorm",
                    "post_attention_layernorm",
                    "self_attn.k_norm",
                    "self_attn.k_proj",
                    "self_attn.q_norm",
                    "self_attn.q_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                )
            ),
        ]
    )


def _vocab_name_for(params: Qwen3MoEParameters) -> str:
    if len(params.split_vocab_order) != 1:
        raise ValueError("HuggingFace mappers can only process a single vocab split")

    return params.split_vocab_order[0]


def mapper_from_huggingface_qwen3_moe(params: Qwen3MoEParameters) -> ModelStateMapper:
    """
    Creates a state mapper for a base Qwen3 MoE model that translates the HuggingFace state dictionary keys
    into the d9d format.

    Args:
        params: Base model parameters.

    Returns:
        A composite state mapper.
    """

    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperRename(
                name_from="embed_tokens.weight", name_to=f"embed_tokens.token_embedding.{vocab_name}.weight"
            ),
            *(
                ModelStateMapperPrefixScope(
                    _mapper_from_huggingface_qwen3_moe_layer(params.layer), prefix=f"layers.{layer_i}."
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )


def mapper_from_huggingface_qwen3_moe_for_causal_lm(params: Qwen3MoEForCausalLMParameters) -> ModelStateMapper:
    """
    Creates a state mapper for a Qwen3 MoE Causal LM model that translates the HuggingFace state dictionary
    keys into the d9d format.

    Args:
        params: Causal LM model parameters.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params.model)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(mapper_from_huggingface_qwen3_moe(params.model), prefix="model."),
            ModelStateMapperRename(name_from="lm_head.weight", name_to=f"lm_head.lm_head.{vocab_name}.weight"),
        ]
    )


def mapper_from_huggingface_qwen3_moe_for_classification(
    params: Qwen3MoEForClassificationParameters,
) -> ModelStateMapper:
    """
    Creates a state mapper for a Qwen3 MoE sequence classification model that translates the HuggingFace
    state dictionary keys into the d9d format.

    Args:
        params: Classification model parameters.

    Returns:
        A composite state mapper.
    """

    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(mapper_from_huggingface_qwen3_moe(params.model), prefix="model."),
            ModelStateMapperRename(name_from="score.weight", name_to="cls_head.weight"),
        ]
    )


def _mapper_to_huggingface_qwen3_moe_layer(params: Qwen3MoELayerParameters) -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            *(
                ModelStateMapperUnstackTensors(
                    source_name=f"mlp.grouped_experts.{proj_type}.weight",
                    target_names=[
                        f"mlp.experts.{expert_i}.{proj_type}.weight" for expert_i in range(params.num_experts)
                    ],
                    unstack_dim=0,
                    transpose_dims=(-1, -2),
                )
                for proj_type in ("down_proj", "gate_proj", "up_proj")
            ),
            ModelStateMapperRename("mlp.router.gate.weight", "mlp.gate.weight"),
            *(
                ModelStateMapperIdentity(f"{param_name}.weight")
                for param_name in (
                    "input_layernorm",
                    "post_attention_layernorm",
                    "self_attn.k_norm",
                    "self_attn.k_proj",
                    "self_attn.q_norm",
                    "self_attn.q_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                )
            ),
        ]
    )


def mapper_to_huggingface_qwen3_moe(params: Qwen3MoEParameters) -> ModelStateMapper:
    """
    Creates a state mapper for a base Qwen3 MoE model that translates the d9d state dictionary keys
    back into the HuggingFace format.

    Args:
        params: Base model parameters.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperRename(
                name_from=f"embed_tokens.token_embedding.{vocab_name}.weight", name_to="embed_tokens.weight"
            ),
            *(
                ModelStateMapperPrefixScope(
                    _mapper_to_huggingface_qwen3_moe_layer(params.layer), prefix=f"layers.{layer_i}."
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )


def mapper_to_huggingface_qwen3_moe_for_causal_lm(params: Qwen3MoEForCausalLMParameters) -> ModelStateMapper:
    """
    Creates a state mapper for a Qwen3 MoE Causal LM model that translates the d9d state dictionary
    keys back into the HuggingFace format.

    Args:
        params: Causal LM model parameters.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params.model)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(mapper_to_huggingface_qwen3_moe(params.model), prefix="model."),
            ModelStateMapperRename(name_from=f"lm_head.lm_head.{vocab_name}.weight", name_to="lm_head.weight"),
        ]
    )


def mapper_to_huggingface_qwen3_moe_for_classification(
    params: Qwen3MoEForClassificationParameters,
) -> ModelStateMapper:
    """
    Creates a state mapper for a Qwen3 MoE sequence classification model that translates the d9d
    state dictionary keys back into the HuggingFace format.

    Args:
        params: Classification model parameters.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(mapper_to_huggingface_qwen3_moe(params.model), prefix="model."),
            ModelStateMapperRename(name_from="cls_head.weight", name_to="score.weight"),
        ]
    )
