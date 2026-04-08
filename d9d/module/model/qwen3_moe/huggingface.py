from enum import StrEnum

from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import (
    ModelStateMapperParallel,
    ModelStateMapperPrefixScope,
    ModelStateMapperSequential,
)
from d9d.model_state.mapper.leaf import (
    ModelStateMapperChunkTensors,
    ModelStateMapperConcatenateTensors,
    ModelStateMapperIdentity,
    ModelStateMapperRename,
    ModelStateMapperStackTensors,
    ModelStateMapperTranspose,
    ModelStateMapperUnstackTensors,
)

from .params import (
    Qwen3MoEForCausalLMParameters,
    Qwen3MoEForClassificationParameters,
    Qwen3MoELayerParameters,
    Qwen3MoEParameters,
)


class Qwen3MoEExpertsFormat(StrEnum):
    """
    Specifies the underlying layout of the experts parameters in Hugging Face.

    Attributes:
        MODULE_LIST: Legacy (v4.x) nn.ModuleList with individual down, up, and gate nn.Linear layers.
        FUSED: New (v5.x) format with 3D expert representations fused into single tensors.
    """

    MODULE_LIST = "module_list"
    FUSED = "fused"


def _experts_mappers_from_huggingface(
    params: Qwen3MoELayerParameters, experts_format: Qwen3MoEExpertsFormat
) -> list[ModelStateMapper]:
    match experts_format:
        case Qwen3MoEExpertsFormat.MODULE_LIST:
            return [
                ModelStateMapperSequential(
                    [
                        ModelStateMapperStackTensors(
                            source_names=[
                                f"mlp.experts.{expert_i}.{proj_type}.weight" for expert_i in range(params.num_experts)
                            ],
                            target_name=f"mlp.grouped_experts.{proj_type}.weight",
                            dim=0,
                        ),
                        ModelStateMapperTranspose(f"mlp.grouped_experts.{proj_type}.weight", dims=(-1, -2)),
                    ]
                )
                for proj_type in ("down_proj", "gate_proj", "up_proj")
            ]
        case Qwen3MoEExpertsFormat.FUSED:
            return [
                ModelStateMapperSequential(
                    [
                        ModelStateMapperTranspose("mlp.experts.gate_up_proj", dims=(-1, -2)),
                        ModelStateMapperChunkTensors(
                            source_name="mlp.experts.gate_up_proj",
                            target_names=[
                                "mlp.grouped_experts.gate_proj.weight",
                                "mlp.grouped_experts.up_proj.weight",
                            ],
                            dim=-1,
                        ),
                    ]
                ),
                ModelStateMapperSequential(
                    [
                        ModelStateMapperTranspose("mlp.experts.down_proj", dims=(-1, -2)),
                        ModelStateMapperRename("mlp.experts.down_proj", "mlp.grouped_experts.down_proj.weight"),
                    ]
                ),
            ]
        case _:
            raise ValueError(f"Unsupported experts format {experts_format}")


def _mapper_from_huggingface_qwen3_moe_layer(
    params: Qwen3MoELayerParameters,
    experts_format: Qwen3MoEExpertsFormat,
) -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            *_experts_mappers_from_huggingface(params, experts_format),
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


def mapper_from_huggingface_qwen3_moe(
    params: Qwen3MoEParameters,
    experts_format: Qwen3MoEExpertsFormat,
) -> ModelStateMapper:
    """
    Creates a state mapper for a base Qwen3 MoE model that translates the HuggingFace state dictionary keys
    into the d9d format.

    Args:
        params: Base model parameters.
        experts_format: Format of the MoE experts storage.

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
                    _mapper_from_huggingface_qwen3_moe_layer(params.layer, experts_format), prefix=f"layers.{layer_i}."
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )


def mapper_from_huggingface_qwen3_moe_for_causal_lm(
    params: Qwen3MoEForCausalLMParameters,
    experts_format: Qwen3MoEExpertsFormat,
) -> ModelStateMapper:
    """
    Creates a state mapper for a Qwen3 MoE Causal LM model that translates the HuggingFace state dictionary
    keys into the d9d format.

    Args:
        params: Causal LM model parameters.
        experts_format: Format of the MoE experts storage.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params.model)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_from_huggingface_qwen3_moe(params.model, experts_format), prefix="model."
            ),
            ModelStateMapperRename(name_from="lm_head.weight", name_to=f"lm_head.lm_head.{vocab_name}.weight"),
        ]
    )


def mapper_from_huggingface_qwen3_moe_for_classification(
    params: Qwen3MoEForClassificationParameters,
    experts_format: Qwen3MoEExpertsFormat,
) -> ModelStateMapper:
    """
    Creates a state mapper for a Qwen3 MoE sequence classification model that translates the HuggingFace
    state dictionary keys into the d9d format.

    Args:
        params: Classification model parameters.
        experts_format: Format of the MoE experts storage.

    Returns:
        A composite state mapper.
    """

    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_from_huggingface_qwen3_moe(params.model, experts_format), prefix="model."
            ),
            ModelStateMapperRename(name_from="score.weight", name_to="cls_head.score.weight"),
        ]
    )


def _experts_mappers_to_huggingface(
    params: Qwen3MoELayerParameters, experts_format: Qwen3MoEExpertsFormat
) -> list[ModelStateMapper]:
    match experts_format:
        case Qwen3MoEExpertsFormat.MODULE_LIST:
            return [
                ModelStateMapperSequential(
                    [
                        ModelStateMapperTranspose(f"mlp.grouped_experts.{proj_type}.weight", dims=(-1, -2)),
                        ModelStateMapperUnstackTensors(
                            source_name=f"mlp.grouped_experts.{proj_type}.weight",
                            target_names=[
                                f"mlp.experts.{expert_i}.{proj_type}.weight" for expert_i in range(params.num_experts)
                            ],
                            dim=0,
                        ),
                    ]
                )
                for proj_type in ("down_proj", "gate_proj", "up_proj")
            ]
        case Qwen3MoEExpertsFormat.FUSED:
            return [
                ModelStateMapperSequential(
                    [
                        ModelStateMapperConcatenateTensors(
                            source_names=[
                                "mlp.grouped_experts.gate_proj.weight",
                                "mlp.grouped_experts.up_proj.weight",
                            ],
                            target_name="mlp.experts.gate_up_proj",
                            dim=-1,
                        ),
                        ModelStateMapperTranspose("mlp.experts.gate_up_proj", dims=(-1, -2)),
                    ]
                ),
                # down proj pipeline
                ModelStateMapperSequential(
                    [
                        ModelStateMapperRename("mlp.grouped_experts.down_proj.weight", "mlp.experts.down_proj"),
                        ModelStateMapperTranspose("mlp.experts.down_proj", dims=(-1, -2)),
                    ]
                ),
            ]
        case _:
            raise ValueError(f"Unsupported experts format {experts_format}")


def _mapper_to_huggingface_qwen3_moe_layer(
    params: Qwen3MoELayerParameters, experts_format: Qwen3MoEExpertsFormat
) -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            *_experts_mappers_to_huggingface(params, experts_format),
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


def mapper_to_huggingface_qwen3_moe(
    params: Qwen3MoEParameters,
    experts_format: Qwen3MoEExpertsFormat,
) -> ModelStateMapper:
    """
    Creates a state mapper for a base Qwen3 MoE model that translates the d9d state dictionary keys
    back into the HuggingFace format.

    Args:
        params: Base model parameters.
        experts_format: Format of the MoE experts storage.

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
                    _mapper_to_huggingface_qwen3_moe_layer(params.layer, experts_format), prefix=f"layers.{layer_i}."
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )


def mapper_to_huggingface_qwen3_moe_for_causal_lm(
    params: Qwen3MoEForCausalLMParameters,
    experts_format: Qwen3MoEExpertsFormat,
) -> ModelStateMapper:
    """
    Creates a state mapper for a Qwen3 MoE Causal LM model that translates the d9d state dictionary
    keys back into the HuggingFace format.

    Args:
        params: Causal LM model parameters.
        experts_format: Format of the MoE experts storage.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params.model)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(mapper_to_huggingface_qwen3_moe(params.model, experts_format), prefix="model."),
            ModelStateMapperRename(name_from=f"lm_head.lm_head.{vocab_name}.weight", name_to="lm_head.weight"),
        ]
    )


def mapper_to_huggingface_qwen3_moe_for_classification(
    params: Qwen3MoEForClassificationParameters,
    experts_format: Qwen3MoEExpertsFormat,
) -> ModelStateMapper:
    """
    Creates a state mapper for a Qwen3 MoE sequence classification model that translates the d9d
    state dictionary keys back into the HuggingFace format.

    Args:
        params: Classification model parameters.
        experts_format: Format of the MoE experts storage.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(mapper_to_huggingface_qwen3_moe(params.model, experts_format), prefix="model."),
            ModelStateMapperRename(name_from="cls_head.score.weight", name_to="score.weight"),
        ]
    )
