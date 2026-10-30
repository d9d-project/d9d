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
from d9d.module.model import SINGLE_HEAD_PREFIX

from .params import (
    Qwen3MoELayerParameters,
    Qwen3MoEParameters,
)


class Qwen3MoEExpertsFormat(StrEnum):
    """Layout of the expert parameters in a Hugging Face Transformers checkpoint.

    Attributes:
        MODULE_LIST: The Transformers v4.x layout: an ``nn.ModuleList`` of experts, each with separate
            ``gate_proj``, ``up_proj`` and ``down_proj`` ``nn.Linear`` layers.
        FUSED: The Transformers v5.x layout: 3D tensors that stack all experts, with ``gate_proj`` and
            ``up_proj`` fused into ``gate_up_proj``.
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
            raise ValueError(f"Unsupported experts format ({experts_format}).")


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
        raise ValueError(
            f"split_vocab_order ({params.split_vocab_order}) must contain a single split. "
            "Hugging Face mappers support only one vocab split."
        )

    return params.split_vocab_order[0]


def mapper_from_huggingface_qwen3_moe(
    params: Qwen3MoEParameters,
    experts_format: Qwen3MoEExpertsFormat,
) -> ModelStateMapper:
    """Creates a state mapper translating base Qwen3 MoE Hugging Face keys into the d9d format.

    Args:
        params: Base model parameters.
        experts_format: Layout of the expert parameters in the Hugging Face checkpoint.

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
                    _mapper_from_huggingface_qwen3_moe_layer(params.layer, experts_format),
                    source_prefix=f"layers.{layer_i}.",
                    target_prefix=f"layers.{layer_i}.",
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )


def mapper_from_huggingface_qwen3_moe_for_causal_lm(
    params: Qwen3MoEParameters,
    experts_format: Qwen3MoEExpertsFormat,
    *,
    head_prefix: str = SINGLE_HEAD_PREFIX,
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 MoE causal LM Hugging Face keys into the d9d format.

    A Hugging Face model has exactly one head, so the mapper must know where it goes in the d9d
    model. The default targets a single-head decoder. To load into a multi-head model, pass that
    head's prefix (``f"heads.{name}."``). The other heads keep their initialization.

    Args:
        params: Base model parameters.
        experts_format: Layout of the expert parameters in the Hugging Face checkpoint.
        head_prefix: FQN prefix of the head that receives the Hugging Face head in the target d9d model.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_from_huggingface_qwen3_moe(params, experts_format),
                source_prefix="model.",
                target_prefix="model.",
            ),
            ModelStateMapperRename(name_from="lm_head.weight", name_to=f"{head_prefix}lm_head.{vocab_name}.weight"),
        ]
    )


def mapper_from_huggingface_qwen3_moe_for_classification(
    params: Qwen3MoEParameters,
    experts_format: Qwen3MoEExpertsFormat,
    *,
    head_prefix: str = SINGLE_HEAD_PREFIX,
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 MoE classification Hugging Face keys into the d9d format.

    Args:
        params: Base model parameters.
        experts_format: Layout of the expert parameters in the Hugging Face checkpoint.
        head_prefix: FQN prefix of the head that receives the Hugging Face head in the target d9d model.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_from_huggingface_qwen3_moe(params, experts_format),
                source_prefix="model.",
                target_prefix="model.",
            ),
            ModelStateMapperRename(name_from="score.weight", name_to=f"{head_prefix}score.weight"),
        ]
    )


def mapper_from_huggingface_qwen3_moe_for_embedding(
    params: Qwen3MoEParameters, experts_format: Qwen3MoEExpertsFormat
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 MoE embedding Hugging Face keys into the d9d format.

    A Hugging Face embedding model is the bare backbone without head weights. The d9d embedding head
    has nothing to load and keeps its initialization.

    Args:
        params: Base model parameters.
        experts_format: Layout of the expert parameters in the Hugging Face checkpoint.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperPrefixScope(
        mapper_from_huggingface_qwen3_moe(params, experts_format), target_prefix="model."
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
                ModelStateMapperSequential(
                    [
                        ModelStateMapperRename("mlp.grouped_experts.down_proj.weight", "mlp.experts.down_proj"),
                        ModelStateMapperTranspose("mlp.experts.down_proj", dims=(-1, -2)),
                    ]
                ),
            ]
        case _:
            raise ValueError(f"Unsupported experts format ({experts_format}).")


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
    """Creates a state mapper translating base Qwen3 MoE d9d keys back into the Hugging Face format.

    Args:
        params: Base model parameters.
        experts_format: Layout of the expert parameters in the Hugging Face checkpoint.

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
                    _mapper_to_huggingface_qwen3_moe_layer(params.layer, experts_format),
                    source_prefix=f"layers.{layer_i}.",
                    target_prefix=f"layers.{layer_i}.",
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )


def mapper_to_huggingface_qwen3_moe_for_causal_lm(
    params: Qwen3MoEParameters,
    experts_format: Qwen3MoEExpertsFormat,
    *,
    head_prefix: str = SINGLE_HEAD_PREFIX,
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 MoE causal LM d9d keys back into the Hugging Face format.

    Args:
        params: Base model parameters.
        experts_format: Layout of the expert parameters in the Hugging Face checkpoint.
        head_prefix: FQN prefix of the head holding the causal LM weights in the source d9d model.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_to_huggingface_qwen3_moe(params, experts_format),
                source_prefix="model.",
                target_prefix="model.",
            ),
            ModelStateMapperRename(name_from=f"{head_prefix}lm_head.{vocab_name}.weight", name_to="lm_head.weight"),
        ]
    )


def mapper_to_huggingface_qwen3_moe_for_classification(
    params: Qwen3MoEParameters,
    experts_format: Qwen3MoEExpertsFormat,
    *,
    head_prefix: str = SINGLE_HEAD_PREFIX,
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 MoE classification d9d keys back into the Hugging Face format.

    Args:
        params: Base model parameters.
        experts_format: Layout of the expert parameters in the Hugging Face checkpoint.
        head_prefix: FQN prefix of the head holding the classification weights in the source d9d model.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_to_huggingface_qwen3_moe(params, experts_format),
                source_prefix="model.",
                target_prefix="model.",
            ),
            ModelStateMapperRename(name_from=f"{head_prefix}score.weight", name_to="score.weight"),
        ]
    )


def mapper_to_huggingface_qwen3_moe_for_embedding(
    params: Qwen3MoEParameters,
    experts_format: Qwen3MoEExpertsFormat,
    *,
    embedding_dim: int | None = None,
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 MoE embedding d9d keys back into the Hugging Face format.

    A Hugging Face embedding model is the bare backbone without head weights, so the mapper cannot
    export an embedding projection.

    Args:
        params: Base model parameters.
        experts_format: Layout of the expert parameters in the Hugging Face checkpoint.
        embedding_dim: The output size of the embedding head projection, or ``None`` if the head has
            no projection.

    Returns:
        A composite state mapper.

    Raises:
        ValueError: If ``embedding_dim`` is set, because Hugging Face has no counterpart for the projection.
    """
    if embedding_dim is not None:
        raise ValueError(
            f"embedding_dim ({embedding_dim}) must be None: Hugging Face has no counterpart for an "
            f"embedding head projection."
        )

    return ModelStateMapperPrefixScope(mapper_to_huggingface_qwen3_moe(params, experts_format), source_prefix="model.")
