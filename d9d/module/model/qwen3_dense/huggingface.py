from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperPrefixScope
from d9d.model_state.mapper.leaf import (
    ModelStateMapperIdentity,
    ModelStateMapperRename,
)

from .params import Qwen3DenseParameters


def _mapper_from_huggingface_qwen3_dense_layer() -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
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
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        ]
    )


def _vocab_name_for(params: Qwen3DenseParameters) -> str:
    if len(params.split_vocab_order) != 1:
        raise ValueError("HuggingFace mappers can only process a single vocab split")

    return params.split_vocab_order[0]


def mapper_from_huggingface_qwen3_dense(params: Qwen3DenseParameters) -> ModelStateMapper:
    """Creates a state mapper translating base Qwen3 Dense HuggingFace keys into the d9d format.

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
                    _mapper_from_huggingface_qwen3_dense_layer(),
                    source_prefix=f"layers.{layer_i}.",
                    target_prefix=f"layers.{layer_i}.",
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )


def _mapper_to_huggingface_qwen3_dense_layer() -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
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
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        ]
    )


def mapper_to_huggingface_qwen3_dense(params: Qwen3DenseParameters) -> ModelStateMapper:
    """Creates a state mapper translating base Qwen3 Dense d9d keys back into the HuggingFace format.

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
                    _mapper_to_huggingface_qwen3_dense_layer(),
                    source_prefix=f"layers.{layer_i}.",
                    target_prefix=f"layers.{layer_i}.",
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )
