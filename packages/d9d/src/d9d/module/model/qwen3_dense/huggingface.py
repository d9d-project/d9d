from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperPrefixScope
from d9d.model_state.mapper.leaf import (
    ModelStateMapperIdentity,
    ModelStateMapperRename,
)
from d9d.module.model import SINGLE_HEAD_PREFIX

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
        raise ValueError(
            f"split_vocab_order ({params.split_vocab_order}) must contain a single split. "
            "Hugging Face mappers support only one vocab split."
        )

    return params.split_vocab_order[0]


def mapper_from_huggingface_qwen3_dense(params: Qwen3DenseParameters) -> ModelStateMapper:
    """Creates a state mapper translating base Qwen3 Dense Hugging Face keys into the d9d format.

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


def mapper_from_huggingface_qwen3_dense_for_causal_lm(
    params: Qwen3DenseParameters, *, head_prefix: str = SINGLE_HEAD_PREFIX
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 Dense causal LM Hugging Face keys into the d9d format.

    A Hugging Face model has exactly one head, so the mapper must know where it goes in the d9d
    model. The default targets a single-head decoder. To load into a multi-head model, pass that
    head's prefix (``f"heads.{name}."``). The other heads keep their initialization.

    Args:
        params: Base model parameters.
        head_prefix: FQN prefix of the head that receives the Hugging Face head in the target d9d model.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_from_huggingface_qwen3_dense(params), source_prefix="model.", target_prefix="model."
            ),
            ModelStateMapperRename(name_from="lm_head.weight", name_to=f"{head_prefix}lm_head.{vocab_name}.weight"),
        ]
    )


def mapper_from_huggingface_qwen3_dense_for_classification(
    params: Qwen3DenseParameters, *, head_prefix: str = SINGLE_HEAD_PREFIX
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 Dense classification Hugging Face keys into the d9d format.

    Args:
        params: Base model parameters.
        head_prefix: FQN prefix of the head that receives the Hugging Face head in the target d9d model.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_from_huggingface_qwen3_dense(params), source_prefix="model.", target_prefix="model."
            ),
            ModelStateMapperRename(name_from="score.weight", name_to=f"{head_prefix}score.weight"),
        ]
    )


def mapper_from_huggingface_qwen3_dense_for_embedding(params: Qwen3DenseParameters) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 Dense embedding Hugging Face keys into the d9d format.

    A Hugging Face embedding model is the bare backbone without head weights. The d9d embedding head
    has nothing to load and keeps its initialization.

    Args:
        params: Base model parameters.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperPrefixScope(mapper_from_huggingface_qwen3_dense(params), target_prefix="model.")


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
    """Creates a state mapper translating base Qwen3 Dense d9d keys back into the Hugging Face format.

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


def mapper_to_huggingface_qwen3_dense_for_causal_lm(
    params: Qwen3DenseParameters, *, head_prefix: str = SINGLE_HEAD_PREFIX
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 Dense causal LM d9d keys back into the Hugging Face format.

    Args:
        params: Base model parameters.
        head_prefix: FQN prefix of the head holding the causal LM weights in the source d9d model.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_to_huggingface_qwen3_dense(params), source_prefix="model.", target_prefix="model."
            ),
            ModelStateMapperRename(name_from=f"{head_prefix}lm_head.{vocab_name}.weight", name_to="lm_head.weight"),
        ]
    )


def mapper_to_huggingface_qwen3_dense_for_classification(
    params: Qwen3DenseParameters, *, head_prefix: str = SINGLE_HEAD_PREFIX
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 Dense classification d9d keys back into the Hugging Face format.

    Args:
        params: Base model parameters.
        head_prefix: FQN prefix of the head holding the classification weights in the source d9d model.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_to_huggingface_qwen3_dense(params), source_prefix="model.", target_prefix="model."
            ),
            ModelStateMapperRename(name_from=f"{head_prefix}score.weight", name_to="score.weight"),
        ]
    )


def mapper_to_huggingface_qwen3_dense_for_embedding(
    params: Qwen3DenseParameters, *, embedding_dim: int | None = None
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3 Dense embedding d9d keys back into the Hugging Face format.

    A Hugging Face embedding model is the bare backbone without head weights, so the mapper cannot
    export an embedding projection.

    Args:
        params: Base model parameters.
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

    return ModelStateMapperPrefixScope(mapper_to_huggingface_qwen3_dense(params), source_prefix="model.")
