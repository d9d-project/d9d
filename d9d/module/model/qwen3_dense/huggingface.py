from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperPrefixScope
from d9d.model_state.mapper.leaf import (
    ModelStateMapperIdentity,
    ModelStateMapperRename,
)
from d9d.module.model import SINGLE_HEAD_PREFIX

from .params import Qwen3DenseLayerParameters, Qwen3DenseParameters


class _Qwen3DenseHuggingFaceRopeParameters(BaseModel):
    # Every key here changes the RoPE frequencies, e.g. a scaling factor, so unknown keys are errors.
    model_config = ConfigDict(extra="forbid")

    rope_type: Literal["default"]
    rope_theta: int


class _Qwen3DenseHuggingFaceConfig(BaseModel):
    # A Hugging Face config also holds metadata that d9d does not use, e.g. the token ids and the dtype.
    model_config = ConfigDict(extra="ignore")

    model_type: Literal["qwen3"]

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    num_hidden_layers: int
    max_position_embeddings: int
    vocab_size: int

    # Transformers 4 writes rope_theta and rope_scaling, Transformers 5 writes rope_parameters instead.
    rope_theta: int | None = None
    rope_scaling: None = None
    rope_parameters: _Qwen3DenseHuggingFaceRopeParameters | None = None

    # These fields enable features that Qwen3DenseModel does not implement. The defaults are the Transformers defaults.
    hidden_act: Literal["silu"] = "silu"
    attention_bias: Literal[False] = False
    attention_dropout: Literal[0] = 0
    use_sliding_window: Literal[False] = False
    layer_types: list[Literal["full_attention"]] | None = None


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


def _rope_base_from_huggingface(config: _Qwen3DenseHuggingFaceConfig) -> int:
    match config.rope_theta, config.rope_parameters:
        case int() as rope_theta, None:
            return rope_theta
        case None, _Qwen3DenseHuggingFaceRopeParameters() as rope_parameters:
            return rope_parameters.rope_theta
        case _:
            raise ValueError(
                f"The Hugging Face config must set exactly one of rope_theta ({config.rope_theta}) "
                f"and rope_parameters ({config.rope_parameters})."
            )


def parameters_from_huggingface_qwen3_dense(
    config: Mapping[str, Any], *, vocab_split_name: str
) -> Qwen3DenseParameters:
    """Builds Qwen3 Dense model parameters from a Hugging Face model config.

    The config can use the layout of Transformers 4 (``rope_theta``) or Transformers 5
    (``rope_parameters``). Fields that d9d does not use, such as the token ids, are ignored.

    Args:
        config: The parsed ``config.json`` of a Hugging Face Qwen3 model.
        vocab_split_name: The name of the vocabulary split that holds the whole Hugging Face vocabulary.

    Returns:
        The model parameters.

    Raises:
        ValueError: If the config misses a field, or if it enables a feature that the d9d model does not
            support, such as sliding window attention, RoPE scaling or attention bias.
    """
    hf_config = _Qwen3DenseHuggingFaceConfig.model_validate(config)

    return Qwen3DenseParameters(
        layer=Qwen3DenseLayerParameters(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            rms_norm_eps=hf_config.rms_norm_eps,
            head_dim=hf_config.head_dim,
        ),
        num_hidden_layers=hf_config.num_hidden_layers,
        rope_base=_rope_base_from_huggingface(hf_config),
        max_position_ids=hf_config.max_position_embeddings,
        split_vocab_size={vocab_split_name: hf_config.vocab_size},
        split_vocab_order=[vocab_split_name],
    )
