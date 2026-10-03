import torch

from d9d.model_state.mapper import ModelStateMapper, StateGroup
from d9d.model_state.mapper.compose import (
    ModelStateMapperParallel,
    ModelStateMapperPrefixScope,
    ModelStateMapperSequential,
)
from d9d.model_state.mapper.leaf import (
    ModelStateMapperIdentity,
    ModelStateMapperRename,
    ModelStateMapperSqueeze,
    ModelStateMapperUnsqueeze,
)
from d9d.module.model import SINGLE_HEAD_PREFIX

from .params import (
    Qwen3p5LayerParameters,
    Qwen3p5Parameters,
    Qwen3p5VisionParameters,
)


class ModelStateMapperSplitQueryGate(ModelStateMapper):
    """Splits the HuggingFace fused query/output-gate projection into separate projections.

    HuggingFace Qwen3.5 stores the query projection and the sigmoid output gate as one merged
    linear layer whose output interleaves ``head_dim`` query features and ``head_dim`` gate
    features per head. d9d keeps them as two separate projections.
    """

    def __init__(self, merged_name: str, query_name: str, gate_name: str, head_dim: int) -> None:
        """Constructs the ModelStateMapperSplitQueryGate object.

        Args:
            merged_name: Source name of the merged projection weight.
            query_name: Target name for the query projection weight.
            gate_name: Target name for the gate projection weight.
            head_dim: Dimension of a single attention head.
        """
        self._merged_name = merged_name
        self._query_name = query_name
        self._gate_name = gate_name
        self._head_dim = head_dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset(
            [
                StateGroup(
                    inputs=frozenset([self._merged_name]),
                    outputs=frozenset([self._query_name, self._gate_name]),
                )
            ]
        )

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        merged = group[self._merged_name].T

        reshaped = merged.view(merged.shape[0], -1, self._head_dim * 2)
        query_chunk, gate_chunk = reshaped.chunk(2, dim=-1)

        return {
            self._query_name: query_chunk.reshape(query_chunk.shape[0], -1).T.contiguous(),
            self._gate_name: gate_chunk.reshape(gate_chunk.shape[0], -1).T.contiguous(),
        }


class ModelStateMapperMergeQueryGate(ModelStateMapper):
    """Merges separate query and output-gate projections into the HuggingFace fused layout.

    The inverse of ``ModelStateMapperSplitQueryGate``.
    """

    def __init__(self, query_name: str, gate_name: str, merged_name: str, head_dim: int) -> None:
        """Constructs the ModelStateMapperMergeQueryGate object.

        Args:
            query_name: Source name of the query projection weight.
            gate_name: Source name of the gate projection weight.
            merged_name: Target name for the merged projection weight.
            head_dim: Dimension of a single attention head.
        """
        self._query_name = query_name
        self._gate_name = gate_name
        self._merged_name = merged_name
        self._head_dim = head_dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset(
            [
                StateGroup(
                    inputs=frozenset([self._query_name, self._gate_name]),
                    outputs=frozenset([self._merged_name]),
                )
            ]
        )

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        query = group[self._query_name].T
        gate = group[self._gate_name].T

        query = query.view(query.shape[0], -1, self._head_dim)
        gate = gate.view(gate.shape[0], -1, self._head_dim)

        merged = torch.cat([query, gate], dim=-1)

        return {self._merged_name: merged.reshape(merged.shape[0], -1).T.contiguous()}


def _mapper_from_huggingface_full_attention_layer(params: Qwen3p5LayerParameters) -> list[ModelStateMapper]:
    return [
        ModelStateMapperSplitQueryGate(
            merged_name="self_attn.q_proj.weight",
            query_name="self_attn.q_proj.weight",
            gate_name="self_attn.gate_proj.weight",
            head_dim=params.head_dim,
        ),
        *(
            ModelStateMapperIdentity(f"self_attn.{param_name}.weight")
            for param_name in ("k_norm", "k_proj", "q_norm", "v_proj", "o_proj")
        ),
    ]


def _mapper_to_huggingface_full_attention_layer(params: Qwen3p5LayerParameters) -> list[ModelStateMapper]:
    return [
        ModelStateMapperMergeQueryGate(
            query_name="self_attn.q_proj.weight",
            gate_name="self_attn.gate_proj.weight",
            merged_name="self_attn.q_proj.weight",
            head_dim=params.head_dim,
        ),
        *(
            ModelStateMapperIdentity(f"self_attn.{param_name}.weight")
            for param_name in ("k_norm", "k_proj", "q_norm", "v_proj", "o_proj")
        ),
    ]


def _mapper_from_huggingface_linear_attention_layer() -> list[ModelStateMapper]:
    return [
        ModelStateMapperSequential(
            [
                ModelStateMapperSqueeze("linear_attn.conv1d.weight", dim=1),
                ModelStateMapperRename("linear_attn.conv1d.weight", "linear_attn.qkv_conv1d.weight"),
            ]
        ),
        ModelStateMapperRename("linear_attn.in_proj_qkv.weight", "linear_attn.qkv_proj.weight"),
        ModelStateMapperRename("linear_attn.in_proj_z.weight", "linear_attn.g_proj.weight"),
        ModelStateMapperRename("linear_attn.in_proj_b.weight", "linear_attn.b_proj.weight"),
        ModelStateMapperRename("linear_attn.in_proj_a.weight", "linear_attn.decay_gate.proj.weight"),
        ModelStateMapperRename("linear_attn.A_log", "linear_attn.decay_gate.A_log"),
        ModelStateMapperRename("linear_attn.dt_bias", "linear_attn.decay_gate.dt_bias"),
        ModelStateMapperRename("linear_attn.norm.weight", "linear_attn.out_norm.weight"),
        ModelStateMapperRename("linear_attn.out_proj.weight", "linear_attn.o_proj.weight"),
    ]


def _mapper_to_huggingface_linear_attention_layer() -> list[ModelStateMapper]:
    return [
        ModelStateMapperSequential(
            [
                ModelStateMapperRename("linear_attn.qkv_conv1d.weight", "linear_attn.conv1d.weight"),
                ModelStateMapperUnsqueeze("linear_attn.conv1d.weight", dim=1),
            ]
        ),
        ModelStateMapperRename("linear_attn.qkv_proj.weight", "linear_attn.in_proj_qkv.weight"),
        ModelStateMapperRename("linear_attn.g_proj.weight", "linear_attn.in_proj_z.weight"),
        ModelStateMapperRename("linear_attn.b_proj.weight", "linear_attn.in_proj_b.weight"),
        ModelStateMapperRename("linear_attn.decay_gate.proj.weight", "linear_attn.in_proj_a.weight"),
        ModelStateMapperRename("linear_attn.decay_gate.A_log", "linear_attn.A_log"),
        ModelStateMapperRename("linear_attn.decay_gate.dt_bias", "linear_attn.dt_bias"),
        ModelStateMapperRename("linear_attn.out_norm.weight", "linear_attn.norm.weight"),
        ModelStateMapperRename("linear_attn.o_proj.weight", "linear_attn.out_proj.weight"),
    ]


def _mapper_mlp() -> list[ModelStateMapper]:
    # HF and d9d SwiGLU share the projection names, so the mapping is symmetric.
    return [ModelStateMapperIdentity(f"mlp.{proj_name}.weight") for proj_name in ("gate_proj", "up_proj", "down_proj")]


def _mapper_from_huggingface_layer(params: Qwen3p5Parameters, layer_idx: int) -> ModelStateMapper:
    if params.is_full_attention_layer(layer_idx):
        mixer_mappers = _mapper_from_huggingface_full_attention_layer(params.layer)
    else:
        mixer_mappers = _mapper_from_huggingface_linear_attention_layer()

    return ModelStateMapperParallel(
        [
            *mixer_mappers,
            *_mapper_mlp(),
            ModelStateMapperIdentity("input_layernorm.weight"),
            ModelStateMapperIdentity("post_attention_layernorm.weight"),
        ]
    )


def _mapper_to_huggingface_layer(params: Qwen3p5Parameters, layer_idx: int) -> ModelStateMapper:
    if params.is_full_attention_layer(layer_idx):
        mixer_mappers = _mapper_to_huggingface_full_attention_layer(params.layer)
    else:
        mixer_mappers = _mapper_to_huggingface_linear_attention_layer()

    return ModelStateMapperParallel(
        [
            *mixer_mappers,
            *_mapper_mlp(),
            ModelStateMapperIdentity("input_layernorm.weight"),
            ModelStateMapperIdentity("post_attention_layernorm.weight"),
        ]
    )


def _mapper_vision_block() -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            *(
                ModelStateMapperIdentity(f"{module_name}.{param_name}")
                for module_name in ("norm1", "norm2", "attn.qkv", "attn.proj")
                for param_name in ("weight", "bias")
            ),
            *(
                ModelStateMapperRename(f"mlp.linear_fc{fc_i}.{param_name}", f"mlp.fc{fc_i}.{param_name}")
                for fc_i in (1, 2)
                for param_name in ("weight", "bias")
            ),
        ]
    )


def _mapper_vision_block_to_huggingface() -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            *(
                ModelStateMapperIdentity(f"{module_name}.{param_name}")
                for module_name in ("norm1", "norm2", "attn.qkv", "attn.proj")
                for param_name in ("weight", "bias")
            ),
            *(
                ModelStateMapperRename(f"mlp.fc{fc_i}.{param_name}", f"mlp.linear_fc{fc_i}.{param_name}")
                for fc_i in (1, 2)
                for param_name in ("weight", "bias")
            ),
        ]
    )


def mapper_from_huggingface_qwen3p5_vision(params: Qwen3p5VisionParameters) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3.5 vision encoder HuggingFace keys into the d9d format.

    Args:
        params: Vision encoder parameters.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperParallel(
        [
            ModelStateMapperIdentity("patch_embed.proj.weight"),
            ModelStateMapperIdentity("patch_embed.proj.bias"),
            ModelStateMapperRename("pos_embed.weight", "pos_embed.pos_embed.weight"),
            *(
                ModelStateMapperPrefixScope(
                    _mapper_vision_block(),
                    source_prefix=f"blocks.{block_i}.",
                    target_prefix=f"blocks.{block_i}.",
                )
                for block_i in range(params.depth)
            ),
            ModelStateMapperIdentity("merger.norm.weight"),
            ModelStateMapperIdentity("merger.norm.bias"),
            *(
                ModelStateMapperRename(f"merger.linear_fc{fc_i}.{param_name}", f"merger.fc{fc_i}.{param_name}")
                for fc_i in (1, 2)
                for param_name in ("weight", "bias")
            ),
        ]
    )


def mapper_to_huggingface_qwen3p5_vision(params: Qwen3p5VisionParameters) -> ModelStateMapper:
    """Creates a state mapper translating d9d Qwen3.5 vision encoder keys into the HuggingFace format.

    Args:
        params: Vision encoder parameters.

    Returns:
        A composite state mapper.
    """
    return ModelStateMapperParallel(
        [
            ModelStateMapperIdentity("patch_embed.proj.weight"),
            ModelStateMapperIdentity("patch_embed.proj.bias"),
            ModelStateMapperRename("pos_embed.pos_embed.weight", "pos_embed.weight"),
            *(
                ModelStateMapperPrefixScope(
                    _mapper_vision_block_to_huggingface(),
                    source_prefix=f"blocks.{block_i}.",
                    target_prefix=f"blocks.{block_i}.",
                )
                for block_i in range(params.depth)
            ),
            ModelStateMapperIdentity("merger.norm.weight"),
            ModelStateMapperIdentity("merger.norm.bias"),
            *(
                ModelStateMapperRename(f"merger.fc{fc_i}.{param_name}", f"merger.linear_fc{fc_i}.{param_name}")
                for fc_i in (1, 2)
                for param_name in ("weight", "bias")
            ),
        ]
    )


def _vocab_name_for(params: Qwen3p5Parameters) -> str:
    if len(params.split_vocab_order) != 1:
        raise ValueError("HuggingFace mappers can only process a single vocab split")

    return params.split_vocab_order[0]


def mapper_from_huggingface_qwen3p5(params: Qwen3p5Parameters) -> ModelStateMapper:
    """Creates a state mapper translating base Qwen3.5 dense HuggingFace keys into the d9d format.

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
                    _mapper_from_huggingface_layer(params, layer_i),
                    source_prefix=f"layers.{layer_i}.",
                    target_prefix=f"layers.{layer_i}.",
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )


def mapper_to_huggingface_qwen3p5(params: Qwen3p5Parameters) -> ModelStateMapper:
    """Creates a state mapper translating d9d base Qwen3.5 dense keys into the HuggingFace format.

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
                    _mapper_to_huggingface_layer(params, layer_i),
                    source_prefix=f"layers.{layer_i}.",
                    target_prefix=f"layers.{layer_i}.",
                )
                for layer_i in range(params.num_hidden_layers)
            ),
            ModelStateMapperIdentity("norm.weight"),
        ]
    )


def mapper_from_huggingface_qwen3p5_for_causal_lm(
    params: Qwen3p5Parameters, *, head_prefix: str = SINGLE_HEAD_PREFIX
) -> ModelStateMapper:
    """Creates a state mapper translating Qwen3.5 dense Causal LM HuggingFace keys into the d9d format.

    Args:
        params: Base backbone parameters.
        head_prefix: FQN prefix of the head that receives the HuggingFace head.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_from_huggingface_qwen3p5(params),
                source_prefix="model.",
                target_prefix="model.",
            ),
            ModelStateMapperRename(name_from="lm_head.weight", name_to=f"{head_prefix}lm_head.{vocab_name}.weight"),
        ]
    )


def mapper_to_huggingface_qwen3p5_for_causal_lm(
    params: Qwen3p5Parameters, *, head_prefix: str = SINGLE_HEAD_PREFIX
) -> ModelStateMapper:
    """Creates a state mapper translating d9d Qwen3.5 dense Causal LM keys into the HuggingFace format.

    Args:
        params: Base backbone parameters.
        head_prefix: FQN prefix of the head that receives the HuggingFace head.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_to_huggingface_qwen3p5(params),
                source_prefix="model.",
                target_prefix="model.",
            ),
            ModelStateMapperRename(name_from=f"{head_prefix}lm_head.{vocab_name}.weight", name_to="lm_head.weight"),
        ]
    )


def mapper_from_huggingface_qwen3p5_for_conditional_generation(
    params: Qwen3p5Parameters,
    vision: Qwen3p5VisionParameters,
    *,
    head_prefix: str = SINGLE_HEAD_PREFIX,
) -> ModelStateMapper:
    """Creates a state mapper translating multimodal Qwen3.5 dense HuggingFace keys into the d9d format.

    The HuggingFace layout scopes the text backbone under ``model.language_model.`` and the vision
    encoder under ``model.visual.``.

    Args:
        params: Base backbone parameters.
        vision: Vision encoder parameters.
        head_prefix: FQN prefix of the head that receives the HuggingFace head.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_from_huggingface_qwen3p5(params),
                source_prefix="model.language_model.",
                target_prefix="model.model.",
            ),
            ModelStateMapperPrefixScope(
                mapper_from_huggingface_qwen3p5_vision(vision),
                source_prefix="model.visual.",
                target_prefix="model.encoder.",
            ),
            ModelStateMapperRename(name_from="lm_head.weight", name_to=f"{head_prefix}lm_head.{vocab_name}.weight"),
        ]
    )


def mapper_to_huggingface_qwen3p5_for_conditional_generation(
    params: Qwen3p5Parameters,
    vision: Qwen3p5VisionParameters,
    *,
    head_prefix: str = SINGLE_HEAD_PREFIX,
) -> ModelStateMapper:
    """Creates a state mapper translating d9d multimodal Qwen3.5 dense keys into the HuggingFace format.

    Args:
        params: Base backbone parameters.
        vision: Vision encoder parameters.
        head_prefix: FQN prefix of the head that receives the HuggingFace head.

    Returns:
        A composite state mapper.
    """
    vocab_name = _vocab_name_for(params)
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                mapper_to_huggingface_qwen3p5(params),
                source_prefix="model.model.",
                target_prefix="model.language_model.",
            ),
            ModelStateMapperPrefixScope(
                mapper_to_huggingface_qwen3p5_vision(vision),
                source_prefix="model.encoder.",
                target_prefix="model.visual.",
            ),
            ModelStateMapperRename(name_from=f"{head_prefix}lm_head.{vocab_name}.weight", name_to="lm_head.weight"),
        ]
    )
