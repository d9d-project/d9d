from collections.abc import Callable
from enum import StrEnum, auto

from d9d.core.dist_context import DistributedContext
from d9d.module.block.head import ClassificationHead, EmbeddingHead, SplitLanguageModellingHead
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model import DecoderBackbone, DecoderWithHeads
from d9d.module.model.qwen3_dense import (
    Qwen3DenseLayerParameters,
    Qwen3DenseModel,
    Qwen3DenseParameters,
)
from d9d.module.model.qwen3_moe import (
    Qwen3MoEExpertsFormat,
    Qwen3MoELayerParameters,
    Qwen3MoEModel,
    Qwen3MoEParameters,
)
from d9d.module.parallelism.model import (
    parallelize_causal_lm_head,
    parallelize_classification_head,
    parallelize_embedding_head,
    parallelize_qwen3_dense_model,
    parallelize_qwen3_moe_model,
)
from d9d.pipelining.api import PipelineStageInfo
from transformers import PretrainedConfig, PreTrainedModel, Qwen3Config, Qwen3MoeConfig

from d9d_test.modules.helper import torch_seed


class ModelCatalogue(StrEnum):
    QWEN3_MOE = auto()
    QWEN3_DENSE = auto()


_HIDDEN_SIZE = 512
_INTERMEDIATE_SIZE_DENSE = 768
_INTERMEDIATE_SIZE_MOE = 256
_NUM_EXPERTS_MOE = 8
_EXPERTS_TOP_K_MOE = 7
_NUM_ATTENTION_HEADS = 16
_NUM_KV_HEADS = 4
_RMS_NORM_EPS = 1e-5
_HEAD_DIM = 32
_ROPE_BASE = 10_000
_MAX_POS_ID = 15_000
_NUM_LAYERS = 8

_VOCAB_SPLIT_SIZE = {"a": 100}
_VOCAB_SPLIT_ORDER = ["a"]
_VOCAB_MERGED = 100

_PAD_TOKEN_ID = 99


# HuggingFace experts layout the d9d MoE mappers translate from/to in these tests.
MOE_EXPERTS_FORMAT = Qwen3MoEExpertsFormat.FUSED


QWEN3_MOE_PARAMETERS = Qwen3MoEParameters(
    layer=Qwen3MoELayerParameters(
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE_MOE,
        num_experts=_NUM_EXPERTS_MOE,
        experts_top_k=_EXPERTS_TOP_K_MOE,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        rms_norm_eps=_RMS_NORM_EPS,
        head_dim=_HEAD_DIM,
    ),
    rope_base=_ROPE_BASE,
    max_position_ids=_MAX_POS_ID,
    num_hidden_layers=_NUM_LAYERS,
    split_vocab_size=_VOCAB_SPLIT_SIZE,
    split_vocab_order=_VOCAB_SPLIT_ORDER,
)

QWEN3_DENSE_PARAMETERS = Qwen3DenseParameters(
    layer=Qwen3DenseLayerParameters(
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE_DENSE,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        rms_norm_eps=_RMS_NORM_EPS,
        head_dim=_HEAD_DIM,
    ),
    rope_base=_ROPE_BASE,
    max_position_ids=_MAX_POS_ID,
    num_hidden_layers=_NUM_LAYERS,
    split_vocab_size=_VOCAB_SPLIT_SIZE,
    split_vocab_order=_VOCAB_SPLIT_ORDER,
)

D9D_MODEL_PARAMETERS: dict[ModelCatalogue, Qwen3MoEParameters | Qwen3DenseParameters] = {
    ModelCatalogue.QWEN3_MOE: QWEN3_MOE_PARAMETERS,
    ModelCatalogue.QWEN3_DENSE: QWEN3_DENSE_PARAMETERS,
}


HF_MODEL_PARAMETERS: dict[ModelCatalogue, PretrainedConfig] = {
    ModelCatalogue.QWEN3_MOE: Qwen3MoeConfig(
        vocab_size=_VOCAB_MERGED,
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        moe_intermediate_size=_INTERMEDIATE_SIZE_MOE,
        num_experts=_NUM_EXPERTS_MOE,
        num_experts_per_tok=_EXPERTS_TOP_K_MOE,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        hidden_act="silu",
        max_position_embeddings=_MAX_POS_ID,
        rms_norm_eps=_RMS_NORM_EPS,
        use_cache=False,
        tie_word_embeddings=False,
        rope_theta=_ROPE_BASE,
        attention_bias=False,
        use_sliding_window=False,
        attention_dropout=0.0,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
        _attn_implementation="flash_attention_4",
        pad_token_id=_PAD_TOKEN_ID,
    ),
    ModelCatalogue.QWEN3_DENSE: Qwen3Config(
        vocab_size=_VOCAB_MERGED,
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE_DENSE,
        num_attention_heads=_NUM_ATTENTION_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        hidden_act="silu",
        max_position_embeddings=_MAX_POS_ID,
        rms_norm_eps=_RMS_NORM_EPS,
        use_cache=False,
        tie_word_embeddings=False,
        rope_theta=_ROPE_BASE,
        attention_bias=False,
        use_sliding_window=False,
        attention_dropout=0.0,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
        _attn_implementation="flash_attention_4",
        pad_token_id=_PAD_TOKEN_ID,
    ),
}


BACKBONE_CLASSES: dict[ModelCatalogue, type[Qwen3MoEModel] | type[Qwen3DenseModel]] = {
    ModelCatalogue.QWEN3_MOE: Qwen3MoEModel,
    ModelCatalogue.QWEN3_DENSE: Qwen3DenseModel,
}

BACKBONE_PARALLELIZE_FN: dict[ModelCatalogue, Callable[..., None]] = {
    ModelCatalogue.QWEN3_MOE: parallelize_qwen3_moe_model,
    ModelCatalogue.QWEN3_DENSE: parallelize_qwen3_dense_model,
}


_HF_INIT_SEED = 131232
_D9D_INIT_SEED = 123213


def hf_model_factory(
    model_class: type[PreTrainedModel],
    config: PretrainedConfig,
    bf16_layers: list[str],
) -> Callable[[], PreTrainedModel]:
    def _build_fn():
        with torch_seed(_HF_INIT_SEED):
            model = model_class(config).cuda().eval()

            for layer_name in bf16_layers:
                model.get_submodule(layer_name).bfloat16()

            return model

    return _build_fn


ComposeDecoder = Callable[[DecoderBackbone, PipelineStageInfo], DecoderWithHeads]
"""Composes a freshly built backbone with the head(s) a suite exercises."""


def make_d9d_model_factory(
    model_type: ModelCatalogue,
    compose: ComposeDecoder,
    enable_checkpointing: bool,
) -> Callable[[PipelineStageInfo], DecoderWithHeads]:
    """Builds a factory that composes the backbone with the suite's heads on the target device."""

    def _build_fn(stage: PipelineStageInfo) -> DecoderWithHeads:
        with torch_seed(_D9D_INIT_SEED):
            backbone = BACKBONE_CLASSES[model_type](
                D9D_MODEL_PARAMETERS[model_type],
                stage,
                hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
                enable_checkpointing=enable_checkpointing,
            )
            model = compose(backbone, stage).cuda().bfloat16()
            model.reset_parameters()
            return model

    return _build_fn


def parallelize_decoder_with_heads(
    model: DecoderWithHeads,
    model_type: ModelCatalogue,
    dist_context: DistributedContext,
    stage: PipelineStageInfo,
) -> None:
    """Parallelizes a composed decoder: the per-family backbone routine, then each head by its type."""
    BACKBONE_PARALLELIZE_FN[model_type](dist_context, model.model, stage)

    if not stage.is_current_stage_last:
        return

    for head in model.heads.values():
        match head:
            case SplitLanguageModellingHead():
                parallelize_causal_lm_head(head, dist_context)
            case ClassificationHead():
                parallelize_classification_head(head, dist_context)
            case EmbeddingHead():
                parallelize_embedding_head(head, dist_context)
            case _:
                raise ValueError(f"Unknown head type: {type(head)}")
