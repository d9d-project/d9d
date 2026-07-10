import dataclasses
from collections.abc import Callable, Mapping
from enum import StrEnum, auto

from d9d.core.dist_context import DistributedContext
from d9d.model_state.mapper import ModelStateMapper
from d9d.module.block.head import AnyHeadConfig, TaskHead, build_head
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model import DecoderWithHeads
from d9d.module.model.qwen3_dense import (
    Qwen3DenseLayerParameters,
    Qwen3DenseModel,
    Qwen3DenseParameters,
    mapper_from_huggingface_qwen3_dense,
    mapper_to_huggingface_qwen3_dense,
)
from d9d.module.model.qwen3_moe import (
    Qwen3MoEExpertsFormat,
    Qwen3MoELayerParameters,
    Qwen3MoEModel,
    Qwen3MoEParameters,
    mapper_from_huggingface_qwen3_moe,
    mapper_to_huggingface_qwen3_moe,
)
from d9d.module.parallelism.model import (
    parallelize_qwen3_dense_model,
    parallelize_qwen3_moe_model,
    parallelize_task_head,
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
_MOE_EXPERTS_FORMAT = Qwen3MoEExpertsFormat.FUSED


D9D_MODEL_PARAMETERS: dict[ModelCatalogue, Qwen3MoEParameters | Qwen3DenseParameters] = {
    ModelCatalogue.QWEN3_MOE: Qwen3MoEParameters(
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
    ),
    ModelCatalogue.QWEN3_DENSE: Qwen3DenseParameters(
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
    ),
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


@dataclasses.dataclass
class _BackboneDims:
    """A lightweight stand-in exposing only the backbone-shared dimensions ``build_head`` reads."""

    hidden_size: int
    split_vocab_size: dict[str, int]
    split_vocab_order: list[str]


def build_head_for(model_type: ModelCatalogue, config: AnyHeadConfig) -> TaskHead:
    """Builds a head instance from the family's backbone dimensions, for state-mapper construction."""
    params = D9D_MODEL_PARAMETERS[model_type]
    stub = _BackboneDims(
        hidden_size=params.layer.hidden_size,
        split_vocab_size=params.split_vocab_size,
        split_vocab_order=params.split_vocab_order,
    )
    return build_head(config, backbone=stub, stage=PipelineStageInfo(current_stage=0, num_stages=1))


def make_d9d_model_factory(
    model_type: ModelCatalogue,
    heads: Mapping[str, AnyHeadConfig],
    enable_checkpointing: bool,
) -> Callable[[PipelineStageInfo], DecoderWithHeads]:
    """Builds a factory that composes the backbone with the given named heads on the target device."""

    def _build_fn(stage: PipelineStageInfo) -> DecoderWithHeads:
        with torch_seed(_D9D_INIT_SEED):
            backbone = BACKBONE_CLASSES[model_type](
                D9D_MODEL_PARAMETERS[model_type],
                stage,
                hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
                enable_checkpointing=enable_checkpointing,
            )
            built_heads = {name: build_head(config, backbone=backbone, stage=stage) for name, config in heads.items()}
            model = DecoderWithHeads(backbone, built_heads, stage).cuda().bfloat16()
            model.reset_parameters()
            return model

    return _build_fn


def parallelize_decoder_with_heads(
    model: DecoderWithHeads,
    model_type: ModelCatalogue,
    dist_context: DistributedContext,
    stage: PipelineStageInfo,
) -> None:
    """Parallelizes a composed decoder: the per-family backbone routine, then each head (HSDP)."""
    BACKBONE_PARALLELIZE_FN[model_type](dist_context, model.model, stage)
    if stage.is_current_stage_last:
        for head in model.heads.values():
            parallelize_task_head(head, dist_context)


def backbone_from_hf_mapper(model_type: ModelCatalogue) -> ModelStateMapper:
    """Builds the backbone-only HuggingFace-to-d9d state mapper for the given family."""
    match model_type:
        case ModelCatalogue.QWEN3_MOE:
            return mapper_from_huggingface_qwen3_moe(
                D9D_MODEL_PARAMETERS[model_type], experts_format=_MOE_EXPERTS_FORMAT
            )
        case ModelCatalogue.QWEN3_DENSE:
            return mapper_from_huggingface_qwen3_dense(D9D_MODEL_PARAMETERS[model_type])
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def backbone_to_hf_mapper(model_type: ModelCatalogue) -> ModelStateMapper:
    """Builds the backbone-only d9d-to-HuggingFace state mapper for the given family."""
    match model_type:
        case ModelCatalogue.QWEN3_MOE:
            return mapper_to_huggingface_qwen3_moe(D9D_MODEL_PARAMETERS[model_type], experts_format=_MOE_EXPERTS_FORMAT)
        case ModelCatalogue.QWEN3_DENSE:
            return mapper_to_huggingface_qwen3_dense(D9D_MODEL_PARAMETERS[model_type])
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
