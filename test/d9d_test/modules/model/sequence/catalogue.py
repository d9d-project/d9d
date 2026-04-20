from collections.abc import Callable
from enum import StrEnum, auto
from typing import TypeVar

from d9d.module.model.qwen3_dense import Qwen3DenseLayerParameters, Qwen3DenseParameters
from d9d.module.model.qwen3_moe import Qwen3MoELayerParameters, Qwen3MoEParameters
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


D9D_MODEL_PARAMETERS = {
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


HF_MODEL_PARAMETERS = {
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


TModel = TypeVar("TModel")


def d9d_model_factory(
    model_class: type[TModel],
    **model_kwargs,
) -> Callable[[PipelineStageInfo], TModel]:
    def _build_fn(stage: PipelineStageInfo):
        with torch_seed(_D9D_INIT_SEED):
            model = model_class(**model_kwargs, stage=stage).cuda().bfloat16()
            model.reset_parameters()
            return model

    return _build_fn
