import copy

import transformers as tr
from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperPrefixScope
from d9d.module.block.head import (
    ClassificationHeadConfig,
    hf_mapper_from_huggingface_cls_head,
    hf_mapper_to_huggingface_cls_head,
)

from d9d_test.modules.model.sequence.catalogue import (
    HF_MODEL_PARAMETERS,
    ModelCatalogue,
    backbone_from_hf_mapper,
    backbone_to_hf_mapper,
    build_head_for,
    hf_model_factory,
    make_d9d_model_factory,
)

NUM_LABELS_CLS = 3

HEAD_NAME_CLS = "cls"
_HEAD_PREFIX = f"heads.{HEAD_NAME_CLS}."


def _hf_config_for(catalogue: ModelCatalogue):
    config = copy.copy(HF_MODEL_PARAMETERS[catalogue])
    config.num_labels = NUM_LABELS_CLS
    return config


HF_MODEL_FACTORY_CLS = {
    ModelCatalogue.QWEN3_MOE: hf_model_factory(
        tr.Qwen3MoeForSequenceClassification,
        config=_hf_config_for(ModelCatalogue.QWEN3_MOE),
        bf16_layers=["model.embed_tokens", "model.layers", "model.norm", "score"],
    ),
    ModelCatalogue.QWEN3_DENSE: hf_model_factory(
        tr.Qwen3ForSequenceClassification,
        config=_hf_config_for(ModelCatalogue.QWEN3_DENSE),
        bf16_layers=["model.embed_tokens", "model.layers", "model.norm", "score"],
    ),
}


def _head_config() -> ClassificationHeadConfig:
    return ClassificationHeadConfig(num_labels=NUM_LABELS_CLS, dropout=0.0)


D9D_MODEL_FACTORIES_CLS = {
    model_type: [
        make_d9d_model_factory(
            model_type,
            heads={HEAD_NAME_CLS: _head_config()},
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ]
    for model_type in ModelCatalogue
}


def _from_hf_mapper(model_type: ModelCatalogue) -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                backbone_from_hf_mapper(model_type), source_prefix="model.", target_prefix="model."
            ),
            hf_mapper_from_huggingface_cls_head(build_head_for(model_type, _head_config()), prefix=_HEAD_PREFIX),
        ]
    )


def _to_hf_mapper(model_type: ModelCatalogue) -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                backbone_to_hf_mapper(model_type), source_prefix="model.", target_prefix="model."
            ),
            hf_mapper_to_huggingface_cls_head(build_head_for(model_type, _head_config()), prefix=_HEAD_PREFIX),
        ]
    )


HF_TO_D9D_MAPPER_CLS = {model_type: _from_hf_mapper(model_type) for model_type in ModelCatalogue}
D9D_TO_HF_MAPPER_CLS = {model_type: _to_hf_mapper(model_type) for model_type in ModelCatalogue}
