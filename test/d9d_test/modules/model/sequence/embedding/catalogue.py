import transformers as tr
from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperPrefixScope
from d9d.module.block.head import (
    EmbeddingHeadConfig,
    hf_mapper_from_huggingface_embedding_head,
    hf_mapper_to_huggingface_embedding_head,
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

HEAD_NAME_EMBEDDING = "embedding"
_HEAD_PREFIX = f"heads.{HEAD_NAME_EMBEDDING}."


HF_MODEL_FACTORY_EMBEDDING = {
    ModelCatalogue.QWEN3_MOE: hf_model_factory(
        tr.Qwen3MoeModel,
        config=HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_MOE],
        bf16_layers=["embed_tokens", "layers", "norm"],
    ),
    ModelCatalogue.QWEN3_DENSE: hf_model_factory(
        tr.Qwen3Model,
        config=HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_DENSE],
        bf16_layers=["embed_tokens", "layers", "norm"],
    ),
}


def _head_config() -> EmbeddingHeadConfig:
    return EmbeddingHeadConfig(embedding_dim=None, normalize=False)


D9D_MODEL_FACTORIES_EMBEDDING = {
    model_type: [
        make_d9d_model_factory(
            model_type,
            heads={HEAD_NAME_EMBEDDING: _head_config()},
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ]
    for model_type in ModelCatalogue
}


def _from_hf_mapper(model_type: ModelCatalogue) -> ModelStateMapper:
    # The HuggingFace embedding reference is the bare backbone (no "model." prefix, no head weights).
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(backbone_from_hf_mapper(model_type), source_prefix="", target_prefix="model."),
            hf_mapper_from_huggingface_embedding_head(build_head_for(model_type, _head_config()), prefix=_HEAD_PREFIX),
        ]
    )


def _to_hf_mapper(model_type: ModelCatalogue) -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(backbone_to_hf_mapper(model_type), source_prefix="model.", target_prefix=""),
            hf_mapper_to_huggingface_embedding_head(build_head_for(model_type, _head_config()), prefix=_HEAD_PREFIX),
        ]
    )


HF_TO_D9D_MAPPER_EMBEDDING = {model_type: _from_hf_mapper(model_type) for model_type in ModelCatalogue}
D9D_TO_HF_MAPPER_EMBEDDING = {model_type: _to_hf_mapper(model_type) for model_type in ModelCatalogue}
