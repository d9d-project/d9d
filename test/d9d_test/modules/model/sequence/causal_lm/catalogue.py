import transformers as tr
from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperPrefixScope
from d9d.module.block.head import (
    CausalLMHeadConfig,
    hf_mapper_from_huggingface_lm_head,
    hf_mapper_to_huggingface_lm_head,
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

HEAD_NAME_LM = "lm"
_HEAD_PREFIX = f"heads.{HEAD_NAME_LM}."


HF_MODEL_FACTORY_CAUSAL_LM = {
    ModelCatalogue.QWEN3_MOE: hf_model_factory(
        tr.Qwen3MoeForCausalLM,
        config=HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_MOE],
        bf16_layers=["model.embed_tokens", "model.layers", "model.norm", "lm_head"],
    ),
    ModelCatalogue.QWEN3_DENSE: hf_model_factory(
        tr.Qwen3ForCausalLM,
        config=HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_DENSE],
        bf16_layers=["model.embed_tokens", "model.layers", "model.norm", "lm_head"],
    ),
}


D9D_MODEL_FACTORIES_CAUSAL_LM = {
    model_type: [
        make_d9d_model_factory(
            model_type,
            heads={HEAD_NAME_LM: CausalLMHeadConfig()},
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
            hf_mapper_from_huggingface_lm_head(build_head_for(model_type, CausalLMHeadConfig()), prefix=_HEAD_PREFIX),
        ]
    )


def _to_hf_mapper(model_type: ModelCatalogue) -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            ModelStateMapperPrefixScope(
                backbone_to_hf_mapper(model_type), source_prefix="model.", target_prefix="model."
            ),
            hf_mapper_to_huggingface_lm_head(build_head_for(model_type, CausalLMHeadConfig()), prefix=_HEAD_PREFIX),
        ]
    )


HF_TO_D9D_MAPPER_CAUSAL_LM = {model_type: _from_hf_mapper(model_type) for model_type in ModelCatalogue}
D9D_TO_HF_MAPPER_CAUSAL_LM = {model_type: _to_hf_mapper(model_type) for model_type in ModelCatalogue}
