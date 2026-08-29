import transformers as tr
from d9d.module.model import DecoderForEmbedding, EmbeddingHeadConfig
from d9d.module.model.qwen3_dense import (
    mapper_from_huggingface_qwen3_dense_for_embedding,
    mapper_to_huggingface_qwen3_dense_for_embedding,
)
from d9d.module.model.qwen3_moe import (
    mapper_from_huggingface_qwen3_moe_for_embedding,
    mapper_to_huggingface_qwen3_moe_for_embedding,
)

from d9d_test.modules.model.sequence.catalogue import (
    HF_MODEL_PARAMETERS,
    MOE_EXPERTS_FORMAT,
    QWEN3_DENSE_PARAMETERS,
    QWEN3_MOE_PARAMETERS,
    ModelCatalogue,
    hf_model_factory,
    make_d9d_model_factory,
)

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
            compose=lambda backbone, stage: DecoderForEmbedding(backbone, _head_config(), stage),
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ]
    # Only the families with a HuggingFace counterpart for this task: the parity suite
    # needs a reference model, and HF_MODEL_FACTORY_EMBEDDING is what defines that set.
    for model_type in HF_MODEL_FACTORY_EMBEDDING
}


# The HuggingFace embedding reference is the bare backbone (no "model." prefix, no head weights).
HF_TO_D9D_MAPPER_EMBEDDING = {
    ModelCatalogue.QWEN3_MOE: mapper_from_huggingface_qwen3_moe_for_embedding(
        QWEN3_MOE_PARAMETERS, experts_format=MOE_EXPERTS_FORMAT
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_from_huggingface_qwen3_dense_for_embedding(QWEN3_DENSE_PARAMETERS),
}

D9D_TO_HF_MAPPER_EMBEDDING = {
    ModelCatalogue.QWEN3_MOE: mapper_to_huggingface_qwen3_moe_for_embedding(
        QWEN3_MOE_PARAMETERS, experts_format=MOE_EXPERTS_FORMAT
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_to_huggingface_qwen3_dense_for_embedding(QWEN3_DENSE_PARAMETERS),
}
