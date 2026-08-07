import copy

import transformers as tr
from d9d.module.model import (
    DEFAULT_HEAD_NAME_CLASSIFICATION,
    ClassificationHeadConfig,
    DecoderForClassification,
)
from d9d.module.model.qwen3_dense import (
    mapper_from_huggingface_qwen3_dense_for_classification,
    mapper_to_huggingface_qwen3_dense_for_classification,
)
from d9d.module.model.qwen3_moe import (
    mapper_from_huggingface_qwen3_moe_for_classification,
    mapper_to_huggingface_qwen3_moe_for_classification,
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

NUM_LABELS_CLS = 3

HEAD_NAME_CLS = DEFAULT_HEAD_NAME_CLASSIFICATION


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
            compose=lambda backbone, stage: DecoderForClassification(backbone, _head_config(), stage),
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ]
    for model_type in ModelCatalogue
}


HF_TO_D9D_MAPPER_CLS = {
    ModelCatalogue.QWEN3_MOE: mapper_from_huggingface_qwen3_moe_for_classification(
        QWEN3_MOE_PARAMETERS, experts_format=MOE_EXPERTS_FORMAT
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_from_huggingface_qwen3_dense_for_classification(QWEN3_DENSE_PARAMETERS),
}

D9D_TO_HF_MAPPER_CLS = {
    ModelCatalogue.QWEN3_MOE: mapper_to_huggingface_qwen3_moe_for_classification(
        QWEN3_MOE_PARAMETERS, experts_format=MOE_EXPERTS_FORMAT
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_to_huggingface_qwen3_dense_for_classification(QWEN3_DENSE_PARAMETERS),
}
