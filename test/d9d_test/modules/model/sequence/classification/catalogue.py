import copy

import transformers as tr
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model.qwen3_dense import (
    Qwen3DenseForClassification,
    Qwen3DenseForClassificationParameters,
    mapper_from_huggingface_qwen3_dense_for_classification,
    mapper_to_huggingface_qwen3_dense_for_classification,
)
from d9d.module.model.qwen3_moe import (
    Qwen3MoEExpertsFormat,
    Qwen3MoEForClassification,
    Qwen3MoEForClassificationParameters,
    mapper_from_huggingface_qwen3_moe_for_classification,
    mapper_to_huggingface_qwen3_moe_for_classification,
)
from d9d.module.parallelism.model.qwen3_dense import parallelize_qwen3_dense_for_classification
from d9d.module.parallelism.model.qwen3_moe import parallelize_qwen3_moe_for_classification

from d9d_test.modules.model.sequence.catalogue import (
    D9D_MODEL_PARAMETERS,
    HF_MODEL_PARAMETERS,
    ModelCatalogue,
    d9d_model_factory,
    hf_model_factory,
)

NUM_LABELS_CLS = 3


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

_D9D_PARAMS = {
    ModelCatalogue.QWEN3_MOE: Qwen3MoEForClassificationParameters(
        model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_MOE],
        num_labels=NUM_LABELS_CLS,
        classifier_dropout=0.0,
    ),
    ModelCatalogue.QWEN3_DENSE: Qwen3DenseForClassificationParameters(
        model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_DENSE],
        num_labels=NUM_LABELS_CLS,
        classifier_dropout=0.0,
    ),
}


D9D_MODEL_FACTORIES_CLS = {
    ModelCatalogue.QWEN3_MOE: [
        d9d_model_factory(
            Qwen3MoEForClassification,
            params=_D9D_PARAMS[ModelCatalogue.QWEN3_MOE],
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ],
    ModelCatalogue.QWEN3_DENSE: [
        d9d_model_factory(
            Qwen3DenseForClassification,
            params=_D9D_PARAMS[ModelCatalogue.QWEN3_DENSE],
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ],
}


HF_TO_D9D_MAPPER_CLS = {
    ModelCatalogue.QWEN3_MOE: mapper_from_huggingface_qwen3_moe_for_classification(
        _D9D_PARAMS[ModelCatalogue.QWEN3_MOE],
        experts_format=Qwen3MoEExpertsFormat.FUSED,
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_from_huggingface_qwen3_dense_for_classification(
        _D9D_PARAMS[ModelCatalogue.QWEN3_DENSE]
    ),
}

D9D_TO_HF_MAPPER_CLS = {
    ModelCatalogue.QWEN3_MOE: mapper_to_huggingface_qwen3_moe_for_classification(
        _D9D_PARAMS[ModelCatalogue.QWEN3_MOE],
        experts_format=Qwen3MoEExpertsFormat.FUSED,
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_to_huggingface_qwen3_dense_for_classification(
        _D9D_PARAMS[ModelCatalogue.QWEN3_DENSE]
    ),
}

D9D_PARALLELIZE_FN = {
    ModelCatalogue.QWEN3_MOE: parallelize_qwen3_moe_for_classification,
    ModelCatalogue.QWEN3_DENSE: parallelize_qwen3_dense_for_classification,
}
