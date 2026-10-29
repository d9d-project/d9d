from d9d.module.model import (
    CausalLMHeadConfig,
    ClassificationHeadConfig,
    DecoderBackbone,
    DecoderWithHeads,
    build_decoder_head,
)
from d9d.pipelining.api import PipelineStageInfo

from d9d_test.modules.model.sequence.catalogue import ModelCatalogue, make_d9d_model_factory

# The headline capability of DEP-0007: two heads on one shared backbone.
HEAD_NAME_LM = "lm"
HEAD_NAME_CLS = "cls"
NUM_LABELS_MULTIHEAD = 4


def _compose(backbone: DecoderBackbone, stage: PipelineStageInfo) -> DecoderWithHeads:
    heads = {
        HEAD_NAME_LM: build_decoder_head(CausalLMHeadConfig(), backbone=backbone),
        HEAD_NAME_CLS: build_decoder_head(
            ClassificationHeadConfig(num_labels=NUM_LABELS_MULTIHEAD, dropout=0.0), backbone=backbone
        ),
    }
    return DecoderWithHeads(backbone, heads, stage)


D9D_MODEL_FACTORIES_MULTIHEAD = {
    model_type: [
        make_d9d_model_factory(
            model_type,
            compose=_compose,
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ]
    for model_type in ModelCatalogue
}
