from d9d.module.block.head import CausalLMHeadConfig, ClassificationHeadConfig

from d9d_test.modules.model.sequence.catalogue import ModelCatalogue, make_d9d_model_factory

# The headline capability of DEP-0007: two heads on one shared backbone.
HEAD_NAME_LM = "lm"
HEAD_NAME_CLS = "cls"
NUM_LABELS_MULTIHEAD = 4


def _heads() -> dict[str, CausalLMHeadConfig | ClassificationHeadConfig]:
    return {
        HEAD_NAME_LM: CausalLMHeadConfig(),
        HEAD_NAME_CLS: ClassificationHeadConfig(num_labels=NUM_LABELS_MULTIHEAD, dropout=0.0),
    }


D9D_MODEL_FACTORIES_MULTIHEAD = {
    model_type: [
        make_d9d_model_factory(
            model_type,
            heads=_heads(),
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ]
    for model_type in ModelCatalogue
}
