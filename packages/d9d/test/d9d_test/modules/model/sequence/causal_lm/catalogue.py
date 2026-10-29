import transformers as tr
from d9d.module.model import DecoderForCausalLM
from d9d.module.model.qwen3_dense import (
    mapper_from_huggingface_qwen3_dense_for_causal_lm,
    mapper_to_huggingface_qwen3_dense_for_causal_lm,
)
from d9d.module.model.qwen3_moe import (
    mapper_from_huggingface_qwen3_moe_for_causal_lm,
    mapper_to_huggingface_qwen3_moe_for_causal_lm,
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
            compose=DecoderForCausalLM,
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ]
    for model_type in ModelCatalogue
}


HF_TO_D9D_MAPPER_CAUSAL_LM = {
    ModelCatalogue.QWEN3_MOE: mapper_from_huggingface_qwen3_moe_for_causal_lm(
        QWEN3_MOE_PARAMETERS, experts_format=MOE_EXPERTS_FORMAT
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_from_huggingface_qwen3_dense_for_causal_lm(QWEN3_DENSE_PARAMETERS),
}

D9D_TO_HF_MAPPER_CAUSAL_LM = {
    ModelCatalogue.QWEN3_MOE: mapper_to_huggingface_qwen3_moe_for_causal_lm(
        QWEN3_MOE_PARAMETERS, experts_format=MOE_EXPERTS_FORMAT
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_to_huggingface_qwen3_dense_for_causal_lm(QWEN3_DENSE_PARAMETERS),
}
