import transformers as tr
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model.qwen3_dense import (
    Qwen3DenseForCausalLM,
    Qwen3DenseForCausalLMParameters,
    mapper_from_huggingface_qwen3_dense_for_causal_lm,
    mapper_to_huggingface_qwen3_dense_for_causal_lm,
)
from d9d.module.model.qwen3_moe import (
    Qwen3MoEForCausalLM,
    Qwen3MoEForCausalLMParameters,
    mapper_from_huggingface_qwen3_moe_for_causal_lm,
    mapper_to_huggingface_qwen3_moe_for_causal_lm,
)
from d9d.module.parallelism.model.qwen3_dense import parallelize_qwen3_dense_for_causal_lm
from d9d.module.parallelism.model.qwen3_moe import parallelize_qwen3_moe_for_causal_lm

from d9d_test.modules.model.sequence.catalogue import (
    D9D_MODEL_PARAMETERS,
    HF_MODEL_PARAMETERS,
    ModelCatalogue,
    d9d_model_factory,
    hf_model_factory,
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
    ModelCatalogue.QWEN3_MOE: [
        d9d_model_factory(
            Qwen3MoEForCausalLM,
            params=Qwen3MoEForCausalLMParameters(model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_MOE]),
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ],
    ModelCatalogue.QWEN3_DENSE: [
        d9d_model_factory(
            Qwen3DenseForCausalLM,
            params=Qwen3DenseForCausalLMParameters(model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_DENSE]),
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=enable_checkpointing,
        )
        for enable_checkpointing in (True, False)
    ],
}


HF_TO_D9D_MAPPER_CAUSAL_LM = {
    ModelCatalogue.QWEN3_MOE: mapper_from_huggingface_qwen3_moe_for_causal_lm(
        Qwen3MoEForCausalLMParameters(model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_MOE])
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_from_huggingface_qwen3_dense_for_causal_lm(
        Qwen3DenseForCausalLMParameters(model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_DENSE])
    ),
}


D9D_TO_HF_MAPPER_CAUSAL_LM = {
    ModelCatalogue.QWEN3_MOE: mapper_to_huggingface_qwen3_moe_for_causal_lm(
        Qwen3MoEForCausalLMParameters(model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_MOE])
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_to_huggingface_qwen3_dense_for_causal_lm(
        Qwen3DenseForCausalLMParameters(model=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_DENSE])
    ),
}


D9D_PARALLELIZE_FN = {
    ModelCatalogue.QWEN3_MOE: parallelize_qwen3_moe_for_causal_lm,
    ModelCatalogue.QWEN3_DENSE: parallelize_qwen3_dense_for_causal_lm,
}
