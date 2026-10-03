import transformers as tr
from d9d.module.model import DecoderForCausalLM
from d9d.module.model.qwen3_5 import (
    mapper_from_huggingface_qwen3p5_for_causal_lm,
    mapper_to_huggingface_qwen3p5_for_causal_lm,
)
from d9d.module.model.qwen3_5_moe import (
    mapper_from_huggingface_qwen3p5_moe_for_causal_lm,
    mapper_to_huggingface_qwen3p5_moe_for_causal_lm,
)
from d9d.module.model.qwen3_dense import (
    mapper_from_huggingface_qwen3_dense_for_causal_lm,
    mapper_to_huggingface_qwen3_dense_for_causal_lm,
)
from d9d.module.model.qwen3_moe import (
    mapper_from_huggingface_qwen3_moe_for_causal_lm,
    mapper_to_huggingface_qwen3_moe_for_causal_lm,
)

from d9d_test.modules.helper import GradTolerance
from d9d_test.modules.model.sequence.catalogue import (
    D9D_MODEL_PARAMETERS,
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
    ModelCatalogue.QWEN3_5: hf_model_factory(
        tr.Qwen3_5ForCausalLM,
        config=HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5],
        bf16_layers=["model.embed_tokens", "model.layers", "model.norm", "lm_head"],
    ),
    ModelCatalogue.QWEN3_5_MOE: hf_model_factory(
        tr.Qwen3_5MoeForCausalLM,
        config=HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE],
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
    ModelCatalogue.QWEN3_5: mapper_from_huggingface_qwen3p5_for_causal_lm(D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5]),
    ModelCatalogue.QWEN3_5_MOE: mapper_from_huggingface_qwen3p5_moe_for_causal_lm(
        D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE]
    ),
}

D9D_TO_HF_MAPPER_CAUSAL_LM = {
    ModelCatalogue.QWEN3_MOE: mapper_to_huggingface_qwen3_moe_for_causal_lm(
        QWEN3_MOE_PARAMETERS, experts_format=MOE_EXPERTS_FORMAT
    ),
    ModelCatalogue.QWEN3_DENSE: mapper_to_huggingface_qwen3_dense_for_causal_lm(QWEN3_DENSE_PARAMETERS),
    ModelCatalogue.QWEN3_5: mapper_to_huggingface_qwen3p5_for_causal_lm(D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5]),
    ModelCatalogue.QWEN3_5_MOE: mapper_to_huggingface_qwen3p5_moe_for_causal_lm(
        D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE]
    ),
}


# Per-model gradient tolerance overrides for the HF parity test.
#
# Qwen3.5: without the `causal_conv1d` CUDA kernel, HuggingFace falls back to an eager
# convolution while d9d uses the fla-core Triton kernel; the short-conv weight gradients are the
# smallest in the model and accumulate visible bf16 noise between the two implementations. The
# looser tolerance covers the conv weights only.
_Q35_CONV_GRAD_TOLERANCES = {
    f"model.layers.{layer_i}.linear_attn.conv1d.weight": GradTolerance(
        tol_angle=0.35, tol_norm_abs=3e-4, tol_norm_rel=1.0
    )
    for layer_i in range(D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE].num_hidden_layers)
}

HF_GRAD_TOLERANCES_CAUSAL_LM: dict[ModelCatalogue, dict[str, GradTolerance] | None] = {
    ModelCatalogue.QWEN3_MOE: None,
    ModelCatalogue.QWEN3_DENSE: None,
    ModelCatalogue.QWEN3_5: _Q35_CONV_GRAD_TOLERANCES,
    ModelCatalogue.QWEN3_5_MOE: _Q35_CONV_GRAD_TOLERANCES,
}
