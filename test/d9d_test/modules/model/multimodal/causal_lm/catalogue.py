import transformers as tr
from d9d.core.dist_context import DistributedContext
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model import DecoderForCausalLM, DecoderWithHead, MultimodalBackbone
from d9d.module.model.qwen3_5_moe import (
    Qwen3p5MoEModel,
    Qwen3p5MoEVisionModel,
    Qwen3p5MoEVisionParameters,
    mapper_from_huggingface_qwen3p5_moe_for_conditional_generation,
    mapper_to_huggingface_qwen3p5_moe_for_conditional_generation,
)
from d9d.module.parallelism.model import parallelize_causal_lm_head
from d9d.module.parallelism.model.qwen3_5_moe import parallelize_qwen3p5_moe_model, parallelize_qwen3p5_moe_vision
from d9d.pipelining.api import PipelineStageInfo
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeVisionConfig,
)

from d9d_test.modules.helper import GradTolerance, torch_seed
from d9d_test.modules.model.sequence.catalogue import (
    D9D_MODEL_PARAMETERS,
    HF_MODEL_PARAMETERS,
    ModelCatalogue,
    MultimodalModelCatalogue,
    hf_model_factory,
)

_VISION_DEPTH = 2
_VISION_HIDDEN_SIZE = 128
_VISION_INTERMEDIATE_SIZE = 256
_VISION_NUM_HEADS = 4
_VISION_IN_CHANNELS = 3
_VISION_PATCH_SIZE = 4
_VISION_TEMPORAL_PATCH_SIZE = 2
_VISION_SPATIAL_MERGE_SIZE = 2
_VISION_NUM_POSITION_EMBEDDINGS = 64
_VISION_MAX_GRID_SIDE = 64
_VISION_NORM_EPS = 1e-6

VISION_FEATURE_DIM = _VISION_IN_CHANNELS * _VISION_TEMPORAL_PATCH_SIZE * _VISION_PATCH_SIZE * _VISION_PATCH_SIZE
VISION_SPATIAL_MERGE_SIZE = _VISION_SPATIAL_MERGE_SIZE

_MEDIA_TOKEN_ID = 98


D9D_VISION_PARAMETERS = {
    MultimodalModelCatalogue.QWEN3_5_MOE: Qwen3p5MoEVisionParameters(
        depth=_VISION_DEPTH,
        hidden_size=_VISION_HIDDEN_SIZE,
        intermediate_size=_VISION_INTERMEDIATE_SIZE,
        num_attention_heads=_VISION_NUM_HEADS,
        in_channels=_VISION_IN_CHANNELS,
        patch_size=_VISION_PATCH_SIZE,
        temporal_patch_size=_VISION_TEMPORAL_PATCH_SIZE,
        spatial_merge_size=_VISION_SPATIAL_MERGE_SIZE,
        out_hidden_size=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE].layer.hidden_size,
        num_position_embeddings=_VISION_NUM_POSITION_EMBEDDINGS,
        max_grid_side=_VISION_MAX_GRID_SIDE,
        norm_eps=_VISION_NORM_EPS,
    ),
}


BACKBONE_FOR_MULTIMODAL = {
    MultimodalModelCatalogue.QWEN3_5_MOE: ModelCatalogue.QWEN3_5_MOE,
}
"""The text backbone each multimodal entry wraps, keyed into the sequence catalogue."""


HF_MODEL_PARAMETERS_MULTIMODAL = {
    MultimodalModelCatalogue.QWEN3_5_MOE: Qwen3_5MoeConfig(
        text_config=HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE],
        vision_config=Qwen3_5MoeVisionConfig(
            depth=_VISION_DEPTH,
            hidden_size=_VISION_HIDDEN_SIZE,
            intermediate_size=_VISION_INTERMEDIATE_SIZE,
            num_heads=_VISION_NUM_HEADS,
            in_channels=_VISION_IN_CHANNELS,
            patch_size=_VISION_PATCH_SIZE,
            temporal_patch_size=_VISION_TEMPORAL_PATCH_SIZE,
            spatial_merge_size=_VISION_SPATIAL_MERGE_SIZE,
            out_hidden_size=D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE].layer.hidden_size,
            num_position_embeddings=_VISION_NUM_POSITION_EMBEDDINGS,
        ),
        image_token_id=_MEDIA_TOKEN_ID,
        video_token_id=_MEDIA_TOKEN_ID + 1,
        vision_start_token_id=_MEDIA_TOKEN_ID - 2,
        vision_end_token_id=_MEDIA_TOKEN_ID - 1,
        _attn_implementation="sdpa",
    ),
}


HF_MODEL_FACTORY_MULTIMODAL = {
    MultimodalModelCatalogue.QWEN3_5_MOE: hf_model_factory(
        tr.Qwen3_5MoeForConditionalGeneration,
        config=HF_MODEL_PARAMETERS_MULTIMODAL[MultimodalModelCatalogue.QWEN3_5_MOE],
        bf16_layers=["model", "lm_head"],
    ),
}


_D9D_INIT_SEED = 123213

VISION_CLASSES = {
    MultimodalModelCatalogue.QWEN3_5_MOE: Qwen3p5MoEVisionModel,
}

BACKBONE_CLASSES_MULTIMODAL = {
    MultimodalModelCatalogue.QWEN3_5_MOE: Qwen3p5MoEModel,
}


def make_multimodal_model_factory(model_type: MultimodalModelCatalogue, enable_checkpointing: bool):
    """Builds a factory composing backbone + modality encoder + causal LM head on the target device.

    This is the multimodal counterpart of ``make_d9d_model_factory``: the vision tower is wrapped
    around the text backbone by ``MultimodalBackbone``, and the task head attaches to that
    composition exactly as it would to a text-only backbone.
    """

    def _build_fn(stage: PipelineStageInfo) -> DecoderForCausalLM:
        with torch_seed(_D9D_INIT_SEED):
            backbone = BACKBONE_CLASSES_MULTIMODAL[model_type](
                D9D_MODEL_PARAMETERS[BACKBONE_FOR_MULTIMODAL[model_type]],
                stage,
                hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
                enable_checkpointing=enable_checkpointing,
            )
            encoder = VISION_CLASSES[model_type](D9D_VISION_PARAMETERS[model_type])
            model = DecoderForCausalLM(MultimodalBackbone(backbone, encoder, stage), stage).cuda().bfloat16()
            model.reset_parameters()
            return model

    return _build_fn


D9D_MODEL_FACTORIES_MULTIMODAL = {
    MultimodalModelCatalogue.QWEN3_5_MOE: [
        make_multimodal_model_factory(MultimodalModelCatalogue.QWEN3_5_MOE, enable_checkpointing=enable_checkpointing)
        for enable_checkpointing in (True, False)
    ],
}


HF_TO_D9D_MAPPER_MULTIMODAL = {
    MultimodalModelCatalogue.QWEN3_5_MOE: mapper_from_huggingface_qwen3p5_moe_for_conditional_generation(
        D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE],
        D9D_VISION_PARAMETERS[MultimodalModelCatalogue.QWEN3_5_MOE],
    ),
}


D9D_TO_HF_MAPPER_MULTIMODAL = {
    MultimodalModelCatalogue.QWEN3_5_MOE: mapper_to_huggingface_qwen3p5_moe_for_conditional_generation(
        D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE],
        D9D_VISION_PARAMETERS[MultimodalModelCatalogue.QWEN3_5_MOE],
    ),
}


_BACKBONE_PARALLELIZE_MULTIMODAL = {
    MultimodalModelCatalogue.QWEN3_5_MOE: parallelize_qwen3p5_moe_model,
}

_VISION_PARALLELIZE_MULTIMODAL = {
    MultimodalModelCatalogue.QWEN3_5_MOE: parallelize_qwen3p5_moe_vision,
}


def parallelize_multimodal_decoder(
    model: DecoderWithHead,
    model_type: MultimodalModelCatalogue,
    dist_context: DistributedContext,
    stage: PipelineStageInfo,
) -> None:
    """Parallelizes a composed multimodal decoder: text backbone, vision encoder, then the head.

    Each part keeps its own routine, mirroring how ``parallelize_decoder`` splits a text-only
    backbone from its heads. ``model.model`` is the ``MultimodalBackbone``, so the wrapped text
    backbone is ``model.model.model`` and the encoder is ``model.model.encoder``.
    """
    _BACKBONE_PARALLELIZE_MULTIMODAL[model_type](dist_context, model.model.model, stage)

    if stage.is_current_stage_first:
        _VISION_PARALLELIZE_MULTIMODAL[model_type](dist_context, model.model.encoder)

    if stage.is_current_stage_last:
        parallelize_causal_lm_head(model.head, dist_context)


# See sequence/causal_lm/catalogue.py for the rationale on the conv1d override.
HF_GRAD_TOLERANCES_MULTIMODAL: dict[MultimodalModelCatalogue, dict[str, GradTolerance] | None] = {
    MultimodalModelCatalogue.QWEN3_5_MOE: {
        f"model.language_model.layers.{layer_i}.linear_attn.conv1d.weight": GradTolerance(
            tol_angle=0.35, tol_norm_abs=3e-4, tol_norm_rel=1.0
        )
        for layer_i in range(D9D_MODEL_PARAMETERS[ModelCatalogue.QWEN3_5_MOE].num_hidden_layers)
    },
}
