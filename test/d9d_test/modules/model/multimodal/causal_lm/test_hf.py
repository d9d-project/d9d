import pytest
from d9d.module.block.head import SequenceCausalLMHeadShared
from d9d.module.model.io import MultimodalSequenceInput, SequenceHeadShared, SequenceShared
from d9d.pipelining.api import PipelineStageInfo
from torch.testing import assert_close

from d9d_test.modules.helper import assert_mapped_gradients_close, clone_module_weights
from d9d_test.modules.model.multimodal.causal_lm.batch import build_multimodal_causal_lm_batch
from d9d_test.modules.model.multimodal.causal_lm.catalogue import (
    D9D_MODEL_FACTORIES_MULTIMODAL,
    D9D_TO_HF_MAPPER_MULTIMODAL,
    HF_GRAD_TOLERANCES_MULTIMODAL,
    HF_MODEL_FACTORY_MULTIMODAL,
    HF_TO_D9D_MAPPER_MULTIMODAL,
    VISION_FEATURE_DIM,
    VISION_SPATIAL_MERGE_SIZE,
)
from d9d_test.modules.model.sequence.catalogue import MultimodalModelCatalogue

_IGNORE_INDEX = -100


@pytest.mark.local
@pytest.mark.parametrize(
    ("model_type", "model_factory_d9d"),
    [
        pytest.param(
            model_type,
            model_factory,
        )
        for model_type, factories in D9D_MODEL_FACTORIES_MULTIMODAL.items()
        for model_factory in factories
    ],
)
def test_consistent_to_hf(model_type: MultimodalModelCatalogue, model_factory_d9d):
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    batch = build_multimodal_causal_lm_batch(
        feature_dim=VISION_FEATURE_DIM,
        spatial_merge_size=VISION_SPATIAL_MERGE_SIZE,
    )

    labels_shift = batch.labels[:, 1:]

    model_hf = HF_MODEL_FACTORY_MULTIMODAL[model_type]()

    outputs_hf = model_hf(
        input_ids=batch.input_ids,
        pixel_values=batch.media.features.bfloat16(),
        image_grid_thw=batch.media.grid_thw,
        position_ids=batch.position_ids,
        labels=batch.labels,
    )
    outputs_hf.loss.backward()

    model_d9d = model_factory_d9d(stage)
    clone_module_weights(from_module=model_hf, to_module=model_d9d, map_with=HF_TO_D9D_MAPPER_MULTIMODAL[model_type])

    outputs_d9d = model_d9d(
        MultimodalSequenceInput(
            input_ids=batch.input_ids[:, :-1],
            media=batch.media,
            media_token_mask=batch.media_token_mask[:, :-1],
        ),
        SequenceHeadShared(
            sequence=SequenceShared(position_ids=batch.position_ids[:, :, :-1]),
            head=SequenceCausalLMHeadShared(labels=labels_shift),
        ),
    )
    loss_d9d = outputs_d9d.logps[labels_shift != _IGNORE_INDEX].mean()
    loss_d9d.backward()

    assert_close(loss_d9d, outputs_hf.loss, atol=1e-3, rtol=0.005)
    assert_mapped_gradients_close(
        from_module=model_d9d,
        to_module=model_hf,
        map_with=D9D_TO_HF_MAPPER_MULTIMODAL[model_type],
        tolerances=HF_GRAD_TOLERANCES_MULTIMODAL[model_type],
    )
