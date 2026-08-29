import pytest
import torch
from d9d.core.dist_context import DeviceMeshParameters
from d9d.module.base import MediaSegments
from d9d.module.block.head import SequenceCausalLMHeadShared, SequenceCausalLMOutput
from d9d.module.model.io import (
    MultimodalSequenceInput,
    SequenceHeadShared,
    SequenceShared,
)
from d9d.pipelining.api import PipelineStageInfo
from d9d.pipelining.factory import PipelineScheduleGPipeConfig, build_schedule
from torch import nn

from d9d_test.modules.helper import (
    all_reduce_over_mesh_groups,
    check_grad_distance_all_local_dist,
    copy_params_local_to_dist,
    microbatch_slice,
    sync_grads_manually,
)
from d9d_test.modules.model.meshes import MESHES_FOR_MODEL_TESTS
from d9d_test.modules.model.multimodal.causal_lm.batch import (
    build_uniform_multimodal_causal_lm_batch,
    shard_uniform_multimodal_batch,
)
from d9d_test.modules.model.multimodal.causal_lm.catalogue import (
    D9D_MODEL_FACTORIES_MULTIMODAL,
    VISION_FEATURE_DIM,
    VISION_SPATIAL_MERGE_SIZE,
    parallelize_multimodal_decoder,
)
from d9d_test.modules.model.sequence.catalogue import MultimodalModelCatalogue

_N_MICROBATCHES = 2
_IGNORE_INDEX = -100


def _slice_media(media: MediaSegments, microbatch_idx: int, n_microbatches: int) -> MediaSegments:
    """Slices a uniform-media packed stream per microbatch (one image per sample)."""
    num_segments = media.grid_thw.shape[0]
    patches_per_sample = media.features.shape[0] // num_segments
    features = media.features.view(num_segments, patches_per_sample, -1)

    return MediaSegments(
        features=microbatch_slice(features, microbatch_idx=microbatch_idx, n_microbatches=n_microbatches).flatten(0, 1),
        grid_thw=microbatch_slice(media.grid_thw, microbatch_idx=microbatch_idx, n_microbatches=n_microbatches),
    )


@pytest.mark.distributed
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
@pytest.mark.parametrize("mesh", MESHES_FOR_MODEL_TESTS)
def test_consistent_to_itself_dist(
    mesh: DeviceMeshParameters, model_type: MultimodalModelCatalogue, model_factory_d9d, dist_ctx_factory
):
    dist_ctx = dist_ctx_factory(mesh)
    stage_global = PipelineStageInfo(current_stage=0, num_stages=1)
    batch_global = build_uniform_multimodal_causal_lm_batch(
        feature_dim=VISION_FEATURE_DIM, spatial_merge_size=VISION_SPATIAL_MERGE_SIZE
    )
    batch_dist = shard_uniform_multimodal_batch(batch_global, dist_ctx)
    loss_delimiter = (batch_global.labels != _IGNORE_INDEX).sum().clamp(min=1)

    dist_loss_accum: list[torch.Tensor] = []

    # Create Global Model and its Outputs
    model_global = model_factory_d9d(stage_global)
    outputs_global = model_global(
        MultimodalSequenceInput(
            input_ids=batch_global.input_ids,
            media=batch_global.media,
            media_token_mask=batch_global.media_token_mask,
        ),
        SequenceHeadShared(
            sequence=SequenceShared(position_ids=batch_global.position_ids),
            head=SequenceCausalLMHeadShared(labels=batch_global.labels),
        ),
    )
    loss_global = outputs_global.logps[batch_global.labels != _IGNORE_INDEX].sum() / loss_delimiter
    loss_global.backward()

    # Create Local Model and PP Schedule
    def _callback(outputs: SequenceCausalLMOutput, microbatch_idx: int) -> torch.Tensor:
        labels_mb = microbatch_slice(batch_dist.labels, microbatch_idx=microbatch_idx, n_microbatches=_N_MICROBATCHES)
        loss_value = outputs.logps[labels_mb != _IGNORE_INDEX].sum() / loss_delimiter
        dist_loss_accum.append(loss_value.detach())
        return loss_value

    def _model_provider(dist_stage: PipelineStageInfo) -> nn.Module:
        model_dist = model_factory_d9d(dist_stage)
        parallelize_multimodal_decoder(model_dist, model_type, dist_ctx, dist_stage)
        copy_params_local_to_dist(model_global, model_dist)
        return model_dist

    schedule_info, models_dist = build_schedule(
        dist_context=dist_ctx,
        schedule_config=PipelineScheduleGPipeConfig(),
        model_provider=_model_provider,
    )

    # Run Local Model
    inputs_microbatches = tuple(
        MultimodalSequenceInput(
            input_ids=microbatch_slice(batch_dist.input_ids, microbatch_idx=i, n_microbatches=_N_MICROBATCHES),
            media=_slice_media(batch_dist.media, microbatch_idx=i, n_microbatches=_N_MICROBATCHES),
            media_token_mask=microbatch_slice(
                batch_dist.media_token_mask, microbatch_idx=i, n_microbatches=_N_MICROBATCHES
            ),
        )
        for i in range(_N_MICROBATCHES)
    )
    shared_microbatches = tuple(
        SequenceHeadShared(
            sequence=SequenceShared(
                position_ids=microbatch_slice(
                    batch_dist.position_ids.transpose(0, 1), microbatch_idx=i, n_microbatches=_N_MICROBATCHES
                ).transpose(0, 1)
            ),
            head=SequenceCausalLMHeadShared(
                labels=microbatch_slice(batch_dist.labels, microbatch_idx=i, n_microbatches=_N_MICROBATCHES)
            ),
        )
        for i in range(_N_MICROBATCHES)
    )
    schedule_info.schedule.step(
        inputs_microbatches=inputs_microbatches,
        shared_microbatches=shared_microbatches,
        callback=_callback,
    )

    # Compare Loss & Grads
    if schedule_info.has_last_stage:
        loss_dist = torch.stack(dist_loss_accum).sum()
        all_reduce_over_mesh_groups(loss_dist, dist_ctx=dist_ctx)
        torch.testing.assert_close(loss_dist, loss_global, atol=1e-3, rtol=0.005)

    for dist_model in models_dist:
        sync_grads_manually(dist_model)
        check_grad_distance_all_local_dist(model_global, dist_model)
