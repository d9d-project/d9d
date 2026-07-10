import pytest
import torch
import torch.nn.functional as F
from d9d.core.dist_context import DeviceMeshParameters
from d9d.module.model.io import (
    SequenceCausalLMHeadShared,
    SequenceHeadsOutput,
    SequenceHeadsShared,
    SequenceInput,
    SequencePoolingHeadShared,
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
from d9d_test.modules.model.sequence.catalogue import ModelCatalogue, parallelize_decoder_with_heads
from d9d_test.modules.model.sequence.multihead.batch import build_multihead_batch, shard_multihead_batch
from d9d_test.modules.model.sequence.multihead.catalogue import (
    D9D_MODEL_FACTORIES_MULTIHEAD,
    HEAD_NAME_CLS,
    HEAD_NAME_LM,
    NUM_LABELS_MULTIHEAD,
)

_N_MICROBATCHES = 2


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("model_type", "model_factory_d9d"),
    [
        pytest.param(model_type, model_factory)
        for model_type, factories in D9D_MODEL_FACTORIES_MULTIHEAD.items()
        for model_factory in factories
    ],
)
@pytest.mark.parametrize("mesh", MESHES_FOR_MODEL_TESTS)
def test_multihead_consistent_to_itself_dist(
    mesh: DeviceMeshParameters, model_type: ModelCatalogue, model_factory_d9d, dist_ctx_factory
):
    dist_ctx = dist_ctx_factory(mesh)
    stage_global = PipelineStageInfo(current_stage=0, num_stages=1)
    batch_global = build_multihead_batch(num_labels=NUM_LABELS_MULTIHEAD)
    batch_dist = shard_multihead_batch(batch_global, dist_ctx)

    lm_delimiter = (batch_global.lm_labels != -100).sum().clamp(min=1)
    total_cls = batch_global.cls_labels.numel()

    dist_loss_accum: list[torch.Tensor] = []

    def _combined_loss(outputs: SequenceHeadsOutput, lm_labels: torch.Tensor, cls_labels: torch.Tensor):
        lm_loss = outputs[HEAD_NAME_LM].logps[lm_labels != -100].sum() / lm_delimiter
        cls_loss = F.cross_entropy(outputs[HEAD_NAME_CLS].scores, cls_labels, reduction="sum") / total_cls
        return lm_loss + cls_loss

    def _build_shared(sequence_shared: SequenceShared, lm_labels: torch.Tensor, pooling_mask: torch.Tensor):
        return SequenceHeadsShared(
            sequence=sequence_shared,
            heads={
                HEAD_NAME_LM: SequenceCausalLMHeadShared(labels=lm_labels),
                HEAD_NAME_CLS: SequencePoolingHeadShared(pooling_mask=pooling_mask),
            },
        )

    # Create Global Model and its Outputs (single stage, both heads active)
    model_global = model_factory_d9d(stage_global)
    outputs_global = model_global(
        SequenceInput(input_ids=batch_global.sequence.input_ids),
        _build_shared(
            SequenceShared(position_ids=batch_global.sequence.position_ids),
            batch_global.lm_labels,
            batch_global.pooling_mask,
        ),
    )
    loss_global = _combined_loss(outputs_global, batch_global.lm_labels, batch_global.cls_labels)
    loss_global.backward()

    # Create Local Model and PP Schedule
    def _callback(outputs: SequenceHeadsOutput, microbatch_idx: int) -> torch.Tensor:
        lm_labels_mb = microbatch_slice(
            batch_dist.lm_labels, microbatch_idx=microbatch_idx, n_microbatches=_N_MICROBATCHES
        )
        cls_labels_mb = microbatch_slice(
            batch_dist.cls_labels, microbatch_idx=microbatch_idx, n_microbatches=_N_MICROBATCHES
        )
        loss_value = _combined_loss(outputs, lm_labels_mb, cls_labels_mb)
        dist_loss_accum.append(loss_value.detach())
        return loss_value

    def _model_provider(dist_stage: PipelineStageInfo) -> nn.Module:
        model_dist = model_factory_d9d(dist_stage)
        parallelize_decoder_with_heads(model_dist, model_type, dist_ctx, dist_stage)
        copy_params_local_to_dist(model_global, model_dist)
        return model_dist

    schedule_info, models_dist = build_schedule(
        dist_context=dist_ctx,
        schedule_config=PipelineScheduleGPipeConfig(),
        model_provider=_model_provider,
    )

    # Run Local Model. input_ids feed the first stage; the shared input reaches every stage and
    # routes position ids to the backbone and labels/pooling mask to their respective heads.
    inputs_microbatches = tuple(
        SequenceInput(
            input_ids=microbatch_slice(batch_dist.sequence.input_ids, microbatch_idx=i, n_microbatches=_N_MICROBATCHES)
        )
        for i in range(_N_MICROBATCHES)
    )
    shared_microbatches = tuple(
        _build_shared(
            SequenceShared(
                position_ids=microbatch_slice(
                    batch_dist.sequence.position_ids, microbatch_idx=i, n_microbatches=_N_MICROBATCHES
                )
            ),
            microbatch_slice(batch_dist.lm_labels, microbatch_idx=i, n_microbatches=_N_MICROBATCHES),
            microbatch_slice(batch_dist.pooling_mask, microbatch_idx=i, n_microbatches=_N_MICROBATCHES),
        )
        for i in range(_N_MICROBATCHES)
    )
    schedule_info.schedule.step(
        inputs_microbatches=inputs_microbatches, shared_microbatches=shared_microbatches, callback=_callback
    )

    # Compare Loss & Grads
    if schedule_info.has_last_stage:
        loss_dist = torch.stack(dist_loss_accum).sum()
        all_reduce_over_mesh_groups(loss_dist, dist_ctx=dist_ctx)
        torch.testing.assert_close(loss_dist, loss_global, atol=1e-3, rtol=0.005)

    for dist_model in models_dist:
        sync_grads_manually(dist_model)
        check_grad_distance_all_local_dist(model_global, dist_model)
