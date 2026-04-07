import pytest
import torch
from d9d.core.dist_context import DeviceMeshParameters
from d9d.pipelining.api import PipelineShardingSpec, PipelineStageInfo
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
from d9d_test.modules.model.sequence.catalogue import ModelCatalogue
from d9d_test.modules.model.sequence.causal_lm.batch import build_causal_lm_batch, shard_causal_lm_batch
from d9d_test.modules.model.sequence.causal_lm.catalogue import D9D_MODEL_FACTORIES_CAUSAL_LM, D9D_PARALLELIZE_FN

_N_MICROBATCHES = 2


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("model_type", "model_factory_d9d"),
    [
        pytest.param(
            model_type,
            model_factory,
        )
        for model_type, factories in D9D_MODEL_FACTORIES_CAUSAL_LM.items()
        for model_factory in factories
    ],
)
@pytest.mark.parametrize("mesh", MESHES_FOR_MODEL_TESTS)
def test_consistent_to_itself_dist(
    mesh: DeviceMeshParameters, model_type: ModelCatalogue, model_factory_d9d, dist_ctx_factory
):
    dist_ctx = dist_ctx_factory(mesh)
    stage_global = PipelineStageInfo(current_stage=0, num_stages=1)
    batch_global = build_causal_lm_batch()
    batch_dist = shard_causal_lm_batch(batch_global, dist_ctx)
    loss_delimiter = (batch_global.labels != -100).sum().clamp(min=1)

    dist_loss_accum: list[torch.Tensor] = []

    # Create Global Model and its Outputs
    model_global = model_factory_d9d(stage_global)
    outputs_global = model_global(
        input_ids=batch_global.sequence.input_ids,
        position_ids=batch_global.sequence.position_ids,
        labels=batch_global.labels,
    )
    loss_global = outputs_global["logps"][batch_global.labels != -100].sum() / loss_delimiter
    loss_global.backward()

    # Create Local Model and PP Schedule
    def _callback(outputs: dict[str, torch.Tensor], microbatch_idx: int) -> torch.Tensor:
        labels_mb = microbatch_slice(batch_dist.labels, microbatch_idx=microbatch_idx, n_microbatches=_N_MICROBATCHES)
        loss_value = outputs["logps"][labels_mb != -100].sum() / loss_delimiter
        dist_loss_accum.append(loss_value.detach())
        return loss_value

    def _model_provider(dist_stage: PipelineStageInfo) -> nn.Module:
        model_dist = model_factory_d9d(dist_stage)
        D9D_PARALLELIZE_FN[model_type](dist_ctx, model_dist, dist_stage)
        copy_params_local_to_dist(model_global, model_dist)
        return model_dist

    schedule_info, models_dist = build_schedule(
        dist_context=dist_ctx,
        n_microbatches=_N_MICROBATCHES,
        schedule_config=PipelineScheduleGPipeConfig(),
        model_provider=_model_provider,
        callback=_callback,
    )

    # Run Local Model
    schedule_info.schedule.configure_buffers(
        inputs={"input_ids": batch_dist.sequence.input_ids},
        kwargs={
            "position_ids": batch_dist.sequence.position_ids,
            "labels": batch_dist.labels,
        },
        sharding_spec=PipelineShardingSpec(),
    )
    schedule_info.schedule.step(
        inputs={"input_ids": batch_dist.sequence.input_ids},
        kwargs={
            "position_ids": batch_dist.sequence.position_ids,
            "labels": batch_dist.labels,
        },
    )

    # Compare Loss & Grads
    if schedule_info.has_last_stage:
        loss_dist = torch.stack(dist_loss_accum).sum()
        all_reduce_over_mesh_groups(loss_dist, dist_ctx=dist_ctx)
        torch.testing.assert_close(loss_dist, loss_global, atol=1e-4, rtol=0.001)

    for dist_model in models_dist:
        sync_grads_manually(dist_model)
        check_grad_distance_all_local_dist(model_global, dist_model)
