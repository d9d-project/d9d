import pytest
import torch
from d9d.core.dist_context import DeviceMeshParameters
from d9d.module.block.head import SequenceEmbeddingOutput, SequencePoolingHeadShared
from d9d.module.model.io import (
    SequenceHeadsOutput,
    SequenceHeadsShared,
    SequenceInput,
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
from d9d_test.modules.model.sequence.embedding.batch import build_embedding_batch, shard_embedding_batch
from d9d_test.modules.model.sequence.embedding.catalogue import D9D_MODEL_FACTORIES_EMBEDDING, HEAD_NAME_EMBEDDING

_N_MICROBATCHES = 2


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("model_type", "model_factory_d9d"),
    [
        pytest.param(
            model_type,
            model_factory,
        )
        for model_type, factories in D9D_MODEL_FACTORIES_EMBEDDING.items()
        for model_factory in factories
    ],
)
@pytest.mark.parametrize("mesh", MESHES_FOR_MODEL_TESTS)
def test_consistent_to_itself_dist(
    mesh: DeviceMeshParameters, model_type: ModelCatalogue, model_factory_d9d, dist_ctx_factory
) -> None:
    dist_ctx = dist_ctx_factory(mesh)
    stage_global = PipelineStageInfo(current_stage=0, num_stages=1)
    batch_global = build_embedding_batch()
    batch_dist = shard_embedding_batch(batch_global, dist_ctx)

    dist_loss_accum: list[torch.Tensor] = []

    # Create Global Model and its Outputs
    model_global = model_factory_d9d(stage_global)
    outputs_global = model_global(
        SequenceInput(input_ids=batch_global.sequence.input_ids),
        SequenceHeadsShared(
            sequence=SequenceShared(position_ids=batch_global.sequence.position_ids),
            heads={HEAD_NAME_EMBEDDING: SequencePoolingHeadShared(pooling_mask=batch_global.pooling_mask)},
        ),
    )

    # Use MSE-style dummy loss for embedding gradient tests
    loss_global = outputs_global[HEAD_NAME_EMBEDDING].embeddings.mean()
    loss_global.backward()

    # Create Local Model and PP Schedule
    def _callback(outputs: SequenceHeadsOutput[SequenceEmbeddingOutput], microbatch_idx: int) -> torch.Tensor:
        embeddings = outputs[HEAD_NAME_EMBEDDING].embeddings
        loss_value = embeddings.sum() / batch_global.pooling_mask.sum() / embeddings.shape[1]
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

    # Run Local Model
    inputs_microbatches = tuple(
        SequenceInput(
            input_ids=microbatch_slice(batch_dist.sequence.input_ids, microbatch_idx=i, n_microbatches=_N_MICROBATCHES)
        )
        for i in range(_N_MICROBATCHES)
    )
    shared_microbatches = tuple(
        SequenceHeadsShared(
            sequence=SequenceShared(
                position_ids=microbatch_slice(
                    batch_dist.sequence.position_ids, microbatch_idx=i, n_microbatches=_N_MICROBATCHES
                )
            ),
            heads={
                HEAD_NAME_EMBEDDING: SequencePoolingHeadShared(
                    pooling_mask=microbatch_slice(
                        batch_dist.pooling_mask, microbatch_idx=i, n_microbatches=_N_MICROBATCHES
                    )
                )
            },
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
