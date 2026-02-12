import pytest
import torch
from d9d.core.dist_context import REGULAR_DOMAIN, DeviceMeshParameters
from d9d.pipelining.api import PipelineStageInfo
from d9d.pipelining.factory import (
    PipelineScheduleInferenceConfig,
    build_schedule,
)
from d9d.pipelining.infra.schedule.component.runtime import OfflinePipelineExecutor

from d9d_test.pipelining.definitions import (
    PipelineModel,
    build_pp_inputs,
    build_pp_model,
)


def _do_standard_forward(stages: list[PipelineModel], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x_in = x
        for stage in stages:
            x_in = stage(x=x_in, y=y)["x"]
    return x_in


@pytest.mark.parametrize("n_microbatches", [1, 2, 4, 8, 16, 32])
@pytest.mark.distributed
def test_inference_e2e(dist_ctx_factory, n_microbatches: int):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(pipeline_parallel=8))
    pp_mesh = dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"]
    n_stages = pp_mesh.size()

    x, y = build_pp_inputs(x_with_grad=False)

    full_stage_modules = [build_pp_model().eval() for _ in range(n_stages)]
    for m in full_stage_modules:
        m.requires_grad_(False)  # noqa: FBT003

    ref_output = _do_standard_forward(full_stage_modules, x, y)

    this_rank_stages = []

    def _model_provider(stage_info: PipelineStageInfo):
        this_rank_stages.append(stage_info.current_stage)
        return full_stage_modules[stage_info.current_stage]

    collected_results: dict[int, torch.Tensor] = {}

    def _result_fn(microbatch_outputs: dict[str, torch.Tensor], microbatch_idx: int):
        collected_results[microbatch_idx] = microbatch_outputs["x"].detach().clone()

    schedule_info, _ = build_schedule(
        dist_context=dist_ctx,
        n_microbatches=n_microbatches,
        schedule_config=PipelineScheduleInferenceConfig(),
        model_provider=_model_provider,
        callback=_result_fn,
    )

    schedule_info.schedule.configure_buffers(inputs={"x": x}, kwargs={"y": y}, sharding_spec=None)

    schedule_info.schedule.step(inputs={"x": x}, kwargs={"y": y})

    if pp_mesh.get_local_rank() == pp_mesh.size() - 1:
        assert len(collected_results) == n_microbatches

        sorted_chunks = [collected_results[i] for i in range(n_microbatches)]
        final_pipeline_output = torch.cat(sorted_chunks, dim=0)

        assert torch.allclose(final_pipeline_output, ref_output)

    for this_stage_i in this_rank_stages:
        assert full_stage_modules[this_stage_i].w1.grad is None


@pytest.mark.local
def test_inference_e2e_local(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    x, y = build_pp_inputs(x_with_grad=False)
    model = build_pp_model().eval()
    model.requires_grad_(False)  # noqa: FBT003

    ref_output = _do_standard_forward([model], x, y)

    # 4. Setup Wrapper for Build Schedule
    def _model_provider(stage_info: PipelineStageInfo):
        assert stage_info.current_stage == 0
        assert stage_info.num_stages == 1
        return model

    collected_results: dict[int, torch.Tensor] = {}

    def _result_fn(microbatch_outputs: dict[str, torch.Tensor], microbatch_idx: int):
        # Offline executor processes everything in one go, always index 0
        collected_results[microbatch_idx] = microbatch_outputs["x"].detach().clone()

    schedule_info, _ = build_schedule(
        dist_context=dist_ctx,
        n_microbatches=4,  # Ignored by offline executor strategies usually, but passed for API
        schedule_config=PipelineScheduleInferenceConfig(),
        model_provider=_model_provider,
        callback=_result_fn,
    )

    assert isinstance(schedule_info.schedule, OfflinePipelineExecutor)

    schedule_info.schedule.configure_buffers(inputs={"x": x}, kwargs={"y": y}, sharding_spec=None)

    schedule_info.schedule.step(inputs={"x": x}, kwargs={"y": y})

    # trace should have exactly one result at index 0 because OfflineExecutor does not shard
    assert len(collected_results) == 1
    assert 0 in collected_results

    output = collected_results[0]

    assert torch.allclose(output, ref_output)

    assert model.w1.grad is None
    assert model.w2.grad is None
    assert model.w3.grad is None
