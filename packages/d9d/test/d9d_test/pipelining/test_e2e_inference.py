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
    _Shared,
    _Transfer,
    build_pp_inputs,
    build_pp_model,
)


def _do_standard_forward(stages: list[PipelineModel], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x_in = x
        for stage in stages:
            x_in = stage(_Transfer(x=x_in), _Shared(y=y)).x
    return x_in


@pytest.mark.parametrize(
    "microbatch_sizes",
    [
        [16],  # single microbatch
        [16, 16],
        [16] * 4,
        [16] * 8,
        [16] * 16,
        [16] * 32,
        [3, 7, 1, 5],  # microbatches within the pack differ in batch size
    ],
)
@pytest.mark.distributed
def test_inference_e2e(dist_ctx_factory, microbatch_sizes: list[int]):
    # Each microbatch's P2P buffers are sized independently, so a pack whose microbatches differ in
    # shape exercises the per-microbatch buffer allocation.
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(pipeline_parallel=8))
    pp_mesh = dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"]
    n_stages = pp_mesh.size()

    torch.manual_seed(4242)
    microbatch_xs = [torch.randn(size, 8, device="cuda") for size in microbatch_sizes]
    microbatch_ys = [torch.randn(size, 8, device="cuda") for size in microbatch_sizes]

    full_stage_modules = [build_pp_model().eval() for _ in range(n_stages)]
    for m in full_stage_modules:
        m.requires_grad_(False)

    ref_outputs = [
        _do_standard_forward(full_stage_modules, x, y) for x, y in zip(microbatch_xs, microbatch_ys, strict=True)
    ]

    this_rank_stages = []

    def _model_provider(stage_info: PipelineStageInfo):
        this_rank_stages.append(stage_info.current_stage)
        return full_stage_modules[stage_info.current_stage]

    collected_results: dict[int, torch.Tensor] = {}

    def _result_fn(microbatch_outputs: _Transfer, microbatch_idx: int):
        collected_results[microbatch_idx] = microbatch_outputs.x.detach().clone()

    schedule_info, _ = build_schedule(
        dist_context=dist_ctx,
        schedule_config=PipelineScheduleInferenceConfig(),
        model_provider=_model_provider,
    )

    inputs_microbatches = tuple(_Transfer(x=x) for x in microbatch_xs)
    shared_microbatches = tuple(_Shared(y=y) for y in microbatch_ys)

    schedule_info.schedule.step(
        inputs_microbatches=inputs_microbatches, shared_microbatches=shared_microbatches, callback=_result_fn
    )

    if pp_mesh.get_local_rank() == pp_mesh.size() - 1:
        assert len(collected_results) == len(microbatch_sizes)
        for idx, (ref, size) in enumerate(zip(ref_outputs, microbatch_sizes, strict=True)):
            assert collected_results[idx].shape[0] == size
            assert torch.allclose(collected_results[idx], ref)

    for this_stage_i in this_rank_stages:
        assert full_stage_modules[this_stage_i].w1.grad is None


@pytest.mark.local
def test_inference_e2e_local(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    x, y = build_pp_inputs(x_with_grad=False)
    model = build_pp_model().eval()
    model.requires_grad_(False)

    ref_output = _do_standard_forward([model], x, y)

    # 4. Setup Wrapper for Build Schedule
    def _model_provider(stage_info: PipelineStageInfo):
        assert stage_info.current_stage == 0
        assert stage_info.num_stages == 1
        return model

    collected_results: dict[int, torch.Tensor] = {}

    def _result_fn(microbatch_outputs: _Transfer, microbatch_idx: int):
        # Offline executor processes everything in one go, always index 0
        collected_results[microbatch_idx] = microbatch_outputs.x.detach().clone()

    schedule_info, _ = build_schedule(
        dist_context=dist_ctx,
        schedule_config=PipelineScheduleInferenceConfig(),
        model_provider=_model_provider,
    )

    assert isinstance(schedule_info.schedule, OfflinePipelineExecutor)

    schedule_info.schedule.step(
        inputs_microbatches=(_Transfer(x=x),), shared_microbatches=(_Shared(y=y),), callback=_result_fn
    )

    # trace should have exactly one result at index 0 because OfflineExecutor does not shard
    assert len(collected_results) == 1
    assert 0 in collected_results

    output = collected_results[0]

    assert torch.allclose(output, ref_output)

    assert model.w1.grad is None
    assert model.w2.grad is None
    assert model.w3.grad is None
