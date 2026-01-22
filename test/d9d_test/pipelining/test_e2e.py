import pytest
import torch
from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.pipelining.api import PipelineShardingSpec, PipelineStageInfo
from d9d.pipelining.factory import (
    AnyPipelineScheduleConfig,
    PipelineSchedule1F1BConfig,
    PipelineScheduleDualPipeVConfig,
    PipelineScheduleGPipeConfig,
    PipelineScheduleLoopedBFSConfig,
    PipelineScheduleZeroBubbleVConfig,
    build_schedule,
)

from d9d_test.pipelining.definitions import (
    PipelineModel,
    build_pp_inputs,
    build_pp_model,
    check_pp_hooks_ran,
    extract_grad,
    register_pp_hooks,
)


def _do_standard_backward(stages: list[PipelineModel], x: torch.Tensor, y: torch.Tensor):
    x_in = x
    for stage in stages:
        x_in = stage(x=x_in, y=y)["x"]
    loss = x_in.sum()
    loss.backward()
    snapshot = [
        {
            "w1": extract_grad(stage.w1),
            "w2": extract_grad(stage.w2),
            "w3": extract_grad(stage.w3),
        }
        for stage in stages
    ]
    for stage in stages:
        stage.zero_grad()
    x.grad = None
    y.grad = None
    return snapshot


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("schedule_config", "stages_per_rank"),
    [
        (PipelineScheduleGPipeConfig(), 1),
        (PipelineScheduleLoopedBFSConfig(num_stages_per_rank=1), 1),
        (PipelineScheduleLoopedBFSConfig(num_stages_per_rank=2), 2),
        (PipelineSchedule1F1BConfig(num_stages_per_rank=1, zero_bubble=False), 1),
        (PipelineSchedule1F1BConfig(num_stages_per_rank=2, zero_bubble=False), 2),
        (PipelineSchedule1F1BConfig(num_stages_per_rank=2, zero_bubble=True), 2),
        (PipelineScheduleZeroBubbleVConfig(), 2),
        (PipelineScheduleDualPipeVConfig(), 2),
    ]
)
@pytest.mark.parametrize(
    "n_microbatches",
    [1, 2, 4, 8, 16, 32]
)
@pytest.mark.parametrize(
    "freeze_w1",
    [True, False]
)
def test_e2e(
        dist_ctx_pp: DistributedContext, schedule_config: AnyPipelineScheduleConfig, stages_per_rank: int,
        n_microbatches: int, freeze_w1: bool
):
    pp_mesh = dist_ctx_pp.mesh_for(REGULAR_DOMAIN)["pp"]
    n_stages = stages_per_rank * pp_mesh.size()
    if n_microbatches < n_stages and isinstance(schedule_config, PipelineScheduleDualPipeVConfig):
        pytest.skip("DualPipeV too small microbatch")
    x, y = build_pp_inputs(x_with_grad=False)
    full_stage_modules = [build_pp_model() for _ in range(n_stages)]
    if freeze_w1:
        for module in full_stage_modules:
            module.w1.requires_grad = False
    snapshot = _do_standard_backward(full_stage_modules, x, y)

    hooks = [register_pp_hooks(x) for x in full_stage_modules]
    this_rank_stages = []

    def _model_provider(stage_info: PipelineStageInfo):
        this_rank_stages.append(stage_info.current_stage)
        return full_stage_modules[stage_info.current_stage]

    loss_seen_microbatches = []

    def _loss_fn(microbatch: dict[str, torch.Tensor], microbatch_idx: int):
        loss_seen_microbatches.append(microbatch_idx)
        return microbatch["x"].sum()

    schedule_info, _ = build_schedule(
        dist_context=dist_ctx_pp,
        n_microbatches=n_microbatches,
        schedule_config=schedule_config,
        model_provider=_model_provider,
        loss_fn=_loss_fn,
        sharding_spec=PipelineShardingSpec()
    )

    not_this_rank_stages = [i for i in range(len(full_stage_modules)) if i not in this_rank_stages]

    schedule_info.schedule.configure_buffers(inputs={"x": x}, kwargs={"y": y})

    schedule_info.schedule.step(inputs={"x": x}, kwargs={"y": y})

    assert x.grad is None
    assert y.grad is None

    for this_stage_i in this_rank_stages:
        this_stage = full_stage_modules[this_stage_i]
        this_snapshot = snapshot[this_stage_i]
        this_hooks = hooks[this_stage_i]
        if freeze_w1:
            assert this_stage.w1.grad is None
        else:
            assert torch.allclose(this_stage.w1.grad, this_snapshot["w1"], rtol=1e-3)
        assert torch.allclose(this_stage.w2.grad, this_snapshot["w2"], rtol=1e-3)
        assert torch.allclose(this_stage.w3.grad, this_snapshot["w3"], rtol=1e-3)
        check_pp_hooks_ran(this_hooks, n_microbatches, override_w1=0 if freeze_w1 else None)

    for not_this_stage_i in not_this_rank_stages:
        not_this_stage = full_stage_modules[not_this_stage_i]
        not_this_hooks = hooks[not_this_stage_i]
        assert not_this_stage.w1.grad is None
        assert not_this_stage.w2.grad is None
        assert not_this_stage.w3.grad is None
        check_pp_hooks_ran(not_this_hooks, 0)
