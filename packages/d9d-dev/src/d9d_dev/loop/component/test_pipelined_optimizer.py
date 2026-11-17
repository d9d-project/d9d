import pytest
import torch
from d9d.dist_context import DeviceMeshParameters
from d9d.loop.component.checkpointer import StateCheckpointer
from d9d.loop.component.garbage_collector import ManualGarbageCollector
from d9d.loop.component.job_schedule import JobSchedule
from d9d.loop.component.pipelined_optimizer import PipelinedOptimizer
from d9d.loop.config import CheckpointingConfig, GarbageCollectionConfig, JobScheduleConfig, StepActionSpecial
from d9d.rng import SeedScope, fork_rng
from torch import nn


class _FakeStream:
    @property
    def total_steps(self) -> int | None:
        return None


class _TwoParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.ones(4))
        self.second = nn.Parameter(torch.ones(4))


_OPTIMIZERS = [
    pytest.param(lambda params: torch.optim.AdamW(params, lr=0.5, weight_decay=0.1), id="adamw"),
    pytest.param(lambda params: torch.optim.Adam(params, lr=0.5, weight_decay=0.1), id="adam_l2"),
    pytest.param(
        lambda params: torch.optim.SGD(params, lr=0.5, momentum=0.9, dampening=0.5, weight_decay=0.1),
        id="sgd_dampening_l2",
    ),
]


def _build(model: nn.Module, make_optimizer) -> PipelinedOptimizer:
    return PipelinedOptimizer(optimizers={0: make_optimizer(list(model.parameters()))})


def _step(optimizer: PipelinedOptimizer, model: nn.Module, used: list[bool]):
    for param, is_used in zip(model.parameters(), used, strict=True):
        param.grad = torch.arange(4.0) if is_used else None
    with fork_rng(SeedScope.root(0)):
        optimizer.step()


@pytest.mark.local
@pytest.mark.parametrize("make_optimizer", _OPTIMIZERS)
@pytest.mark.parametrize(
    "history",
    [
        pytest.param([], id="no_step"),
        pytest.param([[True, False]], id="second_unused"),
        pytest.param([[True, True], [True, True]], id="both_used"),
    ],
)
def test_resume_matches_a_continuous_run(dist_ctx_factory, tmp_path, make_optimizer, history):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=1), stream=_FakeStream())
    checkpointer = StateCheckpointer(
        dist_context=dist_ctx,
        schedule=schedule,
        config=CheckpointingConfig(save_dir=tmp_path, period_steps=1),
        gc=ManualGarbageCollector(GarbageCollectionConfig(period_steps=StepActionSpecial.disable), schedule),
        run_name=None,
    )
    continuous_model, saved_model, restored_model = _TwoParameters(), _TwoParameters(), _TwoParameters()
    continuous = _build(continuous_model, make_optimizer)
    saved = _build(saved_model, make_optimizer)
    for used in history:
        _step(continuous, continuous_model, used)
        _step(saved, saved_model, used)
    schedule.step()
    checkpointer.checkpoint_if_needed(saved)

    restored = _build(restored_model, make_optimizer)
    checkpointer.load_last_checkpoint(restored)
    restored_model.load_state_dict(saved_model.state_dict())

    # A save must not change the optimizer that continues, and a load must restore the same steps.
    for used in ([True, True], [True, True]):
        for optimizer, model in ((continuous, continuous_model), (saved, saved_model), (restored, restored_model)):
            _step(optimizer, model, used)

    for model in (saved_model, restored_model):
        for param, expected in zip(model.parameters(), continuous_model.parameters(), strict=True):
            torch.testing.assert_close(param, expected)
