import time
from typing import Any

import pytest
import torch
from d9d.dist_context import DENSE_DOMAIN, DenseDim, DenseDomainConfig, DeviceMeshParameters, DistributedContext
from d9d.loop.component import TimeoutManager, TrackedModules
from d9d.loop.component.checkpointer import StateCheckpointer
from d9d.loop.component.garbage_collector import ManualGarbageCollector
from d9d.loop.component.job_schedule import JobSchedule
from d9d.loop.config import (
    CheckpointingConfig,
    GarbageCollectionConfig,
    JobScheduleConfig,
    StepActionSpecial,
    TimeoutConfig,
)
from d9d.parallel import parallelize_fsdp
from torch import nn
from torch.distributed.constants import default_pg_nccl_timeout


class _FakeStream:
    @property
    def total_steps(self) -> int | None:
        return None


class _SlowMainProcessState:
    """A ``Stateful`` that makes the main process reach the checkpoint collectives later than the other ranks."""

    def __init__(self, dist_ctx: DistributedContext, delay_seconds: float):
        self._dist_ctx = dist_ctx
        self._delay_seconds = delay_seconds
        self.value = torch.zeros(4)

    def state_dict(self) -> dict[str, Any]:
        if self._dist_ctx.is_main_process:
            time.sleep(self._delay_seconds)
        return {"value": self.value}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.value.copy_(state_dict["value"])


def _build_checkpointer(dist_ctx, save_dir, timeout_manager) -> tuple[StateCheckpointer, JobSchedule]:
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=1), stream=_FakeStream())
    checkpointer = StateCheckpointer(
        dist_context=dist_ctx,
        schedule=schedule,
        config=CheckpointingConfig(save_dir=save_dir, period_steps=1),
        gc=ManualGarbageCollector(GarbageCollectionConfig(period_steps=StepActionSpecial.disable), schedule),
        timeout_manager=timeout_manager,
        run_name=None,
    )
    return checkpointer, schedule


@pytest.mark.distributed
def test_checkpoint_round_trip_inside_inference_mode(dist_ctx_factory, shared_tmp_dir):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8)))
    model = nn.Linear(16, 16, device="cuda")
    parallelize_fsdp(model, dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_CP_SHARD])
    expected = {name: param.full_tensor() for name, param in model.named_parameters()}

    timeout_manager = TimeoutManager(dist_context=dist_ctx, config=TimeoutConfig())
    timeout_manager.set_init()
    checkpointer, schedule = _build_checkpointer(dist_ctx, shared_tmp_dir, timeout_manager)
    state = TrackedModules(dist_ctx, {0: model}, lambda _name, _value: True)

    # The inference loop runs with its checkpoints inside inference mode.
    schedule.step()
    with torch.inference_mode():
        checkpointer.checkpoint_if_needed(state)

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    with torch.inference_mode():
        checkpointer.load_last_checkpoint(state)

    for name, param in model.named_parameters():
        assert torch.equal(param.full_tensor(), expected[name])


@pytest.mark.distributed
def test_checkpoint_io_outlasts_step_timeout(dist_ctx_factory, shared_tmp_dir):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))
    timeout_manager = TimeoutManager(
        dist_context=dist_ctx,
        config=TimeoutConfig(init_timeout=int(default_pg_nccl_timeout.total_seconds()), step_timeout=1),
    )
    timeout_manager.set_init()
    timeout_manager.set_periodic()
    checkpointer, schedule = _build_checkpointer(dist_ctx, shared_tmp_dir, timeout_manager)

    # The main process reaches the save and the load collectives later than the step timeout allows.
    state = _SlowMainProcessState(dist_ctx, delay_seconds=5.0)
    state.value.fill_(7.0)
    try:
        schedule.step()
        checkpointer.checkpoint_if_needed(state)

        state.value.zero_()
        checkpointer.load_last_checkpoint(state)
    finally:
        # set_timeout also changes the default process group, which every later test in the session shares.
        dist_ctx.set_timeout(default_pg_nccl_timeout.total_seconds())

    torch.testing.assert_close(state.value, torch.full((4,), 7.0))
