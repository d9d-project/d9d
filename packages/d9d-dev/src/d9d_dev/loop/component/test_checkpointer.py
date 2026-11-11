import pytest
import torch
from d9d.dist_context import DENSE_DOMAIN, DenseDim, DenseDomainConfig, DeviceMeshParameters
from d9d.loop.component import TrackedModules, build_parameter_runtime
from d9d.loop.component.checkpointer import StateCheckpointer
from d9d.loop.component.garbage_collector import ManualGarbageCollector
from d9d.loop.component.job_schedule import JobSchedule
from d9d.loop.config import (
    CheckpointingConfig,
    GarbageCollectionConfig,
    JobScheduleConfig,
    ParameterRuntimeConfig,
    StepActionSpecial,
)
from d9d.parallel import parallelize_fsdp
from torch import nn


class _FakeStream:
    @property
    def total_steps(self) -> int | None:
        return None


@pytest.mark.distributed
def test_checkpoint_round_trip_inside_inference_mode(dist_ctx_factory, shared_tmp_dir):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8)))
    model = nn.Linear(16, 16, device="cuda")
    parallelize_fsdp(model, dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_CP_SHARD])
    expected = {name: param.full_tensor() for name, param in model.named_parameters()}

    schedule = JobSchedule(config=JobScheduleConfig(total_steps=1), stream=_FakeStream())
    checkpointer = StateCheckpointer(
        dist_context=dist_ctx,
        schedule=schedule,
        config=CheckpointingConfig(save_dir=shared_tmp_dir, period_steps=1),
        gc=ManualGarbageCollector(GarbageCollectionConfig(period_steps=StepActionSpecial.disable), schedule),
        run_name=None,
    )
    state = TrackedModules(
        dist_ctx, {0: model}, lambda _name, _value: True, build_parameter_runtime(ParameterRuntimeConfig())
    )

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
