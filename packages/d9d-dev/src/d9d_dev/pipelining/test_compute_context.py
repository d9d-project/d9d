from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
from d9d.dist_context import BASE_DOMAIN, BaseDim, DeviceMeshParameters
from d9d.pipelining.api import PipelineComputeKind, PipelineStageInfo, PipelineStepHandler, mark_window, run_steps
from d9d.pipelining.factory import PipelineScheduleGPipeConfig, build_program_builder, build_schedule
from d9d.pipelining.runtime import (
    BackwardFullInputComputeAction,
    BackwardWeightComputeAction,
    ForwardComputeAction,
    OfflinePipelineExecutor,
)
from d9d.pipelining.runtime.action import ActionContext
from d9d.pipelining.runtime.callback import PipelineLossHandler
from d9d.rng import SeedScope, fork_rng, fork_rng_child
from torch import nn

from d9d_dev.parallel.runtime_step import make_runtime


class _Recorder(PipelineStepHandler[torch.Tensor]):
    def __init__(self):
        self.events = []

    @contextmanager
    def compute_context(self, kind: PipelineComputeKind, stage_idx: int, microbatch_idx: int):
        self.events.append(("enter", kind, stage_idx, microbatch_idx))
        yield
        self.events.append(("exit", kind, stage_idx, microbatch_idx))

    def process_outputs(self, outputs: torch.Tensor, microbatch_idx: int) -> torch.Tensor:
        self.events.append(("process_outputs", microbatch_idx))
        return outputs.sum()


class _FakeStage:
    def __init__(self, recorder: _Recorder):
        self.info = PipelineStageInfo(num_stages=1, current_stage=0)
        self._recorder = recorder

    def forward_one_chunk(self, microbatch_index, pipeline_inputs, pipeline_shared):
        self._recorder.events.append(("forward", microbatch_index))

    def get_pipeline_output(self, microbatch_index):
        return torch.ones(2)

    def backward_one_chunk(self, microbatch_index, loss, full_backward, final=None):
        self._recorder.events.append(("backward", microbatch_index))

    def backward_weight_one_chunk(self, microbatch_index, final=None):
        self._recorder.events.append(("backward_weight", microbatch_index))


@pytest.mark.local
def test_compute_actions_open_the_compute_context_without_process_outputs():
    recorder = _Recorder()
    ctx = ActionContext(
        pipeline_inputs_microbatches=(None,),
        pipeline_shared_microbatches=(None,),
        stages={0: _FakeStage(recorder)},
        communications=None,
        handler=recorder,
        callback=PipelineLossHandler(recorder),
    )

    ForwardComputeAction(stage_idx=0, microbatch_idx=0).apply(ctx)
    BackwardFullInputComputeAction(stage_idx=0, microbatch_idx=0, full_backward=False).apply(ctx)
    BackwardWeightComputeAction(stage_idx=0, microbatch_idx=0).apply(ctx)

    assert recorder.events == [
        ("enter", PipelineComputeKind.forward, 0, 0),
        ("forward", 0),
        ("exit", PipelineComputeKind.forward, 0, 0),
        ("process_outputs", 0),
        ("enter", PipelineComputeKind.backward, 0, 0),
        ("backward", 0),
        ("exit", PipelineComputeKind.backward, 0, 0),
        ("enter", PipelineComputeKind.backward_weight, 0, 0),
        ("backward_weight", 0),
        ("exit", PipelineComputeKind.backward_weight, 0, 0),
    ]


@pytest.mark.local
def test_offline_executor_opens_the_compute_context_without_process_outputs():
    recorder = _Recorder()
    weight = nn.Parameter(torch.ones(2))
    weight.register_hook(lambda grad: recorder.events.append(("backward",)))

    def _model(inputs: torch.Tensor, shared: None) -> torch.Tensor:
        recorder.events.append(("forward",))
        return inputs * weight

    schedule = OfflinePipelineExecutor(model=_model, do_backward=True)
    schedule.step(
        inputs_microbatches=(torch.ones(2), torch.ones(2)),
        shared_microbatches=(None, None),
        handler=recorder,
    )

    assert recorder.events == [
        event
        for microbatch_idx in range(2)
        for event in [
            ("enter", PipelineComputeKind.forward, 0, microbatch_idx),
            ("forward",),
            ("exit", PipelineComputeKind.forward, 0, microbatch_idx),
            ("process_outputs", microbatch_idx),
            ("enter", PipelineComputeKind.backward, 0, microbatch_idx),
            ("backward",),
            ("exit", PipelineComputeKind.backward, 0, microbatch_idx),
        ]
    ]


class _DropoutLayer(nn.Linear):
    def forward(self, activations, constants):
        return F.dropout(super().forward(activations), 0.5)


class _DropoutChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(mark_window(_DropoutLayer(8, 8, bias=False, device="cuda")) for _ in range(3))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    async def forward_steps(self, steps, inputs, shared):
        for layer in self.layers:
            inputs = await steps.call(layer, inputs, shared)
        return inputs

    def forward(self, inputs, shared):
        return run_steps(self, inputs, shared)


class _ForkingHandler(PipelineStepHandler[torch.Tensor]):
    def __init__(self):
        self.outputs = {}

    def compute_context(self, kind: PipelineComputeKind, stage_idx: int, microbatch_idx: int):
        return fork_rng_child(kind.value, stage_idx, microbatch_idx)

    def process_outputs(self, outputs: torch.Tensor, microbatch_idx: int) -> torch.Tensor:
        self.outputs[microbatch_idx] = outputs.detach().clone()
        return outputs.sum()


@pytest.mark.distributed
def test_window_draws_the_numbers_of_one_compute(dist_ctx_factory):
    # Without pipeline parallelism, the runtime runs one region of all microbatches before the next region.
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))
    model = _DropoutChain()
    runtime = make_runtime()
    schedule_info, _ = build_schedule(
        pp_mesh=dist_ctx.mesh_for(BASE_DOMAIN)[BaseDim.PP],
        program_builder=build_program_builder(PipelineScheduleGPipeConfig()),
        model_provider=lambda _stage: model,
        runtime=runtime,
    )
    inputs = torch.randn(4, 2, 8, device="cuda")
    handler = _ForkingHandler()
    scope = SeedScope.root(0)
    runtime.bind()
    try:
        runtime.zero_grad()
        with fork_rng(scope):
            schedule_info.schedule.step(tuple(inputs), (None,) * len(inputs), handler)
        runtime.wait()
    finally:
        runtime.unbind()

    for microbatch, microbatch_inputs in enumerate(inputs):
        with torch.no_grad(), fork_rng(scope.child("forward", 0, microbatch)):
            expected = model(microbatch_inputs, None)
        assert torch.equal(handler.outputs[microbatch], expected)
