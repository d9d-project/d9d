from contextlib import contextmanager

import pytest
import torch
from d9d.pipelining.api import PipelineComputeKind, PipelineStageInfo, PipelineStepHandler
from d9d.pipelining.runtime import (
    BackwardFullInputComputeAction,
    BackwardWeightComputeAction,
    ForwardComputeAction,
    OfflinePipelineExecutor,
)
from d9d.pipelining.runtime.action import ActionContext
from d9d.pipelining.runtime.callback import PipelineLossHandler
from torch import nn


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

    def backward_one_chunk(self, microbatch_index, loss, full_backward):
        self._recorder.events.append(("backward", microbatch_index))

    def backward_weight_one_chunk(self, microbatch_index):
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
