import pytest
import torch
from d9d.dist_context import DeviceMeshParameters
from d9d.loop.component import PipelineStateHandler
from d9d.loop.component.task_operator.common import build_pipeline_microbatch_inputs
from d9d.loop.component.task_operator.inference import InferenceStepHandler
from d9d.loop.control import (
    BaseTask,
    BuildForwardInputsContext,
    BuildForwardInputsResult,
    InferenceTask,
    ProcessOutputsContext,
)
from d9d.pipelining.api import PipelineComputeKind
from d9d.rng import SeedScope, fork_rng


class _DrawingTask(BaseTask):
    def build_forward_inputs(self, ctx: BuildForwardInputsContext) -> BuildForwardInputsResult:
        return BuildForwardInputsResult(input=torch.rand(4, device="cuda"), shared=None, state=None)


@pytest.mark.local
def test_build_forward_inputs_draws_from_its_seed_scope():
    root = SeedScope.root(0)

    # Draws before the call must not change the numbers of the call.
    torch.rand(16, device="cuda")
    with fork_rng(root):
        inputs, _ = build_pipeline_microbatch_inputs(
            task=_DrawingTask(), pipeline_state=PipelineStateHandler(), pack=["first", "second"]
        )

    for microbatch_idx, actual in enumerate(inputs):
        with fork_rng(root.child("build_forward_inputs", microbatch_idx)):
            expected = torch.rand(4, device="cuda")
        assert torch.equal(actual, expected)


class _DrawingInferenceTask(InferenceTask):
    def __init__(self):
        self.drawn = []

    def build_forward_inputs(self, ctx: BuildForwardInputsContext) -> BuildForwardInputsResult:
        raise NotImplementedError

    def process_outputs(self, ctx: ProcessOutputsContext) -> None:
        self.drawn.append(torch.rand(4, device="cuda"))


@pytest.mark.local
def test_model_compute_draws_from_its_cp_rank_stage_and_microbatch(dist_ctx_factory):
    scope = SeedScope.root(0)
    handler = InferenceStepHandler(
        dist_context=dist_ctx_factory(DeviceMeshParameters()),
        state=PipelineStateHandler(),
        task=_DrawingInferenceTask(),
    )

    with fork_rng(scope), handler.compute_context(PipelineComputeKind.forward, 1, 2):
        actual = torch.rand(4, device="cuda")

    with fork_rng(scope.child("cp", 0, "forward", 1, 2)):
        expected = torch.rand(4, device="cuda")
    assert torch.equal(actual, expected)


@pytest.mark.local
def test_process_outputs_draws_from_the_current_scope(dist_ctx_factory):
    scope = SeedScope.root(0)
    task = _DrawingInferenceTask()
    state = PipelineStateHandler()
    state.store(3, None)
    handler = InferenceStepHandler(dist_context=dist_ctx_factory(DeviceMeshParameters()), state=state, task=task)

    with fork_rng(scope):
        handler.process_outputs(outputs=None, microbatch_idx=3)

    with fork_rng(scope.child("process_outputs", 3)):
        expected = torch.rand(4, device="cuda")
    assert torch.equal(task.drawn[0], expected)
