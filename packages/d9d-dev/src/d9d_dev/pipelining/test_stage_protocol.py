import pytest
import torch
from d9d.pipelining.api import PipelineStageInfo
from d9d.pipelining.infra.stage import PipelineStage
from torch import nn


class _PlainModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor, shared: None) -> torch.Tensor:
        return self.linear(inputs)


@pytest.mark.local
def test_single_stage_accepts_a_module_without_the_protocol():
    stage = PipelineStage(
        info=PipelineStageInfo(num_stages=1, current_stage=0),
        module=_PlainModule(),
        group=None,
        stage_to_host_topology={0: 0},
    )

    stage.configure_buffers(has_backward=True, pipeline_inputs_per_microbatch=(torch.empty(2, 4),))


# Stage 0 of an inference pipeline has no receiver, but it must fail like the other stages.
@pytest.mark.local
@pytest.mark.parametrize("current_stage", [0, 1])
def test_every_stage_of_a_pipeline_requires_the_protocol(current_stage: int):
    with pytest.raises(TypeError, match="stage_transfer_spec"):
        PipelineStage(
            info=PipelineStageInfo(num_stages=2, current_stage=current_stage),
            module=_PlainModule(),
            group=None,
            stage_to_host_topology={0: 0, 1: 1},
        )
