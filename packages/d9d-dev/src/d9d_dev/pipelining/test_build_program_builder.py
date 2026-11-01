import pytest
from d9d.pipelining.factory import (
    PipelineSchedule1F1BConfig,
    PipelineScheduleDualPipeVConfig,
    PipelineScheduleGPipeConfig,
    PipelineScheduleInferenceConfig,
    PipelineScheduleLoopedBFSConfig,
    PipelineScheduleZeroBubbleVConfig,
    build_program_builder,
)
from d9d.pipelining.program.catalogue import (
    DualPipeVPipelineProgramBuilder,
    Interleaved1F1BPipelineProgramBuilder,
    LoopedBFSPipelineProgramBuilder,
    ZeroBubbleVPipelineProgramBuilder,
)


@pytest.mark.local
@pytest.mark.parametrize(
    ("config", "builder_cls", "num_stages_per_rank", "has_backward"),
    [
        (PipelineScheduleInferenceConfig(), LoopedBFSPipelineProgramBuilder, 1, False),
        (PipelineScheduleGPipeConfig(), LoopedBFSPipelineProgramBuilder, 1, True),
        (PipelineScheduleLoopedBFSConfig(num_stages_per_rank=3), LoopedBFSPipelineProgramBuilder, 3, True),
        (
            PipelineSchedule1F1BConfig(num_stages_per_rank=2, zero_bubble=True),
            Interleaved1F1BPipelineProgramBuilder,
            2,
            True,
        ),
        (PipelineScheduleDualPipeVConfig(), DualPipeVPipelineProgramBuilder, 2, True),
        (PipelineScheduleZeroBubbleVConfig(), ZeroBubbleVPipelineProgramBuilder, 2, True),
    ],
)
def test_build_program_builder(config, builder_cls, num_stages_per_rank, has_backward):
    builder = build_program_builder(config)

    assert isinstance(builder, builder_cls)
    assert builder.num_stages_per_rank == num_stages_per_rank
    assert builder.has_backward == has_backward
