from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PipelineScheduleInferenceConfig(BaseModel):
    """Configuration for inference-only pipeline execution.

    This schedule runs all forward passes and no backward passes.

    Attributes:
        schedule: Discriminator field. Always ``"inference"``.
    """

    schedule: Literal["inference"] = "inference"


class PipelineScheduleGPipeConfig(BaseModel):
    """Configuration for GPipe execution.

    This schedule hosts one stage per rank. It runs the forward pass for all microbatches before it
    starts the backward pass.

    Attributes:
        schedule: Discriminator field. Always ``"gpipe"``.
    """

    schedule: Literal["gpipe"] = "gpipe"


class PipelineScheduleLoopedBFSConfig(BaseModel):
    """Configuration for Looped Breadth-First execution.

    This schedule works like GPipe but supports several stages per rank. It runs all work for one
    stage before it moves to the next.

    Attributes:
        schedule: Discriminator field. Always ``"looped_bfs"``.
        num_stages_per_rank: The number of stages hosted on each rank.
    """

    schedule: Literal["looped_bfs"] = "looped_bfs"

    num_stages_per_rank: int


class PipelineSchedule1F1BConfig(BaseModel):
    """Configuration for Interleaved 1F1B and Interleaved Zero Bubble execution.

    This schedule supports several stages per rank. With ``zero_bubble``, it splits the backward pass
    into input-gradient and weight-gradient parts to reduce pipeline bubbles.

    Attributes:
        schedule: Discriminator field. Always ``"1f1b"``.
        num_stages_per_rank: The number of stages hosted on each rank.
        zero_bubble: Whether to use the Interleaved Zero Bubble (ZB1P) variant.
    """

    schedule: Literal["1f1b"] = "1f1b"

    num_stages_per_rank: int
    zero_bubble: bool


class PipelineScheduleZeroBubbleVConfig(BaseModel):
    """Configuration for Zero Bubble V (ZBV) execution.

    This schedule places stages in a V shape and splits the backward pass into input-gradient and
    weight-gradient parts. It always hosts 2 stages per rank.

    Attributes:
        schedule: Discriminator field. Always ``"zero_bubble_v"``.
    """

    schedule: Literal["zero_bubble_v"] = "zero_bubble_v"


class PipelineScheduleDualPipeVConfig(BaseModel):
    """Configuration for DualPipeV execution.

    This bidirectional schedule places stages in a V shape and pairs the forward pass of one
    microbatch with the backward pass of another. It always hosts 2 stages per rank. The number of
    microbatches per step must be at least twice the pipeline-parallel size.

    Attributes:
        schedule: Discriminator field. Always ``"dual_pipe_v"``.
    """

    schedule: Literal["dual_pipe_v"] = "dual_pipe_v"


AnyPipelineScheduleConfig = Annotated[
    PipelineScheduleInferenceConfig
    | PipelineScheduleGPipeConfig
    | PipelineScheduleLoopedBFSConfig
    | PipelineSchedule1F1BConfig
    | PipelineScheduleZeroBubbleVConfig
    | PipelineScheduleDualPipeVConfig,
    Field(discriminator="schedule"),
]
"""Union of all supported pipeline schedule configuration types.

Pydantic selects the config class by the ``schedule`` field.
"""
