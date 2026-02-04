from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PipelineScheduleInferenceConfig(BaseModel):
    """
    Configuration for inference-only pipeline execution.

    This schedule runs all forward passes sequentially without any backward passes.
    """

    schedule: Literal["inference"] = "inference"


class PipelineScheduleGPipeConfig(BaseModel):
    """
    Configuration for GPipe execution.

    This assumes a single stage per rank and processes all microbatches for the
    forward pass before switching to the backward pass.
    """

    schedule: Literal["gpipe"] = "gpipe"


class PipelineScheduleLoopedBFSConfig(BaseModel):
    """
    Configuration for Looped Breadth-First Search execution.

    Similar to GPipe, but supports multiple stages per rank (virtualization).
    It executes all available work for a specific stage before moving to the next.
    """

    schedule: Literal["looped_bfs"] = "looped_bfs"

    num_stages_per_rank: int


class PipelineSchedule1F1BConfig(BaseModel):
    """
    Configuration for Interleaved 1F1B and Interleaved Zero Bubble execution.

    Supports assigning multiple stages per rank and sharding backward to dI and dW
    to reduce pipeline bubbles.
    """

    schedule: Literal["1f1b"] = "1f1b"

    num_stages_per_rank: int
    zero_bubble: bool


class PipelineScheduleZeroBubbleVConfig(BaseModel):
    """
    Configuration for Zero Bubble V (ZBV) execution.

    A specialized V-shape topology schedule that splits backward passes into
    Input and Weight gradients to maximize overlap. Requires exactly 2 stages per rank.
    """
    schedule: Literal["zero_bubble_v"] = "zero_bubble_v"


class PipelineScheduleDualPipeVConfig(BaseModel):
    """
    Configuration for DualPipeV execution.

    A bidirectional pipeline schedule for high-throughput training, utilizing
    V-shape topology and reciprocal forward/backward scheduling.
    """

    schedule: Literal["dual_pipe_v"] = "dual_pipe_v"


AnyPipelineScheduleConfig = Annotated[
    PipelineScheduleInferenceConfig |
    PipelineScheduleGPipeConfig |
    PipelineScheduleLoopedBFSConfig |
    PipelineSchedule1F1BConfig |
    PipelineScheduleZeroBubbleVConfig |
    PipelineScheduleDualPipeVConfig,
    Field(discriminator="schedule")
]
"""Union of all supported pipeline schedule configuration types.

This type alias uses a Pydantic discriminator on the ``schedule`` field to allow
polymorphic validation and serialization of specific schedule configs (e.g.
Inference, GPipe, 1F1B, ZeroBubble, etc.).
"""
