import dataclasses
import typing
from collections.abc import Sequence
from typing import Any

from torch import nn


@dataclasses.dataclass(frozen=True, slots=True)
class StageSegment:
    """One unit of a pipeline stage that runs on its own, such as a decoder layer.

    A stage is a chain of segments. The parameter runtime can cut the autograd graph between them, so every tensor
    with a gradient that a later segment needs must pass through the activations that ``module`` returns.

    Attributes:
        name: The name of the segment, unique within the stage.
        module: Receives the activations of the previous segment and the constants of the microbatch, and returns
            the next activations. It owns every parameter that the segment uses.
    """

    name: str
    module: nn.Module


@typing.runtime_checkable
class ModuleSupportsSegments(typing.Protocol):
    """Protocol for pipeline stage modules that describe their forward pass as a chain of segments."""

    def prepare_segment_constants(self, shared: Any) -> Any:
        """Builds the constants of one microbatch, which every segment receives.

        Args:
            shared: The value broadcast to every stage.

        Returns:
            The constants. They must not contain tensors with a gradient.
        """
        ...

    def stage_segments(self) -> Sequence[StageSegment]:
        """Returns the segments of this stage, in forward order.

        After materialization, every call returns segments with the same names, order and parameter identities.
        """
        ...


class _WholeModule:
    """A stage module without segments, which runs as one segment."""

    def __init__(self, module: nn.Module):
        self._segments = (StageSegment(name="stage", module=module),)

    def prepare_segment_constants(self, shared: Any) -> Any:
        return shared

    def stage_segments(self) -> Sequence[StageSegment]:
        return self._segments


def as_segments(module: nn.Module) -> ModuleSupportsSegments:
    """Returns a stage module as segments.

    Args:
        module: The stage module.

    Returns:
        The module itself if it describes its segments, otherwise one segment with the whole module.
    """
    return module if isinstance(module, ModuleSupportsSegments) else _WholeModule(module)


def run_segments(module: ModuleSupportsSegments, inputs: Any, shared: Any) -> Any:
    """Runs the segments of a stage in order, without cutting the autograd graph.

    A module that describes segments can use it as its ``forward``.

    Args:
        module: The stage module.
        inputs: The stage input.
        shared: The value broadcast to every stage.

    Returns:
        The stage output.
    """
    constants = module.prepare_segment_constants(shared)
    activations = inputs
    for segment in module.stage_segments():
        activations = segment.module(activations, constants)
    return activations
