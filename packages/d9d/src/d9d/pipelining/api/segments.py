import dataclasses
import typing
from collections.abc import Callable, Sequence
from typing import Any

from torch import nn


@dataclasses.dataclass(frozen=True, slots=True)
class StageSegment:
    """One unit of a pipeline stage that the runtime prepares and runs on its own, such as a decoder layer.

    A stage is a chain of segments. The autograd graph is cut between them, so every tensor with a gradient that
    a later segment needs must pass through the activations that ``call`` returns.

    Attributes:
        name: The name of the segment, unique within the stage.
        modules: The modules that hold the parameters of the segment. A segment without parameters has none.
        call: Maps the activations of the previous segment and the constants of the microbatch to the next
            activations.
    """

    name: str
    modules: tuple[nn.Module, ...]
    call: Callable[[Any, Any], Any]


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
        """Returns the segments of this stage, in forward order. The list is the same on every call."""
        ...


class _WholeModule:
    """A stage module without segments, which runs as one segment."""

    def __init__(self, module: nn.Module):
        self._segments = (StageSegment(name="stage", modules=(module,), call=module),)

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
        activations = segment.call(activations, constants)
    return activations
