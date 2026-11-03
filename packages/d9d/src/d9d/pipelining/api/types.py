import abc
from contextlib import AbstractContextManager, nullcontext
from enum import StrEnum
from typing import Generic, TypeVar

import torch

TPipelineInput = TypeVar("TPipelineInput")
TStageTransfer = TypeVar("TStageTransfer")
TSharedInput = TypeVar("TSharedInput")
TPipelineOutput = TypeVar("TPipelineOutput")


class PipelineComputeKind(StrEnum):
    """Kinds of model compute that a pipeline schedule runs for one stage and one microbatch.

    Attributes:
        forward: The forward pass.
        backward: The backward pass for the inputs, and for the weights unless the schedule defers them.
        backward_weight: The deferred backward pass for the weights.
    """

    forward = "forward"
    backward = "backward"
    backward_weight = "backward_weight"


class PipelineStepHandler(abc.ABC, Generic[TPipelineOutput]):
    """Abstract base class for the code that a pipeline step runs around its model compute and on its outputs."""

    def compute_context(
        self, kind: PipelineComputeKind, stage_idx: int, microbatch_idx: int
    ) -> AbstractContextManager[None]:
        """Returns the context to open around one model compute of a stage on a microbatch.

        The context covers the stage module and its autograd graph, but not ``process_outputs()`` and not the
        communications. The default context does nothing.

        Args:
            kind: The kind of the compute.
            stage_idx: The global index of the pipeline stage.
            microbatch_idx: The index of the microbatch.

        Returns:
            The context manager to open while the stage computes.
        """
        return nullcontext()

    @abc.abstractmethod
    def process_outputs(self, outputs: TPipelineOutput, microbatch_idx: int) -> torch.Tensor | None:
        """Handles the ``PipelineOutput`` of the last stage for one microbatch.

        Args:
            outputs: The ``PipelineOutput`` produced by the last stage.
            microbatch_idx: The index of the microbatch.

        Returns:
            The loss of the microbatch if the schedule runs the backward pass, otherwise ``None``.
        """
        ...
